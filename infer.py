import os
import argparse
import pandas as pd
import numpy as np
import pickle
import time
import json
from feature_extraction import process_audio_files
import merge_batches
from sklearn.base import BaseEstimator, ClassifierMixin

# Global dictionary to store timing information
stage_timing = {}

def time_stage(stage_name):
    """Decorator to time a function and add its execution time to stage_timing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Only measure time if enabled
            if os.environ.get("MEASURE_STAGE_TIMING", "0") == "1":
                start_time = time.time()
                
                result = func(*args, **kwargs)
                
                elapsed = time.time() - start_time
                stage_timing[stage_name] = elapsed
                print(f"Stage '{stage_name}' completed in {elapsed:.2f} seconds")
                
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

class GenderAgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gender_model=None, age_model=None, selected_features=None):
        self.gender_model = gender_model
        self.age_model = age_model
        self.selected_features = selected_features
        
    def fit(self, X, y=None):
        # This method is just a placeholder since we're using pre-trained models
        return self
    
    def predict(self, X):
        # Ensure we're only using the selected features
        if self.selected_features is not None:
            if isinstance(X, pd.DataFrame):
                # Make sure all required features exist, fill missing ones with 0
                for feature in self.selected_features:
                    if feature not in X.columns:
                        X[feature] = 0
                X = X[self.selected_features]
        
        # Get gender and age predictions
        gender_preds = self.gender_model.predict(X)
        age_preds = self.age_model.predict(X)
        
        # Initialize output predictions
        final_preds = np.zeros(len(X), dtype=int)
        
        # Apply the conditional logic
        final_preds[(gender_preds == 'male') & (age_preds == 'twenties')] = 0
        final_preds[(gender_preds == 'female') & (age_preds == 'twenties')] = 1
        final_preds[(gender_preds == 'male') & (age_preds == 'fifties')] = 2
        final_preds[(gender_preds == 'female') & (age_preds == 'fifties')] = 3
        
        return final_preds
    
def parse_args():
    parser = argparse.ArgumentParser(description="Audio classification inference pipeline")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pkl file)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save prediction results")
    parser.add_argument("--features_csv", type=str, default="extracted_features.csv", 
                        help="Path to save extracted features CSV")
    return parser.parse_args()

@time_stage("Loading model")
def load_model(model_path):
    """
    Load the pickled model from the given path
    
    Args:
        model_path: Path to the pickled model file
        
    Returns:
        Loaded model object
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@time_stage("Making predictions")
def predict_labels(model, features_df):
    """
    Run prediction on the extracted features
    
    Args:
        model: Loaded model for prediction
        features_df: DataFrame containing features
    
    Returns:
        Dictionary mapping filenames to predicted labels
    """
    # Remove non-feature columns for prediction
    non_feature_cols = ['filename', 'path', 'file_name']
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
    
    # Make sure we have the right feature columns
    X = features_df[feature_cols]
    
    # Map class indices to descriptive labels
    class_labels = {
        0: "Male twenties",
        1: "Female twenties",
        2: "Male fifties",
        3: "Female fifties"
    }
    
    try:
        # Make predictions
        predicted_indices = model.predict(X)
        
        # Convert numerical predictions to descriptive labels
        predicted_labels = [class_labels[idx] for idx in predicted_indices]
        
        # Create a dictionary mapping filenames to predicted labels
        # Use available identifier column (filename, path, or file_name)
        id_col = next((col for col in non_feature_cols if col in features_df.columns), None)
        
        if id_col:
            filenames = features_df[id_col]
        else:
            filenames = [f"sample_{i}" for i in range(len(predicted_labels))]
            
        results = dict(zip(filenames, predicted_labels))
        
        return results
    
    except Exception as e:
        print(f"Prediction error: {e}")
        # For debugging
        print(f"Feature columns: {feature_cols}")
        print(f"First few rows of input features: {X.head()}")
        if hasattr(model, 'selected_features'):
            print(f"Model's selected features: {model.selected_features}")
        raise

@time_stage("Saving results")
def save_results(results, output_dir):
    """
    Save prediction results to text files
    
    Args:
        results: Dictionary mapping filenames to predicted labels
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all results to a single txt file
    with open(os.path.join(output_dir, "all_predictions.txt"), "w") as f:
        for filename, label in results.items():
            f.write(f"{filename}: {label}\n")
    
    print(f"Saved prediction results to {output_dir}")

def main():
    start_time = time.time()
    args = parse_args()
    
    # Step 1: Extract features
    print("Extracting features from audio files...")
    extract_start = time.time()
    process_audio_files(args.audio_dir, args.features_csv)
    extract_time = time.time() - extract_start
    stage_timing["Feature extraction"] = extract_time
    print(f"Features extracted and saved to {args.features_csv} in {extract_time:.2f} seconds")
    
    # Step 2: Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Step 3: Merge feature batches
    merge_start = time.time()
    merge_batches.merge_batch_files()
    merge_time = time.time() - merge_start
    stage_timing["Merging batch files"] = merge_time
    print(f"Merged batch files in {merge_time:.2f} seconds")
    
    # Step 4: Load features CSV
    load_features_start = time.time()
    features_df = pd.read_csv(args.features_csv+"/features.csv")
    load_features_time = time.time() - load_features_start
    stage_timing["Loading features CSV"] = load_features_time
    print(f"Loaded features CSV in {load_features_time:.2f} seconds")
    
    # Step 5: Run predictions
    print("Running predictions...")
    results = predict_labels(model, features_df)
    
    # Step 6: Save results
    print("Saving results...")
    save_results(results, args.output_dir)
    
    # Save timing information
    if os.environ.get("MEASURE_STAGE_TIMING", "0") == "1":
        stage_timing["Total execution time"] = time.time() - start_time
        timing_path = os.path.join(args.output_dir, "stage_timing.json")
        with open(timing_path, 'w') as f:
            json.dump(stage_timing, f, indent=2)
        print(f"Stage timing information saved to {timing_path}")
    
    print("Inference complete!")

if __name__ == "__main__":
    main()
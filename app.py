import gradio as gr
import os
import tempfile
import pandas as pd
import numpy as np
import pickle
from feature_extraction import process_file
import librosa

# First define the GenderAgeClassifier class before loading the model
class GenderAgeClassifier:
    def __init__(self, gender_model=None, age_model=None, selected_features=None):
        self.gender_model = gender_model
        self.age_model = age_model
        self.selected_features = selected_features
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        # Ensure we're only using the selected features
        if self.selected_features is not None:
            if isinstance(X, pd.DataFrame):
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

# Now load the model
model_path = "models/combined_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Class labels
class_labels = {
    0: "Male twenties",
    1: "Female twenties",
    2: "Male fifties",
    3: "Female fifties"
}

def predict(audio_file):
    """Process the uploaded audio file and make a prediction"""
    try:
        # Process the audio file
        features = process_file(audio_file)
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Make sure non-feature columns are excluded
        non_feature_cols = ['filename', 'path', 'file_name']
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
        
        # Get prediction
        X = features_df[feature_cols]
        predicted_idx = model.predict(X)[0]
        predicted_label = class_labels[predicted_idx]
        
        return predicted_label
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
description = """
# Audio Gender and Age Classifier

This model classifies audio recordings into four categories based on the speaker's gender and age:
- Male in their twenties
- Female in their twenties
- Male in their fifties
- Female in their fifties

Upload an MP3 audio file to get a prediction.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio File (MP3)"),
    outputs=gr.Text(label="Predicted Speaker Category"),
    title="Audio Gender and Age Classifier",
    description=description,
    examples=[
        # Optional: Include example audio files
    ]
)

if __name__ == "__main__":
    demo.launch()
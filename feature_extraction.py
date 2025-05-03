# In audio_feature_extractor.py
import pandas as pd
import numpy as np
import os
import librosa
from features_computation import *
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_file(path):
    """Process a single audio file and extract features."""
    try:
        signal, sample_rate = librosa.load(path, sr=None)
        features = extract_all_features(signal, sample_rate)
        features["path"] = os.path.basename(path)
        
        # Add file name (without extension) as a feature
        file_name = os.path.splitext(os.path.basename(path))[0]
        features["file_name"] = file_name
        return features
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def process_audio_files(audio_folder, output_folder, batch_size=10000, max_workers=None):
    """Process audio files in parallel and save features in batches."""
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get audio files
    file_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.mp3')]
    
    total_files = len(file_paths)
    print(f"Found {total_files} audio files to process")
    
    # Process files in parallel using ThreadPoolExecutor
    audio_features_list = []
    processed_count = 0
    batch_num = 1
    
    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Execute process_file on all files in parallel with progress bar
        for features in tqdm(
            executor.map(process_file, file_paths), 
            total=total_files, 
            desc="Processing audio files"
        ):
            if features is not None:
                audio_features_list.append(features)
            
            processed_count += 1
            
            # Save every batch_size files
            if processed_count % batch_size == 0:
                if audio_features_list:
                    audio_df = pd.DataFrame(audio_features_list)
                    save_path = os.path.join(output_folder, f'audio_features_batch_{batch_num}.csv')
                    audio_df.to_csv(save_path, index=False)
                    print(f"Saved batch {batch_num} to {save_path}")
                    
                    audio_features_list = []  # Clear for next batch
                    batch_num += 1

    # Save any remaining files after processing all files
    if audio_features_list:
        audio_df = pd.DataFrame(audio_features_list)
        save_path = os.path.join(output_folder, f'audio_features_batch_{batch_num}.csv')
        audio_df.to_csv(save_path, index=False)
        print(f"Saved final batch {batch_num} to {save_path}")
    
    return f"Processed {processed_count} audio files in {batch_num} batches"

if __name__ == "__main__":
    # Default paths inside the container
    audio_folder = '/audio'
    output_folder = '/output'
    # You can adjust max_workers based on your CPU cores
    # If None, ThreadPoolExecutor will use a default based on CPU count
    process_audio_files(audio_folder, output_folder, max_workers=None)
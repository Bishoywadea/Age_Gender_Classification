import pandas as pd
import glob
import os

def merge_batch_files():
    print("Starting batch file merging...")
    
    # Path to batch files
    batch_files = glob.glob('./output/audio_features_batch_*.csv')
    
    if not batch_files:
        print("No batch files found in /output directory!")
        print(f"Files in /output: {os.listdir('/output')}")
        return False
    
    print(f"Found {len(batch_files)} batch files to merge.")
    
    # Read and combine all batch files
    dfs = []
    for file in batch_files:
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all dataframes
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save the merged dataframe
        output_path = './output/features.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully merged {len(dfs)} files into {output_path}")
        print(f"Merged file contains {len(merged_df)} rows and {len(merged_df.columns)} columns")
        
        # Also create a copy in the working directory for the notebook
        merged_df.to_csv('features.csv', index=False)
        print(f"Also created a copy at ./features.csv")
        
        return True
    else:
        print("No valid CSV files found to merge.")
        return False

if __name__ == "__main__":
    merge_batch_files()
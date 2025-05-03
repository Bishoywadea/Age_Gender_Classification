import subprocess
import time
import os
import argparse
import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with time measurement")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pkl file)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save prediction results")
    parser.add_argument("--features_csv", type=str, default="extracted_features.csv", 
                        help="Path to save extracted features CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Starting inference at {start_datetime}")
    
    # Set up environment variable to enable timing
    env = os.environ.copy()
    env["MEASURE_STAGE_TIMING"] = "1"
    
    # Run the inference script
    command = [
        "python", "infer.py",
        "--audio_dir", args.audio_dir,
        "--model_path", args.model_path,
        "--output_dir", args.output_dir,
        "--features_csv", args.features_csv
    ]
    
    # Execute the command and capture output
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # Get any remaining output
        stdout, stderr = process.communicate()
        
        if stdout:
            print(stdout.strip())
        
        if stderr:
            print("ERRORS:")
            print(stderr.strip())
            
        return_code = process.poll()
        
    except Exception as e:
        print(f"Error running inference script: {e}")
        return_code = -1
    
    # Record end time
    end_time = time.time()
    end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format the time
    time_str = f"{int(hours):02}:{int(minutes):02}:{seconds:.2f}"
    
    # Try to read the stage timing data if it exists
    stage_timing_path = os.path.join(args.output_dir, "stage_timing.json")
    stage_timing = {}
    if os.path.exists(stage_timing_path):
        try:
            with open(stage_timing_path, 'r') as f:
                stage_timing = json.load(f)
        except:
            print("Could not read stage timing data")
    
    # Save timing information to file
    time_output_path = os.path.join(args.output_dir, "time.txt")
    with open(time_output_path, "w") as f:
        f.write(f"Inference started at: {start_datetime}\n")
        f.write(f"Inference completed at: {end_datetime}\n")
        f.write(f"Total execution time: {time_str} (HH:MM:SS.ms)\n")
        f.write(f"Total seconds: {elapsed_time:.2f}\n")
        
        # Add detailed stage timing if available
        if stage_timing:
            f.write("\n--- Stage-by-Stage Timing ---\n")
            for stage, stage_time in stage_timing.items():
                stage_hours, stage_remainder = divmod(stage_time, 3600)
                stage_minutes, stage_seconds = divmod(stage_remainder, 60)
                stage_time_str = f"{int(stage_hours):02}:{int(stage_minutes):02}:{stage_seconds:.2f}"
                f.write(f"{stage}: {stage_time_str} ({stage_time:.2f} seconds) - {(stage_time/elapsed_time)*100:.1f}% of total time\n")
                
        f.write(f"\nExit code: {return_code}\n")
    
    print(f"\nInference completed in {time_str}")
    print(f"Timing information saved to {time_output_path}")

if __name__ == "__main__":
    main()
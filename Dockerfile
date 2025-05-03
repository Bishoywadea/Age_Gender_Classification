FROM python:3.11-slim

WORKDIR /infer

# Copy requirements first (if you have one)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your scripts and model
COPY external_infer.py .
COPY infer.py .
COPY feature_extraction.py .
COPY features_computation.py .
COPY merge_batches.py .

COPY models/combined_model.pkl ./models/

# Create directories
RUN mkdir -p /infer/data /infer/results /output


# Set the entrypoint
ENTRYPOINT ["python", "external_infer.py"]
# Default command (can be overridden)
CMD ["--audio_dir", "./data", "--model_path", "./models/combined_model.pkl", "--output_dir", "./results", "--features_csv", "./output"]
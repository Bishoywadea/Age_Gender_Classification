# Audio Classification Project

## Overview
This project implements an audio classification pipeline that extracts features from audio files, preprocesses the data, trains machine learning models, and evaluates their performance. The pipeline uses XGBoost as the primary classification algorithm after comparison with other methods.

## Project Pipeline

### Preprocessing Module
- **Data Balancing**: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
- **Dimensionality Reduction**: Applied PCA to MFCC features to reduce dimensionality while preserving information
- **Data Visualization**: Created various plots to visualize data distributions and relationships
- **Failed Approaches**: Tested undersampling and oversampling techniques which were less effective than SMOTE

### Feature Extraction/Selection Module
- **Audio Feature Extraction**: Used the Librosa library to extract audio features including:
  - MFCC (Mel-frequency cepstral coefficients)
  - Spectral features
  - Temporal features
- **Feature Selection**: Implemented RFECV (Recursive Feature Elimination with Cross-Validation) to identify the most relevant features

### Model Selection/Training Module
- **Primary Model**: XGBoost was selected after comparing multiple models
- **Alternative Models**: Tested KNN and Random Forest classifiers
- **Performance**: XGBoost significantly outperformed other models (which achieved ~64% accuracy)

### Performance Analysis Module
- **Metrics**: Evaluated models using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- **Validation**: Assessed performance on both training and test datasets to check for overfitting

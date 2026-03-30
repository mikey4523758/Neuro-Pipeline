# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Define the EEGAnalysisPipeline class, which serves
#          as the central orchestrator for the full EEG signal
#          processing workflow. Coordinates data ingestion,
#          preprocessing, feature extraction, model training,
#          and result saving through a single unified interface.
# ============================================================

import os
import json
from src.load_data import fetch_physionet_data, get_epochs
from src.preprocess import apply_preprocessing
from src.features import extract_features
from src.models import train_and_evaluate
from src.visualize import plot_pipeline_results

class EEGAnalysisPipeline:
    """
    A unified interface for the EEG Signal Processing Pipeline.
    Encapsulates the workflow from raw data to model evaluation.
    """
    def __init__(self, config):
        # Store the full configuration dictionary for use across all pipeline stages
        self.config = config

        # Instance variables to hold intermediate and final pipeline outputs.
        # Initialized to None so any stage can be inspected after partial runs.
        self.raw_raw = None        # Original unprocessed Raw EEG object
        self.raw_processed = None  # Preprocessed Raw object (filtered, cleaned)
        self.epochs = None         # Segmented epochs time-locked to motor events
        self.features = None       # 2D feature matrix (n_epochs x n_features)
        self.labels = None         # Corresponding class labels for each epoch
        self.metrics = None        # Accuracy scores from model evaluation
        self.models = None         # Trained model objects (SVM, Random Forest)

    def run(self):
        """Executes the full end-to-end pipeline."""
        print("🚀 Starting Neural Signal Processing Pipeline...")

        # 1. Data Ingestion
        # Download and load raw EEG recordings for the configured subject and runs
        self.raw_raw = fetch_physionet_data(
            self.config['data']['subject'], 
            self.config['data']['runs']
        )

        # 2. Preprocessing
        # We copy to preserve the 'raw_raw' for comparative visualization
        # so the original signal remains available for before/after plots
        self.raw_processed = apply_preprocessing(self.raw_raw.copy(), self.config['preprocessing'])

        # Segment the preprocessed signal into epochs around motor imagery events
        self.epochs = get_epochs(
            self.raw_processed, 
            self.config['data']['tmin'], 
            self.config['data']['tmax']
        )

        # 3. Feature Extraction
        # Compute RMS and band power features from each epoch
        self.features, self.labels = extract_features(self.epochs, self.config['features'])

        # 4. Machine Learning
        # Train and evaluate SVM and Random Forest on the extracted features
        self.metrics, self.models = train_and_evaluate(
            self.features, 
            self.labels, 
            self.config['model']
        )

        # Return metrics so the caller can inspect results without accessing internals
        return self.metrics

    def save_results(self, output_dir="results"):
        """Saves metrics and plots to the specified directory."""

        # Build output subdirectory paths for metrics JSON and figure files
        metrics_path = os.path.join(output_dir, "metrics")
        figures_path = os.path.join(output_dir, "figures")
        
        # Create directories if they don't already exist
        os.makedirs(metrics_path, exist_ok=True)
        os.makedirs(figures_path, exist_ok=True)

        # Save Metrics
        # Serialize the accuracy results dict to a formatted JSON file
        with open(os.path.join(metrics_path, "performance.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

        # Generate and Save Plots
        # Pass both raw and processed signals so the visualizer can show
        # a before/after comparison alongside the model performance charts
        plot_pipeline_results(
            self.raw_raw, 
            self.raw_processed, 
            self.metrics, 
            figures_path
        )
        print(f"✅ Results successfully saved to {output_dir}")
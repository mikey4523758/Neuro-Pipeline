# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Entry point for the EEG Neural Signal Processing
#          Pipeline. Parses a command-line argument for the
#          config file path, loads the YAML configuration,
#          and orchestrates the full pipeline from data
#          ingestion through model evaluation and result saving.
# ============================================================

import yaml
import argparse
from src.pipeline import EEGAnalysisPipeline

def main():
    # Set up the argument parser so the config path can be specified
    # at runtime without modifying source code (e.g. python main.py --config configs/custom.yaml)
    parser = argparse.ArgumentParser(description="EEG Neural Signal Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    # yaml.safe_load parses the YAML file into a Python dict safely,
    # without executing any arbitrary code embedded in the file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize and run the pipeline object
    # All stage coordination (fetch, preprocess, extract, train) happens inside run()
    pipeline = EEGAnalysisPipeline(config)
    pipeline.run()

    # Persist accuracy metrics as JSON and save figures to the results directory
    pipeline.save_results()

# Standard Python guard ensuring main() is only called when this script
# is run directly, not when it's imported as a module elsewhere
if __name__ == "__main__":
    main()
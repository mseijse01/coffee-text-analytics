#!/usr/bin/env python3
"""
Coffee Text Analytics - Main entry point

This script serves as the main entry point for the coffee text analytics project,
orchestrating the complete workflow:

1. Data preprocessing
2. Feature extraction
3. Model training
4. Results visualization

Run this script to execute the complete pipeline or specify which steps to run.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PATHS = {
    "raw_data": "data/raw/coffee_reviews.csv",
    "processed_data": "data/processed/coffee_processed.csv",
    "features_data": "data/processed/coffee_features.csv",
    "models_dir": "models",
    "output_dir": "output",
    "figures_dir": "output/figures",
}


def setup_project():
    """
    Set up project directories and environment.
    """
    # Create directories if they don't exist
    for path in [
        "data/raw",
        "data/processed",
        "models",
        "output",
        "output/figures",
        "notebooks",
    ]:
        os.makedirs(path, exist_ok=True)

    logger.info("Project directories set up successfully")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Coffee Text Analytics pipeline"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocess", "features", "train", "visualize", "all"],
        default=["all"],
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_PATHS["raw_data"],
        help=f"Path to input raw data file (default: {DEFAULT_PATHS['raw_data']})",
    )
    parser.add_argument(
        "--text_columns",
        nargs="+",
        default=["description", "notes"],
        help="Text columns to analyze (default: description notes)",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="rating",
        help="Target column to predict (default: rating)",
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=10,
        help="Number of topics for topic modeling (default: 10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "random_forest", "xgboost"],
        help="Models to train (default: linear random_forest xgboost)",
    )
    return parser.parse_args()


def preprocess_data(args):
    """
    Preprocess raw coffee review data.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from src.data.preprocessing import process_raw_data

    logger.info("Starting data preprocessing step")

    try:
        process_raw_data(
            input_file=args.input_file,
            output_file=DEFAULT_PATHS["processed_data"],
            text_columns=args.text_columns,
        )
        return True
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return False


def extract_features(args):
    """
    Extract features from preprocessed data.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from src.features.feature_extraction import extract_features_from_data

    logger.info("Starting feature extraction step")

    try:
        extract_features_from_data(
            input_file=DEFAULT_PATHS["processed_data"],
            output_file=DEFAULT_PATHS["features_data"],
            n_topics=args.n_topics,
        )
        return True
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return False


def train_models(args):
    """
    Train predictive models.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from src.models.model_training import train_and_evaluate_models

    logger.info("Starting model training step")

    try:
        train_and_evaluate_models(
            input_file=DEFAULT_PATHS["features_data"],
            target_column=args.target_column,
            models_to_train=args.models,
            models_dir=DEFAULT_PATHS["models_dir"],
        )
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False


def visualize_results(args):
    """
    Generate visualizations of results.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from src.visualization.visualize import create_visualizations

    logger.info("Starting results visualization step")

    try:
        create_visualizations(
            features_file=DEFAULT_PATHS["features_data"],
            models_dir=DEFAULT_PATHS["models_dir"],
            output_dir=DEFAULT_PATHS["figures_dir"],
        )
        return True
    except Exception as e:
        logger.error(f"Results visualization failed: {e}")
        return False


def main():
    """
    Main pipeline orchestration function.
    """
    args = parse_args()

    # Set up project directories
    setup_project()

    # Determine which steps to run
    run_all = "all" in args.steps
    run_preprocess = run_all or "preprocess" in args.steps
    run_features = run_all or "features" in args.steps
    run_train = run_all or "train" in args.steps
    run_visualize = run_all or "visualize" in args.steps

    # Execute pipeline steps
    success = True

    if run_preprocess:
        success = preprocess_data(args)
        if not success:
            logger.error("Data preprocessing failed, stopping pipeline")
            return 1

    if run_features and success:
        success = extract_features(args)
        if not success:
            logger.error("Feature extraction failed, stopping pipeline")
            return 1

    if run_train and success:
        success = train_models(args)
        if not success:
            logger.error("Model training failed, stopping pipeline")
            return 1

    if run_visualize and success:
        success = visualize_results(args)
        if not success:
            logger.error("Results visualization failed")
            return 1

    if success:
        logger.info("Pipeline completed successfully!")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Coffee Text Analytics - Main entry point

This script serves as the main entry point for the coffee text analytics project,
orchestrating the complete workflow using modern data processing with Polars and
centralized configuration management:

1. Data preprocessing (Polars-based)
2. Feature extraction (Polars-based with thesis methodology)
3. Model training (Pandas conversion for sklearn compatibility)
4. Results visualization

Run this script to execute the complete pipeline or specify which steps to run.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import polars as pl

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import configuration system
from config import config
from config.validation import validate_config, print_config_summary, check_dependencies
from config.environments import apply_environment_config

# Configure logging using configuration system
logger = logging.getLogger(__name__)


def setup_project():
    """
    Set up project directories and environment using configuration system.
    """
    # Validate configuration
    is_valid = validate_config(config, raise_on_error=False)
    if not is_valid:
        logger.warning("Configuration validation failed, but continuing...")

    # Check dependencies
    deps_available, missing_deps = check_dependencies()
    if not deps_available:
        logger.error(f"Missing dependencies: {missing_deps}")
        logger.error("Please install missing packages before running the pipeline")
        return False

    # Create directories (already done by config initialization)
    logger.info("Project directories set up successfully")

    # Print configuration summary
    print_config_summary(config)

    return True


def parse_args():
    """
    Parse command-line arguments with configuration-aware defaults.
    """
    parser = argparse.ArgumentParser(
        description="Run the Coffee Text Analytics pipeline with centralized configuration"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocess", "features", "train", "visualize", "all"],
        default=["all"],
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "testing", "cicd"],
        default=config.environment,
        help=f"Environment configuration to use (default: {config.environment})",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(config.paths.get_raw_data_path()),
        help=f"Path to input raw data file (default: {config.paths.get_raw_data_path()})",
    )
    parser.add_argument(
        "--text_columns",
        nargs="+",
        default=config.models.text_columns,
        help=f"Text columns to analyze (default: {' '.join(config.models.text_columns)})",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=config.models.target_column,
        help=f"Target column to predict (default: {config.models.target_column})",
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=config.features.n_topics,
        help=f"Number of topics for topic modeling (default: {config.features.n_topics})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.models.models_to_train,
        help=f"Models to train (default: {' '.join(config.models.models_to_train)})",
    )
    parser.add_argument(
        "--validate_config",
        action="store_true",
        help="Validate configuration and exit",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    """
    Apply command-line argument overrides to configuration.

    Args:
        args: Parsed command-line arguments
    """
    global config

    # Apply environment configuration if different
    if args.environment != config.environment:
        logger.info(
            f"Switching from {config.environment} to {args.environment} environment"
        )
        config = apply_environment_config(config, args.environment)
        config.environment = args.environment

    # Apply CLI overrides
    if args.text_columns != config.models.text_columns:
        config.models.text_columns = args.text_columns
        logger.info(f"Text columns overridden: {args.text_columns}")

    if args.target_column != config.models.target_column:
        config.models.target_column = args.target_column
        logger.info(f"Target column overridden: {args.target_column}")

    if args.n_topics != config.features.n_topics:
        config.features.n_topics = args.n_topics
        logger.info(f"Number of topics overridden: {args.n_topics}")

    if args.models != config.models.models_to_train:
        config.models.models_to_train = args.models
        logger.info(f"Models to train overridden: {args.models}")


def preprocess_data(args):
    """
    Preprocess raw coffee review data using Polars-compatible preprocessing.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from data.preprocessing import process_raw_data

    logger.info("Starting data preprocessing step with Polars integration")

    try:
        process_raw_data(
            input_file=args.input_file,
            output_file=str(config.paths.get_processed_data_path()),
            text_columns=config.models.text_columns,
        )
        return True
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return False


def extract_features(args):
    """
    Extract features from preprocessed data using Polars-based feature extraction.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from features.feature_extraction import extract_features_from_data

    logger.info("Starting feature extraction step with Polars")

    try:
        extract_features_from_data(
            input_file=str(config.paths.get_processed_data_path()),
            output_file=str(config.paths.get_features_data_path()),
            n_topics=config.features.n_topics,
        )
        return True
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return False


def train_models(args):
    """
    Train predictive models (converts Polars to Pandas for sklearn compatibility).

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from models.model_training import train_and_evaluate_models

    logger.info(
        "Starting model training step (Polars -> Pandas conversion for sklearn)"
    )

    try:
        # Load features using Polars
        logger.info("Loading features with Polars...")
        features_path = config.paths.get_features_data_path()
        df_polars = pl.read_csv(features_path)

        # Convert to Pandas for sklearn compatibility
        logger.info(
            "Converting Polars DataFrame to Pandas for sklearn compatibility..."
        )
        df_pandas = df_polars.to_pandas()

        # Save temporary pandas file for model training
        temp_pandas_file = str(features_path).replace(".csv", "_pandas.csv")
        df_pandas.to_csv(temp_pandas_file, index=False)

        train_and_evaluate_models(
            input_file=temp_pandas_file,
            target_column=config.models.target_column,
            models_to_train=config.models.models_to_train,
            models_dir=str(config.paths.models),
        )

        # Clean up temporary file
        os.remove(temp_pandas_file)

        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False


def visualize_results(args):
    """
    Generate visualizations and analysis reports.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from visualization.plots import create_analysis_plots

    logger.info("Starting visualization generation")

    try:
        # Load processed data for visualization
        processed_path = config.paths.get_processed_data_path()
        if processed_path.exists():
            df = pl.read_csv(processed_path)

            create_analysis_plots(
                df=df,
                output_dir=str(config.paths.figures),
                target_column=config.models.target_column,
            )

            logger.info(f"Visualizations saved to {config.paths.figures}")
            return True
        else:
            logger.error(f"Processed data not found: {processed_path}")
            return False

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return False


def main():
    """
    Main function that orchestrates the entire pipeline.
    """
    # Parse arguments
    args = parse_args()

    # Handle configuration validation request
    if args.validate_config:
        is_valid = validate_config(config, raise_on_error=False)
        print_config_summary(config)
        sys.exit(0 if is_valid else 1)

    # Set up project
    if not setup_project():
        logger.error("Project setup failed")
        sys.exit(1)

    # Apply CLI overrides to configuration
    apply_cli_overrides(args)

    # Re-validate configuration after overrides
    is_valid = validate_config(config, raise_on_error=False)
    if not is_valid:
        logger.warning("Configuration validation failed after CLI overrides")

    logger.info(f"Starting Coffee Text Analytics pipeline in {config.environment} mode")
    logger.info(f"Steps to run: {args.steps}")

    # Track success of each step
    results = {}

    # Execute pipeline steps
    if "all" in args.steps or "preprocess" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 1: Data Preprocessing")
        logger.info("=" * 60)
        results["preprocess"] = preprocess_data(args)

    if "all" in args.steps or "features" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Extraction")
        logger.info("=" * 60)
        results["features"] = extract_features(args)

    if "all" in args.steps or "train" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 3: Model Training")
        logger.info("=" * 60)
        results["train"] = train_models(args)

    if "all" in args.steps or "visualize" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 4: Visualization")
        logger.info("=" * 60)
        results["visualize"] = visualize_results(args)

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    success_count = sum(results.values())
    total_count = len(results)

    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{step.upper()}: {status}")

    logger.info(
        f"\nOverall: {success_count}/{total_count} steps completed successfully"
    )

    if success_count == total_count:
        logger.info("🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

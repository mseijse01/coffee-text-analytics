#!/usr/bin/env python3
"""
Coffee Text Analytics - Main entry point

This script serves as the main entry point for the coffee text analytics project,
orchestrating the complete workflow using the new component-based architecture:

1. Data preprocessing (Polars-based)
2. Feature extraction (Component-based with CoffeeFeatureManager)
3. Model training (Component-based models with evaluator)
4. Results visualization

Run this script to execute the complete pipeline or specify which steps to run.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import configuration system
from config import config
from config.validation import validate_config, print_config_summary, check_dependencies
from config.environments import apply_environment_config

# Import new component-based architecture
from features import CoffeeFeatureManager
from models import (
    CoffeeLinearRegression,
    CoffeeRidgeRegression,
    CoffeeLassoRegression,
    CoffeeRandomForest,
    CoffeeXGBoost,
    MultinomialInverseRegression,
    CoffeeModelEvaluator,
)

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
        description="Run the Coffee Text Analytics pipeline with component-based architecture"
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
    Extract features from preprocessed data using the new CoffeeFeatureManager.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    logger.info("Starting feature extraction step with CoffeeFeatureManager")

    try:
        # Load preprocessed data
        processed_data_path = config.paths.get_processed_data_path()
        logger.info(f"Loading preprocessed data from {processed_data_path}")

        df = pl.read_csv(processed_data_path)
        logger.info(f"Loaded data shape: {df.shape}")

        # Create feature extraction configuration
        feature_config = {
            "tfidf": {
                "max_features": config.features.max_features,
                "ngram_range": tuple(config.features.ngram_range),
                "models_dir": str(config.paths.models_dir),
            },
            "bert": {
                "batch_size": config.features.bert_batch_size,
                "max_length": config.features.bert_max_length,
            },
            "topics": {
                "n_topics": config.features.n_topics,
                "algorithms": ["lda", "nmf"],
                "models_dir": str(config.paths.models_dir),
            },
            "sentiment": {"batch_size": config.features.bert_batch_size},
            "glove": {"aggregation": "mean"},
        }

        # Initialize feature manager
        feature_manager = CoffeeFeatureManager(feature_config)

        # Combine text columns for feature extraction
        combined_texts = []
        for i in range(len(df)):
            text_parts = []
            for col in config.models.text_columns:
                if col in df.columns:
                    text_value = df[col][i]
                    if text_value and isinstance(text_value, str):
                        text_parts.append(text_value.strip())

            combined_text = " ".join(text_parts) if text_parts else ""
            combined_texts.append(combined_text)

        logger.info(f"Combined {len(combined_texts)} texts for feature extraction")

        # Fit feature extractors
        logger.info("Fitting feature extractors...")
        feature_manager.fit(combined_texts)

        # Extract features
        logger.info("Extracting features...")
        features_df = feature_manager.extract_features(combined_texts)

        # Combine with original data
        if not features_df.is_empty():
            # Convert original data to match features
            result_df = df.hstack(features_df)
        else:
            logger.warning("No features extracted, using original data")
            result_df = df

        # Save features
        features_data_path = config.paths.get_features_data_path()
        logger.info(f"Saving features to {features_data_path}")
        result_df.write_csv(features_data_path)

        # Save feature manager
        feature_manager.save_extractors(str(config.paths.models_dir))

        # Print feature summary
        feature_manager.print_summary()

        logger.info(f"Feature extraction completed. Final shape: {result_df.shape}")
        return True

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def train_models(args):
    """
    Train models using the new component-based architecture.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    logger.info("Starting model training step with component-based models")

    try:
        # Load features data
        features_data_path = config.paths.get_features_data_path()
        logger.info(f"Loading features from {features_data_path}")

        # Load as Polars first, then convert to Pandas for sklearn compatibility
        df_polars = pl.read_csv(features_data_path)
        df = df_polars.to_pandas()
        logger.info(f"Loaded features shape: {df.shape}")

        # Prepare features and target
        target_column = config.models.target_column
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return False

        # Exclude non-feature columns
        exclude_columns = (
            config.models.text_columns
            + [target_column]
            + ["id", "name", "roaster", "roast", "loc", "url"]
        )
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[feature_columns]
        y = df[target_column]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize models based on configuration
        models = {}
        model_configs = {
            "linear": {"scale_features": False},
            "ridge": {"scale_features": True, "cv": 5},
            "lasso": {"scale_features": True, "cv": 5},
            "random_forest": {"tune_hyperparameters": True, "cv": 3},
            "xgboost": {"tune_hyperparameters": True, "cv": 3},
        }

        for model_name in config.models.models_to_train:
            if model_name == "linear":
                models[model_name] = CoffeeLinearRegression(
                    model_configs.get(model_name, {})
                )
            elif model_name == "ridge":
                models[model_name] = CoffeeRidgeRegression(
                    model_configs.get(model_name, {})
                )
            elif model_name == "lasso":
                models[model_name] = CoffeeLassoRegression(
                    model_configs.get(model_name, {})
                )
            elif model_name == "random_forest":
                models[model_name] = CoffeeRandomForest(
                    model_configs.get(model_name, {})
                )
            elif model_name == "xgboost":
                try:
                    models[model_name] = CoffeeXGBoost(
                        model_configs.get(model_name, {})
                    )
                except Exception as e:
                    logger.warning(f"XGBoost not available: {e}")
                    continue

        # Train models
        trained_models = {}
        for name, model in models.items():
            try:
                logger.info(f"Training {name} model...")
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"{name} model trained successfully")
            except Exception as e:
                logger.error(f"Failed to train {name} model: {e}")

        # Initialize evaluator
        evaluator = CoffeeModelEvaluator()

        # Evaluate models
        logger.info("Evaluating models...")
        comparison_results = evaluator.compare_models(trained_models, X_test, y_test)

        # Print results
        print("\n" + "=" * 50)
        print("MODEL COMPARISON RESULTS")
        print("=" * 50)

        summary_metrics = comparison_results["summary_metrics"]
        for metric in ["r2", "rmse", "mae"]:
            print(f"\n{metric.upper()}:")
            for model_name, value in summary_metrics[metric].items():
                print(f"  {model_name}: {value:.4f}")

        print(f"\nBest models:")
        for metric, best_model in comparison_results["best_models"].items():
            print(f"  {metric}: {best_model}")

        # Train MNIR if requested
        if "mnir" in config.models.models_to_train:
            logger.info("Training MNIR model...")
            try:
                # Prepare sensory data (if available)
                sensory_attributes = ["aroma", "acid", "body", "flavor", "aftertaste"]
                sensory_data = {}

                for attr in sensory_attributes:
                    if attr in df.columns:
                        sensory_data[attr] = df[attr].values

                if sensory_data:
                    mnir_config = {
                        "lasso_cv": 5,
                        "random_state": 42,
                        "sensory_attributes": list(sensory_data.keys()),
                    }

                    mnir = MultinomialInverseRegression(mnir_config)
                    mnir.fit(X_train.values, sensory_data)

                    # Generate MNIR report
                    mnir_report = mnir.generate_insights_report()
                    print("\n" + "=" * 50)
                    print("MNIR ANALYSIS RESULTS")
                    print("=" * 50)
                    print(mnir_report)

                    # Save MNIR model
                    mnir_path = config.paths.models_dir / "mnir_model.pkl"
                    mnir.save_model(str(mnir_path))
                    logger.info(f"MNIR model saved to {mnir_path}")
                else:
                    logger.warning("No sensory attributes found for MNIR analysis")

            except Exception as e:
                logger.error(f"MNIR training failed: {e}")

        # Save trained models
        models_dir = config.paths.models_dir
        for name, model in trained_models.items():
            try:
                model_path = models_dir / f"{name}_model.pkl"
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save {name} model: {e}")

        # Save evaluation results
        results_path = config.paths.output_dir / "model_comparison_results.pkl"
        import pickle

        with open(results_path, "wb") as f:
            pickle.dump(comparison_results, f)
        logger.info(f"Evaluation results saved to {results_path}")

        return True

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def visualize_results(args):
    """
    Generate visualizations using the new evaluator.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    logger.info("Starting visualization step")

    try:
        # Load evaluation results
        results_path = config.paths.output_dir / "model_comparison_results.pkl"
        if not results_path.exists():
            logger.error("No evaluation results found. Run training step first.")
            return False

        import pickle

        with open(results_path, "rb") as f:
            comparison_results = pickle.load(f)

        # Initialize evaluator
        evaluator = CoffeeModelEvaluator()

        # Create visualizations
        figures_dir = config.paths.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Model comparison plot
        logger.info("Creating model comparison plot...")
        fig = evaluator.plot_model_comparison(
            comparison_results,
            metric="r2",
            save_path=str(figures_dir / "model_comparison_r2.png"),
        )

        # Feature importance plots for each model
        for model_name, results in comparison_results["individual_results"].items():
            if "feature_importance" in results:
                logger.info(f"Creating feature importance plot for {model_name}...")
                fig = evaluator.plot_feature_importance(
                    results["feature_importance"],
                    model_name=model_name,
                    save_path=str(figures_dir / f"feature_importance_{model_name}.png"),
                )

        # Prediction plots for best model
        best_r2_model = comparison_results["best_models"]["r2"]
        best_results = comparison_results["individual_results"][best_r2_model]

        # Load test data for actual values
        features_data_path = config.paths.get_features_data_path()
        df = pl.read_csv(features_data_path).to_pandas()

        from sklearn.model_selection import train_test_split

        target_column = config.models.target_column
        exclude_columns = (
            config.models.text_columns
            + [target_column]
            + ["id", "name", "roaster", "roast", "loc", "url"]
        )
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"Creating prediction plot for best model: {best_r2_model}...")
        fig = evaluator.plot_predictions(
            y_test,
            best_results["predictions"],
            model_name=best_r2_model,
            save_path=str(figures_dir / f"predictions_{best_r2_model}.png"),
        )

        logger.info(f"Visualizations saved to {figures_dir}")
        return True

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    """
    Main function to orchestrate the complete pipeline.
    """
    # Parse arguments
    args = parse_args()

    # Handle configuration validation
    if args.validate_config:
        is_valid = validate_config(config, raise_on_error=False)
        if is_valid:
            print("✅ Configuration is valid")
            print_config_summary(config)
        else:
            print("❌ Configuration validation failed")
        return

    # Set up project
    if not setup_project():
        logger.error("Project setup failed")
        return

    # Apply CLI overrides
    apply_cli_overrides(args)

    # Determine steps to run
    steps_to_run = args.steps
    if "all" in steps_to_run:
        steps_to_run = ["preprocess", "features", "train", "visualize"]

    # Execute pipeline steps
    success = True

    if "preprocess" in steps_to_run:
        logger.info("=" * 50)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 50)
        success &= preprocess_data(args)

    if "features" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE EXTRACTION")
        logger.info("=" * 50)
        success &= extract_features(args)

    if "train" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 50)
        success &= train_models(args)

    if "visualize" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 4: VISUALIZATION")
        logger.info("=" * 50)
        success &= visualize_results(args)

    # Final status
    if success:
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Results saved to: {config.paths.output_dir}")
    else:
        logger.error("=" * 50)
        logger.error("PIPELINE FAILED!")
        logger.error("=" * 50)
        logger.error("Check the logs above for error details")


if __name__ == "__main__":
    main()

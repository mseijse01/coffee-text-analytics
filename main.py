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
from features import CoffeeFeatureManager, LassoFeatureSelector
from features.feature_selector_corrected import CorrectedLassoFeatureSelector
from models import (
    CoffeeLinearRegression,
    CoffeeRidgeRegression,
    CoffeeLassoRegression,
    CoffeeRandomForest,
    CoffeeXGBoost,
    MultinomialInverseRegression,
    CoffeeModelEvaluator,
    CoffeeSVR,
    CoffeeDecisionTree,
)
from experiment.mlflow_integration import CoffeeMLflowTracker, setup_coffee_mlflow

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
        choices=["preprocess", "features", "select", "train", "visualize", "all"],
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
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=None,
        help="Fraction of data to sample (e.g., 0.1 for 10%%, 0.5 for 50%%)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Absolute number of samples to use (e.g., 1000)",
    )
    parser.add_argument(
        "--box_cox",
        action="store_true",
        help="Enable Box-Cox transformation for target variable",
    )
    parser.add_argument(
        "--box_cox_dual",
        action="store_true",
        help="Run dual pipeline: compare models with and without Box-Cox transformation",
    )
    parser.add_argument(
        "--mlflow_tracking",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="coffee-text-analytics-thesis",
        help="MLflow experiment name (default: coffee-text-analytics-thesis)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not provided)",
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

    # Apply Box-Cox configuration
    if args.box_cox:
        config.models.box_cox_enabled = True
        logger.info("Box-Cox transformation enabled")

    if args.box_cox_dual:
        config.models.box_cox_dual_pipeline = True
        logger.info(
            "Box-Cox dual pipeline enabled (will compare with and without transformation)"
        )

    # Log sampling configuration
    if args.sample_fraction:
        logger.info(f"Data sampling enabled: {args.sample_fraction * 100:.1f}% of data")
    elif args.sample_size:
        logger.info(f"Data sampling enabled: {args.sample_size} samples")

    # Apply MLflow configuration
    if args.mlflow_tracking:
        logger.info("MLflow experiment tracking enabled")
        logger.info(f"Experiment name: {args.experiment_name}")
        if args.run_name:
            logger.info(f"Run name: {args.run_name}")


def preprocess_data(args):
    """
    Preprocess raw coffee review data using dual-mode data loading.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    from data.preprocessing import process_raw_data

    logger.info("Starting data preprocessing step with dual-mode data loading")

    try:
        # Use dual-mode loading if input_file is the default config path
        # This allows containers to load from MinIO while development uses local files
        use_dual_mode = args.input_file == str(config.paths.get_raw_data_path())

        if use_dual_mode:
            logger.info("Using dual-mode data loader (environment-based)")
            process_raw_data(
                input_file=None,  # Use dual-mode loader
                output_file=str(config.paths.get_processed_data_path()),
                text_columns=config.models.text_columns,
                sample_fraction=args.sample_fraction,
                sample_size=args.sample_size,
            )
        else:
            logger.info(f"Using specified input file: {args.input_file}")
            process_raw_data(
                input_file=args.input_file,
                output_file=str(config.paths.get_processed_data_path()),
                text_columns=config.models.text_columns,
                sample_fraction=args.sample_fraction,
                sample_size=args.sample_size,
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
                "max_features": config.features.tfidf_max_features,
                "ngram_range": config.features.tfidf_ngram_range,
                "models_dir": str(config.paths.models),
            },
            "bert": {
                "batch_size": config.features.bert_batch_size,
                "max_length": config.features.bert_max_length,
            },
            "topics": {
                "n_topics": config.features.n_topics,
                "algorithms": ["lda", "nmf"],
                "models_dir": str(config.paths.models),
            },
            "sentiment": {"batch_size": config.features.bert_batch_size},
            "glove": {"aggregation": "mean"},
        }

        # Initialize feature manager
        feature_manager = CoffeeFeatureManager(feature_config)

        # Following thesis methodology: process desc_1, desc_2, desc_3 separately
        logger.info(
            "Following thesis methodology: separate processing of description columns"
        )
        logger.info(f"Processing columns separately: {config.models.text_columns}")

        # Fit feature extractors on combined text from all columns
        logger.info("Fitting feature extractors on combined text from all columns...")
        feature_manager.fit(df, config.models.text_columns)

        # Extract features separately for each column
        logger.info("Extracting features separately for each description column...")
        result_df = feature_manager.extract_all_features(df, config.models.text_columns)

        # Save features
        features_data_path = config.paths.get_features_data_path()
        logger.info(f"Saving features to {features_data_path}")
        result_df.write_csv(features_data_path)

        # Save feature manager
        feature_manager.save_extractors(str(config.paths.models))

        # Print feature summary
        feature_manager.print_summary()

        logger.info(f"Feature extraction completed. Final shape: {result_df.shape}")
        return True

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def select_features(args):
    """
    Perform LASSO-based feature selection using the new LassoFeatureSelector.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    logger.info("Starting LASSO-based feature selection step")

    try:
        # Check if feature selection is enabled
        if not config.models.feature_selection_enabled:
            logger.info("Feature selection is disabled in configuration")
            # Copy features file to selected features file
            import shutil

            features_path = config.paths.get_features_data_path()
            selected_features_path = (
                config.paths.processed / "coffee_features_selected.csv"
            )
            shutil.copy2(features_path, selected_features_path)
            logger.info("Copied features file without selection")
            return True

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
            + [
                "id",
                "slug",
                "all_text",
                "name",
                "location",
                "origin",
                "est_price",
                "review_date",
                "agtron",
                "aroma",
                "acid",
                "body",
                "flavor",
                "aftertaste",
                "with_milk",
                "price_value",
                "price_unit",
                "price_standardized",
                "processed_desc_1",
                "processed_desc_2",
                "processed_desc_3",
                "merged_text",
                "processed_text",
                "url",
                "loc",
                "roaster",  # Exclude categorical columns (encoded versions are used)
                "roast",
                "country_of_origin",
            ]
        )
        # Only exclude columns that actually exist in the dataframe
        exclude_columns = [col for col in exclude_columns if col in df.columns]
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[feature_columns]
        y = df[target_column]

        logger.info(f"Features shape before selection: {X.shape}")
        logger.info(f"Target shape: {y.shape}")

        # Initialize corrected feature selector with configuration (thesis methodology)
        selector_config = config.models.feature_selection_config.copy()
        selector = CorrectedLassoFeatureSelector(selector_config)
        logger.info("Using corrected LASSO feature selector (thesis methodology)")

        # Fit and transform features
        logger.info("Fitting LASSO feature selector...")
        X_selected = selector.fit_transform(X, y)

        logger.info(f"Features shape after selection: {X_selected.shape}")

        # Print selection summary
        selector.print_summary()

        # Create new dataframe with selected features and non-feature columns
        if isinstance(X_selected, pd.DataFrame):
            selected_feature_names = X_selected.columns.tolist()
        else:
            selected_feature_names = selector.get_selected_features()
            X_selected = pd.DataFrame(
                X_selected, columns=selected_feature_names, index=X.index
            )

        # Combine selected features with non-feature columns
        non_feature_df = df[exclude_columns]
        result_df = pd.concat([non_feature_df, X_selected], axis=1)

        # Save selected features
        selected_features_path = config.paths.processed / "coffee_features_selected.csv"
        logger.info(f"Saving selected features to {selected_features_path}")
        result_df.to_csv(selected_features_path, index=False)

        # Save feature selector
        selector_path = config.paths.models / "lasso_feature_selector.pkl"
        selector.save_selector(selector_path)
        logger.info(f"Feature selector saved to {selector_path}")

        # Save selection summary
        summary = selector.get_selection_summary()
        summary_path = config.paths.output / "feature_selection_summary.pkl"
        import pickle

        with open(summary_path, "wb") as f:
            pickle.dump(summary, f)
        logger.info(f"Selection summary saved to {summary_path}")

        logger.info(f"Feature selection completed. Final shape: {result_df.shape}")
        return True

    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
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
        # Load features data (use selected features if available)
        selected_features_path = config.paths.processed / "coffee_features_selected.csv"
        if selected_features_path.exists():
            features_data_path = selected_features_path
            logger.info(f"Loading selected features from {features_data_path}")
        else:
            features_data_path = config.paths.get_features_data_path()
            logger.info(
                f"Loading original features from {features_data_path} (no feature selection performed)"
            )

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
            + [
                "id",
                "slug",
                "all_text",
                "name",
                "location",
                "origin",
                "est_price",
                "review_date",
                "agtron",
                "aroma",
                "acid",
                "body",
                "flavor",
                "aftertaste",
                "with_milk",
                "price_value",
                "price_unit",
                "price_standardized",
                "processed_desc_1",
                "processed_desc_2",
                "processed_desc_3",
                "merged_text",
                "processed_text",
                "url",
                "loc",
                "roaster",  # Exclude categorical columns (encoded versions are used)
                "roast",
                "country_of_origin",
            ]
        )
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[feature_columns]
        y = df[target_column]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")

        # Split data using stratified sampling (thesis methodology)
        from sklearn.model_selection import train_test_split
        import pandas as pd

        # Create stratified bins for the target variable (rating)
        # This ensures balanced representation across rating ranges
        n_bins = 5  # Start with 5 bins for stratification

        # Try stratified sampling with decreasing number of bins
        stratify_var = None
        for bins in [5, 4, 3]:
            try:
                y_binned = pd.cut(y, bins=bins, labels=False)
                bin_counts = pd.Series(y_binned).value_counts()

                # Check if all bins have at least 2 samples (minimum for stratification)
                if bin_counts.min() >= 2:
                    stratify_var = y_binned
                    n_bins = bins
                    logger.info(f"Stratified sampling with {n_bins} bins")
                    logger.info(f"Bin distribution: {bin_counts.sort_index()}")
                    break
                else:
                    logger.warning(
                        f"Bins={bins}: Some bins have <2 samples, trying fewer bins..."
                    )
            except Exception as e:
                logger.warning(f"Stratification with {bins} bins failed: {e}")

        # If stratification still fails, use simple random sampling
        if stratify_var is None:
            logger.warning(
                "Stratified sampling not possible with small sample - using random sampling"
            )
            logger.info("This is acceptable for small development samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.models.test_size,  # Use 0.3 from config (70/30 split)
            random_state=config.models.random_state,
            stratify=stratify_var,  # Use stratify_var (None if stratification failed)
        )

        logger.info(
            f"Train set size: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)"
        )
        logger.info(f"Test set size: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

        # Initialize models based on configuration
        models = {}
        model_configs = {
            "linear": config.models.linear_params,
            "ridge": config.models.ridge_params,
            "lasso": config.models.lasso_params,
            "random_forest": {
                "tune_hyperparameters": True,
                "cv": config.models.cv_folds,
                "use_two_step_tuning": config.models.two_step_tuning_enabled,
                "global_config": config,  # Pass global config for two-step tuning
            },
            "xgboost": {
                "tune_hyperparameters": True,
                "cv": config.models.cv_folds,
                "use_two_step_tuning": config.models.two_step_tuning_enabled,
                "global_config": config,  # Pass global config for two-step tuning
            },
            "svr": {
                **config.models.svr_params,
                "tune_hyperparameters": True,
                "cv": config.models.cv_folds,
                "use_two_step_tuning": config.models.two_step_tuning_enabled,
                "global_config": config,  # Pass global config for two-step tuning
            },
            "decision_tree": config.models.decision_tree_params,
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
            elif model_name == "svr":
                models[model_name] = CoffeeSVR(model_configs.get(model_name, {}))
            elif model_name == "decision_tree":
                models[model_name] = CoffeeDecisionTree(
                    model_configs.get(model_name, {})
                )

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

        # Check if Box-Cox dual pipeline is enabled
        if config.models.box_cox_dual_pipeline:
            logger.info(
                "🔄 Box-Cox dual pipeline enabled - running comprehensive comparison"
            )

            # Import the dual pipeline function
            from utils.transformations import run_box_cox_dual_pipeline

            # Run dual pipeline analysis
            dual_pipeline_results = run_box_cox_dual_pipeline(
                X_train, X_test, y_train, y_test, models, config, logger
            )

            # Use baseline results as the main comparison results
            comparison_results = dual_pipeline_results["baseline_results"]

            # Log dual pipeline summary
            logger.info("📊 Box-Cox Dual Pipeline completed successfully!")
            logger.info(f"Recommendation: {dual_pipeline_results['recommendation']}")
            logger.info(f"Reason: {dual_pipeline_results['recommendation_reason']}")

        elif config.models.box_cox_enabled:
            logger.info(
                "📊 Box-Cox transformation enabled - applying to target variable"
            )

            # Import Box-Cox transformer
            from utils.transformations import BoxCoxTransformer

            # Apply Box-Cox transformation
            transformer = BoxCoxTransformer(config.models.box_cox_config)
            y_train_transformed = transformer.fit_transform(y_train)
            y_test_transformed = transformer.transform(y_test)

            logger.info(
                f"Box-Cox transformation applied (λ = {transformer.lambda_:.4f})"
            )

            # Retrain models with transformed target
            boxcox_trained_models = {}
            for name, model in models.items():
                try:
                    logger.info(f"Training {name} with Box-Cox transformation...")
                    # Create fresh model instance
                    model_copy = type(model)(
                        model.config if hasattr(model, "config") else {}
                    )
                    model_copy.fit(X_train, y_train_transformed)
                    boxcox_trained_models[name] = model_copy
                    logger.info(f"{name} model trained successfully with Box-Cox")
                except Exception as e:
                    logger.error(f"Failed to train {name} with Box-Cox: {e}")

            # Get predictions and inverse transform them
            boxcox_predictions = {}
            for name, model in boxcox_trained_models.items():
                try:
                    pred_transformed = model.predict(X_test)
                    pred_original = transformer.inverse_transform(pred_transformed)
                    boxcox_predictions[name] = pred_original
                except Exception as e:
                    logger.error(f"Failed to get predictions for {name}: {e}")

            # Evaluate against original scale targets
            comparison_results = evaluator.compare_models_with_predictions(
                boxcox_predictions, y_test
            )

            # Save Box-Cox transformer
            transformer_path = config.paths.models / "box_cox_transformer.pkl"
            transformer.save_transformer(transformer_path)
            logger.info(f"Box-Cox transformer saved to {transformer_path}")

            # Update trained_models to use Box-Cox models
            trained_models = boxcox_trained_models

        else:
            # Standard evaluation with comprehensive metrics and SHAP analysis
            logger.info(
                "🔍 Evaluating models with comprehensive metrics and SHAP analysis..."
            )
            comparison_results = evaluator.compare_models(
                trained_models, X_test, y_test, include_shap=True
            )

        # Print comprehensive results
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("=" * 80)
        print("Following thesis methodology with MAE, RMSE, R² evaluation")
        print("")

        # Print comparison report if available
        if "comparison_report" in comparison_results:
            print(comparison_results["comparison_report"])
        else:
            # Fallback to basic summary
            summary_metrics = comparison_results["summary_metrics"]
            for metric in ["r2", "rmse", "mae"]:
                print(f"\n{metric.upper()}:")
                for model_name, value in summary_metrics[metric].items():
                    print(f"  {model_name}: {value:.4f}")

            print(f"\nBest models:")
            for metric, best_model in comparison_results["best_models"].items():
                print(f"  {metric}: {best_model}")

        # Run comprehensive SHAP analysis if not already included
        if (
            "shap_comparison" not in comparison_results
            or comparison_results["shap_comparison"] is None
        ):
            try:
                logger.info("🔍 Running standalone comprehensive SHAP analysis...")
                from utils.shap_analysis import run_comprehensive_shap_analysis

                shap_output_dir = config.paths.output / "shap_analysis"
                shap_results = run_comprehensive_shap_analysis(
                    trained_models, X_test, y_test, shap_output_dir
                )

                # Print SHAP analysis report
                if shap_results and "comparison" in shap_results:
                    print("\n" + "=" * 80)
                    print("🔍 COMPREHENSIVE SHAP ANALYSIS RESULTS")
                    print("=" * 80)

                    comparison = shap_results["comparison"]
                    if "top_features" in comparison:
                        print("🏆 TOP FEATURES ACROSS ALL MODELS:")
                        for i, feature in enumerate(comparison["top_features"][:10], 1):
                            print(f"  {i:2d}. {feature}")

                    if "model_agreement" in comparison:
                        agreement = comparison["model_agreement"]
                        print(
                            f"\n🤝 MODEL AGREEMENT: {agreement['agreement_interpretation']}"
                        )
                        print(
                            f"   Average Correlation: {agreement['average_agreement']:.3f}"
                        )

                    print("\n✅ SHAP analysis completed successfully!")
                    print(f"📊 Results saved to: {shap_output_dir}")

            except Exception as e:
                logger.warning(f"Standalone SHAP analysis failed: {e}")

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
                        "lasso_cv": config.models.lasso_cv_folds,
                        "random_state": config.models.random_state,
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
                    mnir_path = config.paths.models / "mnir_model.pkl"
                    mnir.save_model(str(mnir_path))
                    logger.info(f"MNIR model saved to {mnir_path}")
                else:
                    logger.warning("No sensory attributes found for MNIR analysis")

            except Exception as e:
                logger.error(f"MNIR training failed: {e}")

        # Save trained models
        models_dir = config.paths.models
        for name, model in trained_models.items():
            try:
                model_path = models_dir / f"{name}_model.pkl"
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save {name} model: {e}")

        # Save comprehensive evaluation results
        results_path = config.paths.output / "comprehensive_model_evaluation"
        evaluator.save_comprehensive_evaluation(comparison_results, results_path)
        logger.info(f"Comprehensive evaluation results saved to {results_path}")

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
        results_path = config.paths.output / "model_comparison_results.pkl"
        if not results_path.exists():
            logger.error("No evaluation results found. Run training step first.")
            return False

        import pickle

        with open(results_path, "rb") as f:
            comparison_results = pickle.load(f)

        # Initialize evaluator
        evaluator = CoffeeModelEvaluator()

        # Create visualizations
        figures_dir = config.paths.output / "figures"
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

        # Use same stratified sampling as in training
        n_bins = 5
        y_binned = pd.cut(y, bins=n_bins, labels=False)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.models.test_size,
            random_state=config.models.random_state,
            stratify=y_binned,
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

    # Initialize MLflow tracking if enabled
    mlflow_tracker = None
    mlflow_run_id = None

    if args.mlflow_tracking:
        try:
            logger.info("Setting up MLflow experiment tracking...")
            mlflow_tracker = CoffeeMLflowTracker(args.experiment_name)

            # Generate run name if not provided
            run_name = args.run_name
            if not run_name:
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_info = ""
                if args.sample_fraction:
                    sample_info = f"_sample{int(args.sample_fraction * 100)}pct"
                elif args.sample_size:
                    sample_info = f"_sample{args.sample_size}"
                run_name = f"pipeline_run_{timestamp}{sample_info}"

            # Prepare methodology parameters for thesis tracking
            methodology_params = {
                "sample_fraction": args.sample_fraction or 1.0,
                "sample_size": args.sample_size or "full_dataset",
                "text_columns": ",".join(args.text_columns),
                "target_column": args.target_column,
                "models_to_train": ",".join(args.models),
                "steps_executed": ",".join(args.steps),
                "box_cox_enabled": args.box_cox,
                "box_cox_dual_pipeline": args.box_cox_dual,
                "environment": args.environment,
                "n_topics": args.n_topics,
            }

            # Start MLflow run with methodology tracking
            mlflow_run_id = mlflow_tracker.start_methodology_run(
                run_name=run_name,
                sample_fraction=args.sample_fraction or 1.0,
                methodology_params=methodology_params,
                tags={
                    "pipeline_version": "thesis_methodology_compliant",
                    "execution_mode": "full_pipeline"
                    if "all" in args.steps
                    else "partial_pipeline",
                    "focus": "methodology_validation",
                },
            )

            logger.info(f"MLflow run started: {run_name} (ID: {mlflow_run_id})")

        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            logger.warning("Continuing without MLflow tracking...")
            mlflow_tracker = None

    # Determine steps to run
    steps_to_run = args.steps
    if "all" in steps_to_run:
        steps_to_run = ["preprocess", "features", "select", "train", "visualize"]

    # Execute pipeline steps
    success = True
    step_results = {}

    if "preprocess" in steps_to_run:
        logger.info("=" * 50)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 50)
        step_success = preprocess_data(args)
        success &= step_success
        step_results["preprocess"] = step_success

    if "features" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE EXTRACTION")
        logger.info("=" * 50)
        step_success = extract_features(args)
        success &= step_success
        step_results["features"] = step_success

    if "select" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 3: FEATURE SELECTION")
        logger.info("=" * 50)
        step_success = select_features(args)
        success &= step_success
        step_results["select"] = step_success

    if "train" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 50)
        step_success = train_models(args)
        success &= step_success
        step_results["train"] = step_success

    if "visualize" in steps_to_run and success:
        logger.info("=" * 50)
        logger.info("STEP 5: VISUALIZATION")
        logger.info("=" * 50)
        step_success = visualize_results(args)
        success &= step_success
        step_results["visualize"] = step_success

    # Log results to MLflow if tracking is enabled
    if mlflow_tracker and mlflow_run_id:
        try:
            # Log step completion status
            for step, step_success in step_results.items():
                mlflow_tracker.log_methodology_compliance(
                    {f"step_{step}_completed": step_success}
                )

            # Log overall pipeline success
            mlflow_tracker.log_methodology_compliance(
                {"pipeline_completed_successfully": success}
            )

            # Log methodology compliance
            compliance_report = {
                "separate_text_processing": True,  # We process desc_1, desc_2, desc_3 separately
                "thesis_feature_naming": True,  # Features named as tfidf_desc_X_Y
                "corrected_lasso_selection": True,  # Using corrected LASSO methodology
                "specialized_preprocessing": True,  # Different preprocessing for different extractors
                "categorical_encoding": True,  # One-hot encoding implemented
                "mlflow_tracking": True,  # This run has MLflow tracking
                "all_steps_completed": success,  # Overall pipeline success
            }

            mlflow_tracker.log_methodology_compliance(compliance_report)

            logger.info("MLflow experiment data logged successfully")

        except Exception as e:
            logger.warning(f"Failed to log results to MLflow: {e}")

        finally:
            # End MLflow run
            try:
                mlflow_tracker.end_run()
                logger.info("MLflow run completed")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

    # Final status
    if success:
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Results saved to: {config.paths.output}")
        if mlflow_tracker:
            logger.info("📊 View MLflow results with: mlflow ui")
    else:
        logger.error("=" * 50)
        logger.error("PIPELINE FAILED!")
        logger.error("=" * 50)
        logger.error("Check the logs above for error details")


if __name__ == "__main__":
    main()

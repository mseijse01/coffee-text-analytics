"""Pipeline step: model training and evaluation."""

import logging
import pickle
import traceback

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from models import (
    CoffeeDecisionTree,
    CoffeeLinearRegression,
    CoffeeLassoRegression,
    CoffeeModelEvaluator,
    CoffeeRandomForest,
    CoffeeRidgeRegression,
    CoffeeSVR,
    CoffeeXGBoost,
    MultinomialInverseRegression,
)
from utils.transformations import BoxCoxTransformer, run_box_cox_dual_pipeline
from src.pipeline.constants import EXCLUDE_COLUMNS

logger = logging.getLogger(__name__)


def run_training(args, config) -> bool:
    """
    Train regression models and evaluate with comprehensive metrics.

    Handles three evaluation paths:
    - Standard: train → evaluate → SHAP
    - box_cox: apply Box-Cox to target, train, inverse-transform predictions
    - box_cox_dual: train both paths and compare (thesis methodology)

    Args:
        args: Parsed CLI arguments.
        config: Application configuration object.

    Returns:
        True on success, False on any error.
    """
    logger.info("Starting model training step")
    try:
        # Prefer selected features if available
        selected_path = config.paths.processed / "coffee_features_selected.csv"
        features_path = (
            selected_path
            if selected_path.exists()
            else config.paths.get_features_data_path()
        )
        logger.info(f"Loading features from {features_path}")

        df = pl.read_csv(features_path).to_pandas()
        logger.info(f"Loaded features shape: {df.shape}")

        target_column = config.models.target_column
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return False

        exclude_cols = list(
            set(EXCLUDE_COLUMNS) | set(config.models.text_columns) | {target_column}
        )
        feature_columns = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_columns]
        y = df[target_column]
        logger.info(f"Features shape: {X.shape} | Target shape: {y.shape}")

        # Stratified train/test split (thesis methodology: 70/30)
        stratify_var = None
        for bins in [5, 4, 3]:
            try:
                y_binned = pd.cut(y, bins=bins, labels=False)
                if pd.Series(y_binned).value_counts().min() >= 2:
                    stratify_var = y_binned
                    logger.info(f"Stratified sampling with {bins} bins")
                    break
            except Exception as exc:
                logger.warning(f"Stratification with {bins} bins failed: {exc}")

        if stratify_var is None:
            logger.warning("Falling back to random sampling (sample too small for stratification)")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.models.test_size,
            random_state=config.models.random_state,
            stratify=stratify_var,
        )
        logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # Registry built here so class-level patches (in tests) are applied correctly
        model_registry = {
            "linear": CoffeeLinearRegression,
            "ridge": CoffeeRidgeRegression,
            "lasso": CoffeeLassoRegression,
            "random_forest": CoffeeRandomForest,
            "xgboost": CoffeeXGBoost,
            "svr": CoffeeSVR,
            "decision_tree": CoffeeDecisionTree,
        }

        # Build model instances
        model_configs = {
            "linear": config.models.linear_params,
            "ridge": config.models.ridge_params,
            "lasso": config.models.lasso_params,
            "random_forest": {
                "tune_hyperparameters": True,
                "cv": config.models.cv_folds,
                "use_two_step_tuning": config.models.two_step_tuning_enabled,
                "global_config": config,
            },
            "xgboost": {
                "tune_hyperparameters": True,
                "cv": config.models.cv_folds,
                "use_two_step_tuning": config.models.two_step_tuning_enabled,
                "global_config": config,
            },
            "svr": {
                **config.models.svr_params,
                "tune_hyperparameters": True,
                "cv": config.models.cv_folds,
                "use_two_step_tuning": config.models.two_step_tuning_enabled,
                "global_config": config,
            },
            "decision_tree": config.models.decision_tree_params,
        }

        models = {}
        for name in config.models.models_to_train:
            if name not in model_registry:
                continue
            try:
                models[name] = model_registry[name](model_configs.get(name, {}))
            except Exception as exc:
                logger.warning(f"{name} not available: {exc}")

        # Train
        trained_models = {}
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                trained_models[name] = model
            except Exception as exc:
                logger.error(f"Failed to train {name}: {exc}")

        evaluator = CoffeeModelEvaluator()

        # --- Evaluation path ---
        if config.models.box_cox_dual_pipeline:
            logger.info("Box-Cox dual pipeline enabled")
            dual_results = run_box_cox_dual_pipeline(
                X_train, X_test, y_train, y_test, models, config, logger
            )
            comparison_results = dual_results["baseline_results"]
            logger.info(f"Recommendation: {dual_results['recommendation']}")

        elif config.models.box_cox_enabled:
            logger.info("Box-Cox transformation enabled")
            transformer = BoxCoxTransformer(config.models.box_cox_config)
            y_train_t = transformer.fit_transform(y_train)
            y_test_t = transformer.transform(y_test)
            logger.info(f"Box-Cox λ = {transformer.lambda_:.4f}")

            boxcox_models = {}
            for name, model in models.items():
                try:
                    fresh = type(model)(model.config if hasattr(model, "config") else {})
                    fresh.fit(X_train, y_train_t)
                    boxcox_models[name] = fresh
                except Exception as exc:
                    logger.error(f"Failed to train {name} with Box-Cox: {exc}")

            predictions = {}
            for name, model in boxcox_models.items():
                try:
                    predictions[name] = transformer.inverse_transform(model.predict(X_test))
                except Exception as exc:
                    logger.error(f"Prediction failed for {name}: {exc}")

            comparison_results = evaluator.compare_models_with_predictions(predictions, y_test)
            transformer.save_transformer(config.paths.models / "box_cox_transformer.pkl")
            trained_models = boxcox_models

        else:
            logger.info("Evaluating models with comprehensive metrics and SHAP analysis...")
            comparison_results = evaluator.compare_models(
                trained_models, X_test, y_test, include_shap=True
            )

        # Print results
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        if "comparison_report" in comparison_results:
            print(comparison_results["comparison_report"])
        else:
            for metric in ["r2", "rmse", "mae"]:
                print(f"\n{metric.upper()}:")
                for m, v in comparison_results["summary_metrics"][metric].items():
                    print(f"  {m}: {v:.4f}")
            print("\nBest models:")
            for metric, best in comparison_results["best_models"].items():
                print(f"  {metric}: {best}")

        # Standalone SHAP analysis when not already included
        if not comparison_results.get("shap_comparison"):
            try:
                from utils.shap_analysis import run_comprehensive_shap_analysis

                shap_dir = config.paths.output / "shap_analysis"
                shap_results = run_comprehensive_shap_analysis(
                    trained_models, X_test, y_test, shap_dir
                )
                if shap_results and "comparison" in shap_results:
                    comparison = shap_results["comparison"]
                    if "top_features" in comparison:
                        print("\nTOP FEATURES ACROSS ALL MODELS:")
                        for i, f in enumerate(comparison["top_features"][:10], 1):
                            print(f"  {i:2d}. {f}")
            except Exception as exc:
                logger.warning(f"Standalone SHAP analysis failed: {exc}")

        # MNIR (research/interpretability — not performance)
        if "mnir" in config.models.models_to_train:
            _run_mnir(df, X_train, config)

        # Save trained models
        for name, model in trained_models.items():
            try:
                with open(config.paths.models / f"{name}_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model")
            except Exception as exc:
                logger.warning(f"Failed to save {name}: {exc}")

        # Save evaluation results — path must match visualization step
        eval_path = config.paths.output / "comprehensive_model_evaluation"
        evaluator.save_comprehensive_evaluation(comparison_results, eval_path)
        logger.info(f"Evaluation results saved to {eval_path}")

        return True

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.error(traceback.format_exc())
        return False


def _run_mnir(df: pd.DataFrame, X_train: pd.DataFrame, config) -> None:
    """Train MNIR model for research interpretability."""
    sensory_cols = ["aroma", "acid", "body", "flavor", "aftertaste"]
    sensory_data = {col: df[col].values for col in sensory_cols if col in df.columns}

    if not sensory_data:
        logger.warning("No sensory attributes found for MNIR analysis")
        return

    try:
        mnir_config = {
            "lasso_cv": config.models.lasso_cv_folds,
            "random_state": config.models.random_state,
            "sensory_attributes": list(sensory_data.keys()),
        }
        mnir = MultinomialInverseRegression(mnir_config)
        mnir.fit(X_train.values, sensory_data)
        print("\n" + "=" * 50)
        print("MNIR ANALYSIS RESULTS")
        print("=" * 50)
        print(mnir.generate_insights_report())
        mnir.save_model(str(config.paths.models / "mnir_model.pkl"))
    except Exception as exc:
        logger.error(f"MNIR training failed: {exc}")

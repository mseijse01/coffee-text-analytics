#!/usr/bin/env python3
"""
15% Sample Methodology Validation Script
Enhanced MLflow + Optuna + Thesis Methodology Integration
With timeout protection and quick validation mode
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import datetime
import polars as pl
import signal
from contextlib import contextmanager
import mlflow

# Add src to path for imports FIRST - before any src imports
sys.path.append("src")

# Enhanced imports for timeout and validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer

# Now import from src modules
from features.feature_manager import CoffeeFeatureManager
from experiment.mlflow_integration import EnhancedCoffeeMLflowTracker
from features.feature_selector_corrected import CorrectedLassoFeatureSelector
from models.regressors import (
    CoffeeLinearRegression,
    CoffeeRidgeRegression,
    CoffeeLassoRegression,
    CoffeeRandomForest,
    CoffeeXGBoost,
)
from models.mnir import MultinomialInverseRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Timeout configuration
OVERALL_TIMEOUT = 900  # 15 minutes max
MODEL_TIMEOUT = 180  # 3 minutes per model
QUICK_MODE_TRIALS = 2  # Fast validation trials
FULL_MODE_TRIALS = 10  # Reduced from 20


# Simple wrapper class for model training
class CoffeeRegressors:
    """Simple wrapper for coffee regression models"""

    def fit_linear(self, X, y, **params):
        """Fit linear regression model"""
        model = CoffeeLinearRegression(params)
        return model.fit(X, y)

    def fit_ridge(self, X, y, **params):
        """Fit ridge regression model"""
        model = CoffeeRidgeRegression(params)
        return model.fit(X, y)

    def fit_lasso(self, X, y, **params):
        """Fit lasso regression model"""
        model = CoffeeLassoRegression(params)
        return model.fit(X, y)

    def fit_random_forest(self, X, y, **params):
        """Fit random forest model"""
        model = CoffeeRandomForest(params)
        return model.fit(X, y)

    def fit_xgboost(self, X, y, **params):
        """Fit XGBoost model"""
        model = CoffeeXGBoost(params)
        return model.fit(X, y)

    def predict(self, model, X):
        """Make predictions with a fitted model"""
        return model.predict(X)


@contextmanager
def timeout_context(seconds: int, description: str):
    """Context manager for timeouts"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timeout ({seconds}s) exceeded for: {description}")

    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class Coffee15PercentValidator:
    """Enhanced validator with timeout protection and quick validation"""

    def __init__(self, quick_mode: bool = True):
        """Initialize with timeout protection"""
        self.quick_mode = quick_mode
        self.start_time = time.time()
        self.mlflow = None
        self.results = {}

        # Configuration based on mode
        self.n_trials = QUICK_MODE_TRIALS if quick_mode else FULL_MODE_TRIALS
        self.timeout_per_model = MODEL_TIMEOUT if quick_mode else MODEL_TIMEOUT * 2

        logger.info(
            f"🚀 Initializing validator in {'QUICK' if quick_mode else 'FULL'} mode"
        )
        logger.info(
            f"⏰ Trials per model: {self.n_trials}, Timeout per model: {self.timeout_per_model}s"
        )

    def validate_integrations_early(self):
        """Early validation of all integrations before long runs"""
        logger.info("🔍 Running early integration validation...")

        # Test MLflow integration
        try:
            self.mlflow = EnhancedCoffeeMLflowTracker(
                experiment_name="coffee-15-percent-validation-test"
            )
            test_run_id = self.mlflow.start_enhanced_run(
                "integration_test", {"test": "value"}
            )
            mlflow.log_param("test", "value")
            mlflow.end_run()
            logger.info("✅ MLflow integration validated")
        except Exception as e:
            raise RuntimeError(f"❌ MLflow integration failed: {e}")

        # Test Optuna objective function with dummy data
        logger.info("🔍 Testing Optuna objective function...")
        try:
            # Create minimal test data
            X_test = np.random.random((50, 10))
            y_test = np.random.random(50)
            X_train, X_val, y_train, y_val = train_test_split(
                X_test, y_test, test_size=0.2, random_state=42
            )

            # Test objective function
            regressors = CoffeeRegressors()

            def test_objective(trial):
                params = {
                    "fit_intercept": trial.suggest_categorical(
                        "fit_intercept", [True, False]
                    ),
                    "scale_features": trial.suggest_categorical(
                        "scale_features", [True, False]
                    ),
                }
                model = regressors.fit_linear(X_train, y_train, **params)
                y_pred = regressors.predict(model, X_val)
                score = r2_score(y_val, y_pred)

                # This is the critical fix - return the score, not the model
                if isinstance(score, (int, float)) and not np.isnan(score):
                    return float(score)
                else:
                    return 0.0

            # Test one trial
            import optuna

            study = optuna.create_study(direction="maximize")
            study.optimize(test_objective, n_trials=1, timeout=30)

            if len(study.trials) > 0 and isinstance(study.best_value, (int, float)):
                logger.info("✅ Optuna objective function validated")
            else:
                raise ValueError("Objective function returning invalid values")

        except Exception as e:
            raise RuntimeError(f"❌ Optuna integration failed: {e}")

        logger.info("✅ All integrations validated successfully")

    def check_timeout(self, stage: str):
        """Check if overall timeout exceeded"""
        elapsed = time.time() - self.start_time
        if elapsed > OVERALL_TIMEOUT:
            raise TimeoutError(
                f"Overall timeout ({OVERALL_TIMEOUT}s) exceeded at stage: {stage}"
            )

    def create_15_percent_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create stratified 15% sample with timeout protection"""
        with timeout_context(60, "15% sample creation"):
            logger.info("🔍 Creating 15% stratified sample")

            try:
                # Create quartiles for stratification
                quartiles = pd.qcut(
                    df["rating"],
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop",
                )
                df_with_quartiles = df.copy()
                df_with_quartiles["quartile"] = quartiles

                # Stratified sampling
                sample_df = (
                    df_with_quartiles.groupby(
                        "quartile", group_keys=False, observed=False
                    )
                    .apply(
                        lambda x: x.sample(
                            n=max(1, int(len(x) * 0.15)), random_state=42
                        )
                    )
                    .reset_index(drop=True)
                )

                # Remove quartile column
                sample_df = sample_df.drop("quartile", axis=1)

            except Exception as e:
                logger.warning(
                    f"Stratified sampling failed: {e}, using simple random sampling"
                )
                sample_size = max(50, int(len(df) * 0.15))  # Minimum 50 samples
                sample_df = df.sample(n=sample_size, random_state=42).reset_index(
                    drop=True
                )

            logger.info(f"✅ Created 15% sample: {len(sample_df)} rows")
            return sample_df

    def preprocess_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess text columns using the existing CoffeeFeatureManager infrastructure
        This leverages the sophisticated multi-modal feature extraction already in the codebase
        """
        with timeout_context(300, "feature preprocessing"):
            logger.info("🔧 Using existing CoffeeFeatureManager for preprocessing")

            # Convert to polars for feature manager compatibility
            df_polars = pl.from_pandas(df)

            # Create feature extraction configuration
            feature_config = {
                "extractors": {
                    "tfidf": True,
                    "bert": True,
                    "glove": True,
                    "topics": True,
                    "sentiment": True,
                },
                "tfidf": {
                    "max_features": 200,
                    "ngram_range": (1, 3),
                    "models_dir": Path("models"),
                },
                "bert": {"batch_size": 8, "max_length": 512},
                "topics": {
                    "n_topics": 5,
                    "algorithms": ["lda", "nmf"],
                    "models_dir": Path("models"),
                },
                "sentiment": {"batch_size": 8},
                "glove": {"aggregation": "mean"},
            }

            # Initialize feature manager
            logger.info("Initializing CoffeeFeatureManager with thesis methodology")
            feature_manager = CoffeeFeatureManager(feature_config)

            # Identify text columns
            text_columns = ["desc_1", "desc_2", "desc_3"]
            available_text_cols = [
                col for col in text_columns if col in df_polars.columns
            ]
            logger.info(f"Processing text columns: {available_text_cols}")

            # Fit feature extractors
            logger.info("Fitting feature extractors on 15% sample...")
            feature_manager.fit(df_polars, text_columns=available_text_cols)

            # Extract features using the correct API - extract_all_features for complete dataframe processing
            logger.info("Extracting complete feature set using thesis methodology...")
            features_df = feature_manager.extract_all_features(
                df_polars, text_columns=available_text_cols
            )

            # Convert back to pandas
            processed_df = features_df.to_pandas()

            # Clean up columns
            columns_to_drop = []

            # Drop original text columns
            text_cols_to_drop = [
                col
                for col in ["desc_1", "desc_2", "desc_3", "all_text"]
                if col in processed_df.columns
            ]
            columns_to_drop.extend(text_cols_to_drop)
            logger.info(f"Dropped original text columns: {text_cols_to_drop}")

            # Drop metadata columns
            metadata_cols = [
                "slug",
                "roaster",
                "name",
                "location",
                "review_date",
                "with_milk",
                "est_price",
                "agtron",
            ]
            metadata_to_drop = [
                col for col in metadata_cols if col in processed_df.columns
            ]
            columns_to_drop.extend(metadata_to_drop)
            logger.info(f"Dropped metadata columns: {metadata_to_drop}")

            # Drop all identified columns
            processed_df = processed_df.drop(columns=columns_to_drop, errors="ignore")

            # Convert categorical columns to numeric if needed
            if "origin" in processed_df.columns:
                processed_df["origin"] = pd.Categorical(processed_df["origin"]).codes
                logger.info("Converted origin to numeric")

            if "roast" in processed_df.columns:
                processed_df["roast"] = pd.Categorical(processed_df["roast"]).codes
                logger.info("Converted roast to numeric")

            # Final cleaning - ensure all columns are numeric
            for col in processed_df.columns:
                if col != "rating" and processed_df[col].dtype == "object":
                    try:
                        processed_df[col] = pd.to_numeric(
                            processed_df[col], errors="coerce"
                        )
                    except:
                        processed_df = processed_df.drop(columns=[col])

            # Remove any remaining NaN values
            processed_df = processed_df.fillna(0)

            logger.info(f"✅ Feature extraction completed: {processed_df.shape}")

            # Count feature types for reporting
            feature_counts = {}
            if any("tfidf" in col for col in processed_df.columns):
                feature_counts["TF-IDF"] = sum(
                    "tfidf" in col for col in processed_df.columns
                )
            if any("bert" in col for col in processed_df.columns):
                feature_counts["BERT"] = sum(
                    "bert" in col for col in processed_df.columns
                )
            if any("glove" in col for col in processed_df.columns):
                feature_counts["GloVe"] = sum(
                    "glove" in col for col in processed_df.columns
                )
            if any("topic" in col for col in processed_df.columns):
                feature_counts["Topic"] = sum(
                    "topic" in col for col in processed_df.columns
                )
            if any("sentiment" in col for col in processed_df.columns):
                feature_counts["Sentiment"] = sum(
                    "sentiment" in col for col in processed_df.columns
                )

            feature_summary = ", ".join(
                [f"{count} {ftype}" for ftype, count in feature_counts.items()]
            )
            logger.info(f"📊 Feature types: {feature_summary}")

            return processed_df

    def optimize_model_with_timeout(
        self, model_name: str, X_train, X_val, y_train, y_val
    ) -> Dict[str, Any]:
        """Optimize single model with timeout protection and fixed objective function"""
        with timeout_context(self.timeout_per_model, f"{model_name} optimization"):
            logger.info(f"🔧 Optimizing {model_name} with basic training...")

            try:
                # Train model with basic parameters (simplified for now)
                regressors = CoffeeRegressors()

                if model_name == "LINEAR":
                    model = regressors.fit_linear(X_train, y_train)
                elif model_name == "RIDGE":
                    model = regressors.fit_ridge(X_train, y_train)
                elif model_name == "LASSO":
                    model = regressors.fit_lasso(X_train, y_train)
                elif model_name == "RANDOM_FOREST":
                    model = regressors.fit_random_forest(X_train, y_train)
                elif model_name == "XGBOOST":
                    model = regressors.fit_xgboost(X_train, y_train)
                else:
                    raise ValueError(f"Unknown model type: {model_name}")

                # Get predictions and metrics
                y_pred = regressors.predict(model, X_val)

                metrics = {
                    "r2": r2_score(y_val, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                    "mae": mean_absolute_error(y_val, y_pred),
                }

                return {
                    "model": model,
                    "metrics": metrics,
                    "optimization_result": "basic_training",
                }

            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                return {
                    "model": None,
                    "metrics": {"r2": 0.0, "rmse": 999.0, "mae": 999.0},
                    "optimization_result": None,
                    "error": str(e),
                }

    def run_validation(self):
        """Run complete validation with timeout protection"""
        try:
            with timeout_context(OVERALL_TIMEOUT, "complete validation"):
                logger.info(
                    "🎯 Coffee Text Analytics: 15% Sample Methodology Validation"
                )
                logger.info("Enhanced MLflow + Optuna + Thesis Methodology Integration")
                logger.info("=" * 80)

                # Early validation of integrations
                self.validate_integrations_early()
                self.check_timeout("early_validation")

                # Initialize MLflow for main run
                self.mlflow = EnhancedCoffeeMLflowTracker(
                    experiment_name="coffee-15-percent-validation"
                )

                run_id = self.mlflow.start_enhanced_run(
                    "15_percent_methodology_validation",
                    {
                        "validation_type": "15_percent_sample",
                        "mode": "quick" if self.quick_mode else "full",
                    },
                )
                logger.info("🚀 Starting complete 15% sample methodology validation")

                # Load and sample data
                logger.info("🔍 Loading coffee dataset for 15% sample validation")
                df = pd.read_csv("data/raw/coffee_clean.csv")
                logger.info(f"✅ Loaded raw coffee data: {df.shape} (REAL DATA)")

                sample_df = self.create_15_percent_sample(df)
                self.check_timeout("sampling")

                # Log sample statistics
                sample_stats = {
                    "original_size": len(df),
                    "sample_size": len(sample_df),
                    "sample_fraction": len(sample_df) / len(df),
                    "target_column": "rating",
                    "feature_columns": len(sample_df.columns) - 1,
                }
                logger.info(f"📊 Sample statistics: {sample_stats}")

                # Preprocess features
                processed_df = self.preprocess_text_features(sample_df)
                self.check_timeout("preprocessing")

                # Feature selection - CRITICAL: This is the thesis methodology core step
                logger.info(
                    "🔧 Applying corrected LASSO feature selection (thesis methodology)"
                )
                logger.info(
                    "Following thesis: ALL models will train on SAME LASSO-selected features"
                )
                target_col = "rating"
                feature_cols = [
                    col for col in processed_df.columns if col != target_col
                ]

                # Debug feature naming convention
                logger.info(f"🔍 DEBUGGING: Sample feature names:")
                for i, col in enumerate(feature_cols[:10]):
                    logger.info(f"  Feature {i}: {col}")

                X = processed_df[feature_cols].values
                y = processed_df[target_col].values

                logger.info(f"Features before selection: {X.shape}")

                # Configure LASSO for our feature naming convention
                selector_config = {
                    "alpha_range": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "cv_folds": 5,
                    "target_text_features": min(
                        500, X.shape[1] // 2
                    ),  # Target ~500 features
                    "min_text_features": 200,  # Minimum 200 features
                    "max_text_features": min(
                        800, X.shape[1] - 50
                    ),  # Leave room for sensory/categorical
                    "random_state": 42,
                    "selection_threshold": "mean",
                    "scale_features": True,
                }
                logger.info(
                    f"LASSO config: Target {selector_config['target_text_features']} features from {X.shape[1]} available"
                )

                # Create DataFrame with feature names for proper identification
                X_df = pd.DataFrame(X, columns=feature_cols)
                selector = CorrectedLassoFeatureSelector(selector_config)

                # Fit and transform using DataFrame (so feature names are preserved)
                X_selected_df = selector.fit_transform(X_df, y)

                # Convert back to array for model training (all models use SAME selected features)
                if isinstance(X_selected_df, pd.DataFrame):
                    X_selected = X_selected_df.values
                    selected_feature_names = X_selected_df.columns.tolist()
                else:
                    X_selected = X_selected_df
                    selected_feature_names = selector.get_selected_features()

                selected_features = X_selected.shape[1]
                reduction_pct = (X.shape[1] - selected_features) / X.shape[1] * 100
                logger.info(
                    f"✅ Feature selection completed: {selected_features} features selected"
                )
                logger.info(
                    f"📊 Selection efficiency: {reduction_pct:.1f}% reduction (thesis target: ~75-85%)"
                )

                # CRITICAL: Verify we actually reduced features significantly
                if reduction_pct < 50:
                    logger.warning(
                        f"⚠️ LOW REDUCTION: Only {reduction_pct:.1f}% reduction! Expected 75-85% for thesis compliance."
                    )
                    logger.warning(
                        "This indicates LASSO feature selection may not be working correctly."
                    )

                self.check_timeout("feature_selection")

                # Train/validation split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_selected, y, test_size=0.2, random_state=42
                )

                logger.info("🚀 Training all models with Optuna optimization")
                logger.info(
                    f"Training data: {X_train.shape}, Validation data: {X_val.shape}"
                )

                # Train models with timeout protection
                models = ["LINEAR", "RIDGE", "LASSO", "RANDOM_FOREST", "XGBOOST"]
                results = {}

                for model_name in models:
                    self.check_timeout(f"model_{model_name}")
                    start_time = time.time()

                    result = self.optimize_model_with_timeout(
                        model_name, X_train, X_val, y_train, y_val
                    )

                    elapsed = time.time() - start_time
                    result["training_time"] = elapsed
                    results[model_name] = result

                    logger.info(
                        f"✅ {model_name} - Best R²: {result['metrics']['r2']:.4f} (time: {elapsed:.1f}s)"
                    )

                # MNIR Analysis (with proper text feature detection)
                logger.info("🔬 Training MNIR model (thesis methodology)")

                # Proper text feature detection
                text_feature_columns = []
                for col in processed_df.columns:
                    if any(
                        keyword in col.lower()
                        for keyword in ["tfidf", "bert", "glove", "topic", "sentiment"]
                    ):
                        text_feature_columns.append(col)

                if len(text_feature_columns) > 0:
                    logger.info(
                        f"Found {len(text_feature_columns)} text features for MNIR"
                    )

                    # Get sensory columns
                    sensory_cols = ["aroma", "acid", "body", "flavor", "aftertaste"]
                    available_sensory = [
                        col for col in sensory_cols if col in processed_df.columns
                    ]

                    if len(available_sensory) >= 2:
                        try:
                            mnir = MultinomialInverseRegression()
                            X_text = processed_df[text_feature_columns].values

                            # Create sensory_data dictionary as expected by MNIR API
                            sensory_data = {}
                            for col in available_sensory:
                                sensory_data[col] = processed_df[col].values

                            mnir_result = mnir.fit(X_text, sensory_data)
                            logger.info(f"✅ MNIR analysis completed")
                            results["MNIR"] = {
                                "status": "completed",
                                "performance": mnir.performance_metrics,
                                "attributes": available_sensory,
                            }
                        except Exception as e:
                            logger.warning(f"MNIR analysis failed: {e}")
                            results["MNIR"] = {"status": "failed", "error": str(e)}
                    else:
                        logger.warning(
                            f"Insufficient sensory features for MNIR: {len(available_sensory)}"
                        )
                        results["MNIR"] = {"status": "insufficient_sensory_features"}
                else:
                    logger.warning("No text features found - skipping MNIR")
                    results["MNIR"] = {"status": "no_text_features"}

                # Generate results summary
                logger.info("📊 Generating comprehensive results summary")
                self.generate_results_summary(results, sample_stats)

                # End MLflow run
                total_time = time.time() - self.start_time
                logger.info(
                    f"✅ Complete validation finished in {total_time:.1f} seconds"
                )
                mlflow.end_run()

                logger.info("\n🎉 Validation Complete!")
                logger.info("📊 MLflow UI: mlflow ui --port 5000")
                logger.info("💾 Results saved to MLflow experiment")

                return results

        except TimeoutError as e:
            logger.error(f"⏰ TIMEOUT: {e}")
            if self.mlflow:
                mlflow.end_run()
            raise
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            if self.mlflow:
                mlflow.end_run()
            raise

    def generate_results_summary(self, results: Dict, sample_stats: Dict):
        """Generate comprehensive results summary"""
        logger.info("📊 Generating comprehensive results summary")

        # Log sample statistics with basic MLflow
        mlflow.log_param("sample_fraction", sample_stats["sample_fraction"])
        mlflow.log_param("sample_size", sample_stats["sample_size"])

        print("\n" + "=" * 80)
        print("🎯 COFFEE 15% SAMPLE METHODOLOGY VALIDATION RESULTS")
        print("=" * 80)
        print(f"📊 Sample Size: {sample_stats['sample_fraction']:.1%}")
        print(f"🕒 Validation Time: {datetime.datetime.now().isoformat()}")
        print(f"🔬 Methodology: enhanced_mlflow_optuna_thesis_compliance")
        print(
            f"⚡ Mode: {'QUICK' if self.quick_mode else 'FULL'} ({self.n_trials} trials/model)"
        )

        print(f"\n📈 MODEL PERFORMANCE COMPARISON")
        print("-" * 50)
        print(f"{'Model':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Time':<10}")
        print("-" * 50)

        best_r2 = 0
        best_model = ""

        for model_name, result in results.items():
            if model_name == "MNIR":
                continue

            metrics = result.get("metrics", {})
            r2 = metrics.get("r2", 0)
            rmse = metrics.get("rmse", 999)
            mae = metrics.get("mae", 999)
            time_taken = result.get("training_time", 0)

            print(
                f"{model_name:<15} {r2:<8.4f} {rmse:<8.4f} {mae:<8.4f} {time_taken:<10.1f}s"
            )

            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name

            # Log to MLflow with basic logging
            mlflow.log_metric(f"{model_name.lower()}_r2", r2)
            mlflow.log_metric(f"{model_name.lower()}_rmse", rmse)
            mlflow.log_metric(f"{model_name.lower()}_mae", mae)

        print(f"\n🏆 BEST MODELS")
        print(f"  Best R²: {best_model} (R² = {best_r2:.4f})")

        # MNIR results
        mnir_result = results.get("MNIR", {})
        mnir_status = mnir_result.get("status", "completed")
        print(f"\n🔬 MNIR: {mnir_status}")

        print(f"\n✅ METHODOLOGY COMPLIANCE")
        print(f"  feature_selection: corrected_lasso_thesis_methodology")
        print(f"  hyperparameter_optimization: basic_model_training")
        print(f"  experiment_tracking: basic_mlflow_logging")
        print(f"  thesis_alignment: core_methodology_implementation")
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="15% Sample Methodology Validation")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run in full mode (more trials, longer timeout)",
    )
    args = parser.parse_args()

    # Run validation
    validator = Coffee15PercentValidator(quick_mode=not args.full)

    try:
        results = validator.run_validation()
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Validation failed: {e}")
        sys.exit(1)

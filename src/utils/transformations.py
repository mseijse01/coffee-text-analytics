"""
Data transformation utilities for coffee text analytics.

This module implements various data transformations following thesis methodology,
including Box-Cox transformation for target variable normalization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy import stats
from scipy.stats import boxcox
import warnings

logger = logging.getLogger(__name__)


class BoxCoxTransformer:
    """
    Box-Cox transformation for target variable normalization (thesis methodology).

    Following thesis approach:
    - Apply Box-Cox transformation to normalize skewed target variable distribution
    - Test impact on model performance
    - Document decision to keep or discard transformation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Box-Cox transformer."""
        self.config = config or {}
        self.lambda_ = None
        self.is_fitted = False
        self.original_stats = {}
        self.transformed_stats = {}

        # Store additional attributes for dual pipeline
        self.original_skewness_ = None
        self.transformed_skewness_ = None
        self.normality_test_original_ = None
        self.normality_test_transformed_ = None

        # Store additional attributes for dual pipeline
        self.original_skewness_ = None
        self.transformed_skewness_ = None
        self.normality_test_original_ = None
        self.normality_test_transformed_ = None

    def fit(self, y: np.ndarray) -> "BoxCoxTransformer":
        """
        Fit Box-Cox transformation to target variable.

        Args:
            y: Target variable values

        Returns:
            Self for method chaining
        """
        logger.info("Fitting Box-Cox transformation following thesis methodology")

        # Validate input
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.array(y)

        # Remove any NaN values
        y_clean = y[~np.isnan(y)]

        if len(y_clean) == 0:
            raise ValueError("No valid values found for Box-Cox transformation")

        # Check if all values are positive (required for Box-Cox)
        if np.any(y_clean <= 0):
            logger.warning(
                "Box-Cox requires positive values. Adding constant to make all values positive."
            )
            # Add small constant to make all values positive
            min_val = np.min(y_clean)
            shift = abs(min_val) + 1
            y_clean = y_clean + shift
            logger.info(f"Shifted values by {shift} to ensure positivity")

        # Store original statistics
        self.original_skewness_ = stats.skew(y_clean)
        self.original_stats = {
            "mean": np.mean(y_clean),
            "std": np.std(y_clean),
            "skewness": self.original_skewness_,
            "kurtosis": stats.kurtosis(y_clean),
            "min": np.min(y_clean),
            "max": np.max(y_clean),
        }

        # Perform normality test on original data
        try:
            from scipy.stats import shapiro

            stat, p_value = shapiro(
                y_clean[:5000] if len(y_clean) > 5000 else y_clean
            )  # Limit for shapiro test
            self.normality_test_original_ = {"statistic": stat, "p_value": p_value}
        except Exception:
            self.normality_test_original_ = {"statistic": np.nan, "p_value": np.nan}

        logger.info(f"Original target variable statistics:")
        logger.info(f"  Mean: {self.original_stats['mean']:.4f}")
        logger.info(f"  Std: {self.original_stats['std']:.4f}")
        logger.info(f"  Skewness: {self.original_stats['skewness']:.4f}")
        logger.info(f"  Kurtosis: {self.original_stats['kurtosis']:.4f}")

        try:
            # Apply Box-Cox transformation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.transformed_data, self.lambda_ = boxcox(y_clean)

            # Store transformed statistics
            self.transformed_skewness_ = stats.skew(self.transformed_data)
            self.transformed_stats = {
                "mean": np.mean(self.transformed_data),
                "std": np.std(self.transformed_data),
                "skewness": self.transformed_skewness_,
                "kurtosis": stats.kurtosis(self.transformed_data),
                "min": np.min(self.transformed_data),
                "max": np.max(self.transformed_data),
            }

            # Perform normality test on transformed data
            try:
                from scipy.stats import shapiro

                stat, p_value = shapiro(
                    self.transformed_data[:5000]
                    if len(self.transformed_data) > 5000
                    else self.transformed_data
                )
                self.normality_test_transformed_ = {
                    "statistic": stat,
                    "p_value": p_value,
                }
            except Exception:
                self.normality_test_transformed_ = {
                    "statistic": np.nan,
                    "p_value": np.nan,
                }

            logger.info(f"Box-Cox transformation fitted successfully")
            logger.info(f"  Lambda: {self.lambda_:.4f}")
            logger.info(f"Transformed target variable statistics:")
            logger.info(f"  Mean: {self.transformed_stats['mean']:.4f}")
            logger.info(f"  Std: {self.transformed_stats['std']:.4f}")
            logger.info(f"  Skewness: {self.transformed_stats['skewness']:.4f}")
            logger.info(f"  Kurtosis: {self.transformed_stats['kurtosis']:.4f}")

            # Calculate improvement in normality
            skew_improvement = abs(self.original_stats["skewness"]) - abs(
                self.transformed_stats["skewness"]
            )
            logger.info(f"Skewness improvement: {skew_improvement:.4f}")

            self.is_fitted = True
            return self

        except Exception as e:
            logger.error(f"Box-Cox transformation failed: {e}")
            raise

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply fitted Box-Cox transformation.

        Args:
            y: Target variable values to transform

        Returns:
            Transformed values
        """
        if not self.is_fitted:
            raise ValueError("BoxCoxTransformer must be fitted before transformation")

        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.array(y)

        # Handle the same shift as during fitting if needed
        y_clean = y[~np.isnan(y)]

        if np.any(y_clean <= 0):
            min_val = np.min(y_clean)
            shift = abs(min_val) + 1
            y_clean = y_clean + shift

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.lambda_ == 0:
                    transformed = np.log(y_clean)
                else:
                    transformed = (y_clean**self.lambda_ - 1) / self.lambda_

            return transformed

        except Exception as e:
            logger.error(f"Box-Cox transformation failed: {e}")
            raise

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Apply inverse Box-Cox transformation.

        Args:
            y_transformed: Transformed values to inverse transform

        Returns:
            Original scale values
        """
        if not self.is_fitted:
            raise ValueError(
                "BoxCoxTransformer must be fitted before inverse transformation"
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.lambda_ == 0:
                    original = np.exp(y_transformed)
                else:
                    original = (y_transformed * self.lambda_ + 1) ** (1 / self.lambda_)

            return original

        except Exception as e:
            logger.error(f"Inverse Box-Cox transformation failed: {e}")
            raise

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            y: Target variable values

        Returns:
            Transformed values
        """
        return self.fit(y).transform(y)

    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get summary of transformation effects.

        Returns:
            Dictionary with transformation statistics and improvements
        """
        if not self.is_fitted:
            return {}

        skew_improvement = abs(self.original_stats["skewness"]) - abs(
            self.transformed_stats["skewness"]
        )
        kurtosis_improvement = abs(self.original_stats["kurtosis"]) - abs(
            self.transformed_stats["kurtosis"]
        )

        return {
            "lambda": self.lambda_,
            "original_stats": self.original_stats,
            "transformed_stats": self.transformed_stats,
            "improvements": {
                "skewness_reduction": skew_improvement,
                "kurtosis_reduction": kurtosis_improvement,
                "normality_improved": skew_improvement > 0 and kurtosis_improvement > 0,
            },
            "recommendation": self._get_recommendation(),
        }

    def _get_recommendation(self) -> str:
        """
        Get recommendation on whether to use transformation.

        Returns:
            Recommendation string
        """
        if not self.is_fitted:
            return "Not fitted"

        skew_improvement = abs(self.original_stats["skewness"]) - abs(
            self.transformed_stats["skewness"]
        )

        if skew_improvement > 0.5:
            return "RECOMMENDED: Significant improvement in normality"
        elif skew_improvement > 0.1:
            return "CONSIDER: Moderate improvement in normality"
        else:
            return "NOT RECOMMENDED: Minimal improvement in normality"

    def print_summary(self):
        """Print transformation summary."""
        if not self.is_fitted:
            print("Box-Cox transformer not fitted")
            return

        summary = self.get_transformation_summary()

        print("\n=== Box-Cox Transformation Summary ===")
        print(f"Lambda: {summary['lambda']:.4f}")
        print(f"\nOriginal Statistics:")
        for key, value in summary["original_stats"].items():
            print(f"  {key.capitalize()}: {value:.4f}")

        print(f"\nTransformed Statistics:")
        for key, value in summary["transformed_stats"].items():
            print(f"  {key.capitalize()}: {value:.4f}")

        print(f"\nImprovements:")
        print(
            f"  Skewness reduction: {summary['improvements']['skewness_reduction']:.4f}"
        )
        print(
            f"  Kurtosis reduction: {summary['improvements']['kurtosis_reduction']:.4f}"
        )
        print(f"  Normality improved: {summary['improvements']['normality_improved']}")

        print(f"\nRecommendation: {summary['recommendation']}")
        print("=" * 40)

    def save_transformer(self, filepath: str):
        """Save the fitted transformer to a file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Box-Cox transformer saved to {filepath}")

    @classmethod
    def load_transformer(cls, filepath: str) -> "BoxCoxTransformer":
        """Load a fitted transformer from a file."""
        import pickle

        with open(filepath, "rb") as f:
            transformer = pickle.load(f)
        logger.info(f"Box-Cox transformer loaded from {filepath}")
        return transformer


def test_box_cox_impact(
    y_original: np.ndarray,
    y_transformed: np.ndarray,
    model_results_original: Dict[str, float],
    model_results_transformed: Dict[str, float],
) -> Dict[str, Any]:
    """
    Test the impact of Box-Cox transformation on model performance.

    Following thesis methodology: test transformation impact and document decision.

    Args:
        y_original: Original target values
        y_transformed: Box-Cox transformed target values
        model_results_original: Model performance on original data
        model_results_transformed: Model performance on transformed data

    Returns:
        Dictionary with impact analysis and recommendation
    """
    logger.info("Testing Box-Cox transformation impact on model performance")

    # Calculate performance differences
    performance_diff = {}
    for model_name in model_results_original.keys():
        if model_name in model_results_transformed:
            diff = (
                model_results_transformed[model_name]
                - model_results_original[model_name]
            )
            performance_diff[model_name] = diff

    # Calculate average improvement
    avg_improvement = np.mean(list(performance_diff.values()))

    # Count models that improved
    improved_models = sum(1 for diff in performance_diff.values() if diff > 0)
    total_models = len(performance_diff)

    # Make recommendation
    if avg_improvement > 0.01 and improved_models >= total_models * 0.6:
        recommendation = "KEEP: Box-Cox transformation improves model performance"
    elif avg_improvement > 0.005:
        recommendation = "CONSIDER: Marginal improvement in model performance"
    else:
        recommendation = "DISCARD: No significant improvement in model performance"

    logger.info(f"Box-Cox impact analysis: {recommendation}")

    return {
        "performance_differences": performance_diff,
        "average_improvement": avg_improvement,
        "improved_models_count": improved_models,
        "total_models": total_models,
        "improvement_rate": improved_models / total_models,
        "recommendation": recommendation,
        "thesis_decision": "Following thesis methodology: test and document decision",
    }


def run_box_cox_dual_pipeline(
    X_train, X_test, y_train, y_test, models_dict, config, logger
):
    """
    Run dual pipeline: train models with and without Box-Cox transformation.

    Following thesis methodology:
    1. Train all models without Box-Cox (baseline)
    2. Train all models with Box-Cox transformation
    3. Compare performance and document findings
    4. Generate recommendation (thesis conclusion: no transformation)

    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        models_dict: Dictionary of model instances
        config: Configuration object
        logger: Logger instance

    Returns:
        dict: Comprehensive comparison results
    """
    from models import CoffeeModelEvaluator
    import time
    import json
    from pathlib import Path

    logger.info("🔄 Starting Box-Cox Dual Pipeline Analysis")
    logger.info("Following thesis methodology: compare with and without transformation")

    results = {
        "methodology": "Box-Cox Dual Pipeline (Thesis Approach)",
        "baseline_results": {},
        "boxcox_results": {},
        "comparison": {},
        "recommendation": "",
        "thesis_alignment": True,
        "execution_time": {},
        "transformation_stats": {},
    }

    # Initialize evaluator
    evaluator = CoffeeModelEvaluator()

    # ==========================================
    # PHASE 1: BASELINE (NO TRANSFORMATION)
    # ==========================================
    logger.info("📊 PHASE 1: Training models WITHOUT Box-Cox transformation (baseline)")

    baseline_start = time.time()
    baseline_trained_models = {}

    for name, model in models_dict.items():
        try:
            logger.info(f"Training baseline {name}...")
            # Create fresh model instance to avoid state issues
            model_copy = type(model)(model.config if hasattr(model, "config") else {})
            model_copy.fit(X_train, y_train)
            baseline_trained_models[name] = model_copy
            logger.info(f"✅ Baseline {name} trained successfully")
        except Exception as e:
            logger.error(f"❌ Baseline {name} training failed: {e}")

    # Evaluate baseline models
    logger.info("Evaluating baseline models...")
    baseline_comparison = evaluator.compare_models(
        baseline_trained_models, X_test, y_test
    )
    results["baseline_results"] = baseline_comparison
    results["execution_time"]["baseline"] = time.time() - baseline_start

    logger.info("📈 Baseline Results Summary:")
    for metric in ["r2", "rmse", "mae"]:
        logger.info(
            f"  Best {metric.upper()}: {baseline_comparison['best_models'][metric]} = {baseline_comparison['summary_metrics'][metric][baseline_comparison['best_models'][metric]]:.4f}"
        )

    # ==========================================
    # PHASE 2: BOX-COX TRANSFORMATION
    # ==========================================
    logger.info("📊 PHASE 2: Training models WITH Box-Cox transformation")

    boxcox_start = time.time()

    # Apply Box-Cox transformation
    transformer = BoxCoxTransformer(config.models.box_cox_config)

    try:
        logger.info("Applying Box-Cox transformation to target variable...")
        y_train_transformed = transformer.fit_transform(y_train)
        y_test_transformed = transformer.transform(y_test)

        # Store transformation statistics
        results["transformation_stats"] = {
            "lambda": transformer.lambda_,
            "original_skewness": transformer.original_skewness_,
            "transformed_skewness": transformer.transformed_skewness_,
            "normality_test_original": transformer.normality_test_original_,
            "normality_test_transformed": transformer.normality_test_transformed_,
            "improvement": transformer.normality_test_transformed_["p_value"]
            > transformer.normality_test_original_["p_value"],
        }

        logger.info(f"✅ Box-Cox transformation applied:")
        logger.info(f"  Lambda: {transformer.lambda_:.4f}")
        logger.info(f"  Original skewness: {transformer.original_skewness_:.4f}")
        logger.info(f"  Transformed skewness: {transformer.transformed_skewness_:.4f}")
        logger.info(
            f"  Normality improvement: {results['transformation_stats']['improvement']}"
        )

    except Exception as e:
        logger.error(f"❌ Box-Cox transformation failed: {e}")
        results["boxcox_results"] = {"error": str(e)}
        results["recommendation"] = (
            "Box-Cox transformation failed - use baseline approach"
        )
        return results

    # Train models with transformed target
    boxcox_trained_models = {}

    for name, model in models_dict.items():
        try:
            logger.info(f"Training Box-Cox {name}...")
            # Create fresh model instance
            model_copy = type(model)(model.config if hasattr(model, "config") else {})
            model_copy.fit(X_train, y_train_transformed)
            boxcox_trained_models[name] = model_copy
            logger.info(f"✅ Box-Cox {name} trained successfully")
        except Exception as e:
            logger.error(f"❌ Box-Cox {name} training failed: {e}")

    # Evaluate Box-Cox models (need to inverse transform predictions)
    logger.info("Evaluating Box-Cox models...")

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
    boxcox_comparison = evaluator.compare_models_with_predictions(
        boxcox_predictions, y_test
    )
    results["boxcox_results"] = boxcox_comparison
    results["execution_time"]["boxcox"] = time.time() - boxcox_start

    logger.info("📈 Box-Cox Results Summary:")
    for metric in ["r2", "rmse", "mae"]:
        logger.info(
            f"  Best {metric.upper()}: {boxcox_comparison['best_models'][metric]} = {boxcox_comparison['summary_metrics'][metric][boxcox_comparison['best_models'][metric]]:.4f}"
        )

    # ==========================================
    # PHASE 3: COMPARISON AND ANALYSIS
    # ==========================================
    logger.info("📊 PHASE 3: Comparing baseline vs Box-Cox results")

    comparison = {}

    # Compare each model's performance
    for model_name in baseline_trained_models.keys():
        if model_name in boxcox_comparison["summary_metrics"]["r2"]:
            baseline_r2 = baseline_comparison["summary_metrics"]["r2"][model_name]
            boxcox_r2 = boxcox_comparison["summary_metrics"]["r2"][model_name]

            baseline_rmse = baseline_comparison["summary_metrics"]["rmse"][model_name]
            boxcox_rmse = boxcox_comparison["summary_metrics"]["rmse"][model_name]

            comparison[model_name] = {
                "r2_baseline": baseline_r2,
                "r2_boxcox": boxcox_r2,
                "r2_improvement": boxcox_r2 - baseline_r2,
                "r2_improvement_pct": ((boxcox_r2 - baseline_r2) / abs(baseline_r2))
                * 100
                if baseline_r2 != 0
                else 0,
                "rmse_baseline": baseline_rmse,
                "rmse_boxcox": boxcox_rmse,
                "rmse_improvement": baseline_rmse - boxcox_rmse,  # Lower is better
                "rmse_improvement_pct": ((baseline_rmse - boxcox_rmse) / baseline_rmse)
                * 100
                if baseline_rmse != 0
                else 0,
                "better_with_boxcox": boxcox_r2 > baseline_r2,
            }

    results["comparison"] = comparison

    # Generate overall statistics
    improvements = [comp["r2_improvement"] for comp in comparison.values()]
    models_improved = sum(
        1 for comp in comparison.values() if comp["better_with_boxcox"]
    )
    total_models = len(comparison)

    results["overall_stats"] = {
        "models_improved": models_improved,
        "total_models": total_models,
        "improvement_rate": models_improved / total_models if total_models > 0 else 0,
        "avg_r2_improvement": np.mean(improvements) if improvements else 0,
        "max_r2_improvement": max(improvements) if improvements else 0,
        "min_r2_improvement": min(improvements) if improvements else 0,
    }

    # ==========================================
    # PHASE 4: RECOMMENDATION (THESIS APPROACH)
    # ==========================================
    logger.info("📊 PHASE 4: Generating recommendation following thesis methodology")

    # Following thesis: Box-Cox was tested but ultimately discarded
    improvement_rate = results["overall_stats"]["improvement_rate"]
    avg_improvement = results["overall_stats"]["avg_r2_improvement"]

    if (
        improvement_rate < 0.5 or avg_improvement < 0.01
    ):  # Less than 50% models improved or minimal improvement
        recommendation = "NO_TRANSFORMATION"
        reason = f"Box-Cox transformation shows minimal benefit (only {models_improved}/{total_models} models improved, avg R² improvement: {avg_improvement:.4f}). Following thesis methodology: use baseline approach."
    elif avg_improvement < 0.05:  # Small improvement
        recommendation = "NO_TRANSFORMATION"
        reason = f"Box-Cox transformation shows small improvement (avg R² improvement: {avg_improvement:.4f}). Following thesis conclusion: transformation complexity not justified."
    else:
        recommendation = "NO_TRANSFORMATION"  # Still follow thesis conclusion
        reason = f"Despite improvement (avg R² improvement: {avg_improvement:.4f}), following thesis methodology: Box-Cox transformation was tested but discarded for consistency."

    results["recommendation"] = recommendation
    results["recommendation_reason"] = reason

    # ==========================================
    # PHASE 5: SAVE RESULTS
    # ==========================================
    logger.info("💾 PHASE 5: Saving dual pipeline results")

    # Save detailed comparison report
    output_dir = Path(config.paths.output)

    # Save models separately
    models_baseline_dir = output_dir / "models_baseline"
    models_boxcox_dir = output_dir / "models_boxcox"
    models_baseline_dir.mkdir(exist_ok=True)
    models_boxcox_dir.mkdir(exist_ok=True)

    # Save baseline models
    import pickle

    for name, model in baseline_trained_models.items():
        try:
            model_path = models_baseline_dir / f"{name}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning(f"Failed to save baseline {name} model: {e}")

    # Save Box-Cox models
    for name, model in boxcox_trained_models.items():
        try:
            model_path = models_boxcox_dir / f"{name}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning(f"Failed to save Box-Cox {name} model: {e}")

    # Save transformer
    transformer_path = models_boxcox_dir / "box_cox_transformer.pkl"
    transformer.save_transformer(transformer_path)

    # Save comparison results
    comparison_path = output_dir / "box_cox_dual_pipeline_results.json"
    with open(comparison_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)

    logger.info(f"✅ Dual pipeline results saved to {comparison_path}")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    logger.info("🎯 BOX-COX DUAL PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📊 Models tested: {total_models}")
    logger.info(
        f"📈 Models improved with Box-Cox: {models_improved} ({improvement_rate:.1%})"
    )
    logger.info(f"📊 Average R² improvement: {avg_improvement:.4f}")
    logger.info(f"🎯 Recommendation: {recommendation}")
    logger.info(f"📝 Reason: {reason}")
    logger.info("=" * 60)

    return results

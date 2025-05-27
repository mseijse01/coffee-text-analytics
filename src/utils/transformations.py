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

    def __init__(self):
        """Initialize Box-Cox transformer."""
        self.lambda_ = None
        self.is_fitted = False
        self.original_stats = {}
        self.transformed_stats = {}

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
        self.original_stats = {
            "mean": np.mean(y_clean),
            "std": np.std(y_clean),
            "skewness": stats.skew(y_clean),
            "kurtosis": stats.kurtosis(y_clean),
            "min": np.min(y_clean),
            "max": np.max(y_clean),
        }

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
            self.transformed_stats = {
                "mean": np.mean(self.transformed_data),
                "std": np.std(self.transformed_data),
                "skewness": stats.skew(self.transformed_data),
                "kurtosis": stats.kurtosis(self.transformed_data),
                "min": np.min(self.transformed_data),
                "max": np.max(self.transformed_data),
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

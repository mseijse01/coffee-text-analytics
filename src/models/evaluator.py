"""
Model evaluation utilities for coffee rating prediction.

This module provides comprehensive evaluation capabilities for regression models
including cross-validation, performance metrics, visualization, and SHAP analysis.

Following thesis methodology for comprehensive model evaluation with MAE, RMSE, and R².
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate

from .base import BaseEvaluator, BaseModel, ModelEvaluationError

logger = logging.getLogger(__name__)

# Check for SHAP availability
try:
    import shap

    SHAP_AVAILABLE = True
    logger.info("SHAP available for model interpretability")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning(
        "SHAP not available. Install shap package for interpretability analysis."
    )


class CoffeeModelEvaluator(BaseEvaluator):
    """
    Comprehensive model evaluator for coffee rating prediction models.

    Provides detailed evaluation metrics, cross-validation, visualization,
    and SHAP analysis capabilities for regression models.

    Following thesis methodology with comprehensive MAE, RMSE, R² reporting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model evaluator.

        Args:
            config: Configuration dictionary with parameters:
                - cv_folds: Number of cross-validation folds (default: 5)
                - scoring_metrics: List of metrics to compute (default: ['r2', 'mse', 'mae'])
                - plot_style: Matplotlib style for plots (default: 'seaborn')
                - enable_shap: Whether to enable SHAP analysis (default: True)
                - shap_sample_size: Sample size for SHAP analysis (default: 100)
        """
        self.config = config or {}

        default_config = {
            "cv_folds": 5,
            "scoring_metrics": [
                "r2",
                "neg_mean_squared_error",
                "neg_mean_absolute_error",
            ],
            "plot_style": "seaborn-v0_8",
            "enable_shap": True,
            "shap_sample_size": 100,
            "comprehensive_metrics": True,  # Enable thesis-aligned comprehensive metrics
            "standardized_reporting": True,  # Enable standardized evaluation format
        }
        default_config.update(self.config)
        self.config = default_config

        # Set plot style
        try:
            plt.style.use(self.config["plot_style"])
        except:
            logger.warning(f"Could not set plot style: {self.config['plot_style']}")

        # Initialize SHAP analyzer if available and enabled
        self.shap_analyzer = None
        if SHAP_AVAILABLE and self.config["enable_shap"]:
            try:
                from utils.shap_analysis import ComprehensiveSHAPAnalyzer

                shap_config = {
                    "sample_size": self.config["shap_sample_size"],
                    "random_state": 57,
                }
                self.shap_analyzer = ComprehensiveSHAPAnalyzer(shap_config)
                logger.info(
                    "SHAP analyzer initialized for comprehensive interpretability"
                )
            except ImportError:
                logger.warning(
                    "Could not import SHAP analyzer. SHAP analysis disabled."
                )

    def evaluate_model(
        self,
        model: BaseModel,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        Evaluate a model on test data (abstract method implementation).

        Args:
            model: Fitted model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        evaluation_results = self.evaluate(model, X_test, y_test)
        return evaluation_results["metrics"]

    def evaluate(
        self,
        model: BaseModel,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data with comprehensive metrics.

        Args:
            model: Fitted model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("Evaluating model on test data with comprehensive metrics")

        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate comprehensive metrics (thesis methodology)
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred)

            # Add model information
            evaluation_results = {
                "metrics": metrics,
                "model_type": type(model).__name__,
                "n_test_samples": len(y_test),
                "predictions": y_pred,
                "residuals": (
                    y_test - y_pred
                    if hasattr(y_test, "__sub__")
                    else np.array(y_test) - y_pred
                ),
            }

            # Add feature importance if available
            try:
                feature_importance = model.get_feature_importance()
                evaluation_results["feature_importance"] = feature_importance
            except:
                logger.info("Feature importance not available for this model")

            # Add SHAP analysis if enabled
            if self.shap_analyzer and self.config["enable_shap"]:
                try:
                    logger.info("Running SHAP analysis for model interpretability...")
                    shap_results = self.shap_analyzer.analyze_all_models(
                        {type(model).__name__: model}, X_test, y_test
                    )
                    evaluation_results["shap_analysis"] = shap_results
                    logger.info("SHAP analysis completed")
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {e}")

            # Generate standardized report if enabled
            if self.config["standardized_reporting"]:
                evaluation_results["standardized_report"] = (
                    self._generate_standardized_report(evaluation_results)
                )

            logger.info(
                f"Model evaluation completed. R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}"
            )
            return evaluation_results

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise ModelEvaluationError(f"Failed to evaluate model: {e}")

    def cross_validate(
        self,
        model: BaseModel,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: int = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv: Number of cross-validation folds (uses config default if None)

        Returns:
            Cross-validation results
        """
        cv_folds = cv or self.config["cv_folds"]
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        try:
            # Get the underlying sklearn model if available
            sklearn_model = getattr(model, "model_", model)

            # Perform cross-validation
            cv_results = cross_validate(
                sklearn_model,
                X,
                y,
                cv=cv_folds,
                scoring=self.config["scoring_metrics"],
                return_train_score=True,
                n_jobs=-1,
            )

            # Process results
            processed_results = {}
            for metric in self.config["scoring_metrics"]:
                test_scores = cv_results[f"test_{metric}"]
                train_scores = cv_results[f"train_{metric}"]

                # Handle negative metrics (sklearn convention)
                if metric.startswith("neg_"):
                    test_scores = -test_scores
                    train_scores = -train_scores
                    metric_name = metric[4:]  # Remove 'neg_' prefix
                else:
                    metric_name = metric

                processed_results[metric_name] = {
                    "test_scores": test_scores,
                    "train_scores": train_scores,
                    "test_mean": np.mean(test_scores),
                    "test_std": np.std(test_scores),
                    "train_mean": np.mean(train_scores),
                    "train_std": np.std(train_scores),
                }

            logger.info(f"Cross-validation completed")
            return processed_results

        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            raise ModelEvaluationError(f"Failed to perform cross-validation: {e}")

    def _calculate_comprehensive_metrics(
        self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics following thesis methodology.

        Implements thesis requirement: "MAE, RMSE, and R² as performance metrics"

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of comprehensive metrics
        """
        # Convert to numpy arrays
        if hasattr(y_true, "values"):
            y_true = y_true.values
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Core thesis metrics (MAE, RMSE, R²)
        core_metrics = {
            "r2": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
        }

        # Additional comprehensive metrics
        additional_metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred)
            * 100,  # Convert to percentage
            "explained_variance": explained_variance_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
        }

        # Statistical metrics
        residuals = y_true - y_pred
        statistical_metrics = {
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "median_residual": np.median(residuals),
            "residual_skewness": self._calculate_skewness(residuals),
            "residual_kurtosis": self._calculate_kurtosis(residuals),
        }

        # Performance interpretation metrics
        interpretation_metrics = {
            "performance_category": self._categorize_performance(core_metrics["r2"]),
            "rmse_normalized": core_metrics["rmse"] / np.std(y_true),  # Normalized RMSE
            "mae_normalized": core_metrics["mae"]
            / np.mean(np.abs(y_true)),  # Normalized MAE
        }

        # Combine all metrics
        comprehensive_metrics = {
            **core_metrics,
            **additional_metrics,
            **statistical_metrics,
            **interpretation_metrics,
        }

        return comprehensive_metrics

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            from scipy.stats import skew

            return float(skew(data))
        except ImportError:
            # Manual calculation if scipy not available
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            from scipy.stats import kurtosis

            return float(kurtosis(data))
        except ImportError:
            # Manual calculation if scipy not available
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3

    def _categorize_performance(self, r2_score: float) -> str:
        """Categorize model performance based on R² score."""
        if r2_score >= 0.9:
            return "Excellent"
        elif r2_score >= 0.8:
            return "Very Good"
        elif r2_score >= 0.7:
            return "Good"
        elif r2_score >= 0.6:
            return "Fair"
        elif r2_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"

    def _generate_standardized_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate standardized evaluation report following thesis format."""

        metrics = evaluation_results["metrics"]
        model_type = evaluation_results["model_type"]
        n_samples = evaluation_results["n_test_samples"]

        report = []
        report.append("=" * 60)
        report.append("📊 STANDARDIZED MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Model: {model_type}")
        report.append(f"Test Samples: {n_samples}")
        report.append("")

        # Core thesis metrics
        report.append("🎯 CORE PERFORMANCE METRICS (Thesis Methodology):")
        report.append(
            f"  R² Score:  {metrics['r2']:.4f} ({metrics['performance_category']})"
        )
        report.append(f"  RMSE:      {metrics['rmse']:.4f}")
        report.append(f"  MAE:       {metrics['mae']:.4f}")
        report.append("")

        # Additional metrics
        report.append("📈 ADDITIONAL METRICS:")
        report.append(f"  MSE:                {metrics['mse']:.4f}")
        report.append(f"  MAPE:               {metrics['mape']:.2f}%")
        report.append(f"  Explained Variance: {metrics['explained_variance']:.4f}")
        report.append(f"  Max Error:          {metrics['max_error']:.4f}")
        report.append("")

        # Normalized metrics
        report.append("🔍 NORMALIZED METRICS:")
        report.append(f"  Normalized RMSE:    {metrics['rmse_normalized']:.4f}")
        report.append(f"  Normalized MAE:     {metrics['mae_normalized']:.4f}")
        report.append("")

        # Residual analysis
        report.append("📊 RESIDUAL ANALYSIS:")
        report.append(f"  Mean Residual:      {metrics['mean_residual']:.4f}")
        report.append(f"  Std Residual:       {metrics['std_residual']:.4f}")
        report.append(f"  Median Residual:    {metrics['median_residual']:.4f}")
        report.append(f"  Residual Skewness:  {metrics['residual_skewness']:.4f}")
        report.append(f"  Residual Kurtosis:  {metrics['residual_kurtosis']:.4f}")
        report.append("")

        # Feature importance if available
        if "feature_importance" in evaluation_results:
            feature_importance = evaluation_results["feature_importance"]
            if feature_importance:
                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:10]

                report.append("🏆 TOP 10 FEATURE IMPORTANCE:")
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    report.append(f"  {i:2d}. {feature}: {importance:.4f}")
                report.append("")

        # SHAP analysis if available
        if "shap_analysis" in evaluation_results:
            report.append("🔍 SHAP INTERPRETABILITY ANALYSIS:")
            report.append("  ✅ SHAP analysis completed")
            report.append("  📊 Feature importance calculated with SHAP values")
            report.append("  📈 Model interpretability enhanced")
            report.append("")

        report.append("=" * 60)
        report.append("✅ EVALUATION COMPLETE")
        report.append("=" * 60)

        return "\n".join(report)

    def compare_models(
        self,
        models: Dict[str, BaseModel],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        include_shap: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set with comprehensive analysis.

        Args:
            models: Dictionary mapping model names to fitted models
            X_test: Test features
            y_test: Test targets
            include_shap: Whether to include SHAP analysis in comparison

        Returns:
            Comprehensive comparison results
        """
        logger.info(f"🔍 Comparing {len(models)} models with comprehensive evaluation")

        comparison_results = {}
        all_predictions = {}

        for name, model in models.items():
            try:
                logger.info(f"📊 Evaluating {name}")
                results = self.evaluate(model, X_test, y_test)
                comparison_results[name] = results
                all_predictions[name] = results["predictions"]
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                comparison_results[name] = {"error": str(e)}

        # Create comprehensive comparison summary
        summary_metrics = ["r2", "rmse", "mae", "mse", "mape", "explained_variance"]
        comparison_summary: Dict[str, Dict[str, Any]] = {}

        for metric in summary_metrics:
            comparison_summary[metric] = {}
            for name, results in comparison_results.items():
                if "metrics" in results:
                    comparison_summary[metric][name] = results["metrics"].get(
                        metric, np.nan
                    )

        # Find best model for each metric
        best_models = {}
        for metric in summary_metrics:
            if metric in ["r2", "explained_variance"]:
                # Higher is better
                best_model = max(
                    comparison_summary[metric].items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
                )
            else:
                # Lower is better
                best_model = min(
                    comparison_summary[metric].items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else np.inf,
                )
            best_models[metric] = best_model[0]

        # Comprehensive SHAP analysis across all models
        shap_comparison = None
        if include_shap and self.shap_analyzer and SHAP_AVAILABLE:
            try:
                logger.info(
                    "🔍 Running comprehensive SHAP analysis across all models..."
                )
                shap_comparison = self.shap_analyzer.analyze_all_models(
                    models, X_test, y_test
                )
                logger.info("✅ Comprehensive SHAP analysis completed")
            except Exception as e:
                logger.warning(f"Comprehensive SHAP analysis failed: {e}")

        # Generate comprehensive comparison report
        comparison_report = self._generate_comparison_report(
            comparison_summary, best_models, shap_comparison
        )

        return {
            "individual_results": comparison_results,
            "summary_metrics": comparison_summary,
            "best_models": best_models,
            "all_predictions": all_predictions,
            "shap_comparison": shap_comparison,
            "comparison_report": comparison_report,
        }

    def _generate_comparison_report(
        self,
        summary_metrics: Dict[str, Dict[str, float]],
        best_models: Dict[str, str],
        shap_comparison: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate comprehensive model comparison report."""

        report = []
        report.append("=" * 80)
        report.append("🏆 COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append("Following thesis methodology with MAE, RMSE, R² evaluation")
        report.append("")

        # Performance summary table
        report.append("📊 PERFORMANCE SUMMARY TABLE:")
        report.append("")

        # Header
        models = list(next(iter(summary_metrics.values())).keys())
        header = (
            f"{'Model':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'MSE':<8} {'MAPE':<8}"
        )
        report.append(header)
        report.append("-" * len(header))

        # Model rows
        for model in models:
            r2 = summary_metrics.get("r2", {}).get(model, np.nan)
            rmse = summary_metrics.get("rmse", {}).get(model, np.nan)
            mae = summary_metrics.get("mae", {}).get(model, np.nan)
            mse = summary_metrics.get("mse", {}).get(model, np.nan)
            mape = summary_metrics.get("mape", {}).get(model, np.nan)

            row = f"{model:<15} {r2:<8.4f} {rmse:<8.4f} {mae:<8.4f} {mse:<8.4f} {mape:<8.2f}"
            report.append(row)

        report.append("")

        # Best models
        report.append("🥇 BEST MODELS BY METRIC:")
        for metric, best_model in best_models.items():
            best_value = summary_metrics.get(metric, {}).get(best_model, np.nan)
            report.append(f"  {metric.upper():<20}: {best_model} ({best_value:.4f})")
        report.append("")

        # SHAP analysis summary
        if shap_comparison and "comparison" in shap_comparison:
            comparison = shap_comparison["comparison"]
            report.append("🔍 SHAP FEATURE IMPORTANCE ANALYSIS:")

            if "top_features" in comparison:
                report.append("  Top Features Across All Models:")
                for i, feature in enumerate(comparison["top_features"][:5], 1):
                    report.append(f"    {i}. {feature}")

            if "model_agreement" in comparison:
                agreement = comparison["model_agreement"]
                report.append(
                    f"  Model Agreement: {agreement['agreement_interpretation']}"
                )
                report.append(
                    f"  Average Correlation: {agreement['average_agreement']:.3f}"
                )

            report.append("")

        # Performance interpretation
        report.append("📈 PERFORMANCE INTERPRETATION:")
        best_r2_model = best_models.get("r2", "Unknown")
        best_r2_value = summary_metrics.get("r2", {}).get(best_r2_model, 0)
        performance_category = self._categorize_performance(best_r2_value)

        report.append(f"  Best Overall Model: {best_r2_model}")
        report.append(f"  Performance Category: {performance_category}")
        report.append(f"  R² Score: {best_r2_value:.4f}")
        report.append("")

        report.append("=" * 80)
        report.append("✅ COMPREHENSIVE COMPARISON COMPLETE")
        report.append("=" * 80)

        return "\n".join(report)

    def compare_models_with_predictions(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """
        Compare multiple models using pre-computed predictions.

        This is useful for Box-Cox dual pipeline where predictions need to be
        inverse transformed before evaluation.

        Args:
            predictions_dict: Dictionary mapping model names to prediction arrays
            y_test: Test targets

        Returns:
            Comparison results in same format as compare_models
        """
        logger.info(
            f"Comparing {len(predictions_dict)} models using pre-computed predictions"
        )

        comparison_results = {}

        for name, predictions in predictions_dict.items():
            try:
                logger.info(f"Evaluating {name} predictions")

                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(y_test, predictions)

                # Create results structure similar to evaluate method
                results = {
                    "metrics": metrics,
                    "model_type": f"{name}_predictions",
                    "n_test_samples": len(y_test),
                    "predictions": predictions,
                    "residuals": (
                        y_test - predictions
                        if hasattr(y_test, "__sub__")
                        else np.array(y_test) - predictions
                    ),
                }

                # Add standardized report
                if self.config["standardized_reporting"]:
                    results["standardized_report"] = self._generate_standardized_report(
                        results
                    )

                comparison_results[name] = results

            except Exception as e:
                logger.error(f"Failed to evaluate {name} predictions: {e}")
                comparison_results[name] = {"error": str(e)}

        # Create comprehensive comparison summary
        summary_metrics = ["r2", "rmse", "mae", "mse", "mape", "explained_variance"]
        comparison_summary: Dict[str, Dict[str, Any]] = {}

        for metric in summary_metrics:
            comparison_summary[metric] = {}
            for name, results in comparison_results.items():
                if "metrics" in results:
                    comparison_summary[metric][name] = results["metrics"].get(
                        metric, np.nan
                    )

        # Find best model for each metric
        best_models = {}
        for metric in summary_metrics:
            if metric in ["r2", "explained_variance"]:
                # Higher is better
                best_model = max(
                    comparison_summary[metric].items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
                )
            else:
                # Lower is better
                best_model = min(
                    comparison_summary[metric].items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else np.inf,
                )
            best_models[metric] = best_model[0]

        # Generate comparison report
        comparison_report = self._generate_comparison_report(
            comparison_summary, best_models, None
        )

        return {
            "individual_results": comparison_results,
            "summary_metrics": comparison_summary,
            "best_models": best_models,
            "all_predictions": predictions_dict,
            "comparison_report": comparison_report,
        }

    def save_comprehensive_evaluation(
        self, evaluation_results: Dict[str, Any], filepath: Union[str, Path]
    ) -> None:
        """Save comprehensive evaluation results to file."""

        filepath = Path(filepath)

        # Save full results as pickle
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(evaluation_results, f)

        # Save standardized report as text
        if "comparison_report" in evaluation_results:
            with open(filepath.with_suffix(".txt"), "w") as f:
                f.write(evaluation_results["comparison_report"])
        elif "standardized_report" in evaluation_results:
            with open(filepath.with_suffix(".txt"), "w") as f:
                f.write(evaluation_results["standardized_report"])

        logger.info(f"Comprehensive evaluation results saved to {filepath}")

    def plot_predictions(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot predicted vs actual values.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for the title
            save_path: Path to save the plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Convert to numpy arrays
        if hasattr(y_true, "values"):
            y_true = y_true.values
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Predicted vs Actual scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(f"{model_name}: Predicted vs Actual")

        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        ax1.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color="r", linestyle="--")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title(f"{model_name}: Residuals Plot")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        model_name: str = "Model",
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            model_name: Name of the model for the title
            top_n: Number of top features to display
            save_path: Path to save the plot (optional)

        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        top_features = sorted_features[:top_n]

        features, importances = zip(*top_features)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))

        bars = ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{model_name}: Top {top_n} Feature Importance")

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(
                bar.get_width() + max(importances) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.3f}",
                ha="left",
                va="center",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_model_comparison(
        self,
        comparison_results: Dict[str, Any],
        metric: str = "r2",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparison of multiple models.

        Args:
            comparison_results: Results from compare_models method
            metric: Metric to compare (default: 'r2')
            save_path: Path to save the plot (optional)

        Returns:
            Matplotlib figure
        """
        summary_metrics = comparison_results["summary_metrics"]

        if metric not in summary_metrics:
            raise ValueError(f"Metric '{metric}' not found in comparison results")

        model_names = list(summary_metrics[metric].keys())
        metric_values = list(summary_metrics[metric].values())

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(model_names, metric_values)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Model Comparison: {metric.upper()}")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(metric_values) * 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Highlight best model
        best_model = comparison_results["best_models"].get(metric)
        if best_model and best_model in model_names:
            best_idx = model_names.index(best_model)
            bars[best_idx].set_color("gold")
            bars[best_idx].set_edgecolor("orange")
            bars[best_idx].set_linewidth(2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            evaluation_results: Results from evaluate method

        Returns:
            Formatted report string
        """
        report = []
        report.append("=== Model Evaluation Report ===\n")

        # Model information
        report.append(f"Model Type: {evaluation_results['model_type']}")
        report.append(f"Test Samples: {evaluation_results['n_test_samples']}\n")

        # Performance metrics
        metrics = evaluation_results["metrics"]
        report.append("=== Performance Metrics ===")
        report.append(f"R² Score: {metrics['r2']:.4f}")
        report.append(f"RMSE: {metrics['rmse']:.4f}")
        report.append(f"MAE: {metrics['mae']:.4f}")
        report.append(f"MAPE: {metrics['mape']:.2f}%")
        report.append(f"Max Error: {metrics['max_error']:.4f}")
        report.append(f"Explained Variance: {metrics['explained_variance']:.4f}\n")

        # Residual analysis
        report.append("=== Residual Analysis ===")
        report.append(f"Mean Residual: {metrics['mean_residual']:.4f}")
        report.append(f"Std Residual: {metrics['std_residual']:.4f}\n")

        # Feature importance (if available)
        if "feature_importance" in evaluation_results:
            report.append("=== Top 10 Important Features ===")
            feature_importance = evaluation_results["feature_importance"]
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )

            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                report.append(f"{i:2d}. {feature}: {importance:.4f}")

        return "\n".join(report)

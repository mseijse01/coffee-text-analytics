"""
Model evaluation utilities for coffee rating prediction.

This module provides comprehensive evaluation capabilities for regression models
including cross-validation, performance metrics, and visualization.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, Union
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseEvaluator, BaseModel, ModelEvaluationError

logger = logging.getLogger(__name__)

# Check for SHAP availability
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class CoffeeModelEvaluator(BaseEvaluator):
    """
    Comprehensive model evaluator for coffee rating prediction models.

    Provides detailed evaluation metrics, cross-validation, and visualization
    capabilities for regression models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model evaluator.

        Args:
            config: Configuration dictionary with parameters:
                - cv_folds: Number of cross-validation folds (default: 5)
                - scoring_metrics: List of metrics to compute (default: ['r2', 'mse', 'mae'])
                - plot_style: Matplotlib style for plots (default: 'seaborn')
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
        }
        default_config.update(self.config)
        self.config = default_config

        # Set plot style
        try:
            plt.style.use(self.config["plot_style"])
        except:
            logger.warning(f"Could not set plot style: {self.config['plot_style']}")

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
        Evaluate a model on test data.

        Args:
            model: Fitted model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model on test data")

        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)

            # Add model information
            evaluation_results = {
                "metrics": metrics,
                "model_type": type(model).__name__,
                "n_test_samples": len(y_test),
                "predictions": y_pred,
                "residuals": y_test - y_pred
                if hasattr(y_test, "__sub__")
                else np.array(y_test) - y_pred,
            }

            # Add feature importance if available
            try:
                feature_importance = model.get_feature_importance()
                evaluation_results["feature_importance"] = feature_importance
            except:
                logger.info("Feature importance not available for this model")

            logger.info(f"Model evaluation completed. R² = {metrics['r2']:.4f}")
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

            # Add overall summary
            processed_results["summary"] = {
                "cv_folds": cv_folds,
                "model_type": type(model).__name__,
                "n_samples": len(y),
            }

            logger.info("Cross-validation completed successfully")
            return processed_results

        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            raise ModelEvaluationError(f"Failed to perform cross-validation: {e}")

    def _calculate_metrics(
        self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        if hasattr(y_true, "values"):
            y_true = y_true.values
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred)
            * 100,  # Convert to percentage
        }

        # Additional metrics
        residuals = y_true - y_pred
        metrics.update(
            {
                "mean_residual": np.mean(residuals),
                "std_residual": np.std(residuals),
                "max_error": np.max(np.abs(residuals)),
                "explained_variance": 1 - (np.var(residuals) / np.var(y_true)),
            }
        )

        return metrics

    def compare_models(
        self,
        models: Dict[str, BaseModel],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.

        Args:
            models: Dictionary mapping model names to fitted models
            X_test: Test features
            y_test: Test targets

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models")

        comparison_results = {}
        all_predictions = {}

        for name, model in models.items():
            try:
                logger.info(f"Evaluating {name}")
                results = self.evaluate(model, X_test, y_test)
                comparison_results[name] = results
                all_predictions[name] = results["predictions"]
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                comparison_results[name] = {"error": str(e)}

        # Create comparison summary
        summary_metrics = ["r2", "rmse", "mae", "mape"]
        comparison_summary = {}

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

        return {
            "individual_results": comparison_results,
            "summary_metrics": comparison_summary,
            "best_models": best_models,
            "all_predictions": all_predictions,
        }

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

                # Calculate metrics directly
                metrics = self._calculate_metrics(y_test, predictions)

                # Create results structure similar to evaluate method
                results = {
                    "metrics": metrics,
                    "model_type": f"{name}_predictions",
                    "n_test_samples": len(y_test),
                    "predictions": predictions,
                    "residuals": y_test - predictions
                    if hasattr(y_test, "__sub__")
                    else np.array(y_test) - predictions,
                }

                comparison_results[name] = results

            except Exception as e:
                logger.error(f"Failed to evaluate {name} predictions: {e}")
                comparison_results[name] = {"error": str(e)}

        # Create comparison summary
        summary_metrics = ["r2", "rmse", "mae", "mape"]
        comparison_summary = {}

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

        return {
            "individual_results": comparison_results,
            "summary_metrics": comparison_summary,
            "best_models": best_models,
            "all_predictions": predictions_dict,
        }

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

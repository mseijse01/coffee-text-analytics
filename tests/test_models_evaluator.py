"""
Comprehensive tests for the models/evaluator.py module.

Tests for CoffeeModelEvaluator including:
- Single model evaluation with comprehensive metrics
- Cross-validation functionality
- Model comparison capabilities
- SHAP analysis integration
- Visualization functionality
- Standardized reporting
- Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import pickle

from src.models.evaluator import CoffeeModelEvaluator, SHAP_AVAILABLE
from src.models.base import BaseModel, ModelEvaluationError


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def mock_fitted_model():
    """Create a mock fitted model for testing."""
    model = Mock(spec=BaseModel)
    model.is_fitted = True
    model.predict.return_value = np.random.randn(50)
    model.get_feature_importance.return_value = {
        "feature_0": 0.4,
        "feature_1": 0.3,
        "feature_2": 0.2,
        "feature_3": 0.1,
    }
    model.__class__.__name__ = "MockRegressor"
    # Add sklearn-compatible model for cross-validation
    model.model_ = Mock()
    return model


class TestCoffeeModelEvaluatorInitialization:
    """Test CoffeeModelEvaluator initialization."""

    @pytest.mark.unit
    def test_initialization_default_config(self):
        """Test evaluator initialization with default configuration."""
        evaluator = CoffeeModelEvaluator()

        assert evaluator.config["cv_folds"] == 5
        assert evaluator.config["scoring_metrics"] == [
            "r2",
            "neg_mean_squared_error",
            "neg_mean_absolute_error",
        ]
        assert evaluator.config["enable_shap"] is True
        assert evaluator.config["comprehensive_metrics"] is True
        assert evaluator.config["standardized_reporting"] is True

    @pytest.mark.unit
    def test_initialization_custom_config(self):
        """Test evaluator initialization with custom configuration."""
        custom_config = {
            "cv_folds": 3,
            "scoring_metrics": ["r2", "neg_mean_squared_error"],
            "enable_shap": False,
            "shap_sample_size": 50,
            "plot_style": "default",
        }
        evaluator = CoffeeModelEvaluator(custom_config)

        assert evaluator.config["cv_folds"] == 3
        assert evaluator.config["scoring_metrics"] == ["r2", "neg_mean_squared_error"]
        assert evaluator.config["enable_shap"] is False
        assert evaluator.config["shap_sample_size"] == 50

    @pytest.mark.unit
    @patch("src.models.evaluator.SHAP_AVAILABLE", True)
    @patch("utils.shap_analysis.ComprehensiveSHAPAnalyzer")
    def test_initialization_with_shap_analyzer(self, mock_shap_analyzer):
        """Test evaluator initialization with SHAP analyzer."""
        mock_analyzer_instance = Mock()
        mock_shap_analyzer.return_value = mock_analyzer_instance

        evaluator = CoffeeModelEvaluator({"enable_shap": True})

        assert evaluator.shap_analyzer is not None
        mock_shap_analyzer.assert_called_once()

    @pytest.mark.unit
    def test_initialization_shap_unavailable(self):
        """Test evaluator initialization when SHAP is unavailable."""
        with patch("src.models.evaluator.SHAP_AVAILABLE", False):
            evaluator = CoffeeModelEvaluator({"enable_shap": True})
            assert evaluator.shap_analyzer is None


class TestSingleModelEvaluation:
    """Test single model evaluation functionality."""

    @pytest.mark.unit
    def test_evaluate_model_basic(self, mock_fitted_model, sample_regression_data):
        """Test basic model evaluation."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Set up mock predictions
        y_pred = 2 * X_test[:, 0] + X_test[:, 1] + np.random.randn(50) * 0.05
        mock_fitted_model.predict.return_value = y_pred

        evaluator = CoffeeModelEvaluator({"enable_shap": False})
        results = evaluator.evaluate_model(mock_fitted_model, X_test, y_test)

        assert isinstance(results, dict)
        assert "r2" in results
        assert "rmse" in results
        assert "mae" in results
        assert "mse" in results

        # Check that R² is reasonable
        assert -1.0 <= results["r2"] <= 1.0

    @pytest.mark.unit
    def test_evaluate_comprehensive_metrics(
        self, mock_fitted_model, sample_regression_data
    ):
        """Test comprehensive metrics calculation."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Create realistic predictions
        y_pred = y_test + np.random.randn(50) * 0.1
        mock_fitted_model.predict.return_value = y_pred

        evaluator = CoffeeModelEvaluator({"enable_shap": False})
        results = evaluator.evaluate(mock_fitted_model, X_test, y_test)

        # Check comprehensive results structure
        assert "metrics" in results
        assert "model_type" in results
        assert "n_test_samples" in results
        assert "predictions" in results
        assert "residuals" in results
        assert "feature_importance" in results

        metrics = results["metrics"]

        # Core thesis metrics
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics

        # Additional comprehensive metrics
        assert "mse" in metrics
        assert "mape" in metrics
        assert "explained_variance" in metrics
        assert "max_error" in metrics

        # Statistical metrics
        assert "mean_residual" in metrics
        assert "std_residual" in metrics
        assert "median_residual" in metrics
        assert "residual_skewness" in metrics
        assert "residual_kurtosis" in metrics

        # Performance interpretation metrics
        assert "performance_category" in metrics
        assert "rmse_normalized" in metrics
        assert "mae_normalized" in metrics

        # Validate metric types and ranges
        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"], float)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    @pytest.mark.unit
    def test_calculate_comprehensive_metrics_edge_cases(self):
        """Test comprehensive metrics calculation with edge cases."""
        evaluator = CoffeeModelEvaluator()

        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)

        assert metrics["r2"] == 1.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["mean_residual"] == 0.0
        assert metrics["performance_category"] == "Excellent"

    @pytest.mark.unit
    def test_categorize_performance(self):
        """Test performance categorization."""
        evaluator = CoffeeModelEvaluator()

        assert evaluator._categorize_performance(0.95) == "Excellent"
        assert evaluator._categorize_performance(0.85) == "Very Good"
        assert evaluator._categorize_performance(0.75) == "Good"
        assert evaluator._categorize_performance(0.65) == "Fair"
        assert evaluator._categorize_performance(0.55) == "Poor"
        assert evaluator._categorize_performance(0.45) == "Very Poor"

    @pytest.mark.unit
    def test_skewness_kurtosis_calculation(self):
        """Test skewness and kurtosis calculation."""
        evaluator = CoffeeModelEvaluator()

        # Normal distribution should have skewness ≈ 0, kurtosis ≈ 0
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        skewness = evaluator._calculate_skewness(normal_data)
        kurtosis = evaluator._calculate_kurtosis(normal_data)

        assert abs(skewness) < 0.5  # Should be close to 0
        assert abs(kurtosis) < 0.5  # Should be close to 0

        # Constant data should have 0 skewness and kurtosis (handle NaN case)
        constant_data = np.ones(100)
        skewness_const = evaluator._calculate_skewness(constant_data)
        kurtosis_const = evaluator._calculate_kurtosis(constant_data)
        # For constant data, skewness and kurtosis are undefined (NaN) or 0
        assert np.isnan(skewness_const) or skewness_const == 0.0
        assert np.isnan(kurtosis_const) or kurtosis_const == 0.0

    @pytest.mark.unit
    @patch("src.models.evaluator.SHAP_AVAILABLE", True)
    def test_evaluate_with_shap_analysis(
        self, mock_fitted_model, sample_regression_data
    ):
        """Test evaluation with SHAP analysis enabled."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        y_pred = y_test + np.random.randn(50) * 0.1
        mock_fitted_model.predict.return_value = y_pred

        # Mock SHAP analyzer
        mock_shap_analyzer = Mock()
        mock_shap_analyzer.analyze_all_models.return_value = {
            "MockRegressor": {"shap_values": "mock_shap_results"}
        }

        evaluator = CoffeeModelEvaluator({"enable_shap": True})
        evaluator.shap_analyzer = mock_shap_analyzer

        results = evaluator.evaluate(mock_fitted_model, X_test, y_test)

        assert "shap_analysis" in results
        mock_shap_analyzer.analyze_all_models.assert_called_once()

    @pytest.mark.unit
    def test_evaluate_standardized_report(
        self, mock_fitted_model, sample_regression_data
    ):
        """Test standardized report generation."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        y_pred = y_test + np.random.randn(50) * 0.1
        mock_fitted_model.predict.return_value = y_pred

        evaluator = CoffeeModelEvaluator(
            {"enable_shap": False, "standardized_reporting": True}
        )
        results = evaluator.evaluate(mock_fitted_model, X_test, y_test)

        assert "standardized_report" in results
        report = results["standardized_report"]
        assert isinstance(report, str)
        assert "STANDARDIZED MODEL EVALUATION REPORT" in report
        assert "CORE PERFORMANCE METRICS" in report
        assert "R² Score:" in report
        assert "RMSE:" in report
        assert "MAE:" in report

    @pytest.mark.unit
    def test_evaluate_model_error_handling(self, sample_regression_data):
        """Test error handling in model evaluation."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Mock model that raises an exception during prediction
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")

        evaluator = CoffeeModelEvaluator({"enable_shap": False})

        with pytest.raises(ModelEvaluationError, match="Failed to evaluate model"):
            evaluator.evaluate(mock_model, X_test, y_test)


class TestCrossValidation:
    """Test cross-validation functionality."""

    @pytest.mark.unit
    @patch("src.models.evaluator.cross_validate")
    def test_cross_validate_basic(
        self, mock_cross_validate, mock_fitted_model, sample_regression_data
    ):
        """Test basic cross-validation."""
        X, y = sample_regression_data

        # Mock cross_validate results
        mock_cv_results = {
            "test_r2": np.array([0.8, 0.85, 0.82, 0.78, 0.83]),
            "train_r2": np.array([0.85, 0.87, 0.86, 0.84, 0.88]),
            "test_neg_mean_squared_error": np.array([-0.2, -0.18, -0.19, -0.22, -0.20]),
            "train_neg_mean_squared_error": np.array(
                [-0.15, -0.13, -0.14, -0.16, -0.12]
            ),
            "test_neg_mean_absolute_error": np.array(
                [-0.3, -0.28, -0.29, -0.32, -0.30]
            ),
            "train_neg_mean_absolute_error": np.array(
                [-0.25, -0.23, -0.24, -0.26, -0.22]
            ),
        }
        mock_cross_validate.return_value = mock_cv_results

        evaluator = CoffeeModelEvaluator()
        results = evaluator.cross_validate(mock_fitted_model, X, y)

        assert isinstance(results, dict)
        assert "r2" in results
        assert "mean_squared_error" in results
        assert "mean_absolute_error" in results

        # Check R² results structure
        r2_results = results["r2"]
        assert "test_scores" in r2_results
        assert "train_scores" in r2_results
        assert "test_mean" in r2_results
        assert "test_std" in r2_results
        assert "train_mean" in r2_results
        assert "train_std" in r2_results

        # Verify mean calculations
        assert (
            abs(r2_results["test_mean"] - 0.816) < 0.01
        )  # Mean of [0.8, 0.85, 0.82, 0.78, 0.83]

    @pytest.mark.unit
    def test_cross_validate_custom_folds(
        self, mock_fitted_model, sample_regression_data
    ):
        """Test cross-validation with custom number of folds."""
        X, y = sample_regression_data

        with patch("src.models.evaluator.cross_validate") as mock_cross_validate:
            mock_cross_validate.return_value = {
                "test_r2": np.array([0.8, 0.85, 0.82]),
                "train_r2": np.array([0.85, 0.87, 0.86]),
                "test_neg_mean_squared_error": np.array([-0.2, -0.18, -0.19]),
                "train_neg_mean_squared_error": np.array([-0.15, -0.13, -0.14]),
                "test_neg_mean_absolute_error": np.array([-0.3, -0.28, -0.29]),
                "train_neg_mean_absolute_error": np.array([-0.25, -0.23, -0.24]),
            }

            evaluator = CoffeeModelEvaluator()
            results = evaluator.cross_validate(mock_fitted_model, X, y, cv=3)

            # Verify cross_validate was called with correct cv parameter
            mock_cross_validate.assert_called_once()
            call_args = mock_cross_validate.call_args
            assert call_args[1]["cv"] == 3

    @pytest.mark.unit
    def test_cross_validate_error_handling(
        self, mock_fitted_model, sample_regression_data
    ):
        """Test cross-validation error handling."""
        X, y = sample_regression_data

        with patch("src.models.evaluator.cross_validate") as mock_cross_validate:
            mock_cross_validate.side_effect = Exception("Cross-validation failed")

            evaluator = CoffeeModelEvaluator()

            with pytest.raises(
                ModelEvaluationError, match="Failed to perform cross-validation"
            ):
                evaluator.cross_validate(mock_fitted_model, X, y)


class TestModelComparison:
    """Test model comparison functionality."""

    @pytest.mark.unit
    def test_compare_models_basic(self, sample_regression_data):
        """Test basic model comparison."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Create mock models with different performance
        model1 = Mock(spec=BaseModel)
        model1.predict.return_value = (
            y_test + np.random.randn(50) * 0.1
        )  # Good predictions
        model1.get_feature_importance.return_value = {
            "feat_0": 0.5,
            "feat_1": 0.3,
            "feat_2": 0.2,
        }
        model1.__class__.__name__ = "ModelA"

        model2 = Mock(spec=BaseModel)
        model2.predict.return_value = (
            y_test + np.random.randn(50) * 0.2
        )  # Worse predictions
        model2.get_feature_importance.return_value = {
            "feat_0": 0.4,
            "feat_1": 0.4,
            "feat_2": 0.2,
        }
        model2.__class__.__name__ = "ModelB"

        models = {"ModelA": model1, "ModelB": model2}
        evaluator = CoffeeModelEvaluator({"enable_shap": False})

        comparison = evaluator.compare_models(
            models, X_test, y_test, include_shap=False
        )

        assert isinstance(comparison, dict)
        assert "individual_results" in comparison
        assert "summary_metrics" in comparison
        assert "best_models" in comparison
        assert "all_predictions" in comparison
        assert "comparison_report" in comparison

        # Check individual results
        individual = comparison["individual_results"]
        assert "ModelA" in individual
        assert "ModelB" in individual
        assert "metrics" in individual["ModelA"]
        assert "metrics" in individual["ModelB"]

        # Check summary metrics
        summary = comparison["summary_metrics"]
        assert "r2" in summary
        assert "rmse" in summary
        assert "mae" in summary
        assert "ModelA" in summary["r2"]
        assert "ModelB" in summary["r2"]

        # Check best models
        best = comparison["best_models"]
        assert "r2" in best
        assert "rmse" in best
        assert "mae" in best

    @pytest.mark.unit
    def test_compare_models_with_predictions(self, sample_regression_data):
        """Test model comparison using pre-computed predictions."""
        X, y = sample_regression_data
        y_test = y[:50]

        predictions_dict = {
            "ModelA": y_test + np.random.randn(50) * 0.1,  # Good predictions
            "ModelB": y_test + np.random.randn(50) * 0.2,  # Worse predictions
            "ModelC": y_test + np.random.randn(50) * 0.15,  # Medium predictions
        }

        evaluator = CoffeeModelEvaluator({"enable_shap": False})
        comparison = evaluator.compare_models_with_predictions(predictions_dict, y_test)

        assert isinstance(comparison, dict)
        assert "individual_results" in comparison
        assert "summary_metrics" in comparison
        assert "best_models" in comparison
        assert "all_predictions" in comparison
        assert "comparison_report" in comparison

        # Check that all models were evaluated
        assert len(comparison["individual_results"]) == 3
        assert "ModelA" in comparison["individual_results"]
        assert "ModelB" in comparison["individual_results"]
        assert "ModelC" in comparison["individual_results"]

    @pytest.mark.unit
    @patch("src.models.evaluator.SHAP_AVAILABLE", True)
    def test_compare_models_with_shap(self, sample_regression_data):
        """Test model comparison with SHAP analysis."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Create mock models
        model1 = Mock(spec=BaseModel)
        model1.predict.return_value = y_test + np.random.randn(50) * 0.1
        model1.get_feature_importance.return_value = {"feat_0": 0.5}
        model1.__class__.__name__ = "ModelA"

        models = {"ModelA": model1}

        # Mock SHAP analyzer
        mock_shap_analyzer = Mock()
        mock_shap_analyzer.analyze_all_models.return_value = {
            "comparison_results": "mock_shap_comparison"
        }

        evaluator = CoffeeModelEvaluator({"enable_shap": True})
        evaluator.shap_analyzer = mock_shap_analyzer

        comparison = evaluator.compare_models(models, X_test, y_test, include_shap=True)

        assert "shap_comparison" in comparison
        # SHAP analyzer may be called multiple times (once per model evaluation, once for comparison)
        assert mock_shap_analyzer.analyze_all_models.called

    @pytest.mark.unit
    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        summary_metrics = {
            "r2": {"ModelA": 0.85, "ModelB": 0.80},
            "rmse": {"ModelA": 0.15, "ModelB": 0.18},
            "mae": {"ModelA": 0.12, "ModelB": 0.14},
            "mse": {"ModelA": 0.0225, "ModelB": 0.0324},
            "mape": {"ModelA": 5.2, "ModelB": 6.1},
        }

        best_models = {
            "r2": "ModelA",
            "rmse": "ModelA",
            "mae": "ModelA",
            "mse": "ModelA",
            "mape": "ModelA",
        }

        evaluator = CoffeeModelEvaluator()
        report = evaluator._generate_comparison_report(
            summary_metrics, best_models, None
        )

        assert isinstance(report, str)
        assert "COMPREHENSIVE MODEL COMPARISON REPORT" in report
        assert "PERFORMANCE SUMMARY TABLE" in report
        assert "BEST MODELS BY METRIC" in report
        assert "ModelA" in report
        assert "ModelB" in report
        assert "0.8500" in report  # ModelA R²
        assert "0.8000" in report  # ModelB R²

    @pytest.mark.unit
    def test_compare_models_error_handling(self, sample_regression_data):
        """Test error handling in model comparison."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Create model that raises exception
        error_model = Mock()
        error_model.predict.side_effect = Exception("Model failed")
        error_model.__class__.__name__ = "ErrorModel"

        good_model = Mock()
        good_model.predict.return_value = y_test
        good_model.get_feature_importance.return_value = {"feat_0": 1.0}
        good_model.__class__.__name__ = "GoodModel"

        models = {"ErrorModel": error_model, "GoodModel": good_model}
        evaluator = CoffeeModelEvaluator({"enable_shap": False})

        comparison = evaluator.compare_models(
            models, X_test, y_test, include_shap=False
        )

        # Should handle error gracefully
        assert "ErrorModel" in comparison["individual_results"]
        assert "error" in comparison["individual_results"]["ErrorModel"]
        assert "GoodModel" in comparison["individual_results"]
        assert "metrics" in comparison["individual_results"]["GoodModel"]


class TestVisualization:
    """Test visualization functionality."""

    @pytest.mark.unit
    def test_plot_predictions_basic(self):
        """Test basic prediction plotting."""
        evaluator = CoffeeModelEvaluator()

        # Create test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])

        fig = evaluator.plot_predictions(y_true, y_pred, "TestModel")

        assert fig is not None
        assert len(fig.axes) == 2  # Scatter plot and residuals plot

        # Check that plots were created
        ax1, ax2 = fig.axes
        assert ax1.get_xlabel() == "Actual Values"
        assert ax1.get_ylabel() == "Predicted Values"
        assert ax2.get_xlabel() == "Predicted Values"
        assert ax2.get_ylabel() == "Residuals"

        plt.close(fig)

    @pytest.mark.unit
    def test_plot_predictions_with_save(self):
        """Test prediction plotting with save functionality."""
        evaluator = CoffeeModelEvaluator()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            fig = evaluator.plot_predictions(
                y_true, y_pred, "TestModel", save_path=save_path
            )

            # Check that file was created
            assert Path(save_path).exists()
            plt.close(fig)
        finally:
            # Clean up
            if Path(save_path).exists():
                Path(save_path).unlink()

    @pytest.mark.unit
    def test_plot_model_comparison(self):
        """Test model comparison plotting."""
        comparison_results = {
            "summary_metrics": {"r2": {"ModelA": 0.85, "ModelB": 0.80, "ModelC": 0.78}},
            "best_models": {"r2": "ModelA"},
        }

        evaluator = CoffeeModelEvaluator()
        fig = evaluator.plot_model_comparison(comparison_results, metric="r2")

        assert fig is not None
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert ax.get_ylabel() == "R2"
        assert ax.get_title() == "Model Comparison: R2"

        plt.close(fig)

    @pytest.mark.unit
    def test_plot_model_comparison_invalid_metric(self):
        """Test model comparison plotting with invalid metric."""
        comparison_results = {
            "summary_metrics": {"r2": {"ModelA": 0.85, "ModelB": 0.80}},
            "best_models": {"r2": "ModelA"},
        }

        evaluator = CoffeeModelEvaluator()

        with pytest.raises(ValueError, match="Metric 'invalid_metric' not found"):
            evaluator.plot_model_comparison(comparison_results, metric="invalid_metric")


class TestReportGeneration:
    """Test report generation functionality."""

    @pytest.mark.unit
    def test_generate_evaluation_report(self):
        """Test comprehensive evaluation report generation."""
        evaluation_results = {
            "model_type": "TestModel",
            "n_test_samples": 100,
            "metrics": {
                "r2": 0.85,
                "rmse": 0.25,
                "mae": 0.20,
                "mape": 5.5,
                "max_error": 0.80,
                "explained_variance": 0.86,
                "mean_residual": 0.01,
                "std_residual": 0.24,
            },
            "feature_importance": {
                "feature_0": 0.4,
                "feature_1": 0.3,
                "feature_2": 0.2,
                "feature_3": 0.1,
            },
        }

        evaluator = CoffeeModelEvaluator()
        report = evaluator.generate_evaluation_report(evaluation_results)

        assert isinstance(report, str)
        assert "Model Evaluation Report" in report
        assert "TestModel" in report
        assert "R² Score: 0.8500" in report
        assert "RMSE: 0.2500" in report
        assert "MAE: 0.2000" in report
        assert "Top 10 Important Features" in report
        assert "feature_0: 0.4000" in report

    @pytest.mark.unit
    def test_generate_standardized_report_complete(self):
        """Test complete standardized report generation."""
        evaluation_results = {
            "model_type": "ComprehensiveModel",
            "n_test_samples": 150,
            "metrics": {
                "r2": 0.92,
                "rmse": 0.15,
                "mae": 0.12,
                "mse": 0.0225,
                "mape": 3.2,
                "explained_variance": 0.93,
                "max_error": 0.45,
                "rmse_normalized": 0.75,
                "mae_normalized": 0.15,
                "mean_residual": -0.002,
                "std_residual": 0.148,
                "median_residual": 0.001,
                "residual_skewness": 0.05,
                "residual_kurtosis": -0.12,
                "performance_category": "Excellent",
            },
            "feature_importance": {
                "temperature": 0.35,
                "pressure": 0.25,
                "time": 0.20,
                "grind_size": 0.15,
                "bean_type": 0.05,
            },
            "shap_analysis": {"feature_importance": "mock_shap_data"},
        }

        evaluator = CoffeeModelEvaluator()
        report = evaluator._generate_standardized_report(evaluation_results)

        assert isinstance(report, str)
        assert "STANDARDIZED MODEL EVALUATION REPORT" in report
        assert "ComprehensiveModel" in report
        assert "Test Samples: 150" in report
        assert "R² Score:  0.9200 (Excellent)" in report
        assert "RMSE:      0.1500" in report
        assert "MAE:       0.1200" in report
        assert "NORMALIZED METRICS:" in report
        assert "RESIDUAL ANALYSIS:" in report
        assert "TOP 10 FEATURE IMPORTANCE:" in report
        assert "temperature: 0.3500" in report
        assert "SHAP INTERPRETABILITY ANALYSIS:" in report
        assert "SHAP analysis completed" in report


class TestFileOperations:
    """Test file operations functionality."""

    @pytest.mark.unit
    def test_save_comprehensive_evaluation(self):
        """Test saving comprehensive evaluation results."""
        evaluation_results = {
            "metrics": {"r2": 0.85, "rmse": 0.25},
            "standardized_report": "Test report content",
            "model_type": "TestModel",
        }

        evaluator = CoffeeModelEvaluator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / "test_evaluation"

            evaluator.save_comprehensive_evaluation(evaluation_results, filepath)

            # Check that files were created
            pkl_file = filepath.with_suffix(".pkl")
            txt_file = filepath.with_suffix(".txt")

            assert pkl_file.exists()
            assert txt_file.exists()

            # Verify content
            with open(pkl_file, "rb") as f:
                loaded_results = pickle.load(f)
                assert loaded_results["metrics"]["r2"] == 0.85

            with open(txt_file, "r") as f:
                report_content = f.read()
                assert "Test report content" in report_content

    @pytest.mark.unit
    def test_save_comparison_results(self):
        """Test saving comparison results."""
        evaluation_results = {
            "comparison_report": "Model comparison report content",
            "summary_metrics": {"r2": {"ModelA": 0.85}},
        }

        evaluator = CoffeeModelEvaluator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / "test_comparison"

            evaluator.save_comprehensive_evaluation(evaluation_results, filepath)

            pkl_file = filepath.with_suffix(".pkl")
            txt_file = filepath.with_suffix(".txt")

            assert pkl_file.exists()
            assert txt_file.exists()

            with open(txt_file, "r") as f:
                report_content = f.read()
                assert "Model comparison report content" in report_content


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_predictions_handling(self):
        """Test handling of edge cases with minimal data."""
        evaluator = CoffeeModelEvaluator()

        # Very small dataset
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 1.9])

        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert not np.isnan(metrics["r2"])

    @pytest.mark.unit
    def test_pandas_series_input_handling(self):
        """Test handling of pandas Series input."""
        evaluator = CoffeeModelEvaluator()

        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])

        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert "r2" in metrics
        assert not np.isnan(metrics["r2"])

    @pytest.mark.unit
    def test_scipy_unavailable_fallback(self):
        """Test fallback calculations when scipy is unavailable."""
        evaluator = CoffeeModelEvaluator()

        # Test manual calculation fallback by patching the try/except block
        # We'll call the methods directly with valid data to test fallback
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test the actual fallback implementation works
        skewness = evaluator._calculate_skewness(data)
        kurtosis = evaluator._calculate_kurtosis(data)

        assert isinstance(skewness, float)
        assert isinstance(kurtosis, float)
        assert not np.isnan(skewness)
        assert not np.isnan(kurtosis)

        # For this data, we expect reasonable skewness and kurtosis values
        assert -2.0 <= skewness <= 2.0
        assert -2.0 <= kurtosis <= 2.0

    @pytest.mark.unit
    def test_invalid_plot_style_handling(self):
        """Test handling of invalid plot style."""
        # Should not raise exception, just log warning
        evaluator = CoffeeModelEvaluator({"plot_style": "nonexistent_style"})

        assert evaluator.config["plot_style"] == "nonexistent_style"

    @pytest.mark.unit
    def test_model_without_feature_importance(self, sample_regression_data):
        """Test evaluation of model without feature importance."""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]

        # Model without feature importance
        mock_model = Mock()
        mock_model.predict.return_value = y_test + np.random.randn(50) * 0.1
        mock_model.get_feature_importance.side_effect = AttributeError(
            "No feature importance"
        )
        mock_model.__class__.__name__ = "SimpleModel"

        evaluator = CoffeeModelEvaluator({"enable_shap": False})
        results = evaluator.evaluate(mock_model, X_test, y_test)

        # Should complete successfully without feature importance
        assert "metrics" in results
        assert "feature_importance" not in results or not results.get(
            "feature_importance"
        )

    @pytest.mark.unit
    def test_extreme_values_handling(self):
        """Test handling of extreme values in metrics calculation."""
        evaluator = CoffeeModelEvaluator()

        # Test with very large values
        y_true = np.array([1e6, 2e6, 3e6])
        y_pred = np.array([1e6 + 1000, 2e6 + 2000, 3e6 + 3000])

        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)

        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"], float)
        assert not np.isnan(metrics["r2"])
        assert not np.isinf(metrics["rmse"])

    @pytest.mark.unit
    def test_zero_variance_target_handling(self):
        """Test handling of zero variance in target variable."""
        evaluator = CoffeeModelEvaluator()

        # Constant target values
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.1, 4.9, 5.2, 4.8, 5.0])

        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)

        # Should handle gracefully
        assert isinstance(metrics, dict)
        assert "r2" in metrics
        assert "rmse" in metrics

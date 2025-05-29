"""
Tests for MLflow integration functionality.

This module tests the CoffeeMLflowTracker class and its integration
with the coffee text analytics pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.experiment.mlflow_integration import CoffeeMLflowTracker


class TestCoffeeMLflowTracker:
    """Test suite for CoffeeMLflowTracker class."""

    @pytest.fixture
    def temp_mlruns_dir(self):
        """Create a temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def tracker(self, temp_mlruns_dir):
        """Create a CoffeeMLflowTracker instance with temporary tracking."""
        with patch("mlflow.set_tracking_uri") as mock_set_uri:
            mock_set_uri.return_value = None
            tracker = CoffeeMLflowTracker()
            yield tracker

    def test_tracker_initialization(self, tracker):
        """Test that tracker initializes correctly."""
        assert tracker.experiment_name == "coffee-text-analytics-thesis"
        assert tracker.logger is not None

    @pytest.mark.mlflow
    def test_setup_experiment(self, tracker):
        """Test experiment setup functionality."""
        with patch("mlflow.create_experiment") as mock_create:
            with patch("mlflow.get_experiment_by_name") as mock_get:
                mock_get.return_value = None
                mock_create.return_value = "123"

                tracker.setup_experiment()

                mock_get.assert_called_once_with("coffee-text-analytics-thesis")
                mock_create.assert_called_once()

    @pytest.mark.mlflow
    def test_start_methodology_run(self, tracker):
        """Test starting a methodology run with proper parameters."""
        methodology_params = {
            "text_columns": ["desc_1", "desc_2", "desc_3"],
            "feature_selection_method": "corrected_lasso",
            "specialized_preprocessing": True,
            "box_cox_enabled": False,
        }

        with patch("mlflow.start_run") as mock_start:
            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock()

            with patch("mlflow.log_params") as mock_log_params:
                with patch("mlflow.set_tags") as mock_set_tags:
                    run_id = tracker.start_methodology_run(
                        run_name="test_run",
                        sample_fraction=0.15,
                        methodology_params=methodology_params,
                    )

                    mock_start.assert_called_once()
                    mock_log_params.assert_called()
                    mock_set_tags.assert_called()

    @pytest.mark.mlflow
    def test_log_feature_extraction(self, tracker):
        """Test feature extraction logging."""
        feature_counts = {
            "tfidf_desc_1": 5000,
            "tfidf_desc_2": 5000,
            "tfidf_desc_3": 5000,
            "bert_desc_1": 768,
            "sentiment_features": 18,
            "topic_features": 20,
            "sensory_features": 5,
        }

        with patch("mlflow.log_metrics") as mock_log_metrics:
            tracker.log_feature_extraction(feature_counts)
            mock_log_metrics.assert_called_once()

            # Verify the logged metrics include total features
            logged_metrics = mock_log_metrics.call_args[0][0]
            assert "total_features" in logged_metrics
            assert logged_metrics["total_features"] == sum(feature_counts.values())

    @pytest.mark.mlflow
    def test_log_model_performance(self, tracker):
        """Test model performance logging."""
        metrics = {"r2": 0.682, "mae": 0.245, "rmse": 0.334}

        model_params = {"n_estimators": 100, "max_depth": 6, "random_state": 57}

        with patch("mlflow.log_metrics") as mock_log_metrics:
            with patch("mlflow.log_params") as mock_log_params:
                tracker.log_model_performance(
                    model_name="xgboost", metrics=metrics, model_params=model_params
                )

                mock_log_metrics.assert_called()
                mock_log_params.assert_called()

    @pytest.mark.mlflow
    def test_log_methodology_compliance(self, tracker):
        """Test methodology compliance tracking."""
        compliance_report = {
            "separate_text_processing": True,
            "thesis_feature_naming": True,
            "corrected_lasso_selection": True,
            "specialized_preprocessing": True,
            "categorical_encoding": True,
            "two_step_hyperparameter_tuning": True,
            "comprehensive_shap_analysis": True,
            "complete_evaluation_metrics": True,
        }

        with patch("mlflow.log_params") as mock_log_params:
            with patch("mlflow.log_metric") as mock_log_metric:
                tracker.log_methodology_compliance(compliance_report)

                mock_log_params.assert_called()
                mock_log_metric.assert_called()

                # Verify compliance score calculation
                compliance_score = sum(compliance_report.values()) / len(
                    compliance_report
                )
                mock_log_metric.assert_called_with(
                    "methodology_compliance_score", compliance_score
                )

    @pytest.mark.mlflow
    def test_log_storage_efficiency(self, tracker):
        """Test storage efficiency tracking."""
        traditional_size = 2048.5  # MB
        mlflow_size = 0.172  # MB

        with patch("mlflow.log_metrics") as mock_log_metrics:
            tracker.log_storage_efficiency(traditional_size, mlflow_size)

            mock_log_metrics.assert_called_once()
            logged_metrics = mock_log_metrics.call_args[0][0]

            assert "traditional_storage_mb" in logged_metrics
            assert "mlflow_storage_mb" in logged_metrics
            assert "storage_reduction_percent" in logged_metrics

            # Verify calculation
            expected_reduction = (
                (traditional_size - mlflow_size) / traditional_size
            ) * 100
            assert logged_metrics["storage_reduction_percent"] == pytest.approx(
                expected_reduction, rel=1e-3
            )

    @pytest.mark.mlflow
    def test_end_run(self, tracker):
        """Test ending MLflow run."""
        with patch("mlflow.end_run") as mock_end_run:
            tracker.end_run()
            mock_end_run.assert_called_once()

    def test_setup_coffee_mlflow_function(self):
        """Test the setup function creates tracker correctly."""
        from src.experiment.mlflow_integration import setup_coffee_mlflow

        with patch(
            "src.experiment.mlflow_integration.CoffeeMLflowTracker"
        ) as mock_tracker:
            setup_coffee_mlflow()
            mock_tracker.assert_called_once_with()


class TestMLflowIntegrationValidation:
    """Integration tests for MLflow with coffee analytics pipeline."""

    @pytest.mark.integration
    @pytest.mark.mlflow
    def test_validation_script_execution(self):
        """Test that the validation script runs without errors."""
        from src.experiment.mlflow_integration import validate_methodology_with_mlflow

        # This should run without throwing exceptions
        try:
            validate_methodology_with_mlflow()
        except Exception as e:
            pytest.fail(f"MLflow validation failed: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_mlflow_with_small_pipeline_run(self):
        """Test MLflow integration with a small pipeline run."""
        # This would be a full integration test with actual pipeline
        # For now, we'll just verify the imports work
        try:
            from src.experiment.mlflow_integration import CoffeeMLflowTracker

            tracker = CoffeeMLflowTracker()
            assert tracker is not None
        except Exception as e:
            pytest.fail(f"MLflow pipeline integration failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

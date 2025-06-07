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

        # Create a mock run object with the expected structure
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_12345"

        with patch("mlflow.start_run") as mock_start:
            mock_start.return_value = mock_run

            with patch("mlflow.log_params") as mock_log_params:
                run_id = tracker.start_methodology_run(
                    run_name="test_run",
                    sample_fraction=0.15,
                    methodology_params=methodology_params,
                )

                # Verify MLflow calls
                mock_start.assert_called_once()
                # log_params is called twice in the actual implementation
                assert mock_log_params.call_count == 2
                assert run_id == "test_run_id_12345"

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
            with patch("mlflow.log_metric") as mock_log_metric:
                tracker.log_feature_extraction(feature_counts)

                # Verify log_metrics was called with feature counts
                mock_log_metrics.assert_called_once()
                logged_metrics = mock_log_metrics.call_args[0][0]

                # Check that feature counts were logged with proper prefix
                assert "features_tfidf_desc_1" in logged_metrics
                assert logged_metrics["features_tfidf_desc_1"] == 5000

                # Verify total_features was logged separately
                mock_log_metric.assert_called_once_with(
                    "total_features", sum(feature_counts.values())
                )

    @pytest.mark.mlflow
    def test_log_model_performance(self, tracker):
        """Test model performance logging."""
        model_name = "xgboost"
        metrics = {"r2": 0.682, "mae": 0.234, "rmse": 0.445}
        model_params = {"n_estimators": 100, "max_depth": 6}

        with patch("mlflow.log_metrics") as mock_log_metrics:
            with patch("mlflow.log_params") as mock_log_params:
                tracker.log_model_performance(model_name, metrics, model_params)

                mock_log_metrics.assert_called_once()
                mock_log_params.assert_called_once()

                # Verify prefixed metrics
                logged_metrics = mock_log_metrics.call_args[0][0]
                assert "xgboost_r2" in logged_metrics
                assert logged_metrics["xgboost_r2"] == 0.682

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

        with patch("mlflow.log_metrics") as mock_log_metrics:
            with patch("mlflow.log_metric") as mock_log_metric:
                tracker.log_methodology_compliance(compliance_report)

                # log_metrics is called once with compliance metrics
                mock_log_metrics.assert_called_once()
                # log_metric is called once with overall compliance
                mock_log_metric.assert_called_once()

                # Verify compliance metrics were converted to numeric
                logged_metrics = mock_log_metrics.call_args[0][0]
                assert "compliance_separate_text_processing" in logged_metrics
                assert logged_metrics["compliance_separate_text_processing"] == 1.0

    @pytest.mark.mlflow
    def test_log_storage_efficiency(self, tracker):
        """Test storage efficiency tracking."""
        traditional_size = 2048.5  # MB
        mlflow_size = 0.172  # MB

        with patch("mlflow.log_metrics") as mock_log_metrics:
            tracker.log_storage_efficiency(traditional_size, mlflow_size)

            mock_log_metrics.assert_called_once()
            logged_metrics = mock_log_metrics.call_args[0][0]

            # Check correct key names based on actual implementation
            assert "storage_traditional_mb" in logged_metrics
            assert "storage_mlflow_mb" in logged_metrics
            assert "storage_reduction_percent" in logged_metrics
            assert logged_metrics["storage_traditional_mb"] == traditional_size
            assert logged_metrics["storage_mlflow_mb"] == mlflow_size

    @pytest.mark.mlflow
    def test_end_run(self, tracker):
        """Test ending MLflow run."""
        with patch("mlflow.end_run") as mock_end_run:
            tracker.end_run()
            mock_end_run.assert_called_once()

    @pytest.mark.mlflow
    def test_setup_coffee_mlflow_function(self):
        """Test convenience setup function."""
        from src.experiment.mlflow_integration import setup_coffee_mlflow

        with patch("mlflow.set_tracking_uri"):
            tracker = setup_coffee_mlflow()
            assert isinstance(tracker, CoffeeMLflowTracker)


class TestMLflowIntegrationValidation:
    """Test MLflow integration validation scenarios."""

    @pytest.mark.integration
    @pytest.mark.mlflow
    def test_validation_script_execution(self):
        """Test that the validation script runs without errors."""
        from src.experiment.mlflow_integration import validate_methodology_with_mlflow

        # This should run without throwing exceptions
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = MagicMock()
            mock_run.info.run_id = "test_validation_run_id"
            mock_start_run.return_value = mock_run

            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        try:
                            validate_methodology_with_mlflow()
                        except Exception as e:
                            pytest.fail(f"MLflow validation failed: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_mlflow_with_small_pipeline_run(self):
        """Test MLflow integration with a small pipeline run."""
        tracker = CoffeeMLflowTracker()

        with patch("mlflow.start_run") as mock_start_run:
            mock_run = MagicMock()
            mock_run.info.run_id = "integration_test_run"
            mock_start_run.return_value = mock_run

            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        # Simulate a small pipeline run
                        methodology_params = {
                            "sample_fraction": 0.05,
                            "text_columns": "desc_1",
                            "models": "linear,ridge",
                        }

                        run_id = tracker.start_methodology_run(
                            run_name="integration_test",
                            sample_fraction=0.05,
                            methodology_params=methodology_params,
                        )

                        # Log some dummy metrics
                        tracker.log_feature_extraction({"tfidf": 100, "sensory": 5})
                        tracker.log_model_performance("linear", {"r2": 0.75})
                        tracker.log_storage_efficiency(100.0, 1.0)

                        assert run_id == "integration_test_run"


if __name__ == "__main__":
    pytest.main([__file__])

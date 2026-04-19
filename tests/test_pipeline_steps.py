#!/usr/bin/env python3
"""
TDD tests for src/pipeline/ step modules.

These tests define the interface contract for the pipeline refactor.
They will fail (ImportError) until src/pipeline/ is implemented — that is expected.

Interface contract:
    src/pipeline/constants.py  — EXCLUDE_COLUMNS list
    src/pipeline/preprocess.py — run_preprocessing(args, config) -> bool
    src/pipeline/features.py   — run_feature_extraction(args, config) -> bool
    src/pipeline/selection.py  — run_feature_selection(args, config) -> bool
    src/pipeline/training.py   — run_training(args, config) -> bool
    src/pipeline/visualization.py — run_visualization(args, config) -> bool
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.pipeline.constants import EXCLUDE_COLUMNS
from src.pipeline.preprocess import run_preprocessing
from src.pipeline.features import run_feature_extraction
from src.pipeline.selection import run_feature_selection
from src.pipeline.training import run_training
from src.pipeline.visualization import run_visualization


# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temp directory structure mirroring the real layout."""
    (tmp_path / "processed").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "figures").mkdir()
    return tmp_path


@pytest.fixture
def mock_config(tmp_dirs):
    """Mock config object with temp paths and sensible defaults."""
    cfg = MagicMock()

    cfg.paths.get_raw_data_path.return_value = tmp_dirs / "coffee_clean.csv"
    cfg.paths.get_processed_data_path.return_value = (
        tmp_dirs / "processed" / "coffee_processed.csv"
    )
    cfg.paths.get_features_data_path.return_value = (
        tmp_dirs / "processed" / "coffee_features.csv"
    )
    cfg.paths.processed = tmp_dirs / "processed"
    cfg.paths.models = tmp_dirs / "models"
    cfg.paths.output = tmp_dirs / "output"

    cfg.models.text_columns = ["desc_1", "desc_2", "desc_3"]
    cfg.models.target_column = "rating"
    cfg.models.feature_selection_enabled = True
    cfg.models.feature_selection_config = {}
    cfg.models.models_to_train = ["linear"]
    cfg.models.test_size = 0.3
    cfg.models.random_state = 42
    cfg.models.cv_folds = 5
    cfg.models.box_cox_enabled = False
    cfg.models.box_cox_dual_pipeline = False
    cfg.models.two_step_tuning_enabled = False
    cfg.models.linear_params = {}
    cfg.models.ridge_params = {}
    cfg.models.lasso_params = {}
    cfg.models.svr_params = {}
    cfg.models.decision_tree_params = {}
    cfg.models.lasso_cv_folds = 5
    cfg.models.box_cox_config = {}

    cfg.features.tfidf_max_features = 100
    cfg.features.tfidf_ngram_range = (1, 2)
    cfg.features.bert_batch_size = 8
    cfg.features.bert_max_length = 128
    cfg.features.n_topics = 5

    return cfg


@pytest.fixture
def mock_args(tmp_dirs):
    """Minimal args namespace covering what each step reads."""
    return argparse.Namespace(
        input_file=str(tmp_dirs / "coffee_clean.csv"),
        sample_fraction=None,
        sample_size=None,
        box_cox=False,
        box_cox_dual=False,
        text_columns=["desc_1", "desc_2", "desc_3"],
        target_column="rating",
        n_topics=5,
        models=["linear"],
    )


@pytest.fixture
def sample_feature_df():
    """Small DataFrame that looks like real extracted features."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "rating": np.random.uniform(84, 97, n),
            "desc_1": ["coffee text"] * n,
            "desc_2": ["aroma notes"] * n,
            "desc_3": ["flavor notes"] * n,
            "aroma": np.random.uniform(8, 10, n),
            "acid": np.random.uniform(8, 10, n),
            "body": np.random.uniform(8, 10, n),
            "flavor": np.random.uniform(8, 10, n),
            "aftertaste": np.random.uniform(8, 10, n),
            "tfidf_desc_1_word1": np.random.randn(n),
            "tfidf_desc_1_word2": np.random.randn(n),
            "tfidf_desc_2_word1": np.random.randn(n),
            "bert_desc_1_0": np.random.randn(n),
        }
    )


# ============================================================
# EXCLUDE_COLUMNS contract
# ============================================================


class TestExcludeColumns:
    """EXCLUDE_COLUMNS must be a single canonical list used by both selection and training."""

    def test_is_list(self):
        assert isinstance(EXCLUDE_COLUMNS, list)

    def test_contains_target(self):
        assert "rating" in EXCLUDE_COLUMNS

    def test_contains_text_columns(self):
        for col in ["desc_1", "desc_2", "desc_3"]:
            assert col in EXCLUDE_COLUMNS, f"Expected '{col}' in EXCLUDE_COLUMNS"

    def test_contains_sensory_attributes(self):
        for col in ["aroma", "acid", "body", "flavor", "aftertaste"]:
            assert col in EXCLUDE_COLUMNS, f"Expected sensory col '{col}' in EXCLUDE_COLUMNS"

    def test_contains_metadata_columns(self):
        for col in ["id", "slug", "url", "name", "roaster", "roast"]:
            assert col in EXCLUDE_COLUMNS, f"Expected metadata col '{col}' in EXCLUDE_COLUMNS"

    def test_no_feature_columns(self):
        """Feature columns (tfidf_, bert_, etc.) must NOT be excluded."""
        for col in EXCLUDE_COLUMNS:
            assert not col.startswith(("tfidf_", "bert_", "glove_", "topic_")), (
                f"Feature column '{col}' should not be in EXCLUDE_COLUMNS"
            )

    def test_no_duplicates(self):
        assert len(EXCLUDE_COLUMNS) == len(set(EXCLUDE_COLUMNS))


# ============================================================
# run_preprocessing
# ============================================================


class TestRunPreprocessing:
    """run_preprocessing(args, config) -> bool"""

    def test_returns_true_on_success(self, mock_args, mock_config):
        with patch("src.pipeline.preprocess.process_raw_data") as mock_proc:
            result = run_preprocessing(mock_args, mock_config)
        assert result is True
        mock_proc.assert_called_once()

    def test_passes_output_path_from_config(self, mock_args, mock_config):
        with patch("src.pipeline.preprocess.process_raw_data") as mock_proc:
            run_preprocessing(mock_args, mock_config)
        _, kwargs = mock_proc.call_args
        assert kwargs.get("output_file") == str(
            mock_config.paths.get_processed_data_path()
        )

    def test_passes_sample_fraction(self, mock_args, mock_config):
        mock_args.sample_fraction = 0.1
        with patch("src.pipeline.preprocess.process_raw_data") as mock_proc:
            run_preprocessing(mock_args, mock_config)
        _, kwargs = mock_proc.call_args
        assert kwargs.get("sample_fraction") == 0.1

    def test_passes_sample_size(self, mock_args, mock_config):
        mock_args.sample_size = 50
        with patch("src.pipeline.preprocess.process_raw_data") as mock_proc:
            run_preprocessing(mock_args, mock_config)
        _, kwargs = mock_proc.call_args
        assert kwargs.get("sample_size") == 50

    def test_returns_false_on_exception(self, mock_args, mock_config):
        with patch(
            "src.pipeline.preprocess.process_raw_data",
            side_effect=RuntimeError("disk full"),
        ):
            result = run_preprocessing(mock_args, mock_config)
        assert result is False


# ============================================================
# run_feature_extraction
# ============================================================


class TestRunFeatureExtraction:
    """run_feature_extraction(args, config) -> bool"""

    def test_returns_true_on_success(self, mock_args, mock_config, sample_feature_df, tmp_dirs):
        processed_path = mock_config.paths.get_processed_data_path()
        pl.from_pandas(sample_feature_df).write_csv(processed_path)

        mock_manager = MagicMock()
        mock_manager.extract_all_features.return_value = pl.from_pandas(sample_feature_df)

        with patch("src.pipeline.features.CoffeeFeatureManager", return_value=mock_manager):
            result = run_feature_extraction(mock_args, mock_config)

        assert result is True

    def test_calls_fit_and_extract(self, mock_args, mock_config, sample_feature_df):
        processed_path = mock_config.paths.get_processed_data_path()
        pl.from_pandas(sample_feature_df).write_csv(processed_path)

        mock_manager = MagicMock()
        mock_manager.extract_all_features.return_value = pl.from_pandas(sample_feature_df)

        with patch("src.pipeline.features.CoffeeFeatureManager", return_value=mock_manager):
            run_feature_extraction(mock_args, mock_config)

        mock_manager.fit.assert_called_once()
        mock_manager.extract_all_features.assert_called_once()

    def test_returns_false_when_input_missing(self, mock_args, mock_config):
        # processed CSV does not exist
        result = run_feature_extraction(mock_args, mock_config)
        assert result is False

    def test_returns_false_on_exception(self, mock_args, mock_config, sample_feature_df):
        processed_path = mock_config.paths.get_processed_data_path()
        pl.from_pandas(sample_feature_df).write_csv(processed_path)

        with patch(
            "src.pipeline.features.CoffeeFeatureManager",
            side_effect=RuntimeError("CUDA OOM"),
        ):
            result = run_feature_extraction(mock_args, mock_config)

        assert result is False


# ============================================================
# run_feature_selection
# ============================================================


class TestRunFeatureSelection:
    """run_feature_selection(args, config) -> bool"""

    def test_returns_true_on_success(self, mock_args, mock_config, sample_feature_df):
        features_path = mock_config.paths.get_features_data_path()
        pl.from_pandas(sample_feature_df).write_csv(features_path)

        mock_selector = MagicMock()
        mock_selector.fit_transform.return_value = sample_feature_df[
            ["tfidf_desc_1_word1", "tfidf_desc_1_word2"]
        ]
        mock_selector.get_selected_features.return_value = [
            "tfidf_desc_1_word1",
            "tfidf_desc_1_word2",
        ]
        # get_selection_summary must return something picklable
        mock_selector.get_selection_summary.return_value = {"n_selected": 2, "n_total": 5}

        with patch("src.pipeline.selection.LassoFeatureSelector", return_value=mock_selector):
            result = run_feature_selection(mock_args, mock_config)

        assert result is True

    def test_skips_selection_when_disabled(self, mock_args, mock_config, sample_feature_df):
        mock_config.models.feature_selection_enabled = False
        features_path = mock_config.paths.get_features_data_path()
        pl.from_pandas(sample_feature_df).write_csv(features_path)

        with patch("src.pipeline.selection.LassoFeatureSelector") as mock_sel_cls:
            result = run_feature_selection(mock_args, mock_config)

        assert result is True
        mock_sel_cls.assert_not_called()

    def test_returns_false_when_target_missing(self, mock_args, mock_config, sample_feature_df):
        df_no_target = sample_feature_df.drop(columns=["rating"])
        features_path = mock_config.paths.get_features_data_path()
        pl.from_pandas(df_no_target).write_csv(features_path)

        result = run_feature_selection(mock_args, mock_config)

        assert result is False

    def test_returns_false_when_input_missing(self, mock_args, mock_config):
        result = run_feature_selection(mock_args, mock_config)
        assert result is False

    def test_returns_false_on_exception(self, mock_args, mock_config, sample_feature_df):
        features_path = mock_config.paths.get_features_data_path()
        pl.from_pandas(sample_feature_df).write_csv(features_path)

        with patch(
            "src.pipeline.selection.LassoFeatureSelector",
            side_effect=RuntimeError("lasso failed"),
        ):
            result = run_feature_selection(mock_args, mock_config)

        assert result is False


# ============================================================
# run_training
# ============================================================


class TestRunTraining:
    """run_training(args, config) -> bool"""

    def _write_features(self, mock_config, df):
        """Helper: write feature CSV where training will look for it."""
        selected_path = mock_config.paths.processed / "coffee_features_selected.csv"
        pl.from_pandas(df).write_csv(selected_path)

    def test_returns_true_standard_path(self, mock_args, mock_config, sample_feature_df):
        self._write_features(mock_config, sample_feature_df)

        mock_model = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.compare_models.return_value = {
            "summary_metrics": {"r2": {"linear": 0.85}, "rmse": {"linear": 0.5}, "mae": {"linear": 0.4}},
            "best_models": {"r2": "linear", "rmse": "linear", "mae": "linear"},
            "individual_results": {"linear": {"predictions": np.zeros(6)}},
            "comparison_report": "Test report",
        }

        with (
            patch("src.pipeline.training.CoffeeLinearRegression", return_value=mock_model),
            patch("src.pipeline.training.CoffeeModelEvaluator", return_value=mock_evaluator),
        ):
            result = run_training(mock_args, mock_config)

        assert result is True
        mock_model.fit.assert_called_once()

    def test_returns_false_when_target_missing(self, mock_args, mock_config, sample_feature_df):
        df_no_target = sample_feature_df.drop(columns=["rating"])
        self._write_features(mock_config, df_no_target)

        result = run_training(mock_args, mock_config)

        assert result is False

    def test_returns_false_when_input_missing(self, mock_args, mock_config):
        result = run_training(mock_args, mock_config)
        assert result is False

    def test_box_cox_path_calls_transformer(self, mock_args, mock_config, sample_feature_df):
        mock_config.models.box_cox_enabled = True
        self._write_features(mock_config, sample_feature_df)

        mock_model = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.compare_models_with_predictions.return_value = {
            "summary_metrics": {"r2": {"linear": 0.85}, "rmse": {"linear": 0.5}, "mae": {"linear": 0.4}},
            "best_models": {"r2": "linear"},
            "individual_results": {},
            "comparison_report": "Test report",
        }
        mock_transformer = MagicMock()
        mock_transformer.lambda_ = 0.5
        mock_transformer.fit_transform.return_value = np.zeros(14)
        mock_transformer.transform.return_value = np.zeros(6)
        mock_transformer.inverse_transform.return_value = np.zeros(6)

        with (
            patch("src.pipeline.training.CoffeeLinearRegression", return_value=mock_model),
            patch("src.pipeline.training.CoffeeModelEvaluator", return_value=mock_evaluator),
            patch("src.pipeline.training.BoxCoxTransformer", return_value=mock_transformer),
        ):
            result = run_training(mock_args, mock_config)

        assert result is True
        mock_transformer.fit_transform.assert_called_once()

    def test_box_cox_dual_path_calls_dual_pipeline(self, mock_args, mock_config, sample_feature_df):
        mock_config.models.box_cox_dual_pipeline = True
        self._write_features(mock_config, sample_feature_df)

        mock_model = MagicMock()
        dual_result = {
            "baseline_results": {
                "summary_metrics": {"r2": {"linear": 0.85}, "rmse": {"linear": 0.5}, "mae": {"linear": 0.4}},
                "best_models": {"r2": "linear"},
                "individual_results": {},
                "comparison_report": "Dual report",
            },
            "recommendation": "No Box-Cox",
            "recommendation_reason": "No improvement",
        }
        mock_evaluator = MagicMock()

        with (
            patch("src.pipeline.training.CoffeeLinearRegression", return_value=mock_model),
            patch("src.pipeline.training.CoffeeModelEvaluator", return_value=mock_evaluator),
            patch(
                "src.pipeline.training.run_box_cox_dual_pipeline",
                return_value=dual_result,
            ) as mock_dual,
        ):
            result = run_training(mock_args, mock_config)

        assert result is True
        mock_dual.assert_called_once()

    def test_saves_evaluation_results(self, mock_args, mock_config, sample_feature_df):
        self._write_features(mock_config, sample_feature_df)

        mock_model = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.compare_models.return_value = {
            "summary_metrics": {"r2": {"linear": 0.85}, "rmse": {"linear": 0.5}, "mae": {"linear": 0.4}},
            "best_models": {"r2": "linear"},
            "individual_results": {},
            "comparison_report": "Test report",
        }

        with (
            patch("src.pipeline.training.CoffeeLinearRegression", return_value=mock_model),
            patch("src.pipeline.training.CoffeeModelEvaluator", return_value=mock_evaluator),
        ):
            run_training(mock_args, mock_config)

        mock_evaluator.save_comprehensive_evaluation.assert_called_once()

    def test_returns_false_on_exception(self, mock_args, mock_config, sample_feature_df):
        self._write_features(mock_config, sample_feature_df)

        with patch(
            "src.pipeline.training.CoffeeLinearRegression",
            side_effect=RuntimeError("model init failed"),
        ):
            result = run_training(mock_args, mock_config)

        assert result is False


# ============================================================
# run_visualization
# ============================================================


class TestRunVisualization:
    """run_visualization(args, config) -> bool"""

    def _write_eval_results(self, mock_config, results):
        """Write the pkl file that visualization will load."""
        import pickle

        pkl_path = mock_config.paths.output / "comprehensive_model_evaluation.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)

    def test_returns_false_when_no_eval_results(self, mock_args, mock_config):
        """Visualization must fail clearly when training hasn't run."""
        result = run_visualization(mock_args, mock_config)
        assert result is False

    def test_returns_true_on_success(self, mock_args, mock_config, sample_feature_df):
        eval_results = {
            "best_models": {"r2": "linear"},
            "individual_results": {
                "linear": {"predictions": np.zeros(6), "feature_importance": {}}
            },
        }
        self._write_eval_results(mock_config, eval_results)

        # Write features so visualize can reconstruct test split
        features_path = mock_config.paths.get_features_data_path()
        pl.from_pandas(sample_feature_df).write_csv(features_path)

        mock_evaluator = MagicMock()
        mock_evaluator.plot_model_comparison.return_value = MagicMock()
        mock_evaluator.plot_predictions.return_value = MagicMock()

        with patch("src.pipeline.visualization.CoffeeModelEvaluator", return_value=mock_evaluator):
            result = run_visualization(mock_args, mock_config)

        assert result is True

    def test_returns_false_on_exception(self, mock_args, mock_config):
        eval_results = {"best_models": {"r2": "linear"}, "individual_results": {}}
        self._write_eval_results(mock_config, eval_results)

        with patch(
            "src.pipeline.visualization.CoffeeModelEvaluator",
            side_effect=RuntimeError("plot failed"),
        ):
            result = run_visualization(mock_args, mock_config)

        assert result is False

    def test_loads_from_correct_pkl_path(self, mock_args, mock_config, sample_feature_df):
        """Verify visualization loads from the same path training saves to."""
        import pickle

        eval_results = {
            "best_models": {"r2": "linear"},
            "individual_results": {"linear": {"predictions": np.zeros(6), "feature_importance": {}}},
        }

        # Write to the TRAINING output path — visualization must find it there
        training_output_path = mock_config.paths.output / "comprehensive_model_evaluation.pkl"
        with open(training_output_path, "wb") as f:
            pickle.dump(eval_results, f)

        features_path = mock_config.paths.get_features_data_path()
        pl.from_pandas(sample_feature_df).write_csv(features_path)

        mock_evaluator = MagicMock()
        mock_evaluator.plot_model_comparison.return_value = MagicMock()
        mock_evaluator.plot_predictions.return_value = MagicMock()

        with patch("src.pipeline.visualization.CoffeeModelEvaluator", return_value=mock_evaluator):
            result = run_visualization(mock_args, mock_config)

        # If it loaded from the wrong path, it would return False
        assert result is True

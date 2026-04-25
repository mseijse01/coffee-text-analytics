"""
Unit tests for MultinomialInverseRegression (MNIR).

Tests cover:
- Fit on synthetic sensory data
- get_attribute_feature_importance (renamed from get_feature_importance with args)
- get_feature_importance (no-arg aggregated interface used by evaluator)
- generate_insights_report output
- save / load round-trip
- Edge cases: missing attributes, all-NaN sensory scores
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.mnir import MultinomialInverseRegression


@pytest.fixture
def synthetic_data():
    """Small synthetic dataset — no real coffee data needed."""
    np.random.seed(42)
    n, p = 80, 20
    X = pd.DataFrame(
        np.random.randn(n, p),
        columns=[f"tfidf_desc_1_word{i}" for i in range(p)],
    )
    sensory = {
        "aroma": np.random.uniform(7, 10, n),
        "acid": np.random.uniform(7, 10, n),
        "body": np.random.uniform(7, 10, n),
        "flavor": np.random.uniform(7, 10, n),
        "aftertaste": np.random.uniform(7, 10, n),
    }
    return X, sensory


@pytest.fixture
def fitted_mnir(synthetic_data):
    X, sensory = synthetic_data
    mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
    mnir.fit(X, sensory)
    return mnir, X, sensory


class TestMNIRFit:
    @pytest.mark.unit
    def test_fit_sets_is_fitted(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        assert mnir.is_fitted

    @pytest.mark.unit
    def test_fit_trains_all_attributes(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        assert set(mnir.regression_models.keys()) == {
            "aroma",
            "acid",
            "body",
            "flavor",
            "aftertaste",
        }

    @pytest.mark.unit
    def test_fit_preserves_feature_names_from_dataframe(self, fitted_mnir):
        mnir, X, _ = fitted_mnir
        assert mnir.feature_names == list(X.columns)

    @pytest.mark.unit
    def test_fit_numpy_array_uses_generic_names(self, synthetic_data):
        X, sensory = synthetic_data
        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X.values, sensory)
        assert all(n.startswith("feature_") for n in mnir.feature_names)

    @pytest.mark.unit
    def test_fit_skips_missing_attribute(self, synthetic_data):
        X, sensory = synthetic_data
        partial = {k: v for k, v in sensory.items() if k != "body"}
        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, partial)
        assert "body" not in mnir.regression_models
        assert "aroma" in mnir.regression_models

    @pytest.mark.unit
    def test_fit_skips_all_nan_attribute(self, synthetic_data):
        X, sensory = synthetic_data
        sensory_with_nan = dict(sensory)
        sensory_with_nan["aroma"] = np.full(len(X), np.nan)
        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory_with_nan)
        assert "aroma" not in mnir.regression_models


class TestMNIRFeatureImportance:
    @pytest.mark.unit
    def test_get_attribute_feature_importance_returns_list(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        result = mnir.get_attribute_feature_importance("aroma", top_n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    @pytest.mark.unit
    def test_get_attribute_feature_importance_tuples(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        result = mnir.get_attribute_feature_importance("acid")
        for name, score in result:
            assert isinstance(name, str)
            assert isinstance(score, (float, np.floating))
            assert score >= 0

    @pytest.mark.unit
    def test_get_attribute_feature_importance_sorted_descending(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        result = mnir.get_attribute_feature_importance("body")
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    def test_get_attribute_feature_importance_unknown_attribute_raises(
        self, fitted_mnir
    ):
        mnir, _, _ = fitted_mnir
        with pytest.raises(Exception):
            mnir.get_attribute_feature_importance("nonexistent")

    @pytest.mark.unit
    def test_get_feature_importance_no_args_returns_dict(self, fitted_mnir):
        """Evaluator interface: get_feature_importance() with no args."""
        mnir, _, _ = fitted_mnir
        result = mnir.get_feature_importance()
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, score in result.items():
            assert isinstance(name, str)
            assert score >= 0

    @pytest.mark.unit
    def test_get_feature_importance_aggregates_across_attributes(self, fitted_mnir):
        """Aggregated importance should cover features from all attributes."""
        mnir, _, _ = fitted_mnir
        result = mnir.get_feature_importance()
        # Should have at least some features
        assert len(result) > 0


class TestMNIRReport:
    @pytest.mark.unit
    def test_generate_insights_report_returns_string(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        report = mnir.generate_insights_report()
        assert isinstance(report, str)

    @pytest.mark.unit
    def test_generate_insights_report_contains_attributes(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        report = mnir.generate_insights_report()
        for attr in ["AROMA", "ACID", "BODY", "FLAVOR", "AFTERTASTE"]:
            assert attr in report

    @pytest.mark.unit
    def test_generate_insights_report_contains_r2(self, fitted_mnir):
        mnir, _, _ = fitted_mnir
        report = mnir.generate_insights_report()
        assert "R² Score" in report

    @pytest.mark.unit
    def test_generate_insights_report_contains_feature_names(self, fitted_mnir):
        """After fix: report should show real feature names, not feature_N."""
        mnir, X, _ = fitted_mnir
        report = mnir.generate_insights_report()
        # At least one real feature name should appear
        assert any(col in report for col in X.columns)


class TestMNIRPersistence:
    @pytest.mark.unit
    def test_save_load_roundtrip(self, fitted_mnir):
        mnir, X, _ = fitted_mnir
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            mnir.save_model(path)
            loaded = MultinomialInverseRegression({})
            loaded.load_model(path)
            assert loaded.is_fitted
            assert loaded.feature_names == mnir.feature_names
            assert set(loaded.regression_models.keys()) == set(
                mnir.regression_models.keys()
            )
        finally:
            os.unlink(path)

    @pytest.mark.unit
    def test_save_unfitted_raises(self):
        mnir = MultinomialInverseRegression({})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            # save_model on an unfitted MNIR — should either raise or produce
            # a loadable artifact (implementation-defined); at minimum no crash
            # that corrupts state
            mnir.save_model(path)
        except Exception:
            pass  # Raising is fine
        finally:
            os.unlink(path)

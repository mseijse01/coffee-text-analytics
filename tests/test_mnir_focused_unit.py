"""
Focused unit tests for MNIR fixes.

Tests the critical bug fixes without depending on full feature extraction pipeline:
- Sensory data index alignment (df.loc[X.index, col])
- Feature name preservation
- Save/load roundtrip
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
def synthetic_features_and_sensory():
    """Create synthetic feature matrix and sensory data (already aligned)."""
    np.random.seed(42)
    n = 100

    # Feature matrix with real names
    X = pd.DataFrame(
        np.random.randn(n, 30), columns=[f"tfidf_desc_1_word{i}" for i in range(30)]
    )

    # Sensory data aligned with X's index
    sensory = {
        "aroma": np.random.uniform(7, 10, n),
        "acid": np.random.uniform(7, 10, n),
        "body": np.random.uniform(7, 10, n),
        "flavor": np.random.uniform(7, 10, n),
        "aftertaste": np.random.uniform(7, 10, n),
    }

    return X, sensory


class TestMNIRSensoryDataAlignment:
    """Test the critical fix: sensory data must align with X.index."""

    @pytest.mark.unit
    def test_sensory_data_alignment_with_matching_indices(
        self, synthetic_features_and_sensory
    ):
        """MNIR should fit when sensory data indices match feature matrix."""
        X, sensory = synthetic_features_and_sensory

        # Create MNIR and fit — should not raise
        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        assert mnir.is_fitted
        assert len(mnir.regression_models) == 5

    @pytest.mark.unit
    def test_sensory_data_alignment_with_mismatched_indices_fails(
        self, synthetic_features_and_sensory
    ):
        """MNIR should fail when sensory data size doesn't match X."""
        X, sensory = synthetic_features_and_sensory

        # Create mismatched sensory data (different size)
        sensory_wrong = {
            col: np.random.uniform(7, 10, len(X) * 2) for col in sensory.keys()
        }

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})

        # Should fail due to shape mismatch
        with pytest.raises((ValueError, AssertionError, IndexError)):
            mnir.fit(X, sensory_wrong)

    @pytest.mark.unit
    def test_sensory_data_partial_subset(self, synthetic_features_and_sensory):
        """MNIR should handle sensory data with fewer columns."""
        X, sensory = synthetic_features_and_sensory

        # Only provide some sensory attributes
        partial_sensory = {"aroma": sensory["aroma"], "flavor": sensory["flavor"]}

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, partial_sensory)

        # Should only have models for provided attributes
        assert "aroma" in mnir.regression_models
        assert "flavor" in mnir.regression_models
        assert "body" not in mnir.regression_models


class TestMNIRFeatureNamePreservation:
    """Test that real feature names are preserved and used."""

    @pytest.mark.unit
    def test_feature_names_preserved_from_dataframe(
        self, synthetic_features_and_sensory
    ):
        """MNIR should preserve DataFrame column names."""
        X, sensory = synthetic_features_and_sensory
        original_columns = list(X.columns)

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        # Feature names should match original DataFrame columns
        assert mnir.feature_names == original_columns

    @pytest.mark.unit
    def test_feature_importance_uses_real_names(self, synthetic_features_and_sensory):
        """Feature importance should use real names, not generic feature_N."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        # Get feature importance
        importance = mnir.get_attribute_feature_importance("aroma", top_n=5)

        # All feature names should contain "tfidf", not be generic "feature_"
        for feature_name, score in importance:
            assert "tfidf" in feature_name, f"Expected real name, got {feature_name}"
            assert not feature_name.startswith("feature_")

    @pytest.mark.unit
    def test_insights_report_contains_real_features(
        self, synthetic_features_and_sensory
    ):
        """Generated insights report should contain real feature names."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        report = mnir.generate_insights_report()

        # Report should contain TF-IDF feature names
        assert "tfidf_desc_1_word" in report


class TestMNIRPersistenceWithRealNames:
    """Test save/load preserves feature names and models."""

    @pytest.mark.unit
    def test_save_load_preserves_feature_names(self, synthetic_features_and_sensory):
        """Save/load should preserve feature names."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        original_names = list(mnir.feature_names)
        original_models = set(mnir.regression_models.keys())

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            mnir.save_model(path)
            loaded = MultinomialInverseRegression({})
            loaded.load_model(path)

            assert loaded.feature_names == original_names
            assert set(loaded.regression_models.keys()) == original_models
        finally:
            os.unlink(path)

    @pytest.mark.unit
    def test_predictions_work_after_load(self, synthetic_features_and_sensory):
        """Loaded model should be able to make predictions."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            mnir.save_model(path)
            loaded = MultinomialInverseRegression({})
            loaded.load_model(path)

            # Make prediction on a sample
            test_sample = X.iloc[:5]
            preds = loaded.predict(test_sample, "aroma")

            assert len(preds) == 5
            assert all(isinstance(p, (int, float, np.number)) for p in preds)
        finally:
            os.unlink(path)


class TestMNIRMethodShadowing:
    """Test that get_feature_importance methods don't shadow each other."""

    @pytest.mark.unit
    def test_get_attribute_feature_importance_per_attribute(
        self, synthetic_features_and_sensory
    ):
        """get_attribute_feature_importance should work for specific attributes."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        # Per-attribute method
        aroma_importance = mnir.get_attribute_feature_importance("aroma", top_n=5)
        assert isinstance(aroma_importance, list)
        assert len(aroma_importance) <= 5
        assert all(
            isinstance(score, (int, float, np.number)) for _, score in aroma_importance
        )

    @pytest.mark.unit
    def test_get_feature_importance_aggregated(self, synthetic_features_and_sensory):
        """get_feature_importance() should aggregate across all attributes."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        # Aggregated method (no arguments)
        aggregated = mnir.get_feature_importance()
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0
        assert all(
            isinstance(score, (int, float, np.number)) for score in aggregated.values()
        )

    @pytest.mark.unit
    def test_both_methods_complement_each_other(self, synthetic_features_and_sensory):
        """Aggregated should include features from all per-attribute calls."""
        X, sensory = synthetic_features_and_sensory

        mnir = MultinomialInverseRegression({"lasso_cv": 2, "lasso_max_iter": 200})
        mnir.fit(X, sensory)

        # Get aggregated
        aggregated = mnir.get_feature_importance()

        # Get all per-attribute features
        all_per_attribute = set()
        for attr in mnir.regression_models.keys():
            per_attr = mnir.get_attribute_feature_importance(attr, top_n=len(X.columns))
            all_per_attribute.update(fname for fname, _ in per_attr)

        # Aggregated should have same features as union of per-attribute
        assert set(aggregated.keys()) == all_per_attribute

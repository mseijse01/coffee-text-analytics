#!/usr/bin/env python3
"""
Contract and Specification Tests for Feature Selectors - POLARS-FIRST EDITION

These tests verify that feature selectors conform to their POLARS-FIRST design,
supporting Polars DataFrames as the preferred input type with pandas/numpy fallback.

FOCUS:
- Polars DataFrame support validation
- Type flexibility (Polars + pandas + numpy)
- Performance verification for Polars paths
- Mixed input type handling
"""

import os
import tempfile
import time
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.feature_selector import LassoFeatureSelector


class TestPolarsFirstArchitecture:
    """Test Polars-first architecture support."""

    def test_polars_dataframe_support(self):
        """
        CORE TEST: Validate that Polars DataFrames are properly supported.

        This is the primary intended behavior - Polars should work seamlessly.
        """
        # Create realistic Polars test data
        X_polars = pl.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "feature_3": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        y_polars = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Should work with Polars inputs
        selector = LassoFeatureSelector({"cv_folds": 3})
        result = selector.fit_select_features(X_polars, y_polars)

        # Validate successful operation
        assert isinstance(
            result, LassoFeatureSelector
        ), "Should return self for method chaining"
        assert selector.is_fitted_, "Should be fitted after processing"

        # Validate transform works
        X_transformed = selector.transform(X_polars)
        assert hasattr(
            X_transformed, "shape"
        ), "Transform result should have shape attribute"
        assert (
            X_transformed.shape[0] == X_polars.shape[0]
        ), "Should preserve sample count"
        assert (
            X_transformed.shape[1] <= X_polars.shape[1]
        ), "Should reduce or maintain feature count"

        # Validate methods work
        selected_features = selector.get_selected_features()
        assert isinstance(
            selected_features, list
        ), "Should return list of feature names"
        assert len(selected_features) > 0, "Should select some features"

    def test_mixed_polars_pandas_inputs(self):
        """Test that mixed Polars/pandas inputs work correctly."""
        X_polars = pl.DataFrame(
            {
                "tfidf_feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "bert_feature_1": [0.6, 0.7, 0.8, 0.9, 1.0],
                "categorical_1": [1, 0, 1, 0, 1],
            }
        )
        y_pandas = pd.Series([92.0, 93.0, 91.0, 94.0, 92.5])

        # Should handle mixed types gracefully
        selector = LassoFeatureSelector({"cv_folds": 3})
        result = selector.fit_select_features(X_polars, y_pandas)
        assert isinstance(result, LassoFeatureSelector)

        # Transform should work
        X_transformed = selector.transform(X_polars)
        assert X_transformed.shape[0] == X_polars.shape[0]

    def test_polars_vs_pandas_feature_consistency(self):
        """Test that Polars and pandas inputs produce consistent results."""
        # Create identical data in both formats
        data_dict = {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "feature_3": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feature_4": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }

        X_polars = pl.DataFrame(data_dict)
        X_pandas = X_polars.to_pandas()
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Same configuration for both
        config = {"cv_folds": 3, "random_state": 42}

        # Process with Polars
        selector_polars = LassoFeatureSelector(config)
        selector_polars.fit_select_features(X_polars, y)
        features_polars = set(selector_polars.get_selected_features())

        # Process with pandas
        selector_pandas = LassoFeatureSelector(config)
        selector_pandas.fit_select_features(X_pandas, y)
        features_pandas = set(selector_pandas.get_selected_features())

        # Should produce consistent feature selection
        # (Allow for some variation due to numerical differences)
        intersection = features_polars & features_pandas
        union = features_polars | features_pandas

        if len(union) > 0:
            consistency_ratio = len(intersection) / len(union)
            assert (
                consistency_ratio >= 0.7
            ), f"Feature selection should be mostly consistent between Polars and pandas (got {consistency_ratio:.2f})"

    def test_corrected_selector_polars_support(self):
        """Test LassoFeatureSelector with Polars inputs."""
        X_polars = pl.DataFrame(
            {
                "tfidf_desc_1_coffee": [0.1, 0.2, 0.3, 0.4],
                "tfidf_desc_2_flavor": [0.2, 0.3, 0.4, 0.5],
                "bert_desc_1_emb0": [0.8, 0.9, 1.0, 1.1],
                "sentiment_desc_1_pos": [0.7, 0.8, 0.9, 0.6],
                "aroma": [8.0, 8.5, 9.0, 7.5],
                "origin_ethiopia": [1, 0, 1, 0],
            }
        )
        y_polars = pl.Series([92.0, 93.0, 91.0, 94.0])

        # Should work with Polars inputs
        selector = LassoFeatureSelector({"cv_folds": 3})
        result = selector.fit_select_features(X_polars, y_polars)

        assert isinstance(result, LassoFeatureSelector)
        assert len(selector.get_selected_features()) > 0
        assert len(selector.get_text_features()) > 0


class TestTypeFlexibilityAndCompatibility:
    """Test support for multiple input types."""

    def test_all_supported_input_types(self):
        """Test that all documented input types work correctly."""
        # Create test data in all supported formats
        data_values = [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0],
            [6.0, 9.0],
        ]
        target_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        # Test pandas DataFrame + pandas Series
        X_pandas = pd.DataFrame(data_values, columns=["feature_1", "feature_2"])
        y_pandas = pd.Series(target_values)

        selector1 = LassoFeatureSelector({"cv_folds": 3})
        result1 = selector1.fit_select_features(X_pandas, y_pandas)
        assert isinstance(result1, LassoFeatureSelector)

        # Test numpy array + numpy array
        X_numpy = np.array(data_values)
        y_numpy = np.array(target_values)

        selector2 = LassoFeatureSelector({"cv_folds": 3})
        result2 = selector2.fit_select_features(X_numpy, y_numpy)
        assert isinstance(result2, LassoFeatureSelector)

        # Test Polars DataFrame + Polars Series
        X_polars = pl.DataFrame(data_values, schema=["feature_1", "feature_2"])
        y_polars = pl.Series(target_values)

        selector3 = LassoFeatureSelector({"cv_folds": 3})
        result3 = selector3.fit_select_features(X_polars, y_polars)
        assert isinstance(result3, LassoFeatureSelector)

        # All should produce valid results
        for selector in [selector1, selector2, selector3]:
            assert len(selector.get_selected_features()) > 0
            assert isinstance(selector.get_feature_importance(), dict)

    def test_transform_type_preservation(self):
        """Test that transform preserves appropriate types."""
        # Create test data
        X_pandas = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]})
        X_polars = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]})
        X_numpy = np.array([[1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9]])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Test pandas preservation
        selector_pd = LassoFeatureSelector({"cv_folds": 3})
        selector_pd.fit_select_features(X_pandas, y)
        result_pd = selector_pd.transform(X_pandas)
        assert isinstance(
            result_pd, pd.DataFrame
        ), "pandas input should return pandas DataFrame"

        # Test numpy preservation
        selector_np = LassoFeatureSelector({"cv_folds": 3})
        selector_np.fit_select_features(X_numpy, y)
        result_np = selector_np.transform(X_numpy)
        assert isinstance(
            result_np, np.ndarray
        ), "numpy input should return numpy array"

        # Test Polars handling (implementation-dependent - just check it works)
        selector_pl = LassoFeatureSelector({"cv_folds": 3})
        selector_pl.fit_select_features(X_polars, y)
        result_pl = selector_pl.transform(X_polars)
        assert hasattr(
            result_pl, "shape"
        ), "Polars transform should return array-like object"


class TestPerformanceAndEfficiency:
    """Test performance characteristics of Polars vs pandas paths."""

    def test_polars_performance_baseline(self):
        """Test that Polars processing completes in reasonable time."""
        # Create larger dataset for performance testing
        n_samples, n_features = 500, 20

        # Generate data
        data_dict = {
            f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)
        }
        X_polars = pl.DataFrame(data_dict)
        y_polars = pl.Series(np.random.randn(n_samples))

        # Time the operation
        start_time = time.time()
        selector = LassoFeatureSelector({"cv_folds": 3})
        selector.fit_select_features(X_polars, y_polars)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (generous bound for CI)
        assert (
            elapsed_time < 30.0
        ), f"Polars processing took too long: {elapsed_time:.2f}s"

        # Should produce meaningful results
        assert len(selector.get_selected_features()) > 0
        assert selector.is_fitted_

    def test_memory_efficiency_with_polars(self):
        """Test that Polars inputs don't cause memory issues."""
        # Create moderately large dataset
        n_samples, n_features = 200, 50

        data_dict = {
            f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)
        }
        X_polars = pl.DataFrame(data_dict)
        y_polars = pl.Series(np.random.randn(n_samples))

        # Should handle without memory errors
        selector = LassoFeatureSelector({"cv_folds": 3})
        try:
            selector.fit_select_features(X_polars, y_polars)
            X_transformed = selector.transform(X_polars)

            # Validate results
            assert X_transformed.shape[0] == n_samples
            assert X_transformed.shape[1] <= n_features

        except MemoryError:
            pytest.fail("Memory error with Polars DataFrame processing")


class TestErrorHandlingAndEdgeCases:
    """Test proper error handling while supporting Polars."""

    def test_invalid_input_types_are_rejected(self):
        """Test that truly invalid types are still rejected."""
        selector = LassoFeatureSelector({"cv_folds": 3})

        # These should still fail
        invalid_inputs = [
            "string_input",
            123,
            None,
            {"dict": "input"},
            ["list", "input"],
        ]

        for invalid_X in invalid_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                selector.fit_select_features(invalid_X, [1, 2, 3])

    def test_empty_dataframes_handling(self):
        """Test handling of empty DataFrames in both formats."""
        # Empty pandas DataFrame
        X_pandas_empty = pd.DataFrame()
        y_pandas_empty = pd.Series(dtype=float)

        selector1 = LassoFeatureSelector()
        with pytest.raises((ValueError, IndexError)):
            selector1.fit_select_features(X_pandas_empty, y_pandas_empty)

        # Empty Polars DataFrame
        X_polars_empty = pl.DataFrame()
        y_polars_empty = pl.Series(dtype=pl.Float64)

        selector2 = LassoFeatureSelector()
        with pytest.raises((ValueError, IndexError)):
            selector2.fit_select_features(X_polars_empty, y_polars_empty)

    def test_dimension_mismatches_with_polars(self):
        """Test dimension mismatch handling with Polars inputs."""
        X_polars = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        y_wrong_size = pl.Series([1, 2])  # Wrong size

        selector = LassoFeatureSelector()
        with pytest.raises((ValueError, IndexError)):
            selector.fit_select_features(X_polars, y_wrong_size)


class TestDocumentationAndSpecification:
    """Test that behavior matches intended Polars-first specification."""

    def test_polars_first_design_intent(self):
        """
        Test that the Polars-first design intent is properly implemented.

        This validates the architectural decision to prioritize Polars.
        """
        # Create test data that should work optimally with Polars
        X_polars = pl.DataFrame(
            {
                "tfidf_desc_1_coffee": [0.1, 0.2, 0.3, 0.4, 0.5],
                "bert_desc_1_0": [0.8, 0.9, 1.0, 1.1, 1.2],
                "sentiment_pos": [0.6, 0.7, 0.8, 0.9, 0.5],
                "categorical_origin": [1, 0, 1, 0, 1],
            }
        )
        y_polars = pl.Series([92.0, 93.0, 91.0, 94.0, 92.5])

        # Both selectors should handle this seamlessly
        for SelectorClass in [LassoFeatureSelector, LassoFeatureSelector]:
            selector = SelectorClass({"cv_folds": 3})

            # Should work without conversion
            result = selector.fit_select_features(X_polars, y_polars)
            assert isinstance(result, SelectorClass)

            # Should produce meaningful feature selection
            selected_features = selector.get_selected_features()
            assert len(selected_features) > 0
            assert all(isinstance(f, str) for f in selected_features)

            # Should handle transform
            X_transformed = selector.transform(X_polars)
            assert X_transformed.shape[0] == X_polars.shape[0]

    def test_backward_compatibility_maintained(self):
        """Test that pandas/numpy support is maintained for backward compatibility."""
        # Create test data in legacy formats
        X_pandas = pd.DataFrame(
            {"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]}
        )
        X_numpy = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        # Should still work with legacy formats
        for X_input in [X_pandas, X_numpy]:
            selector = LassoFeatureSelector({"cv_folds": 3})
            result = selector.fit_select_features(X_input, y)
            assert isinstance(result, LassoFeatureSelector)
            assert len(selector.get_selected_features()) > 0

"""
Tests for the categorical feature encoder functionality.

This module tests the CategoricalFeatureEncoder class and its integration
with the coffee text analytics pipeline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.features.categorical_encoder import CategoricalFeatureEncoder


class TestCategoricalFeatureEncoder:
    """Test suite for CategoricalFeatureEncoder class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing categorical encoding."""
        return pd.DataFrame(
            {
                "roaster": [
                    "Blue Bottle",
                    "Stumptown",
                    "Blue Bottle",
                    "Blue Bottle",
                    "Counter Culture",
                    "Intelligentsia",
                    "Stumptown",
                    "Blue Bottle",
                    "Stumptown",
                    "Counter Culture",
                ],
                "country_of_origin": [
                    "Ethiopia",
                    "Colombia",
                    "Kenya",
                    "Ethiopia",
                    "Guatemala",
                    "Brazil",
                    "Colombia",
                    "Ethiopia",
                    "Kenya",
                    "Colombia",
                ],
                "roast": [
                    "Light",
                    "Medium",
                    "Medium",
                    "Light",
                    "Dark",
                    "Medium",
                    "Light",
                    "Medium",
                    "Dark",
                    "Light",
                ],
                "cupper_points": [
                    85.5,
                    88.2,
                    84.1,
                    87.3,
                    86.8,
                    89.1,
                    85.9,
                    86.4,
                    88.7,
                    87.5,
                ],
            }
        )

    @pytest.fixture
    def encoder(self):
        """Create a CategoricalFeatureEncoder instance."""
        return CategoricalFeatureEncoder()

    def test_encoder_initialization(self, encoder):
        """Test that encoder initializes correctly."""
        assert encoder.config is not None
        assert encoder.default_config is not None
        assert "roast" in encoder.config
        assert "country_of_origin" in encoder.config
        assert "roaster" in encoder.config
        assert not encoder.is_fitted

    @pytest.mark.unit
    def test_fit_roast_encoding(self, encoder, sample_data):
        """Test fitting the roast encoder (should use standard one-hot)."""
        encoder.fit(sample_data)

        assert encoder.is_fitted
        assert "roast" in encoder.encoders
        assert isinstance(encoder.encoders["roast"], OneHotEncoder)

        # Should have roast feature names
        assert "roast" in encoder.feature_names
        roast_features = encoder.feature_names["roast"]
        assert len(roast_features) >= 3  # At least Dark, Light, Medium

    @pytest.mark.unit
    def test_fit_country_encoding(self, encoder, sample_data):
        """Test fitting the country encoder (top-K strategy)."""
        encoder.fit(sample_data)

        assert "country_of_origin" in encoder.encoders
        assert "country_of_origin" in encoder.category_mappings

        # Should have country feature names
        country_features = encoder.feature_names["country_of_origin"]
        assert len(country_features) > 0

    @pytest.mark.unit
    def test_fit_roaster_encoding(self, encoder, sample_data):
        """Test fitting the roaster encoder (frequency grouping strategy)."""
        encoder.fit(sample_data)

        assert "roaster" in encoder.encoders
        assert "roaster" in encoder.category_mappings

        # Should have roaster feature names
        roaster_features = encoder.feature_names["roaster"]
        assert len(roaster_features) > 0

    @pytest.mark.unit
    def test_transform_all_categories(self, encoder, sample_data):
        """Test transformation of all categorical features."""
        encoder.fit(sample_data)
        result = encoder.transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

        # Check that we have encoded features (should also keep original columns)
        roast_cols = [
            col for col in result.columns if col.startswith("roast_") and col != "roast"
        ]
        country_cols = [
            col
            for col in result.columns
            if col.startswith("country_") and col != "country_of_origin"
        ]
        roaster_cols = [
            col
            for col in result.columns
            if col.startswith("roaster_") and col != "roaster"
        ]

        assert len(roast_cols) > 0
        assert len(country_cols) > 0
        assert len(roaster_cols) > 0

        # All encoded columns should be binary (0 or 1) - test only the prefixed columns
        encoded_cols = roast_cols + country_cols + roaster_cols
        for col in encoded_cols:
            assert (
                result[col].isin([0, 1]).all()
            ), f"Column {col} should be binary but contains: {result[col].unique()}"

    @pytest.mark.unit
    def test_transform_specific_roast_categories(self, encoder, sample_data):
        """Test that roast categories are transformed correctly."""
        encoder.fit(sample_data)
        result = encoder.transform(sample_data)

        # Check specific roast encoding
        roast_cols = [col for col in result.columns if col.startswith("roast_")]

        # Should have roast columns
        assert len(roast_cols) >= 3  # At least Dark, Light, Medium

        # Check that each row has exactly one roast category active
        roast_data = result[roast_cols]
        assert (roast_data.sum(axis=1) == 1).all()

    @pytest.mark.unit
    def test_transform_with_new_categories(self, encoder, sample_data):
        """Test transformation when new data contains unseen categories."""
        encoder.fit(sample_data)

        # Create test data with new categories
        new_data = pd.DataFrame(
            {
                "roaster": ["New Roaster", "Blue Bottle"],
                "country_of_origin": ["New Country", "Ethiopia"],
                "roast": ["Extra Light", "Medium"],
                "cupper_points": [85.0, 86.0],
            }
        )

        result = encoder.transform(new_data)

        # Should handle new categories gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(new_data)

        # Check that we have the expected feature columns
        roaster_cols = [col for col in result.columns if col.startswith("roaster_")]
        country_cols = [col for col in result.columns if col.startswith("country_")]

        assert len(roaster_cols) > 0
        assert len(country_cols) > 0

    @pytest.mark.unit
    def test_fit_transform(self, encoder, sample_data):
        """Test the fit_transform convenience method."""
        result = encoder.fit_transform(sample_data)

        assert encoder.is_fitted
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

        # Should have the same result as separate fit and transform
        encoder2 = CategoricalFeatureEncoder()
        encoder2.fit(sample_data)
        result2 = encoder2.transform(sample_data)

        # Check that both results have the same shape and similar structure
        assert result.shape == result2.shape

    @pytest.mark.unit
    def test_get_feature_names(self, encoder, sample_data):
        """Test getting the feature names after encoding."""
        encoder.fit(sample_data)
        feature_names = encoder.get_feature_names()

        assert isinstance(feature_names, dict)
        assert len(feature_names) > 0

        # Should include all categories
        assert "roast" in feature_names
        assert "country_of_origin" in feature_names
        assert "roaster" in feature_names

        # Each category should have a list of feature names
        for col, names in feature_names.items():
            assert isinstance(names, list)
            assert len(names) > 0

    @pytest.mark.unit
    def test_error_when_not_fitted(self, encoder, sample_data):
        """Test that transform raises error when encoder is not fitted."""
        with pytest.raises((ValueError, AttributeError)):
            encoder.transform(sample_data)

    @pytest.mark.unit
    def test_get_encoding_summary(self, encoder, sample_data):
        """Test the encoding summary functionality."""
        encoder.fit(sample_data)
        summary = encoder.get_encoding_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0

        # Should contain information about each encoded column
        for col in ["roast", "country_of_origin", "roaster"]:
            if col in summary:
                assert "method" in summary[col]
                assert "n_features" in summary[col]

    @pytest.mark.methodology
    def test_thesis_compliance_encoding_strategies(self, encoder, sample_data):
        """Test that encoding strategies match thesis methodology."""
        encoder.fit(sample_data)
        result = encoder.transform(sample_data)

        # Roast: Standard one-hot (low cardinality)
        roast_cols = [col for col in result.columns if col.startswith("roast_")]
        assert len(roast_cols) >= 3  # Should have at least the basic roast levels

        # Country: Should include country-prefixed columns
        country_cols = [col for col in result.columns if col.startswith("country_")]
        assert len(country_cols) > 0

        # Roaster: Should include roaster-prefixed columns
        roaster_cols = [col for col in result.columns if col.startswith("roaster_")]
        assert len(roaster_cols) > 0

    @pytest.mark.integration
    def test_encoder_with_realistic_data_size(self, encoder):
        """Test encoder with realistic data size and distributions."""
        # Create larger, more realistic dataset
        np.random.seed(42)
        n_samples = 500

        roasters = ["Blue Bottle", "Stumptown", "Counter Culture", "Intelligentsia"] + [
            f"Small_Roaster_{i}" for i in range(20)
        ]  # Total: 24 roasters
        countries = ["Ethiopia", "Colombia", "Kenya", "Guatemala", "Brazil"] + [
            f"Country_{i}" for i in range(30)
        ]  # Total: 35 countries
        roasts = ["Light", "Medium", "Dark", "Medium-Light", "Medium-Dark"]

        # Create data with realistic frequency distribution (no probabilities to avoid size mismatch)
        data = pd.DataFrame(
            {
                "roaster": np.random.choice(roasters, size=n_samples),
                "country_of_origin": np.random.choice(countries, size=n_samples),
                "roast": np.random.choice(roasts, size=n_samples),
                "cupper_points": np.random.uniform(80, 95, n_samples),
            }
        )

        result = encoder.fit_transform(data)

        # Should handle the complexity properly
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_samples

        # Check reasonable number of features
        encoded_cols = [
            col
            for col in result.columns
            if col.startswith(("roast_", "country_", "roaster_"))
        ]
        assert 10 <= len(encoded_cols) <= 80  # Reasonable range for this encoding

    @pytest.mark.performance
    def test_encoding_performance(self, encoder):
        """Test that encoding is reasonably fast for large datasets."""
        import time

        # Create large dataset
        n_samples = 10000
        large_data = pd.DataFrame(
            {
                "roaster": np.random.choice(["A", "B", "C"] * 50, n_samples),
                "country_of_origin": np.random.choice(["X", "Y", "Z"] * 40, n_samples),
                "roast": np.random.choice(["Light", "Medium", "Dark"], n_samples),
                "cupper_points": np.random.uniform(80, 95, n_samples),
            }
        )

        start_time = time.time()
        result = encoder.fit_transform(large_data)
        execution_time = time.time() - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert execution_time < 5.0
        assert len(result) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

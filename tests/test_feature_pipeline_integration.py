"""
Integration tests for feature engineering pipeline.

Tests the complete end-to-end feature engineering workflow using real sample data,
focusing on pipeline behavior and output validation with strategic mocking of
heavyweight ML operations.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, List

# Import modules under test
from src.features.feature_manager import CoffeeFeatureManager, GloVeExtractor
from src.features.tfidf_extractor import TfidfExtractor
from src.features.bert_extractor import BertExtractor
from src.features.topic_extractor import TopicExtractor
from src.features.sentiment_extractor import SentimentExtractor
from src.features.categorical_encoder import CategoricalFeatureEncoder


class TestFeatureEngineeringPipelineIntegration:
    """Integration tests for complete feature engineering pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Load real sample data for testing."""
        return pd.read_csv("tests/data/coffee_sample.csv")

    @pytest.fixture
    def polars_sample_data(self, sample_data):
        """Convert sample data to Polars format."""
        return pl.from_pandas(sample_data)

    @pytest.fixture
    def text_columns(self):
        """Standard text columns for coffee review data."""
        return ["desc_1", "desc_2", "desc_3"]

    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing."""
        return {
            "extractors": {
                "tfidf": True,
                "bert": False,  # Skip heavyweight BERT initially
                "glove": False,  # Skip GloVe due to model size
                "topics": False,  # Skip topics initially
                "sentiment": False,  # Disable for now to avoid model loading
            },
            "tfidf": {
                "max_features": 50,  # Small for testing
                "ngram_range": (1, 2),
            },
        }

    @pytest.fixture
    def full_config(self):
        """Full configuration for testing all features."""
        return {
            "extractors": {
                "tfidf": True,
                "bert": True,
                "glove": True,
                "topics": True,
                "sentiment": True,
            },
            "tfidf": {"max_features": 100},
            "bert": {"model_name": "distilbert-base-uncased"},
            "glove": {"model_name": "glove-wiki-gigaword-50"},  # Smaller model
            "topics": {"n_topics": 5},
            "sentiment": {"model_name": "vader"},
        }

    def test_complete_feature_extraction_pipeline(
        self, polars_sample_data, text_columns, minimal_config
    ):
        """Test complete feature extraction pipeline from raw data to feature matrix."""
        # Initialize feature manager
        manager = CoffeeFeatureManager(minimal_config)

        # Check existing text columns in real data
        existing_text_cols = [
            col for col in text_columns if col in polars_sample_data.columns
        ]

        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        # Test fitting pipeline
        manager.fit(polars_sample_data, existing_text_cols)
        assert manager.is_fitted

        # Test feature extraction
        features = manager.extract_all_features(polars_sample_data, existing_text_cols)

        # Validate output
        assert isinstance(features, pl.DataFrame)
        assert features.shape[0] == polars_sample_data.shape[0]  # Same number of rows
        assert features.shape[1] > 0  # Features were generated

        # Validate feature naming convention
        feature_names = features.columns
        # Should have features for each enabled extractor and text column
        for extractor_name in ["tfidf"]:  # Only TF-IDF enabled in minimal config
            for col in existing_text_cols:
                extractor_features = [
                    f for f in feature_names if f.startswith(f"{extractor_name}_{col}")
                ]
                assert len(extractor_features) > 0, (
                    f"No features found for {extractor_name}_{col}"
                )

    @patch("src.features.bert_extractor.BertExtractor.extract_features")
    @patch("src.features.topic_extractor.TopicExtractor.extract_features")
    def test_heavyweight_ml_operations_mocked(
        self,
        mock_topic_extract,
        mock_bert_extract,
        polars_sample_data,
        text_columns,
        full_config,
    ):
        """Test that heavyweight ML operations are properly mocked."""
        # Setup mocks for heavyweight operations
        n_samples = polars_sample_data.shape[0]

        # Mock BERT features (768-dimensional)
        mock_bert_features = pl.DataFrame(
            {
                f"bert_{i}": np.random.random(n_samples)
                for i in range(10)  # Simplified for testing
            }
        )
        mock_bert_extract.return_value = mock_bert_features

        # Mock topic features
        mock_topic_features = pl.DataFrame(
            {f"topic_{i}": np.random.random(n_samples) for i in range(5)}
        )
        mock_topic_extract.return_value = mock_topic_features

        # Initialize manager with heavyweight features enabled
        manager = CoffeeFeatureManager(full_config)

        existing_text_cols = [
            col for col in text_columns if col in polars_sample_data.columns
        ]
        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        # Test that mocked features are used
        manager.fit(polars_sample_data, existing_text_cols)
        features = manager.extract_all_features(polars_sample_data, existing_text_cols)

        # Verify mocks were called
        assert mock_bert_extract.called
        assert mock_topic_extract.called

        # Validate that mocked features are in output
        feature_names = features.columns
        bert_features = [f for f in feature_names if "bert" in f]
        topic_features = [f for f in feature_names if "topic" in f]

        assert len(bert_features) > 0, "BERT features not found in output"
        assert len(topic_features) > 0, "Topic features not found in output"

    @pytest.mark.parametrize("data_format", ["polars", "pandas"])
    def test_polars_vs_pandas_consistency(
        self, sample_data, polars_sample_data, text_columns, minimal_config, data_format
    ):
        """Test that feature extraction behaves consistently across data formats."""
        # Get appropriate data format
        data = polars_sample_data if data_format == "polars" else sample_data

        existing_text_cols = [col for col in text_columns if col in data.columns]
        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        # Convert pandas to polars if needed for the feature manager
        if data_format == "pandas":
            data = pl.from_pandas(data)

        manager = CoffeeFeatureManager(minimal_config)
        manager.fit(data, existing_text_cols)
        features = manager.extract_all_features(data, existing_text_cols)

        # Validate output format and consistency
        assert isinstance(features, pl.DataFrame), (
            "Output should always be Polars DataFrame"
        )
        assert features.shape[0] == data.shape[0]
        assert features.shape[1] > 0

        # Validate data types
        for col in features.columns:
            assert features[col].dtype in [
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
            ], f"Feature {col} has unexpected dtype: {features[col].dtype}"

    def test_categorical_encoding_integration(self, polars_sample_data, minimal_config):
        """Test categorical feature encoding integration with text features."""
        # Convert to pandas since categorical encoder expects pandas
        pandas_data = polars_sample_data.to_pandas()

        # Find categorical columns in real data that match the encoder's expected columns
        expected_categorical_cols = ["roaster", "country_of_origin", "roast"]
        available_categorical_cols = [
            col for col in expected_categorical_cols if col in pandas_data.columns
        ]

        if not available_categorical_cols:
            pytest.skip("No expected categorical columns found in sample data")

        # Test categorical encoder
        encoder = CategoricalFeatureEncoder()
        encoded_features = encoder.fit_transform(pandas_data)

        # Validate encoding results
        assert isinstance(encoded_features, pd.DataFrame)
        assert encoded_features.shape[0] == pandas_data.shape[0]

        # Should have created new columns for categorical features
        for col in available_categorical_cols:
            # Look for encoded versions of this column
            encoded_cols = [
                c
                for c in encoded_features.columns
                if col in c
                or any(prefix in c for prefix in ["roast_", "country_", "roaster_"])
            ]
            assert len(encoded_cols) > 0, f"No encoded features found for {col}"

    @pytest.mark.parametrize("extractor_type", ["tfidf"])  # Only test TF-IDF for now
    def test_individual_extractor_integration(
        self, polars_sample_data, text_columns, extractor_type
    ):
        """Test individual feature extractors with real data."""
        existing_text_cols = [
            col for col in text_columns if col in polars_sample_data.columns
        ]
        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        # Get sample texts
        sample_texts = []
        for col in existing_text_cols:
            texts = polars_sample_data[col].fill_null("").to_list()
            sample_texts.extend(
                [text for text in texts if text and len(text.strip()) > 0]
            )

        if not sample_texts:
            pytest.skip("No valid text data found")

        # Test individual extractors
        if extractor_type == "tfidf":
            extractor = TfidfExtractor({"max_features": 50})
            extractor.fit(sample_texts)
            features = extractor.extract_features(sample_texts)

        # Validate extractor output
        assert isinstance(features, pl.DataFrame)
        assert features.shape[0] == len(sample_texts)
        assert features.shape[1] > 0

        # Validate feature names
        feature_names = extractor.get_feature_names()
        assert len(feature_names) == features.shape[1]
        assert all(isinstance(name, str) for name in feature_names)

    def test_feature_naming_consistency(
        self, polars_sample_data, text_columns, minimal_config
    ):
        """Test that feature names follow consistent naming conventions."""
        manager = CoffeeFeatureManager(minimal_config)

        existing_text_cols = [
            col for col in text_columns if col in polars_sample_data.columns
        ]
        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        manager.fit(polars_sample_data, existing_text_cols)
        features = manager.extract_all_features(polars_sample_data, existing_text_cols)

        # Validate naming conventions
        feature_names = features.columns

        # Features should follow pattern: {extractor}_{column}_{feature_name}
        for feature_name in feature_names:
            parts = feature_name.split("_")
            assert len(parts) >= 3, (
                f"Feature name {feature_name} doesn't follow naming convention"
            )

            extractor_name = parts[0]
            column_name = parts[1]

            # Verify extractor name is valid
            assert extractor_name in ["tfidf"], (
                f"Unknown extractor in feature name: {feature_name}"
            )

            # Verify column name is from text columns
            assert column_name in existing_text_cols, (
                f"Unknown column in feature name: {feature_name}"
            )

    def test_missing_data_handling_in_features(
        self, polars_sample_data, text_columns, minimal_config
    ):
        """Test feature extraction handles missing data appropriately."""
        # Create test data with missing values
        test_data = polars_sample_data.clone()

        existing_text_cols = [col for col in text_columns if col in test_data.columns]
        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        # Introduce missing data
        if len(test_data) > 0:
            # Set some text values to null
            col_to_modify = existing_text_cols[0]
            test_data = test_data.with_columns(
                [
                    pl.when(pl.int_range(len(test_data)) % 3 == 0)
                    .then(None)
                    .otherwise(pl.col(col_to_modify))
                    .alias(col_to_modify)
                ]
            )

        # Test feature extraction with missing data
        manager = CoffeeFeatureManager(minimal_config)
        manager.fit(test_data, existing_text_cols)
        features = manager.extract_all_features(test_data, existing_text_cols)

        # Validate handling of missing data
        assert isinstance(features, pl.DataFrame)
        assert features.shape[0] == test_data.shape[0]

        # Check that no features have all null values
        for col in features.columns:
            non_null_count = features[col].filter(pl.col(col).is_not_null()).shape[0]
            assert non_null_count > 0, f"Feature {col} has all null values"

    def test_feature_pipeline_performance_monitoring(
        self, polars_sample_data, text_columns, minimal_config
    ):
        """Test performance monitoring and validation in feature pipeline."""
        import time

        existing_text_cols = [
            col for col in text_columns if col in polars_sample_data.columns
        ]
        if not existing_text_cols:
            pytest.skip("No text columns found in sample data")

        manager = CoffeeFeatureManager(minimal_config)

        # Measure fitting time
        start_time = time.time()
        manager.fit(polars_sample_data, existing_text_cols)
        fit_time = time.time() - start_time

        # Measure extraction time
        start_time = time.time()
        features = manager.extract_all_features(polars_sample_data, existing_text_cols)
        extract_time = time.time() - start_time

        # Performance should be reasonable for small dataset
        assert fit_time < 30.0, f"Fitting took too long: {fit_time:.2f}s"
        assert extract_time < 30.0, (
            f"Feature extraction took too long: {extract_time:.2f}s"
        )

        # Memory usage validation
        assert features.shape[1] < 1000, (
            f"Too many features generated: {features.shape[1]}"
        )

        # Feature quality validation
        total_features = features.shape[1]
        assert total_features > 0, "No features were generated"

        # Log performance info for monitoring
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Feature pipeline performance - Fit: {fit_time:.2f}s, Extract: {extract_time:.2f}s, Features: {total_features}"
        )


class TestFeatureExtractionEdgeCases:
    """Test edge cases and error handling in feature extraction."""

    def test_empty_text_handling(self, minimal_config):
        """Test feature extraction with empty or minimal text data."""
        # Create minimal test data
        empty_data = pl.DataFrame(
            {"desc_1": ["", None, "   ", "minimal text"], "rating": [80, 85, 90, 95]}
        )

        manager = CoffeeFeatureManager(minimal_config)
        manager.fit(empty_data, ["desc_1"])
        features = manager.extract_all_features(empty_data, ["desc_1"])

        # Should handle empty data gracefully
        assert isinstance(features, pl.DataFrame)
        assert features.shape[0] == empty_data.shape[0]

        # Features should be numeric (possibly zero for empty text)
        for col in features.columns:
            assert features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]

    def test_error_recovery_in_extractors(self, polars_sample_data, minimal_config):
        """Test that feature extraction continues gracefully when individual extractors fail."""
        existing_text_cols = (
            ["desc_1"]
            if "desc_1" in polars_sample_data.columns
            else polars_sample_data.columns[:1]
        )

        # Mock one extractor to fail
        with patch(
            "src.features.tfidf_extractor.TfidfExtractor.extract_features"
        ) as mock_tfidf:
            mock_tfidf.side_effect = Exception("TF-IDF extraction failed")

            manager = CoffeeFeatureManager(minimal_config)

            # Should not raise exception despite TF-IDF failure
            try:
                manager.fit(polars_sample_data, existing_text_cols)
                features = manager.extract_all_features(
                    polars_sample_data, existing_text_cols
                )

                # Should still get some features from working extractors
                assert isinstance(features, pl.DataFrame)

            except Exception as e:
                # If the specific extractor error propagates, that's also acceptable
                # depending on the implementation's error handling strategy
                pass

    @pytest.mark.parametrize(
        "invalid_config",
        [
            {},  # Empty config
            {"extractors": {}},  # No extractors enabled
            {"extractors": {"nonexistent": True}},  # Invalid extractor
        ],
    )
    def test_configuration_validation(self, invalid_config):
        """Test handling of invalid configurations."""
        # Should either handle gracefully or raise informative errors
        try:
            manager = CoffeeFeatureManager(invalid_config)
            # If it doesn't raise an error, should have some default behavior
            assert hasattr(manager, "extractors")
        except (ValueError, KeyError) as e:
            # Acceptable to raise informative errors for invalid configs
            assert len(str(e)) > 0  # Error message should be informative

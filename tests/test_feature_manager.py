"""
Tests for the CoffeeFeatureManager class.

This module tests the unified feature extraction manager and its integration
with all feature extractors following thesis methodology.
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import tempfile
import shutil

from src.features.feature_manager import CoffeeFeatureManager, GloVeExtractor
from src.features.base import ExtractorError


class TestCoffeeFeatureManager:
    """Test suite for CoffeeFeatureManager class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for feature manager."""
        return {
            "extractors": {
                "tfidf": True,
                "bert": True,
                "topics": True,
                "sentiment": True,
                "glove": False,  # Disable to avoid dependency issues in tests
            },
            "tfidf": {
                "max_features": 100,
                "ngram_range": (1, 2),
            },
            "bert": {
                "model_name": "distilbert-base-uncased",
                "max_length": 128,
            },
            "topics": {
                "n_topics": 10,
                "method": "lda",
            },
            "sentiment": {
                "model_type": "vader",
            },
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "desc_1": [
                    "bright acidic coffee with citrus notes",
                    "smooth dark roast with chocolate undertones",
                    "light roast with floral and fruity flavors",
                    "medium roast balanced coffee",
                    "complex coffee with wine-like characteristics",
                ],
                "desc_2": [
                    "grown in ethiopian highlands",
                    "processed using washed method",
                    "single origin from colombia",
                    "blend of central american beans",
                    "naturally processed brazilian coffee",
                ],
                "desc_3": [
                    "excellent morning coffee",
                    "perfect for espresso",
                    "great for pour over brewing",
                    "versatile all-day coffee",
                    "unique and memorable cup",
                ],
                "roaster": [
                    "Blue Bottle",
                    "Stumptown",
                    "Counter Culture",
                    "Intelligentsia",
                    "Blue Bottle",
                ],
                "country_of_origin": [
                    "Ethiopia",
                    "Colombia",
                    "Guatemala",
                    "Brazil",
                    "Ethiopia",
                ],
                "roast": ["Light", "Dark", "Light", "Medium", "Medium"],
                "cupper_points": [88.5, 86.2, 89.1, 87.3, 85.8],
            }
        )

    @pytest.fixture
    def feature_manager(self, sample_config):
        """Create a CoffeeFeatureManager instance."""
        return CoffeeFeatureManager(sample_config)

    def test_feature_manager_initialization(self, feature_manager, sample_config):
        """Test that feature manager initializes correctly."""
        assert feature_manager.config == sample_config
        assert hasattr(feature_manager, "extractors")
        assert hasattr(feature_manager, "categorical_encoder")
        assert not feature_manager.is_fitted

        # Check that enabled extractors are initialized
        expected_extractors = ["tfidf", "bert", "topics", "sentiment"]
        for extractor_name in expected_extractors:
            assert extractor_name in feature_manager.extractors

        # GloVe should not be initialized (disabled in config)
        assert "glove" not in feature_manager.extractors

    @pytest.mark.unit
    def test_feature_manager_fit(self, feature_manager, sample_data):
        """Test fitting the feature manager on data."""
        text_columns = ["desc_1", "desc_2", "desc_3"]

        # Use patch context manager to mock the fit methods at class level
        with (
            patch.multiple(
                "src.features.tfidf_extractor.TfidfExtractor",
                fit=Mock(return_value=None),
            ),
            patch.multiple(
                "src.features.bert_extractor.BertExtractor", fit=Mock(return_value=None)
            ),
            patch.multiple(
                "src.features.topic_extractor.TopicExtractor",
                fit=Mock(return_value=None),
            ),
            patch.multiple(
                "src.features.sentiment_extractor.SentimentExtractor",
                fit=Mock(return_value=None),
            ),
        ):
            # Mock categorical encoder
            mock_cat_fit = Mock(return_value=feature_manager.categorical_encoder)
            feature_manager.categorical_encoder.fit = mock_cat_fit

            # Fit the manager
            feature_manager.fit(sample_data, text_columns)

            # Verify it's fitted
            assert feature_manager.is_fitted

            # Verify categorical encoder was fitted
            feature_manager.categorical_encoder.fit.assert_called_once()

    @pytest.mark.unit
    def test_create_extractor(self, feature_manager):
        """Test creating different types of extractors."""
        # Test TF-IDF extractor creation
        tfidf_config = {"max_features": 100}
        tfidf_extractor = feature_manager._create_extractor("tfidf", tfidf_config)
        assert tfidf_extractor is not None
        assert hasattr(tfidf_extractor, "extract_features")

        # Test BERT extractor creation
        bert_config = {"model_name": "distilbert-base-uncased"}
        bert_extractor = feature_manager._create_extractor("bert", bert_config)
        assert bert_extractor is not None
        assert hasattr(bert_extractor, "extract_features")

        # Test unknown extractor
        with pytest.raises(ValueError, match="Unknown extractor type"):
            feature_manager._create_extractor("unknown", {})

    @pytest.mark.unit
    def test_extract_features_not_fitted(self, feature_manager):
        """Test that extract_features raises error when not fitted."""
        texts = ["sample text"]

        with pytest.raises(ExtractorError, match="must be fitted"):
            feature_manager.extract_features(texts)

    @pytest.mark.unit
    def test_extract_features_basic(self, feature_manager, sample_data):
        """Test basic feature extraction functionality."""
        # Mock fitted state
        feature_manager.is_fitted = True

        # Create unique mock features for each extractor to avoid column name conflicts
        extractors_list = list(feature_manager.extractors.items())

        for i, (name, extractor) in enumerate(extractors_list):
            # Create unique feature names for each extractor
            mock_features = pl.DataFrame(
                {
                    f"{name}_feature_1": [1.0, 2.0, 3.0],
                    f"{name}_feature_2": [0.5, 1.5, 2.5],
                }
            )
            extractor.extract_features = Mock(return_value=mock_features)

        texts = ["text1", "text2", "text3"]
        result = feature_manager.extract_features(texts)

        # Should combine features from all extractors
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3  # Same as input texts

        # Should have features from all extractors
        expected_columns = len(extractors_list) * 2  # 2 features per extractor
        assert result.shape[1] == expected_columns

        # Verify all extractors were called
        for extractor in feature_manager.extractors.values():
            extractor.extract_features.assert_called_once_with(texts)

    @pytest.mark.unit
    def test_extract_features_empty_result(self, feature_manager):
        """Test handling when extractors return empty results."""
        feature_manager.is_fitted = True

        # Mock extractors to return empty DataFrames
        for extractor in feature_manager.extractors.values():
            extractor.extract_features = Mock(return_value=pl.DataFrame())

        texts = ["text1", "text2"]
        result = feature_manager.extract_features(texts)

        # Should return empty DataFrame when no features extracted
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    @pytest.mark.integration
    def test_extract_all_features_integration(self, feature_manager, sample_data):
        """Test the main extract_all_features method with thesis methodology."""
        # Mock fitted state and methods
        feature_manager.is_fitted = True
        feature_manager.categorical_encoder.is_fitted = True

        # Mock extract_features_for_column to return mock features
        def mock_extract_for_column(texts, col_name):
            n_texts = len(texts)
            return pl.DataFrame(
                {
                    f"{col_name}_tfidf_0": [1.0] * n_texts,
                    f"{col_name}_bert_0": [0.5] * n_texts,
                    f"{col_name}_sentiment_positive": [0.8] * n_texts,
                }
            )

        feature_manager.extract_features_for_column = Mock(
            side_effect=mock_extract_for_column
        )

        # Mock categorical encoder transform
        def mock_categorical_transform(df_pandas):
            # Add mock categorical features
            result = df_pandas.copy()
            result["roast_Light"] = [1, 0, 1, 0, 0]
            result["roast_Dark"] = [0, 1, 0, 0, 0]
            result["roast_Medium"] = [0, 0, 0, 1, 1]
            result["country_Ethiopia"] = [1, 0, 0, 0, 1]
            result["country_Colombia"] = [0, 1, 0, 0, 0]
            result["roaster_Blue_Bottle"] = [1, 0, 0, 0, 1]
            return result

        feature_manager.categorical_encoder.transform = Mock(
            side_effect=mock_categorical_transform
        )
        feature_manager.categorical_encoder.get_feature_names = Mock(
            return_value={
                "roast": ["roast_Light", "roast_Dark", "roast_Medium"],
                "country_of_origin": ["country_Ethiopia", "country_Colombia"],
                "roaster": ["roaster_Blue_Bottle"],
            }
        )

        # Extract features
        text_columns = ["desc_1", "desc_2", "desc_3"]
        result = feature_manager.extract_all_features(sample_data, text_columns)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_data)

        # Should have original columns plus extracted features plus categorical
        original_cols = set(sample_data.columns)
        result_cols = set(result.columns)
        new_cols = result_cols - original_cols

        # Should have features from each text column plus categorical
        assert len(new_cols) > 0

        # Verify extract_features_for_column was called for each text column
        assert feature_manager.extract_features_for_column.call_count == len(
            text_columns
        )

    @pytest.mark.methodology
    def test_thesis_compliance_separate_processing(self, feature_manager, sample_data):
        """Test that each desc column is processed separately per thesis methodology."""
        feature_manager.is_fitted = True
        feature_manager.categorical_encoder.is_fitted = True

        # Track calls to extract_features_for_column
        call_log = []

        def track_extract_calls(texts, col_name):
            call_log.append((col_name, len(texts)))
            return pl.DataFrame({f"{col_name}_feature": [1.0] * len(texts)})

        feature_manager.extract_features_for_column = Mock(
            side_effect=track_extract_calls
        )
        feature_manager.categorical_encoder.transform = Mock(
            return_value=sample_data.to_pandas()
        )
        feature_manager.categorical_encoder.get_feature_names = Mock(return_value={})

        # Extract features
        text_columns = ["desc_1", "desc_2", "desc_3"]
        feature_manager.extract_all_features(sample_data, text_columns)

        # Verify each column was processed separately
        assert len(call_log) == 3
        expected_calls = [("desc_1", 5), ("desc_2", 5), ("desc_3", 5)]
        assert call_log == expected_calls

    @pytest.mark.unit
    def test_get_feature_names(self, feature_manager):
        """Test getting feature names from all extractors."""
        feature_manager.is_fitted = True

        # Mock extractors to return feature names and set is_fitted = True
        for i, (name, extractor) in enumerate(feature_manager.extractors.items()):
            extractor.is_fitted = True  # This is crucial - the method checks this
            mock_get_names = Mock(
                return_value=[f"{name}_feature_{j}" for j in range(3)]
            )
            extractor.get_feature_names = mock_get_names

        result = feature_manager.get_feature_names()

        assert isinstance(result, dict)
        for extractor_name in feature_manager.extractors:
            assert extractor_name in result
            assert len(result[extractor_name]) == 3

    @pytest.mark.unit
    def test_get_feature_counts(self, feature_manager):
        """Test getting feature counts from all extractors."""
        feature_manager.is_fitted = True

        # Mock extractors to return feature counts and set is_fitted = True
        expected_counts = {"tfidf": 100, "bert": 768, "topics": 10, "sentiment": 18}
        for name, extractor in feature_manager.extractors.items():
            extractor.is_fitted = True  # This is crucial - the method checks this
            mock_get_count = Mock(return_value=expected_counts.get(name, 50))
            extractor.get_feature_count = mock_get_count

        result = feature_manager.get_feature_counts()

        assert isinstance(result, dict)
        for extractor_name in feature_manager.extractors:
            assert extractor_name in result
            assert isinstance(result[extractor_name], int)

    @pytest.mark.unit
    def test_get_total_feature_count(self, feature_manager):
        """Test getting total feature count."""
        # Mock get_feature_counts
        mock_get_counts = Mock(
            return_value={"tfidf": 100, "bert": 768, "topics": 10, "sentiment": 18}
        )
        feature_manager.get_feature_counts = mock_get_counts

        total = feature_manager.get_total_feature_count()
        assert total == 896  # 100 + 768 + 10 + 18

    @pytest.mark.unit
    def test_get_extractor_info(self, feature_manager):
        """Test getting comprehensive extractor information."""
        feature_manager.is_fitted = True

        # Mock extractors with complete interface
        for name, extractor in feature_manager.extractors.items():
            mock_get_count = Mock(return_value=50)
            extractor.get_feature_count = mock_get_count
            extractor.config = {"test_param": "test_value"}

            # Mock get_model_info method that's called in get_extractor_info
            mock_model_info = Mock(
                return_value={
                    "model_name": f"test_{name}_model",
                    "is_fitted": True,
                    "additional_info": "test_value",
                }
            )
            extractor.get_model_info = mock_model_info

        result = feature_manager.get_extractor_info()

        assert isinstance(result, dict)
        for extractor_name in feature_manager.extractors:
            assert extractor_name in result
            assert "feature_count" in result[extractor_name]
            assert "config" in result[extractor_name]

    @pytest.mark.integration
    def test_save_and_load_extractors(self, feature_manager):
        """Test saving and loading extractor models."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = temp_dir

            # Mock fitted extractors
            feature_manager.is_fitted = True
            for extractor in feature_manager.extractors.values():
                extractor.is_fitted = True  # Set fitted state
                mock_save = Mock()
                extractor._save_vectorizer = (
                    mock_save  # This is the actual method called
                )

            # Test saving
            feature_manager.save_extractors(models_dir)

            # Verify _save_vectorizer was called on all extractors
            for extractor in feature_manager.extractors.values():
                extractor._save_vectorizer.assert_called_once()

    @pytest.mark.unit
    def test_specialized_preprocessing_integration(self, feature_manager):
        """Test that specialized preprocessing is applied correctly."""

        # Mock the _apply_specialized_preprocessing method
        def mock_specialized_preprocessing(texts, extractor_name):
            return [f"processed_for_{extractor_name}: {text}" for text in texts]

        mock_preprocess = Mock(side_effect=mock_specialized_preprocessing)
        feature_manager._apply_specialized_preprocessing = mock_preprocess
        feature_manager.is_fitted = True

        # Mock an extractor
        mock_extractor = Mock()
        mock_extractor.extract_features = Mock(
            return_value=pl.DataFrame({"feature": [1, 2, 3]})
        )
        feature_manager.extractors = {"test_extractor": mock_extractor}

        # Test extract_features_for_column which should use specialized preprocessing
        texts = ["text1", "text2", "text3"]
        feature_manager.extract_features_for_column(texts, "desc_1")

        # Verify specialized preprocessing was applied
        feature_manager._apply_specialized_preprocessing.assert_called()

    @pytest.mark.performance
    def test_feature_extraction_performance(self, feature_manager):
        """Test that feature extraction completes in reasonable time."""
        import time

        feature_manager.is_fitted = True

        # Create unique mock features for each extractor to avoid column name conflicts
        extractors_list = list(feature_manager.extractors.items())

        for i, (name, extractor) in enumerate(extractors_list):
            # Create unique feature names for each extractor
            fast_features = pl.DataFrame({f"{name}_feature": list(range(1000))})
            extractor.extract_features = Mock(return_value=fast_features)

        texts = ["sample text"] * 1000

        start_time = time.time()
        result = feature_manager.extract_features(texts)
        execution_time = time.time() - start_time

        # Should complete quickly with mocked extractors
        assert execution_time < 1.0
        assert len(result) == 1000

    @pytest.mark.error_handling
    def test_extractor_failure_handling(self, feature_manager):
        """Test that feature manager handles individual extractor failures gracefully."""
        feature_manager.is_fitted = True

        # Get extractors as list to ensure consistent ordering
        extractors_list = list(feature_manager.extractors.items())

        # Mock first extractor to fail
        failing_name, failing_extractor = extractors_list[0]
        failing_extractor.extract_features = Mock(
            side_effect=Exception("Extractor failed")
        )

        # Mock remaining extractors to succeed with unique column names
        for i, (name, extractor) in enumerate(extractors_list[1:], 1):
            success_features = pl.DataFrame({f"{name}_feature": [1, 2, 3]})
            extractor.extract_features = Mock(return_value=success_features)

        texts = ["text1", "text2", "text3"]
        result = feature_manager.extract_features(texts)

        # Should still return features from working extractors
        assert isinstance(result, pl.DataFrame)
        # Should have features from working extractors only
        assert not result.is_empty()

        # Should have features from successful extractors (excluding the failing one)
        expected_columns = len(extractors_list) - 1  # All except the failing one
        assert result.shape[1] == expected_columns


class TestGloVeExtractor:
    """Test suite for GloVeExtractor class."""

    @pytest.fixture
    def glove_config(self):
        """Create GloVe extractor configuration."""
        return {
            "model_name": "glove-wiki-gigaword-50",  # Smaller model for testing
            "vector_dimension": 50,
            "aggregation": "mean",
        }

    @pytest.mark.unit
    @patch("src.features.feature_manager.GENSIM_AVAILABLE", False)
    def test_glove_extractor_without_gensim(self, glove_config):
        """Test GloVe extractor when gensim is not available."""
        extractor = GloVeExtractor(glove_config)

        assert extractor.glove_model_ is None

        # Should return empty DataFrame when gensim not available
        texts = ["sample text"]
        result = extractor.extract_features(texts)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    @pytest.mark.unit
    @patch("src.features.feature_manager.GENSIM_AVAILABLE", True)
    def test_glove_extractor_with_mocked_model(self, glove_config):
        """Test GloVe extractor with mocked gensim model."""
        with patch("src.features.feature_manager.api") as mock_api:
            # Mock the GloVe model
            mock_model = Mock()
            mock_model.__contains__ = Mock(
                side_effect=lambda word: word in ["coffee", "good"]
            )
            mock_model.__getitem__ = Mock(return_value=np.random.rand(50))
            mock_api.load.return_value = mock_model

            extractor = GloVeExtractor(glove_config)
            extractor.fit(["sample text"])

            # Test feature extraction
            texts = ["coffee is good", "unknown words here"]
            result = extractor.extract_features(texts)

            assert isinstance(result, pl.DataFrame)
            assert len(result) == 2
            assert result.shape[1] == 50  # 50-dimensional vectors

    @pytest.mark.unit
    def test_glove_extractor_feature_names(self, glove_config):
        """Test GloVe extractor feature names."""
        extractor = GloVeExtractor(glove_config)

        feature_names = extractor.get_feature_names()
        assert len(feature_names) == 50
        assert all(name.startswith("glove_") for name in feature_names)

        feature_count = extractor.get_feature_count()
        assert feature_count == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unified feature extraction manager for coffee review text analysis.

This module orchestrates all feature extractors and provides a single interface
for extracting comprehensive features following the thesis methodology.
"""

import polars as pl
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from .base import BaseExtractor, ExtractorError
from .tfidf_extractor import TfidfExtractor
from .bert_extractor import BertExtractor
from .topic_extractor import TopicExtractor
from .sentiment_extractor import SentimentExtractor

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import gensim.downloader as api

    GENSIM_AVAILABLE = True
    logger.info("Gensim available - GloVe embeddings enabled")
except ImportError:
    logger.warning("Gensim not installed. GloVe embeddings will be limited.")
    GENSIM_AVAILABLE = False


class GloVeExtractor(BaseExtractor):
    """
    GloVe embeddings extractor for word-level semantics.

    From thesis: "GloVe embeddings (300-dimensional) using pre-trained vectors"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GloVe extractor."""
        super().__init__(config)

        default_config = {
            "model_name": "glove-wiki-gigaword-300",
            "vector_dimension": 300,
            "aggregation": "mean",  # 'mean', 'sum', 'max'
        }
        default_config.update(self.config)
        self.config = default_config

        self.glove_model_ = None
        self.vector_dimension = self.config["vector_dimension"]
        self.feature_names_ = [f"glove_{i}" for i in range(self.vector_dimension)]

        if GENSIM_AVAILABLE:
            self._load_glove_model()

    def _load_glove_model(self) -> None:
        """Load GloVe model."""
        try:
            logger.info(f"Loading GloVe model: {self.config['model_name']}")
            self.glove_model_ = api.load(self.config["model_name"])
            logger.info("GloVe model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GloVe model: {e}")
            self.glove_model_ = None

    def fit(self, texts: List[str]) -> "GloVeExtractor":
        """Fit GloVe extractor (pre-trained model)."""
        self.is_fitted = True
        return self

    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """Extract GloVe embeddings from texts."""
        if not GENSIM_AVAILABLE or self.glove_model_ is None:
            logger.warning("GloVe not available, returning empty DataFrame")
            return pl.DataFrame()

        if not texts:
            return pl.DataFrame()

        try:
            embeddings = []
            for text in texts:
                embedding = self._get_text_embedding(text)
                embeddings.append(embedding)

            # Convert to Polars DataFrame
            embeddings_array = np.array(embeddings)
            feature_data = {
                f"glove_{i}": embeddings_array[:, i]
                for i in range(embeddings_array.shape[1])
            }

            return pl.DataFrame(feature_data)

        except Exception as e:
            logger.error(f"Error extracting GloVe features: {e}")
            return pl.DataFrame()

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get GloVe embedding for a text."""
        words = text.lower().split()
        word_embeddings = []

        for word in words:
            if word in self.glove_model_:
                word_embeddings.append(self.glove_model_[word])

        if not word_embeddings:
            return np.zeros(self.vector_dimension)

        word_embeddings = np.array(word_embeddings)

        if self.config["aggregation"] == "mean":
            return np.mean(word_embeddings, axis=0)
        elif self.config["aggregation"] == "sum":
            return np.sum(word_embeddings, axis=0)
        elif self.config["aggregation"] == "max":
            return np.max(word_embeddings, axis=0)
        else:
            return np.mean(word_embeddings, axis=0)

    def get_feature_names(self) -> List[str]:
        """Get GloVe feature names."""
        return self.feature_names_.copy()

    def get_feature_count(self) -> int:
        """Get number of GloVe features."""
        return self.vector_dimension


class CoffeeFeatureManager:
    """
    Unified feature extraction manager for coffee reviews.

    This manager orchestrates all feature extractors and provides a single
    interface for comprehensive feature extraction following thesis methodology.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature manager.

        Args:
            config: Configuration dictionary with extractor settings
        """
        self.config = config or {}

        # Initialize extractors
        self.extractors = {}
        self.is_fitted = False

        # Default extractor configurations
        default_extractors = {
            "tfidf": True,
            "bert": True,
            "glove": True,
            "topics": True,
            "sentiment": True,
        }

        extractor_config = self.config.get("extractors", default_extractors)

        # Initialize enabled extractors
        if extractor_config.get("tfidf", True):
            tfidf_config = self.config.get("tfidf", {})
            self.extractors["tfidf"] = TfidfExtractor(tfidf_config)

        if extractor_config.get("bert", True):
            bert_config = self.config.get("bert", {})
            self.extractors["bert"] = BertExtractor(bert_config)

        if extractor_config.get("glove", True):
            glove_config = self.config.get("glove", {})
            self.extractors["glove"] = GloVeExtractor(glove_config)

        if extractor_config.get("topics", True):
            topics_config = self.config.get("topics", {})
            self.extractors["topics"] = TopicExtractor(topics_config)

        if extractor_config.get("sentiment", True):
            sentiment_config = self.config.get("sentiment", {})
            self.extractors["sentiment"] = SentimentExtractor(sentiment_config)

        logger.info(
            f"Initialized feature manager with extractors: {list(self.extractors.keys())}"
        )

    def fit(self, texts: List[str]) -> "CoffeeFeatureManager":
        """
        Fit all extractors to the training texts.

        Args:
            texts: List of training texts

        Returns:
            Self for method chaining
        """
        logger.info("Fitting all feature extractors")

        for name, extractor in self.extractors.items():
            try:
                logger.info(f"Fitting {name} extractor")
                extractor.fit(texts)
                logger.info(f"{name} extractor fitted successfully")
            except Exception as e:
                logger.error(f"Failed to fit {name} extractor: {e}")
                # Continue with other extractors

        self.is_fitted = True
        logger.info("All feature extractors fitted")
        return self

    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract all features from texts.

        Args:
            texts: List of texts to process

        Returns:
            Polars DataFrame with all extracted features
        """
        if not self.is_fitted:
            raise ExtractorError("Feature manager must be fitted before extraction")

        logger.info(f"Extracting features for {len(texts)} texts")

        # Extract features from each extractor
        feature_dfs = []

        for name, extractor in self.extractors.items():
            try:
                logger.info(f"Extracting {name} features")
                features_df = extractor.extract_features(texts)

                if not features_df.is_empty():
                    feature_dfs.append(features_df)
                    logger.info(f"{name} features extracted: {features_df.shape}")
                else:
                    logger.warning(f"{name} extractor returned empty DataFrame")

            except Exception as e:
                logger.error(f"Failed to extract {name} features: {e}")
                # Continue with other extractors

        # Combine all features
        if not feature_dfs:
            logger.warning("No features extracted from any extractor")
            return pl.DataFrame()

        # Concatenate horizontally
        combined_df = feature_dfs[0]
        for df in feature_dfs[1:]:
            combined_df = combined_df.hstack(df)

        logger.info(f"Combined features shape: {combined_df.shape}")
        return combined_df

    def extract_all_features(
        self, df: pl.DataFrame, text_columns: List[str] = ["desc_1", "desc_2", "desc_3"]
    ) -> pl.DataFrame:
        """
        Extract features from multiple text columns and combine with original data.

        Args:
            df: Input DataFrame with text columns
            text_columns: List of text column names to process

        Returns:
            DataFrame with original data plus extracted features
        """
        logger.info(f"Extracting features from columns: {text_columns}")

        # Combine text columns
        combined_texts = []
        for i in range(len(df)):
            text_parts = []
            for col in text_columns:
                if col in df.columns:
                    text_value = df[col][i]
                    if text_value and isinstance(text_value, str):
                        text_parts.append(text_value.strip())

            combined_text = " ".join(text_parts) if text_parts else ""
            combined_texts.append(combined_text)

        # Extract features
        features_df = self.extract_features(combined_texts)

        # Combine with original data
        if not features_df.is_empty():
            result_df = df.hstack(features_df)
        else:
            result_df = df

        logger.info(f"Final DataFrame shape: {result_df.shape}")
        return result_df

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get feature names from all extractors.

        Returns:
            Dictionary mapping extractor names to their feature names
        """
        feature_names = {}
        for name, extractor in self.extractors.items():
            if extractor.is_fitted:
                feature_names[name] = extractor.get_feature_names()
        return feature_names

    def get_feature_counts(self) -> Dict[str, int]:
        """
        Get feature counts from all extractors.

        Returns:
            Dictionary mapping extractor names to their feature counts
        """
        feature_counts = {}
        for name, extractor in self.extractors.items():
            if extractor.is_fitted:
                # Try to get feature count, fallback to feature names length
                if hasattr(extractor, "get_feature_count"):
                    feature_counts[name] = extractor.get_feature_count()
                elif hasattr(extractor, "feature_names_"):
                    feature_counts[name] = len(extractor.feature_names_)
                else:
                    feature_counts[name] = 0
        return feature_counts

    def get_total_feature_count(self) -> int:
        """Get total number of features across all extractors."""
        return sum(self.get_feature_counts().values())

    def get_extractor_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all extractors.

        Returns:
            Dictionary with extractor information
        """
        info = {}
        for name, extractor in self.extractors.items():
            # Get feature count safely
            feature_count = 0
            if extractor.is_fitted:
                if hasattr(extractor, "get_feature_count"):
                    feature_count = extractor.get_feature_count()
                elif hasattr(extractor, "feature_names_"):
                    feature_count = len(extractor.feature_names_)

            extractor_info = {
                "is_fitted": extractor.is_fitted,
                "feature_count": feature_count,
                "config": extractor.get_config()
                if hasattr(extractor, "get_config")
                else extractor.config,
            }

            # Add specific info for certain extractors
            if hasattr(extractor, "get_model_info"):
                extractor_info.update(extractor.get_model_info())

            info[name] = extractor_info

        return info

    def save_extractors(self, models_dir: str = "models") -> None:
        """
        Save all fitted extractors.

        Args:
            models_dir: Directory to save models
        """
        logger.info(f"Saving extractors to {models_dir}")

        for name, extractor in self.extractors.items():
            if extractor.is_fitted and hasattr(extractor, "_save_vectorizer"):
                try:
                    # Update models directory in config
                    extractor.config["models_dir"] = models_dir
                    if hasattr(extractor, "_save_vectorizer"):
                        extractor._save_vectorizer()
                    elif hasattr(extractor, "_save_models"):
                        extractor._save_models()
                    logger.info(f"Saved {name} extractor")
                except Exception as e:
                    logger.warning(f"Failed to save {name} extractor: {e}")

    def load_extractors(self, models_dir: str = "models") -> "CoffeeFeatureManager":
        """
        Load previously fitted extractors.

        Args:
            models_dir: Directory containing saved models

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading extractors from {models_dir}")

        for name, extractor in self.extractors.items():
            try:
                if hasattr(extractor, "load_vectorizer"):
                    extractor.load_vectorizer()
                elif hasattr(extractor, "load_models"):
                    extractor.load_models(models_dir)
                logger.info(f"Loaded {name} extractor")
            except Exception as e:
                logger.warning(f"Failed to load {name} extractor: {e}")

        self.is_fitted = True
        return self

    def print_summary(self) -> None:
        """Print a summary of the feature manager."""
        print("\n=== Coffee Feature Manager Summary ===")
        print(f"Fitted: {self.is_fitted}")
        print(f"Active extractors: {list(self.extractors.keys())}")

        if self.is_fitted:
            feature_counts = self.get_feature_counts()
            total_features = sum(feature_counts.values())

            print(f"Total features: {total_features}")
            print("\nFeature breakdown:")
            for name, count in feature_counts.items():
                percentage = (count / total_features * 100) if total_features > 0 else 0
                print(f"  {name}: {count} features ({percentage:.1f}%)")

        print("=" * 40)

"""
Unified feature extraction manager for coffee review text analysis.

This module orchestrates all feature extractors and provides a single interface
for extracting comprehensive features following the thesis methodology.
"""

import polars as pl
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from .base import BaseExtractor, ExtractorError
from .tfidf_extractor import TfidfExtractor
from .bert_extractor import BertExtractor
from .topic_extractor import TopicExtractor
from .sentiment_extractor import SentimentExtractor

# Import specialized preprocessing functions
from data.preprocessing import (
    preprocess_text_for_embeddings,
    preprocess_text_for_topics,
)
from .categorical_encoder import CategoricalFeatureEncoder

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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Coffee Feature Manager.

        Args:
            config: Configuration dictionary containing parameters for each extractor
        """
        self.config = config
        self.extractors = {}
        self.is_fitted = False

        # Initialize categorical encoder with thesis-compliant configuration
        self.categorical_encoder = CategoricalFeatureEncoder()

        logger.info("Initialized CoffeeFeatureManager")
        logger.info(f"Configuration: {config}")
        logger.info("Categorical encoder initialized with thesis methodology")

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

    def fit(self, df: pl.DataFrame, text_columns: List[str]) -> None:
        """
        Fit all feature extractors on the dataset including categorical encoder.

        Args:
            df: Input DataFrame
            text_columns: List of text column names to process
        """
        logger.info("Fitting CoffeeFeatureManager on dataset")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Text columns: {text_columns}")

        # Convert polars to pandas for sklearn compatibility
        df_pandas = df.to_pandas()

        # Combine all text for fitting extractors
        all_texts = []
        for col_name in text_columns:
            if col_name in df.columns:
                column_texts = df[col_name].to_list()
                all_texts.extend([str(text) if text else "" for text in column_texts])

        logger.info(f"Total texts for fitting: {len(all_texts)}")

        # Fit each configured extractor
        extractors_config = self.config.get("extractors", {})
        for extractor_name in ["tfidf", "bert", "topics", "sentiment", "glove"]:
            # Only process if extractor is enabled
            if extractors_config.get(extractor_name, False):
                logger.info(f"Fitting {extractor_name} extractor")
                extractor_specific_config = self.config.get(extractor_name, {})
                extractor = self._create_extractor(
                    extractor_name, extractor_specific_config
                )

                # Skip if extractor creation failed (e.g., GloVe not available)
                if extractor is None:
                    logger.warning(
                        f"Skipping {extractor_name} extractor - not available"
                    )
                    continue

                extractor.fit(all_texts)
                self.extractors[extractor_name] = extractor
                logger.info(f"✅ {extractor_name} extractor fitted")

        # Fit categorical encoder
        logger.info("Fitting categorical encoder")
        self.categorical_encoder.fit(df_pandas)
        logger.info("✅ Categorical encoder fitted")

        self.is_fitted = True
        logger.info("✅ All extractors fitted successfully")

    def _create_extractor(self, extractor_name: str, extractor_config: Dict[str, Any]):
        """
        Create an extractor instance based on name and configuration.

        Args:
            extractor_name: Name of the extractor (tfidf, bert, topics, sentiment, glove)
            extractor_config: Configuration dictionary for the extractor

        Returns:
            Initialized extractor instance
        """
        if extractor_name == "tfidf":
            return TfidfExtractor(extractor_config)
        elif extractor_name == "bert":
            return BertExtractor(extractor_config)
        elif extractor_name == "topics":
            return TopicExtractor(extractor_config)
        elif extractor_name == "sentiment":
            return SentimentExtractor(extractor_config)
        elif extractor_name == "glove":
            # Handle GloVe extractor if available
            try:
                from .glove_extractor import GloVeExtractor

                return GloVeExtractor(extractor_config)
            except ImportError:
                logger.warning("GloVe extractor not available - skipping")
                return None
        else:
            raise ValueError(f"Unknown extractor type: {extractor_name}")

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
        Extract features from multiple text columns separately plus categorical features.

        Following thesis methodology, each description column is processed separately
        to capture distinct semantic content:
        - desc_1: Tasting notes (flavor descriptors)
        - desc_2: Contextual information (brewing, origin details)
        - desc_3: Reviewer's conclusion ("bottom line")

        Additionally includes categorical features:
        - roast: One-hot encoded roast levels
        - country_of_origin: Top-K countries + Other
        - roaster: Frequency-grouped roasters + Other

        Args:
            df: Input DataFrame with text columns
            text_columns: List of text column names to process separately

        Returns:
            DataFrame with original data plus extracted features from each column
        """
        logger.info(f"Extracting features separately from columns: {text_columns}")
        logger.info("Following thesis methodology: separate processing per desc column")
        logger.info("Including categorical features: roast, country_of_origin, roaster")

        # Start with original data
        result_df = df

        # Process each text column separately
        for col_name in text_columns:
            if col_name not in df.columns:
                logger.warning(f"Column '{col_name}' not found in DataFrame")
                continue

            logger.info(f"Processing text column: {col_name}")

            # Extract texts from this column
            column_texts = []
            for i in range(len(df)):
                text_value = df[col_name][i]
                if text_value and isinstance(text_value, str):
                    column_texts.append(text_value.strip())
                else:
                    column_texts.append("")

            # Extract features for this column
            column_features_df = self.extract_features_for_column(
                column_texts, col_name
            )

            # Combine with result
            if not column_features_df.is_empty():
                result_df = result_df.hstack(column_features_df)
                logger.info(
                    f"Added {column_features_df.shape[1]} features from {col_name}"
                )
            else:
                logger.warning(f"No features extracted from {col_name}")

        # Extract categorical features
        logger.info("Extracting categorical features")
        try:
            # Convert to pandas for categorical encoder
            df_pandas = df.to_pandas()

            # Apply categorical encoding
            df_with_categorical = self.categorical_encoder.transform(df_pandas)

            # Get only the new categorical features (exclude original columns)
            categorical_feature_names = []
            for col_features in self.categorical_encoder.get_feature_names().values():
                categorical_feature_names.extend(col_features)

            # Extract only the categorical feature columns
            categorical_features_df = df_with_categorical[categorical_feature_names]

            # Convert back to polars
            categorical_features_pl = pl.from_pandas(categorical_features_df)

            # Add to result
            result_df = result_df.hstack(categorical_features_pl)

            logger.info(f"Added {len(categorical_feature_names)} categorical features")
            logger.info(f"Categorical features: {categorical_feature_names}")

        except Exception as e:
            logger.error(f"Failed to extract categorical features: {e}")
            logger.warning("Continuing without categorical features")

        logger.info(f"Final DataFrame shape: {result_df.shape}")
        logger.info(f"Total features added: {result_df.shape[1] - df.shape[1]}")
        return result_df

    def extract_features_for_column(
        self, texts: List[str], column_name: str
    ) -> pl.DataFrame:
        """
        Extract features from texts for a specific column with proper naming.

        Following thesis methodology, applies specialized preprocessing:
        - Topic modeling: Retain stopwords, remove punctuation
        - Embeddings/Sentiment: Remove stopwords, retain punctuation
        - TF-IDF: Remove stopwords, retain punctuation

        Args:
            texts: List of texts from the column
            column_name: Name of the source column (e.g., 'desc_1')

        Returns:
            DataFrame with features named according to thesis convention
        """
        logger.info(
            f"Extracting features for column '{column_name}' from {len(texts)} texts"
        )
        logger.info(
            "Following thesis methodology: specialized preprocessing per extractor type"
        )

        feature_dfs = []

        # Extract features using each extractor with specialized preprocessing
        for name, extractor in self.extractors.items():
            try:
                logger.info(f"Extracting {name} features for {column_name}")

                # Apply specialized preprocessing based on extractor type (thesis methodology)
                preprocessed_texts = self._apply_specialized_preprocessing(texts, name)

                # Extract features using preprocessed texts
                features_df = extractor.extract_features(preprocessed_texts)

                if not features_df.is_empty():
                    # Rename columns to include source column (thesis naming convention)
                    renamed_df = self._rename_features_for_column(
                        features_df, name, column_name
                    )
                    feature_dfs.append(renamed_df)
                    logger.info(
                        f"{name} features extracted for {column_name}: {renamed_df.shape}"
                    )
                else:
                    logger.warning(
                        f"{name} extractor returned empty DataFrame for {column_name}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to extract {name} features for {column_name}: {e}"
                )
                # Continue with other extractors

        # Combine all features for this column
        if not feature_dfs:
            logger.warning(f"No features extracted for column {column_name}")
            return pl.DataFrame()

        # Concatenate horizontally
        combined_df = feature_dfs[0]
        for df in feature_dfs[1:]:
            combined_df = combined_df.hstack(df)

        logger.info(f"Combined features for {column_name}: {combined_df.shape}")
        return combined_df

    def _rename_features_for_column(
        self, features_df: pl.DataFrame, extractor_name: str, column_name: str
    ) -> pl.DataFrame:
        """
        Rename feature columns to follow thesis naming convention.

        Converts generic names like 'tfidf_0' to 'tfidf_desc_1_0'
        following the thesis methodology.

        Args:
            features_df: DataFrame with generic feature names
            extractor_name: Name of the extractor (e.g., 'tfidf')
            column_name: Source column name (e.g., 'desc_1')

        Returns:
            DataFrame with renamed columns
        """
        # Create mapping for column renaming
        column_mapping = {}

        for col in features_df.columns:
            # Extract the feature index/identifier from the original name
            if col.startswith(f"{extractor_name}_"):
                # Get the suffix after the extractor name
                suffix = col[len(f"{extractor_name}_") :]
                # Create new name following thesis convention
                new_name = f"{extractor_name}_{column_name}_{suffix}"
                column_mapping[col] = new_name
            else:
                # Handle cases where column doesn't follow expected pattern
                new_name = f"{extractor_name}_{column_name}_{col}"
                column_mapping[col] = new_name

        # Rename columns
        renamed_df = features_df.rename(column_mapping)

        logger.debug(
            f"Renamed {len(column_mapping)} features for {extractor_name}_{column_name}"
        )
        return renamed_df

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

    def _apply_specialized_preprocessing(
        self, texts: List[str], extractor_name: str
    ) -> List[str]:
        """
        Apply specialized preprocessing based on extractor type.

        Following thesis methodology:
        - Topic modeling: Retain stopwords, remove punctuation
        - Embeddings/TF-IDF/Sentiment: Remove stopwords, retain punctuation

        Args:
            texts: List of texts to preprocess
            extractor_name: Name of the extractor (e.g., 'topics', 'tfidf', 'bert', 'sentiment')

        Returns:
            List of preprocessed texts
        """
        logger.debug(f"Applying specialized preprocessing for {extractor_name}")

        preprocessed_texts = []

        # Determine preprocessing type based on extractor
        if extractor_name in ["topics", "topic"]:
            # Topic modeling: retain stopwords, remove punctuation
            for text in texts:
                if isinstance(text, str) and text.strip():
                    preprocessed_texts.append(preprocess_text_for_topics(text))
                else:
                    preprocessed_texts.append("")
            logger.debug(
                f"Applied topic modeling preprocessing (retain stopwords, remove punctuation)"
            )
        else:
            # Embeddings/TF-IDF/Sentiment: remove stopwords, retain punctuation
            for text in texts:
                if isinstance(text, str) and text.strip():
                    preprocessed_texts.append(preprocess_text_for_embeddings(text))
                else:
                    preprocessed_texts.append("")
            logger.debug(
                f"Applied embeddings/TF-IDF/sentiment preprocessing (remove stopwords, retain punctuation)"
            )

        return preprocessed_texts

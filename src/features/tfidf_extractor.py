"""
TF-IDF feature extractor for coffee review text analysis.

This module provides TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction
following the thesis methodology with robust error handling.
"""

import logging
import pickle
import os
from typing import List, Dict, Any, Optional
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseSparseExtractor

# Import centralized exceptions using absolute import
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import (
    TfidfExtractionError,
    ExtractorNotFittedError,
    ExtractorConfigError,
    ModelSaveError,
    ModelLoadError,
    handle_exception,
    validate_not_none,
    validate_not_empty,
    require_dependency,
)

logger = logging.getLogger(__name__)


class TfidfExtractor(BaseSparseExtractor):
    """
    TF-IDF feature extractor with robust error handling.

    Extracts TF-IDF features from text documents using scikit-learn's TfidfVectorizer
    with comprehensive error handling and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TF-IDF extractor with configuration.

        Args:
            config: Configuration dictionary with TF-IDF parameters
        """
        # Set default configuration
        default_config = {
            "max_features": 5000,
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english",
            "lowercase": True,
            "strip_accents": "unicode",
            "analyzer": "word",
            "token_pattern": r"(?u)\b\w\w+\b",
            "models_dir": "models",
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)
        self.vectorizer_ = None

        # Validate dependencies
        try:
            require_dependency("sklearn.feature_extraction.text", "TfidfVectorizer")
        except Exception as e:
            handle_exception(
                e,
                context={"extractor": "TfidfExtractor"},
                reraise_as=ExtractorConfigError,
                message="Required dependency scikit-learn not available",
            )

    def _validate_config(self) -> None:
        """
        Validate TF-IDF extractor configuration.

        Raises:
            ExtractorConfigError: If configuration is invalid
        """
        super()._validate_config()

        # Validate max_features
        max_features = self.config.get("max_features")
        if max_features is not None and (
            not isinstance(max_features, int) or max_features <= 0
        ):
            raise ExtractorConfigError(
                "max_features must be a positive integer",
                context={"max_features": max_features},
            )

        # Validate ngram_range
        ngram_range = self.config.get("ngram_range")
        if ngram_range is not None:
            if (
                not isinstance(ngram_range, tuple)
                or len(ngram_range) != 2
                or not all(isinstance(x, int) and x > 0 for x in ngram_range)
                or ngram_range[0] > ngram_range[1]
            ):
                raise ExtractorConfigError(
                    "ngram_range must be a tuple of two positive integers (min, max) with min <= max",
                    context={"ngram_range": ngram_range},
                )

        # Validate min_df and max_df
        min_df = self.config.get("min_df")
        if min_df is not None and not (
            isinstance(min_df, (int, float)) and min_df >= 0
        ):
            raise ExtractorConfigError(
                "min_df must be a non-negative number", context={"min_df": min_df}
            )

        max_df = self.config.get("max_df")
        if max_df is not None and not (isinstance(max_df, (int, float)) and max_df > 0):
            raise ExtractorConfigError(
                "max_df must be a positive number", context={"max_df": max_df}
            )

    def fit(self, texts: List[str]) -> "TfidfExtractor":
        """
        Fit the TF-IDF vectorizer to training texts.

        Args:
            texts: List of training texts

        Returns:
            Self for method chaining

        Raises:
            TfidfExtractionError: If fitting fails
        """
        logger.info(f"Fitting TF-IDF extractor with config: {self.config}")

        # Validate inputs
        self._validate_texts(texts)

        if not texts:
            logger.warning("Empty text list provided for TF-IDF fitting")
            self.is_fitted = True
            self.feature_names_ = []
            return self

        try:
            # Filter out empty/invalid texts
            valid_texts = []
            for i, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    valid_texts.append(text.strip())
                else:
                    logger.debug(f"Skipping invalid text at index {i}: {type(text)}")

            if not valid_texts:
                raise TfidfExtractionError(
                    "No valid texts found for TF-IDF fitting",
                    context={"total_texts": len(texts), "valid_texts": 0},
                )

            logger.info(
                f"Fitting TF-IDF on {len(valid_texts)} valid texts out of {len(texts)} total"
            )

            # Create and fit vectorizer
            self.vectorizer_ = TfidfVectorizer(
                max_features=self.config["max_features"],
                ngram_range=self.config["ngram_range"],
                min_df=self.config["min_df"],
                max_df=self.config["max_df"],
                stop_words=self.config["stop_words"],
                lowercase=self.config["lowercase"],
                strip_accents=self.config["strip_accents"],
                analyzer=self.config["analyzer"],
                token_pattern=self.config["token_pattern"],
            )

            # Fit the vectorizer
            self.vectorizer_.fit(valid_texts)

            # Store feature names
            self.feature_names_ = [
                f"tfidf_{name}" for name in self.vectorizer_.get_feature_names_out()
            ]

            self.is_fitted = True
            logger.info(
                f"TF-IDF extractor fitted successfully with {len(self.feature_names_)} features"
            )

            return self

        except Exception as e:
            handle_exception(
                e,
                context={
                    "extractor": "TfidfExtractor",
                    "total_texts": len(texts),
                    "config": self.config,
                },
                reraise_as=TfidfExtractionError,
                message="Failed to fit TF-IDF extractor",
            )

    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract TF-IDF features from texts.

        Args:
            texts: List of texts to process

        Returns:
            Polars DataFrame with TF-IDF features

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
            TfidfExtractionError: If extraction fails
        """
        self._check_fitted()
        self._validate_texts(texts)

        if not texts:
            logger.warning("Empty text list provided for TF-IDF extraction")
            return pl.DataFrame()

        try:
            # Filter out empty/invalid texts and track indices
            valid_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
                else:
                    logger.debug(f"Skipping invalid text at index {i}: {type(text)}")

            if not valid_texts:
                logger.warning("No valid texts found for TF-IDF extraction")
                # Return empty DataFrame with correct shape
                empty_data = {name: [0.0] * len(texts) for name in self.feature_names_}
                return pl.DataFrame(empty_data)

            logger.info(
                f"Extracting TF-IDF features from {len(valid_texts)} valid texts"
            )

            # Transform texts
            tfidf_matrix = self.vectorizer_.transform(valid_texts)

            # Convert to dense array
            tfidf_dense = tfidf_matrix.toarray()

            logger.info(f"TF-IDF matrix shape: {tfidf_dense.shape}")

            # Create full matrix with zeros for invalid texts
            full_matrix = np.zeros((len(texts), tfidf_dense.shape[1]))
            full_matrix[valid_indices] = tfidf_dense

            # Create Polars DataFrame
            tfidf_data = {
                name: full_matrix[:, i] for i, name in enumerate(self.feature_names_)
            }

            tfidf_df = pl.DataFrame(tfidf_data)

            logger.info(f"TF-IDF features extracted successfully: {tfidf_df.shape}")
            return tfidf_df

        except Exception as e:
            handle_exception(
                e,
                context={
                    "extractor": "TfidfExtractor",
                    "total_texts": len(texts),
                    "feature_count": len(self.feature_names_),
                },
                reraise_as=TfidfExtractionError,
                message="Failed to extract TF-IDF features",
            )

    def save_extractor(self, models_dir: Optional[str] = None) -> None:
        """
        Save the fitted TF-IDF extractor.

        Args:
            models_dir: Directory to save the extractor. If None, uses config directory.

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
            ModelSaveError: If saving fails
        """
        self._check_fitted()

        if models_dir is None:
            models_dir = self.config["models_dir"]

        try:
            os.makedirs(models_dir, exist_ok=True)

            # Save vectorizer
            vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
            with open(vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer_, f)

            # Save feature names
            features_path = os.path.join(models_dir, "tfidf_features.pkl")
            with open(features_path, "wb") as f:
                pickle.dump(self.feature_names_, f)

            logger.info(f"TF-IDF extractor saved to {models_dir}")

        except Exception as e:
            handle_exception(
                e,
                context={"extractor": "TfidfExtractor", "models_dir": models_dir},
                reraise_as=ModelSaveError,
                message="Failed to save TF-IDF extractor",
            )

    def load_extractor(self, models_dir: Optional[str] = None) -> "TfidfExtractor":
        """
        Load a previously fitted TF-IDF extractor.

        Args:
            models_dir: Directory containing the extractor. If None, uses config directory.

        Returns:
            Self for method chaining

        Raises:
            ModelLoadError: If loading fails
        """
        if models_dir is None:
            models_dir = self.config["models_dir"]

        try:
            # Load vectorizer
            vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
            with open(vectorizer_path, "rb") as f:
                self.vectorizer_ = pickle.load(f)

            # Load feature names
            features_path = os.path.join(models_dir, "tfidf_features.pkl")
            with open(features_path, "rb") as f:
                self.feature_names_ = pickle.load(f)

            self.is_fitted = True
            logger.info(f"TF-IDF extractor loaded from {models_dir}")

            return self

        except Exception as e:
            handle_exception(
                e,
                context={"extractor": "TfidfExtractor", "models_dir": models_dir},
                reraise_as=ModelLoadError,
                message="Failed to load TF-IDF extractor",
            )

    def get_vocabulary(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.

        Returns:
            Dictionary mapping terms to indices

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
        """
        self._check_fitted()

        if self.vectorizer_ is None:
            return {}

        return self.vectorizer_.vocabulary_

    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the extracted features.

        Returns:
            Dictionary with feature statistics

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
        """
        self._check_fitted()

        stats = {
            "feature_count": len(self.feature_names_),
            "vocabulary_size": len(self.get_vocabulary()) if self.vectorizer_ else 0,
            "ngram_range": self.config["ngram_range"],
            "max_features": self.config["max_features"],
            "min_df": self.config["min_df"],
            "max_df": self.config["max_df"],
        }

        if self.vectorizer_:
            stats.update(
                {
                    "stop_words_count": len(self.vectorizer_.stop_words_)
                    if self.vectorizer_.stop_words_
                    else 0,
                    "idf_min": float(np.min(self.vectorizer_.idf_))
                    if hasattr(self.vectorizer_, "idf_")
                    else None,
                    "idf_max": float(np.max(self.vectorizer_.idf_))
                    if hasattr(self.vectorizer_, "idf_")
                    else None,
                    "idf_mean": float(np.mean(self.vectorizer_.idf_))
                    if hasattr(self.vectorizer_, "idf_")
                    else None,
                }
            )

        return stats

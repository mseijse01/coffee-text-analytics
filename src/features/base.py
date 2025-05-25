"""
Base classes for feature extraction components.

This module provides abstract base classes that define the interface for all
feature extractors in the coffee text analytics project.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import polars as pl

# Import centralized exceptions using absolute import
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import (
    FeatureExtractionError,
    ExtractorNotFittedError,
    ExtractorConfigError,
    handle_exception,
    validate_not_none,
    validate_not_empty,
)

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    Defines the common interface and behavior for feature extraction components.
    All extractors must implement fit() and extract_features() methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self.feature_names_ = []

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate the extractor configuration.

        Raises:
            ExtractorConfigError: If configuration is invalid
        """
        # Base validation - subclasses can override
        if not isinstance(self.config, dict):
            raise ExtractorConfigError(
                "Configuration must be a dictionary",
                context={"config_type": type(self.config)},
            )

    @abstractmethod
    def fit(self, texts: List[str]) -> "BaseExtractor":
        """
        Fit the extractor to training texts.

        Args:
            texts: List of training texts

        Returns:
            Self for method chaining

        Raises:
            FeatureExtractionError: If fitting fails
        """
        pass

    @abstractmethod
    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract features from texts.

        Args:
            texts: List of texts to process

        Returns:
            Polars DataFrame with extracted features

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
            FeatureExtractionError: If extraction fails
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get the names of extracted features.

        Returns:
            List of feature names

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
        """
        if not self.is_fitted:
            raise ExtractorNotFittedError(
                "Extractor must be fitted before getting feature names",
                context={"extractor_type": self.__class__.__name__},
            )
        return self.feature_names_

    def _check_fitted(self) -> None:
        """
        Check if the extractor is fitted.

        Raises:
            ExtractorNotFittedError: If extractor is not fitted
        """
        if not self.is_fitted:
            raise ExtractorNotFittedError(
                f"{self.__class__.__name__} must be fitted before use",
                context={"extractor_type": self.__class__.__name__},
            )

    def _validate_texts(self, texts: List[str]) -> None:
        """
        Validate input texts.

        Args:
            texts: List of texts to validate

        Raises:
            FeatureExtractionError: If texts are invalid
        """
        validate_not_none(
            texts, "texts", context={"extractor": self.__class__.__name__}
        )

        if not isinstance(texts, list):
            raise FeatureExtractionError(
                "Texts must be a list",
                context={
                    "extractor": self.__class__.__name__,
                    "texts_type": type(texts),
                },
            )

        if not texts:
            logger.warning(f"{self.__class__.__name__}: Empty text list provided")


class BaseVectorExtractor(BaseExtractor):
    """
    Base class for extractors that produce dense vector features.

    Examples: BERT embeddings, GloVe embeddings
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_size = self.config.get("vector_size", 768)

    def _validate_config(self) -> None:
        """Validate vector extractor configuration."""
        super()._validate_config()

        vector_size = self.config.get("vector_size")
        if vector_size is not None and (
            not isinstance(vector_size, int) or vector_size <= 0
        ):
            raise ExtractorConfigError(
                "vector_size must be a positive integer",
                context={"vector_size": vector_size},
            )


class BaseSparseExtractor(BaseExtractor):
    """
    Base class for extractors that produce sparse features.

    Examples: TF-IDF, bag-of-words
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_features = self.config.get("max_features", 5000)

    def _validate_config(self) -> None:
        """Validate sparse extractor configuration."""
        super()._validate_config()

        max_features = self.config.get("max_features")
        if max_features is not None and (
            not isinstance(max_features, int) or max_features <= 0
        ):
            raise ExtractorConfigError(
                "max_features must be a positive integer",
                context={"max_features": max_features},
            )


class BaseTopicExtractor(BaseExtractor):
    """
    Base class for topic modeling extractors.

    Examples: LDA, NMF topic modeling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_topics = self.config.get("n_topics", 10)

    def _validate_config(self) -> None:
        """Validate topic extractor configuration."""
        super()._validate_config()

        n_topics = self.config.get("n_topics")
        if n_topics is not None and (not isinstance(n_topics, int) or n_topics <= 0):
            raise ExtractorConfigError(
                "n_topics must be a positive integer", context={"n_topics": n_topics}
            )


# Legacy compatibility - keep the old exception names for backward compatibility
# but they now inherit from the centralized exceptions
ExtractorError = FeatureExtractionError  # Alias for backward compatibility

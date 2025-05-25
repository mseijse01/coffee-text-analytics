"""
Feature extraction module for coffee review text analysis.

This module provides a comprehensive feature extraction pipeline following
the thesis methodology with component-based architecture.
"""

# Base classes
from .base import (
    BaseExtractor,
    BaseVectorExtractor,
    BaseTopicExtractor,
    BaseSparseExtractor,
    ExtractorError,
    ExtractorNotFittedError,
    ExtractorConfigError,
)

# Individual extractors
from .tfidf_extractor import TfidfExtractor
from .bert_extractor import BertExtractor
from .topic_extractor import TopicExtractor
from .sentiment_extractor import SentimentExtractor

# Unified manager
from .feature_manager import CoffeeFeatureManager, GloVeExtractor

# Legacy CoffeeFeatureExtractor has been removed - use CoffeeFeatureManager instead


__all__ = [
    # Base classes
    "BaseExtractor",
    "BaseVectorExtractor",
    "BaseTopicExtractor",
    "BaseSparseExtractor",
    "ExtractorError",
    "ExtractorNotFittedError",
    "ExtractorConfigError",
    # Individual extractors
    "TfidfExtractor",
    "BertExtractor",
    "TopicExtractor",
    "SentimentExtractor",
    "GloVeExtractor",
    # Unified manager
    "CoffeeFeatureManager",
]

"""
Feature extraction module for coffee review text analysis.

This module provides a comprehensive feature extraction pipeline following
the thesis methodology with component-based architecture.
"""

# Base classes
from .base import (
    BaseExtractor,
    BaseSparseExtractor,
    BaseTopicExtractor,
    BaseVectorExtractor,
    ExtractorConfigError,
    ExtractorError,
    ExtractorNotFittedError,
)
from .bert_extractor import BertExtractor

# Unified manager
from .feature_manager import CoffeeFeatureManager, GloVeExtractor

# Feature selection
from .feature_selector import LassoFeatureSelector
from .sentiment_extractor import SentimentExtractor

# Individual extractors
from .tfidf_extractor import TfidfExtractor
from .topic_extractor import TopicExtractor

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
    # Feature selection
    "LassoFeatureSelector",
]

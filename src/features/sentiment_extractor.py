"""
Sentiment analysis extractor for coffee review text analysis.

This module implements sentiment analysis following the thesis methodology:
- DistilBERT-based sentiment classification
- Positive/negative probability scores
- Polars DataFrame output for efficient processing
"""

import polars as pl
import numpy as np
import logging
from typing import List, Dict, Optional, Any

from .base import BaseExtractor, ExtractorError

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from transformers import (
        pipeline,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available - sentiment analysis enabled")
except ImportError:
    logger.warning("Transformers not installed. Sentiment analysis will be limited.")
    TRANSFORMERS_AVAILABLE = False


class SentimentExtractor(BaseExtractor):
    """
    Sentiment analysis extractor following thesis methodology.

    From thesis: "Sentiment scores (positive/negative probabilities)"

    This extractor analyzes sentiment in coffee reviews using DistilBERT
    and outputs results as Polars DataFrames for efficient processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment extractor.

        Args:
            config: Configuration dictionary with parameters:
                - model_name: Sentiment model name (default: distilbert-base-uncased-finetuned-sst-2-english)
                - batch_size: Batch size for processing (default: 16)
                - max_length: Maximum sequence length (default: 512)
                - device: Device to use (default: auto-detect)
        """
        super().__init__(config)

        # Set default configuration
        default_config = {
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "batch_size": 16,
            "max_length": 512,
            "device": 0 if TRANSFORMERS_AVAILABLE else -1,  # 0 for GPU, -1 for CPU
        }
        default_config.update(self.config)
        self.config = default_config

        # Initialize sentiment pipeline
        self.sentiment_pipeline_ = None
        self.feature_names_ = ["sentiment_positive", "sentiment_negative"]

        if TRANSFORMERS_AVAILABLE:
            self._load_sentiment_model()
        else:
            logger.warning(
                "Transformers not available. Sentiment extractor will be limited."
            )

    def _load_sentiment_model(self) -> None:
        """Load sentiment analysis pipeline."""
        try:
            logger.info(f"Loading sentiment model: {self.config['model_name']}")

            self.sentiment_pipeline_ = pipeline(
                "sentiment-analysis",
                model=self.config["model_name"],
                tokenizer=self.config["model_name"],
                device=self.config["device"],
                return_all_scores=True,
                max_length=self.config["max_length"],
                truncation=True,
            )

            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise ExtractorError(f"Failed to load sentiment model: {e}")

    def fit(self, texts: List[str]) -> "SentimentExtractor":
        """
        Fit the sentiment extractor (no training needed for pre-trained models).

        Args:
            texts: List of training texts (not used for sentiment analysis)

        Returns:
            Self for method chaining
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Transformers not available. Sentiment extractor cannot be fitted."
            )
            return self

        # Sentiment model is pre-trained, so we just mark as fitted
        self.is_fitted = True
        logger.info("Sentiment extractor fitted (using pre-trained model)")
        return self

    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract sentiment features from texts.

        Args:
            texts: List of texts to process

        Returns:
            Polars DataFrame with sentiment features (positive/negative probabilities)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Sentiment analysis not available, returning empty DataFrame"
            )
            return pl.DataFrame()

        if not self.is_fitted:
            raise ExtractorError(
                "Sentiment extractor must be fitted before feature extraction"
            )

        if not texts or all(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            logger.warning("Empty or invalid text input for sentiment analysis")
            return pl.DataFrame()

        logger.info(f"Extracting sentiment features for {len(texts)} texts")

        try:
            # Process texts in batches
            batch_size = self.config["batch_size"]
            all_sentiments = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_sentiments = self._analyze_batch_sentiment(batch_texts)
                all_sentiments.extend(batch_sentiments)

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

            # Convert to Polars DataFrame
            positive_scores = [s["positive"] for s in all_sentiments]
            negative_scores = [s["negative"] for s in all_sentiments]

            sentiment_df = pl.DataFrame(
                {
                    "sentiment_positive": positive_scores,
                    "sentiment_negative": negative_scores,
                }
            )

            logger.info(
                f"Created Polars DataFrame with sentiment features: {sentiment_df.shape}"
            )
            return sentiment_df

        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
            raise ExtractorError(f"Failed to extract sentiment features: {e}")

    def _analyze_batch_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: Batch of texts to analyze

        Returns:
            List of sentiment dictionaries with positive/negative scores
        """
        try:
            # Get sentiment predictions
            results = self.sentiment_pipeline_(texts)

            # Convert to standardized format
            batch_sentiments = []
            for result in results:
                # result is a list of scores for each label
                sentiment_dict = {}
                for score_dict in result:
                    label = score_dict["label"].lower()
                    score = score_dict["score"]

                    # Map labels to positive/negative
                    if label in ["positive", "pos"]:
                        sentiment_dict["positive"] = score
                    elif label in ["negative", "neg"]:
                        sentiment_dict["negative"] = score

                # Ensure both positive and negative scores exist
                if "positive" not in sentiment_dict:
                    sentiment_dict["positive"] = 1.0 - sentiment_dict.get(
                        "negative", 0.5
                    )
                if "negative" not in sentiment_dict:
                    sentiment_dict["negative"] = 1.0 - sentiment_dict.get(
                        "positive", 0.5
                    )

                batch_sentiments.append(sentiment_dict)

            return batch_sentiments

        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            raise ExtractorError(f"Failed to analyze batch sentiment: {e}")

    def get_feature_names(self) -> List[str]:
        """Get sentiment feature names."""
        return self.feature_names_.copy()

    def get_feature_count(self) -> int:
        """Get the number of sentiment features."""
        return len(self.feature_names_)

    def analyze_single_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment for a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with positive and negative sentiment scores
        """
        if not self.is_fitted:
            raise ExtractorError("Sentiment extractor must be fitted before analysis")

        sentiments = self._analyze_batch_sentiment([text])
        return sentiments[0]

    def get_dominant_sentiment(self, text: str) -> str:
        """
        Get the dominant sentiment (positive/negative) for a text.

        Args:
            text: Text to analyze

        Returns:
            'positive' or 'negative'
        """
        sentiment_scores = self.analyze_single_text(text)
        return (
            "positive"
            if sentiment_scores["positive"] > sentiment_scores["negative"]
            else "negative"
        )

    def get_sentiment_confidence(self, text: str) -> float:
        """
        Get the confidence score for the dominant sentiment.

        Args:
            text: Text to analyze

        Returns:
            Confidence score (0-1)
        """
        sentiment_scores = self.analyze_single_text(text)
        return max(sentiment_scores["positive"], sentiment_scores["negative"])

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the sentiment model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config["model_name"],
            "batch_size": self.config["batch_size"],
            "max_length": self.config["max_length"],
            "device": self.config["device"],
            "feature_names": self.feature_names_,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "is_fitted": self.is_fitted,
        }

    def get_sentiment_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get sentiment statistics for a collection of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with sentiment statistics
        """
        if not self.is_fitted:
            raise ExtractorError("Sentiment extractor must be fitted to get statistics")

        sentiment_df = self.extract_features(texts)

        if sentiment_df.is_empty():
            return {}

        positive_scores = sentiment_df["sentiment_positive"].to_numpy()
        negative_scores = sentiment_df["sentiment_negative"].to_numpy()

        # Determine dominant sentiments
        dominant_positive = (positive_scores > negative_scores).sum()
        dominant_negative = len(texts) - dominant_positive

        return {
            "total_texts": len(texts),
            "positive_dominant": int(dominant_positive),
            "negative_dominant": int(dominant_negative),
            "positive_percentage": float(dominant_positive / len(texts) * 100),
            "negative_percentage": float(dominant_negative / len(texts) * 100),
            "avg_positive_score": float(positive_scores.mean()),
            "avg_negative_score": float(negative_scores.mean()),
            "std_positive_score": float(positive_scores.std()),
            "std_negative_score": float(negative_scores.std()),
        }

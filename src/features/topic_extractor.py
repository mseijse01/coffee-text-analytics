"""
Topic modeling extractor for coffee review text analysis.

This module implements topic modeling following the thesis methodology:
- LDA (Latent Dirichlet Allocation) for probabilistic topic modeling
- NMF (Non-negative Matrix Factorization) for linear topic modeling
- 10 topics per model as specified in thesis
- Polars DataFrame output for efficient processing
"""

import polars as pl
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from .base import BaseTopicExtractor, ExtractorError

logger = logging.getLogger(__name__)


class TopicExtractor(BaseTopicExtractor):
    """
    Topic modeling extractor following thesis methodology.

    From thesis: "LDA and NMF topic modeling (10 topics each)"

    This extractor discovers latent topics in coffee reviews using both
    LDA and NMF algorithms and outputs results as Polars DataFrames.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize topic extractor.

        Args:
            config: Configuration dictionary with parameters:
                - n_topics: Number of topics to discover (default: 10)
                - algorithms: List of algorithms to use (default: ['lda', 'nmf'])
                - max_features: Max features for TF-IDF (default: 1000)
                - max_iter: Maximum iterations for fitting (default: 100)
                - random_state: Random state for reproducibility (default: 42)
                - models_dir: Directory to save fitted models (default: 'models')
        """
        super().__init__(config)

        # Set default configuration
        default_config = {
            "n_topics": 10,
            "algorithms": ["lda", "nmf"],
            "max_features": 1000,
            "max_iter": 100,
            "random_state": 42,
            "models_dir": "models",
        }
        default_config.update(self.config)
        self.config = default_config

        # Initialize models
        self.vectorizer_ = None
        self.lda_model_ = None
        self.nmf_model_ = None
        self.topics_ = {}
        self.topic_words_ = {}

        # Ensure models directory exists
        os.makedirs(self.config["models_dir"], exist_ok=True)

    def fit(self, texts: List[str]) -> "TopicExtractor":
        """
        Fit topic models to the training texts.

        Args:
            texts: List of training texts

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting topic models with config: {self.config}")

        # Handle empty inputs
        if not texts or all(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            logger.warning("Empty or invalid text input for topic modeling")
            self.is_fitted = True
            return self

        try:
            # Create TF-IDF vectorizer for topic modeling (thesis methodology)
            # Following thesis: retain stopwords, remove punctuation for topic modeling
            self.vectorizer_ = TfidfVectorizer(
                max_features=self.config["max_features"],
                stop_words=None,  # Retain stopwords for topic modeling (thesis methodology)
                min_df=1 if len(texts) < 10 else 2,
                max_df=1.0 if len(texts) < 20 else 0.95,
                token_pattern=r"(?u)\b\w\w+\b",  # Only words, no punctuation
            )

            # Fit vectorizer and transform texts
            tfidf_matrix = self.vectorizer_.fit_transform(texts)
            logger.info(f"TF-IDF matrix for topic modeling: {tfidf_matrix.shape}")

            # Fit topic models
            algorithms = self.config["algorithms"]

            if "lda" in algorithms:
                self._fit_lda(tfidf_matrix)

            if "nmf" in algorithms:
                self._fit_nmf(tfidf_matrix)

            # Extract topics and create feature names
            self._extract_topics()
            self._create_feature_names()

            # Save models
            self._save_models()

            self.is_fitted = True
            logger.info("Topic models fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting topic models: {e}")
            raise ExtractorError(f"Failed to fit topic models: {e}")

    def _fit_lda(self, tfidf_matrix) -> None:
        """Fit LDA model."""
        logger.info("Fitting LDA model")

        self.lda_model_ = LatentDirichletAllocation(
            n_components=self.config["n_topics"],
            max_iter=self.config["max_iter"],
            random_state=self.config["random_state"],
            n_jobs=-1,
        )

        self.lda_model_.fit(tfidf_matrix)
        logger.info("LDA model fitted successfully")

    def _fit_nmf(self, tfidf_matrix) -> None:
        """Fit NMF model."""
        logger.info("Fitting NMF model")

        self.nmf_model_ = NMF(
            n_components=self.config["n_topics"],
            max_iter=self.config["max_iter"],
            random_state=self.config["random_state"],
        )

        self.nmf_model_.fit(tfidf_matrix)
        logger.info("NMF model fitted successfully")

    def _extract_topics(self) -> None:
        """Extract topics from fitted models."""
        feature_names = self.vectorizer_.get_feature_names_out()

        # Extract LDA topics
        if self.lda_model_ is not None:
            self.topics_["lda"] = []
            for topic_idx, topic in enumerate(self.lda_model_.components_):
                top_words_idx = topic.argsort()[-10:][::-1]  # Top 10 words
                top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
                self.topics_["lda"].append(top_words)

        # Extract NMF topics
        if self.nmf_model_ is not None:
            self.topics_["nmf"] = []
            for topic_idx, topic in enumerate(self.nmf_model_.components_):
                top_words_idx = topic.argsort()[-10:][::-1]  # Top 10 words
                top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
                self.topics_["nmf"].append(top_words)

    def _create_feature_names(self) -> None:
        """Create feature names for topic distributions."""
        self.feature_names_ = []

        if "lda" in self.config["algorithms"]:
            for i in range(self.config["n_topics"]):
                self.feature_names_.append(f"lda_topic_{i}")

        if "nmf" in self.config["algorithms"]:
            for i in range(self.config["n_topics"]):
                self.feature_names_.append(f"nmf_topic_{i}")

    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract topic features from texts.

        Args:
            texts: List of texts to process

        Returns:
            Polars DataFrame with topic distribution features
        """
        if not self.is_fitted:
            raise ExtractorError(
                "Topic extractor must be fitted before feature extraction"
            )

        if not texts or all(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            logger.warning("Empty or invalid text input for topic extraction")
            return pl.DataFrame()

        try:
            # Transform texts to TF-IDF
            tfidf_matrix = self.vectorizer_.transform(texts)

            feature_data = {}

            # Extract LDA topic distributions
            if self.lda_model_ is not None:
                lda_distributions = self.lda_model_.transform(tfidf_matrix)
                for i in range(self.config["n_topics"]):
                    feature_data[f"lda_topic_{i}"] = lda_distributions[:, i]

            # Extract NMF topic distributions
            if self.nmf_model_ is not None:
                nmf_distributions = self.nmf_model_.transform(tfidf_matrix)
                for i in range(self.config["n_topics"]):
                    feature_data[f"nmf_topic_{i}"] = nmf_distributions[:, i]

            # Create Polars DataFrame
            topic_df = pl.DataFrame(feature_data)
            logger.info(
                f"Created Polars DataFrame with topic features: {topic_df.shape}"
            )

            return topic_df

        except Exception as e:
            logger.error(f"Error extracting topic features: {e}")
            raise ExtractorError(f"Failed to extract topic features: {e}")

    def get_feature_names(self) -> List[str]:
        """Get topic feature names."""
        return self.feature_names_.copy()

    def get_topics(self) -> List[List[Tuple[str, float]]]:
        """Get discovered topics with top words and weights."""
        all_topics = []

        if "lda" in self.topics_:
            all_topics.extend(self.topics_["lda"])

        if "nmf" in self.topics_:
            all_topics.extend(self.topics_["nmf"])

        return all_topics

    def get_topic_count(self) -> int:
        """Get the total number of topics."""
        count = 0
        if "lda" in self.config["algorithms"]:
            count += self.config["n_topics"]
        if "nmf" in self.config["algorithms"]:
            count += self.config["n_topics"]
        return count

    def get_lda_topics(self) -> List[List[Tuple[str, float]]]:
        """Get LDA topics specifically."""
        return self.topics_.get("lda", [])

    def get_nmf_topics(self) -> List[List[Tuple[str, float]]]:
        """Get NMF topics specifically."""
        return self.topics_.get("nmf", [])

    def print_topics(self, n_words: int = 10) -> None:
        """
        Print discovered topics in a readable format.

        Args:
            n_words: Number of top words to display per topic
        """
        if not self.is_fitted:
            raise ExtractorError("Topic extractor must be fitted to print topics")

        # Print LDA topics
        if "lda" in self.topics_:
            print("\n=== LDA Topics ===")
            for i, topic in enumerate(self.topics_["lda"]):
                words = [f"{word}({weight:.3f})" for word, weight in topic[:n_words]]
                print(f"Topic {i}: {', '.join(words)}")

        # Print NMF topics
        if "nmf" in self.topics_:
            print("\n=== NMF Topics ===")
            for i, topic in enumerate(self.topics_["nmf"]):
                words = [f"{word}({weight:.3f})" for word, weight in topic[:n_words]]
                print(f"Topic {i}: {', '.join(words)}")

    def get_topic_distribution(self, text: str) -> Dict[str, np.ndarray]:
        """
        Get topic distribution for a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with topic distributions for each algorithm
        """
        if not self.is_fitted:
            raise ExtractorError("Topic extractor must be fitted to get distributions")

        tfidf_vector = self.vectorizer_.transform([text])
        distributions = {}

        if self.lda_model_ is not None:
            distributions["lda"] = self.lda_model_.transform(tfidf_vector)[0]

        if self.nmf_model_ is not None:
            distributions["nmf"] = self.nmf_model_.transform(tfidf_vector)[0]

        return distributions

    def _save_models(self) -> None:
        """Save fitted models to disk."""
        try:
            models_dir = self.config["models_dir"]

            # Save vectorizer
            with open(os.path.join(models_dir, "topic_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.vectorizer_, f)

            # Save LDA model
            if self.lda_model_ is not None:
                with open(os.path.join(models_dir, "lda_model.pkl"), "wb") as f:
                    pickle.dump(self.lda_model_, f)

            # Save NMF model
            if self.nmf_model_ is not None:
                with open(os.path.join(models_dir, "nmf_model.pkl"), "wb") as f:
                    pickle.dump(self.nmf_model_, f)

            # Save topics
            with open(os.path.join(models_dir, "topics.pkl"), "wb") as f:
                pickle.dump(self.topics_, f)

            logger.info("Topic models saved successfully")

        except Exception as e:
            logger.warning(f"Failed to save topic models: {e}")

    def load_models(self, models_dir: Optional[str] = None) -> "TopicExtractor":
        """
        Load previously fitted models.

        Args:
            models_dir: Directory containing the models. If None, uses config directory.

        Returns:
            Self for method chaining
        """
        if models_dir is None:
            models_dir = self.config["models_dir"]

        try:
            # Load vectorizer
            with open(os.path.join(models_dir, "topic_vectorizer.pkl"), "rb") as f:
                self.vectorizer_ = pickle.load(f)

            # Load LDA model if exists
            lda_path = os.path.join(models_dir, "lda_model.pkl")
            if os.path.exists(lda_path):
                with open(lda_path, "rb") as f:
                    self.lda_model_ = pickle.load(f)

            # Load NMF model if exists
            nmf_path = os.path.join(models_dir, "nmf_model.pkl")
            if os.path.exists(nmf_path):
                with open(nmf_path, "rb") as f:
                    self.nmf_model_ = pickle.load(f)

            # Load topics
            with open(os.path.join(models_dir, "topics.pkl"), "rb") as f:
                self.topics_ = pickle.load(f)

            # Recreate feature names
            self._create_feature_names()
            self.is_fitted = True

            logger.info("Topic models loaded successfully")
            return self

        except Exception as e:
            logger.error(f"Failed to load topic models: {e}")
            raise ExtractorError(f"Failed to load models from {models_dir}: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the topic models.

        Returns:
            Dictionary with model information
        """
        return {
            "n_topics": self.config["n_topics"],
            "algorithms": self.config["algorithms"],
            "max_features": self.config["max_features"],
            "max_iter": self.config["max_iter"],
            "total_topics": self.get_topic_count(),
            "is_fitted": self.is_fitted,
            "has_lda": self.lda_model_ is not None,
            "has_nmf": self.nmf_model_ is not None,
        }

"""
Feature extraction utilities for coffee review text data.

This module includes:
- Topic modeling (LDA and NMF)
- Text embeddings generation
- Sentiment analysis
- Feature combination
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not installed. Sentiment analysis will be limited.")
    TRANSFORMERS_AVAILABLE = False

try:
    import gensim
    from gensim.corpora import Dictionary

    GENSIM_AVAILABLE = True
except ImportError:
    logger.warning(
        "Gensim not installed. Some topic modeling features will be limited."
    )
    GENSIM_AVAILABLE = False


def tfidf_vectorization(
    texts: List[str], max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 3)
):
    """
    Convert text corpus to TF-IDF matrix.

    Args:
        texts: List of preprocessed text documents
        max_features: Maximum number of features to extract
        ngram_range: Range of n-grams to extract

    Returns:
        Tuple: (vectorizer, tfidf_matrix)
    """
    logger.info(
        f"Performing TF-IDF vectorization with max_features={max_features}, ngram_range={ngram_range}"
    )

    # Handle empty inputs
    if not texts or all(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        logger.warning("Empty or invalid text input for TF-IDF vectorization")
        return TfidfVectorizer(), np.zeros((0, 0))

    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        return TfidfVectorizer(), np.zeros((0, 0))


def perform_topic_modeling(
    tfidf_matrix, n_topics=10, model_type="lda", random_state=42
):
    """
    Perform topic modeling on TF-IDF matrix.

    Args:
        tfidf_matrix: TF-IDF matrix of documents
        n_topics: Number of topics to extract
        model_type: Type of topic model ('lda' or 'nmf')
        random_state: Random state for reproducibility

    Returns:
        Trained topic model
    """
    if tfidf_matrix.shape[0] == 0:
        logger.error(f"Empty matrix provided for {model_type.upper()} topic modeling")
        return None

    logger.info(f"Training {model_type.upper()} topic model with {n_topics} topics")

    try:
        if model_type.lower() == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method="online",
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type.lower() == "nmf":
            model = NMF(n_components=n_topics, random_state=random_state, max_iter=1000)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None

        model.fit(tfidf_matrix)
        return model
    except Exception as e:
        logger.error(f"Error during {model_type.upper()} topic modeling: {e}")
        return None


def extract_topic_features(df, text_col, n_topics=10):
    """
    Extract topic distribution features from text.

    Args:
        df: DataFrame containing text data
        text_col: Name of column containing processed text
        n_topics: Number of topics for modeling

    Returns:
        DataFrame with topic features added
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    if text_col not in result.columns:
        logger.error(f"Text column '{text_col}' not found in DataFrame")
        return result

    try:
        # Vectorize text
        texts = result[text_col].fillna("").tolist()
        vectorizer, tfidf_matrix = tfidf_vectorization(texts)

        # Train topic models
        lda_model = perform_topic_modeling(tfidf_matrix, n_topics, "lda")
        nmf_model = perform_topic_modeling(tfidf_matrix, n_topics, "nmf")

        if lda_model:
            # Get topic distributions for LDA
            lda_topics = lda_model.transform(tfidf_matrix)

            # Add topic distributions to DataFrame
            for i in range(n_topics):
                result[f"lda_topic_{i + 1}"] = lda_topics[:, i]

            result["dominant_lda_topic"] = np.argmax(lda_topics, axis=1) + 1
            logger.info(f"Added {n_topics} LDA topic features")

        if nmf_model:
            # Get topic distributions for NMF
            nmf_topics = nmf_model.transform(tfidf_matrix)

            # Add topic distributions to DataFrame
            for i in range(n_topics):
                result[f"nmf_topic_{i + 1}"] = nmf_topics[:, i]

            result["dominant_nmf_topic"] = np.argmax(nmf_topics, axis=1) + 1
            logger.info(f"Added {n_topics} NMF topic features")

        # Save models
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        if lda_model:
            with open(os.path.join(models_dir, "lda_model.pkl"), "wb") as f:
                pickle.dump(lda_model, f)

        if nmf_model:
            with open(os.path.join(models_dir, "nmf_model.pkl"), "wb") as f:
                pickle.dump(nmf_model, f)

        logger.info("Saved topic modeling artifacts")

        return result
    except Exception as e:
        logger.error(f"Error extracting topic features: {e}")
        return result


def perform_sentiment_analysis(texts):
    """
    Perform sentiment analysis on a list of texts.

    Args:
        texts: List of texts to analyze

    Returns:
        List of sentiment scores (0-1 scale, where 1 is positive)
    """
    try:
        if TRANSFORMERS_AVAILABLE:
            logger.info(
                f"Performing transformer-based sentiment analysis on {len(texts)} texts"
            )

            # Filter empty texts
            valid_texts = [
                text for text in texts if isinstance(text, str) and text.strip()
            ]

            if not valid_texts:
                logger.warning("No valid texts for sentiment analysis")
                return [0.5] * len(texts)  # Neutral sentiment for all

            # Initialize sentiment pipeline with small distilbert model
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )

            # Process in batches
            batch_size = 32
            results = []

            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i : i + batch_size]
                batch_results = sentiment_pipeline(batch)

                # Extract scores (convert to 0-1 scale where 1 is positive)
                batch_scores = []
                for result in batch_results:
                    if result["label"] == "POSITIVE":
                        batch_scores.append(result["score"])
                    else:
                        batch_scores.append(1 - result["score"])

                results.extend(batch_scores)

            # Handle any texts that were skipped
            if len(results) < len(texts):
                results.extend([0.5] * (len(texts) - len(results)))

            return results
        else:
            # Fallback to simple lexicon-based approach
            logger.info("Using basic sentiment analysis (transformers not available)")

            positive_words = {
                "good",
                "great",
                "excellent",
                "delicious",
                "sweet",
                "balanced",
                "rich",
                "complex",
                "perfect",
            }
            negative_words = {
                "bad",
                "bitter",
                "sour",
                "harsh",
                "disappointing",
                "unbalanced",
                "weak",
                "stale",
            }

            scores = []
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    scores.append(0.5)  # Neutral
                    continue

                words = text.lower().split()
                pos_count = sum(1 for word in words if word in positive_words)
                neg_count = sum(1 for word in words if word in negative_words)

                if pos_count == 0 and neg_count == 0:
                    scores.append(0.5)  # Neutral
                else:
                    score = (
                        pos_count / (pos_count + neg_count)
                        if (pos_count + neg_count) > 0
                        else 0.5
                    )
                    scores.append(score)

            return scores
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {e}")
        return [0.5] * len(texts)  # Default to neutral


def extract_text_features(df, text_col):
    """
    Extract basic text features like length, word count, etc.

    Args:
        df: DataFrame containing text data
        text_col: Name of column containing text

    Returns:
        DataFrame with text features added
    """
    result = df.copy()

    if text_col not in result.columns:
        logger.warning(f"Text column '{text_col}' not found in DataFrame")
        return result

    try:
        # Fill NA values
        result[text_col] = result[text_col].fillna("")

        # Extract text length features
        result[f"{text_col}_char_count"] = result[text_col].apply(len)
        result[f"{text_col}_word_count"] = result[text_col].apply(
            lambda x: len(str(x).split())
        )
        result[f"{text_col}_avg_word_length"] = result[text_col].apply(
            lambda x: np.mean([len(w) for w in str(x).split()])
            if len(str(x).split()) > 0
            else 0
        )

        logger.info(f"Added basic text features from column: {text_col}")
        return result
    except Exception as e:
        logger.error(f"Error extracting text features: {e}")
        return result


def extract_features_from_data(input_file, output_file, n_topics=10):
    """
    Main function to extract features from preprocessed data.

    Args:
        input_file: Path to preprocessed data file
        output_file: Path to save features
        n_topics: Number of topics for topic modeling
    """
    try:
        # Load preprocessed data
        logger.info(f"Loading preprocessed data from {input_file}")
        df = pd.read_csv(input_file)

        if df.empty:
            raise ValueError(f"No data found in {input_file}")

        # Ensure text column exists
        text_col = "processed_text"
        if text_col not in df.columns:
            logger.warning(f"'{text_col}' column not found, looking for 'merged_text'")
            text_col = "merged_text"

            if text_col not in df.columns:
                raise ValueError("No text column found for feature extraction")

        # Extract basic text features
        df = extract_text_features(df, text_col)

        # Extract topic modeling features
        df = extract_topic_features(df, text_col, n_topics)

        # Extract sentiment
        logger.info("Performing sentiment analysis")
        df["sentiment_score"] = perform_sentiment_analysis(
            df[text_col].fillna("").tolist()
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Save features
        df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        raise

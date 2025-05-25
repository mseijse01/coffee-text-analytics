"""
Advanced feature extraction for coffee review text analysis using Polars.

This module implements the complete feature extraction pipeline described in the thesis:
- TF-IDF vectorization with unigrams, bigrams, and trigrams (5000 features)
- BERT embeddings using DistilBERT (768-dimensional vectors)
- GloVe embeddings (300-dimensional vectors)
- Topic modeling using LDA and NMF
- Sentiment analysis using DistilBERT
- Text-based feature engineering

Uses Polars for efficient data manipulation and showcases modern data processing techniques.
Aligns with thesis methodology for comprehensive text analysis.
"""

import polars as pl
import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path

# Core ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertModel,
        DistilBertForSequenceClassification,
        pipeline,
    )
    import torch

    TRANSFORMERS_AVAILABLE = True
    logger.info(
        "Transformers available - BERT embeddings and sentiment analysis enabled"
    )
except ImportError:
    logger.warning("Transformers not installed. BERT features will be limited.")
    TRANSFORMERS_AVAILABLE = False

try:
    import gensim.downloader as api

    GENSIM_AVAILABLE = True
    logger.info("Gensim available - GloVe embeddings enabled")
except ImportError:
    logger.warning("Gensim not installed. GloVe embeddings will be limited.")
    GENSIM_AVAILABLE = False


class CoffeeFeatureExtractor:
    """
    Comprehensive feature extraction for coffee reviews using Polars and matching thesis methodology.

    This implementation showcases modern data processing with Polars while extracting:
    - TF-IDF features (unigrams, bigrams, trigrams) - 5000 features per text column
    - BERT embeddings (768-dimensional) using DistilBERT
    - GloVe embeddings (300-dimensional) using pre-trained vectors
    - LDA and NMF topic modeling (10 topics each)
    - Sentiment scores (positive/negative probabilities)

    As described in the thesis: "A diverse set of features, including flavor attributes,
    categorical variables such as country of origin and roast level, and text-based features
    derived from BERT embeddings, GloVe vectors, and LDA topics, were used to predict coffee ratings."
    """

    def __init__(self, models_dir: str = "models"):
        """Initialize the feature extractor with all required models."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # Initialize BERT models if available
        if TRANSFORMERS_AVAILABLE:
            logger.info("Loading BERT models for semantic analysis...")
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("BERT models loaded successfully")

        # Initialize GloVe model if available
        if GENSIM_AVAILABLE:
            logger.info("Loading GloVe embeddings for word-level semantics...")
            try:
                self.glove_model = api.load("glove-wiki-gigaword-300")
                logger.info("GloVe embeddings loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load GloVe embeddings: {e}")
                self.glove_model = None
        else:
            self.glove_model = None

    def extract_tfidf_features(
        self,
        texts: List[str],
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 3),
    ) -> Tuple[pl.DataFrame, TfidfVectorizer]:
        """
        Extract TF-IDF features as described in thesis using Polars for efficient processing.

        From thesis: "TF-IDF vectorization with unigrams, bigrams, and trigrams (5000 features)"

        Args:
            texts: List of preprocessed text documents
            max_features: Maximum number of features (thesis uses 5000)
            ngram_range: N-gram range (thesis uses (1,3) for unigrams, bigrams, trigrams)

        Returns:
            Tuple of (polars_dataframe_with_tfidf_features, fitted_vectorizer)
        """
        logger.info(
            f"Extracting TF-IDF features with Polars: max_features={max_features}, ngram_range={ngram_range}"
        )

        # Handle empty inputs
        if not texts or all(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            logger.warning("Empty or invalid text input for TF-IDF")
            return pl.DataFrame(), TfidfVectorizer()

        try:
            # Adjust parameters for small datasets and identical texts
            min_df = 1 if len(texts) < 10 else 2
            max_df = 1.0 if len(texts) < 20 else 0.95

            # Use sklearn for TF-IDF computation (industry standard)
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words="english",  # Remove common stopwords
                min_df=min_df,  # Adjust for small datasets
                max_df=max_df,  # Adjust for small datasets
            )

            tfidf_matrix = vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

            # Convert to Polars DataFrame for efficient processing
            feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_data = {
                name: tfidf_matrix[:, i].toarray().flatten()
                for i, name in enumerate(feature_names)
            }

            tfidf_df = pl.DataFrame(tfidf_data)
            logger.info(
                f"Created Polars DataFrame with TF-IDF features: {tfidf_df.shape}"
            )

            # Save vectorizer for future use
            vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)

            return tfidf_df, vectorizer

        except Exception as e:
            logger.error(f"Error during TF-IDF extraction: {e}")
            return pl.DataFrame(), TfidfVectorizer()

    def extract_bert_embeddings(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract BERT embeddings using DistilBERT as described in thesis.

        From thesis: "BERT embeddings using DistilBERT (768-dimensional vectors)"

        Args:
            texts: List of preprocessed text documents

        Returns:
            Polars DataFrame with BERT embedding features (768 columns)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("BERT not available, returning empty DataFrame")
            return pl.DataFrame()

        logger.info(f"Extracting BERT embeddings for {len(texts)} texts using Polars")
        embeddings = []

        try:
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    logger.info(f"Processing BERT embedding {i}/{len(texts)}")

                if not isinstance(text, str) or not text.strip():
                    # Use zero embedding for empty text
                    embeddings.append(np.zeros(768))
                    continue

                # Tokenize and encode
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )

                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(embedding)

            # Convert to Polars DataFrame
            embeddings_array = np.vstack(embeddings)
            bert_data = {
                f"bert_{i}": embeddings_array[:, i]
                for i in range(embeddings_array.shape[1])
            }

            bert_df = pl.DataFrame(bert_data)
            logger.info(f"BERT embeddings extracted: {bert_df.shape}")
            return bert_df

        except Exception as e:
            logger.error(f"Error during BERT embedding extraction: {e}")
            return pl.DataFrame()

    def extract_glove_embeddings(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract GloVe embeddings as described in thesis.

        From thesis: "GloVe embeddings (300-dimensional vectors)"

        Args:
            texts: List of preprocessed text documents

        Returns:
            Polars DataFrame with GloVe embedding features (300 columns)
        """
        if not self.glove_model:
            logger.warning("GloVe not available, returning empty DataFrame")
            return pl.DataFrame()

        logger.info(f"Extracting GloVe embeddings for {len(texts)} texts using Polars")
        embeddings = []

        try:
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    embeddings.append(np.zeros(300))
                    continue

                # Tokenize and get word embeddings
                words = text.split()
                word_embeddings = []

                for word in words:
                    if word in self.glove_model:
                        word_embeddings.append(self.glove_model[word])

                if word_embeddings:
                    # Average word embeddings to get document embedding
                    doc_embedding = np.mean(word_embeddings, axis=0)
                else:
                    # Use zero embedding if no words found
                    doc_embedding = np.zeros(300)

                embeddings.append(doc_embedding)

            # Convert to Polars DataFrame
            embeddings_array = np.vstack(embeddings)
            glove_data = {
                f"glove_{i}": embeddings_array[:, i]
                for i in range(embeddings_array.shape[1])
            }

            glove_df = pl.DataFrame(glove_data)
            logger.info(f"GloVe embeddings extracted: {glove_df.shape}")
            return glove_df

        except Exception as e:
            logger.error(f"Error during GloVe embedding extraction: {e}")
            return pl.DataFrame()

    def extract_topic_features(
        self, texts: List[str], n_topics: int = 10
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Extract topic features using LDA and NMF as described in thesis.

        From thesis: "Topic modeling using LDA and NMF"

        Args:
            texts: List of preprocessed text documents
            n_topics: Number of topics to extract (thesis uses 10)

        Returns:
            Tuple of (lda_topics_df, nmf_topics_df) as Polars DataFrames
        """
        logger.info(f"Extracting topic features with {n_topics} topics using Polars")

        try:
            # Use TF-IDF for topic modeling input
            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)

            if tfidf_matrix.shape[1] == 0:
                logger.warning("Empty TF-IDF matrix, returning empty DataFrames")
                return pl.DataFrame(), pl.DataFrame()

            # LDA topic modeling
            logger.info("Training LDA model...")
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method="online",
                random_state=42,
                n_jobs=-1,
            )
            lda_topics = lda_model.fit_transform(tfidf_matrix)

            # NMF topic modeling
            logger.info("Training NMF model...")
            nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=1000)
            nmf_topics = nmf_model.fit_transform(tfidf_matrix)

            # Convert to Polars DataFrames
            lda_data = {f"lda_topic_{i}": lda_topics[:, i] for i in range(n_topics)}
            nmf_data = {f"nmf_topic_{i}": nmf_topics[:, i] for i in range(n_topics)}

            lda_df = pl.DataFrame(lda_data)
            nmf_df = pl.DataFrame(nmf_data)

            # Save models
            lda_path = os.path.join(self.models_dir, "lda_model.pkl")
            nmf_path = os.path.join(self.models_dir, "nmf_model.pkl")

            with open(lda_path, "wb") as f:
                pickle.dump(lda_model, f)
            with open(nmf_path, "wb") as f:
                pickle.dump(nmf_model, f)

            logger.info(
                f"Topic features extracted - LDA: {lda_df.shape}, NMF: {nmf_df.shape}"
            )
            return lda_df, nmf_df

        except Exception as e:
            logger.error(f"Error during topic modeling: {e}")
            return pl.DataFrame(), pl.DataFrame()

    def extract_sentiment_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract sentiment features using DistilBERT as described in thesis.

        From thesis: "Sentiment analysis using DistilBERT"

        Args:
            texts: List of preprocessed text documents

        Returns:
            Polars DataFrame with sentiment features (positive/negative probabilities)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Sentiment analysis not available, returning empty DataFrame"
            )
            return pl.DataFrame()

        logger.info(
            f"Extracting sentiment features for {len(texts)} texts using Polars"
        )

        try:
            # Initialize sentiment pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True,
            )

            sentiments = []
            batch_size = 32

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Filter empty texts
                processed_batch = []
                for text in batch_texts:
                    if isinstance(text, str) and text.strip():
                        processed_batch.append(text)
                    else:
                        processed_batch.append("neutral")  # Placeholder for empty text

                # Get sentiment scores
                batch_results = sentiment_pipeline(processed_batch)

                for result in batch_results:
                    # Extract positive and negative probabilities
                    pos_score = next(
                        r["score"] for r in result if r["label"] == "POSITIVE"
                    )
                    neg_score = next(
                        r["score"] for r in result if r["label"] == "NEGATIVE"
                    )
                    sentiments.append([pos_score, neg_score])

            # Convert to Polars DataFrame
            sentiment_data = {
                "sentiment_positive": [s[0] for s in sentiments],
                "sentiment_negative": [s[1] for s in sentiments],
            }

            sentiment_df = pl.DataFrame(sentiment_data)
            logger.info(f"Sentiment features extracted: {sentiment_df.shape}")
            return sentiment_df

        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            return pl.DataFrame()

    def extract_all_features(
        self,
        df: pl.DataFrame,
        text_columns: List[str] = ["desc_1", "desc_2", "desc_3"],
        n_topics: int = 10,
    ) -> pl.DataFrame:
        """
        Extract all features for the coffee reviews dataset using Polars as described in thesis.

        This method implements the complete feature extraction pipeline:
        "A diverse set of features, including flavor attributes, categorical variables
        such as country of origin and roast level, and text-based features derived from
        BERT embeddings, GloVe vectors, and LDA topics, were used to predict coffee ratings."

        Args:
            df: Polars DataFrame containing coffee review data
            text_columns: List of text columns to process
            n_topics: Number of topics for topic modeling

        Returns:
            Polars DataFrame with all extracted features added
        """
        logger.info(
            f"Starting comprehensive feature extraction for columns: {text_columns} using Polars"
        )

        # Start with the original DataFrame
        result_df = df.clone()
        feature_dfs = []

        for col in text_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame, skipping")
                continue

            logger.info(f"Processing column: {col}")
            texts = df[col].fill_null("").to_list()

            # 1. TF-IDF Features
            logger.info(f"Extracting TF-IDF features for {col}")
            tfidf_df, vectorizer = self.extract_tfidf_features(texts)
            if not tfidf_df.is_empty():
                # Rename columns to include source column
                tfidf_df = tfidf_df.rename(
                    {name: f"{col}_{name}" for name in tfidf_df.columns}
                )
                feature_dfs.append(tfidf_df)

            # 2. BERT Embeddings
            logger.info(f"Extracting BERT embeddings for {col}")
            bert_df = self.extract_bert_embeddings(texts)
            if not bert_df.is_empty():
                # Rename columns to include source column
                bert_df = bert_df.rename(
                    {name: f"{col}_{name}" for name in bert_df.columns}
                )
                feature_dfs.append(bert_df)

            # 3. GloVe Embeddings
            logger.info(f"Extracting GloVe embeddings for {col}")
            glove_df = self.extract_glove_embeddings(texts)
            if not glove_df.is_empty():
                # Rename columns to include source column
                glove_df = glove_df.rename(
                    {name: f"{col}_{name}" for name in glove_df.columns}
                )
                feature_dfs.append(glove_df)

            # 4. Topic Features
            logger.info(f"Extracting topic features for {col}")
            lda_df, nmf_df = self.extract_topic_features(texts, n_topics)
            if not lda_df.is_empty():
                lda_df = lda_df.rename(
                    {name: f"{col}_{name}" for name in lda_df.columns}
                )
                feature_dfs.append(lda_df)
            if not nmf_df.is_empty():
                nmf_df = nmf_df.rename(
                    {name: f"{col}_{name}" for name in nmf_df.columns}
                )
                feature_dfs.append(nmf_df)

            # 5. Sentiment Features
            logger.info(f"Extracting sentiment features for {col}")
            sentiment_df = self.extract_sentiment_features(texts)
            if not sentiment_df.is_empty():
                sentiment_df = sentiment_df.rename(
                    {name: f"{col}_{name}" for name in sentiment_df.columns}
                )
                feature_dfs.append(sentiment_df)

            logger.info(f"Completed feature extraction for {col}")

        # Combine all feature DataFrames using Polars horizontal concatenation
        if feature_dfs:
            # Add row indices to ensure proper alignment
            for i, feature_df in enumerate(feature_dfs):
                feature_dfs[i] = feature_df.with_row_count("row_idx")

            # Concatenate horizontally
            combined_features = feature_dfs[0]
            for feature_df in feature_dfs[1:]:
                combined_features = combined_features.join(
                    feature_df, on="row_idx", how="inner"
                )

            # Remove the row index column
            combined_features = combined_features.drop("row_idx")

            # Add row indices to original DataFrame and join
            result_df = result_df.with_row_count("row_idx")
            combined_features = combined_features.with_row_count("row_idx")

            result_df = result_df.join(
                combined_features, on="row_idx", how="inner"
            ).drop("row_idx")

        logger.info(
            f"Feature extraction complete using Polars. Final shape: {result_df.shape}"
        )
        return result_df


def extract_features_from_data(input_file: str, output_file: str, n_topics: int = 10):
    """
    Main function to extract features from preprocessed data using Polars.

    Args:
        input_file: Path to preprocessed data file
        output_file: Path to save features
        n_topics: Number of topics for topic modeling
    """
    try:
        # Load preprocessed data using Polars
        logger.info(f"Loading preprocessed data from {input_file} using Polars")
        df = pl.read_csv(input_file)

        if df.is_empty():
            raise ValueError(f"No data found in {input_file}")

        # Initialize feature extractor
        extractor = CoffeeFeatureExtractor()

        # Extract all features
        df_with_features = extractor.extract_all_features(df, n_topics=n_topics)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Save features using Polars
        df_with_features.write_csv(output_file)
        logger.info(f"Features saved to {output_file} using Polars")

        return df_with_features

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        raise

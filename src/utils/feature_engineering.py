import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    DistilBertTokenizer,
    TFDistilBertModel,
    TFDistilBertForSequenceClassification,
)
from sklearn.decomposition import LatentDirichletAllocation, NMF
import tensorflow as tf


class CoffeeFeatureEngineering:
    """
    Feature engineering for coffee review analysis. Extracts meaningful features from review text
    to predict coffee ratings and understand key quality factors.

    Features extracted:
    - TF-IDF: Captures important coffee descriptors and terminology
    - BERT embeddings: Captures semantic meaning in reviews
    - Topics: Identifies themes like origin, processing, and flavor profiles
    - Sentiment: Measures reviewer attitudes and satisfaction
    """

    def __init__(
        self,
        desc_columns: List[str] = [
            "processed_desc_1",
            "processed_desc_2",
            "processed_desc_3",
        ],
    ):
        """Initialize coffee feature engineering pipeline."""
        self.desc_columns = desc_columns

        # Initialize models
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        self.sentiment_model = TFDistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    def extract_coffee_descriptors(
        self, texts: List[str], max_features: int = 1000
    ) -> pl.DataFrame:
        """
        Extract important coffee-related terms and descriptors using TF-IDF.
        Focuses on capturing flavor notes, processing methods, and origin descriptors.
        """
        print("Extracting coffee descriptors with TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Capture compound descriptors like "bright acidity"
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Get feature names for interpretability
        feature_names = [f"descriptor_{i}" for i in range(tfidf_matrix.shape[1])]
        self.top_terms = vectorizer.get_feature_names_out()

        return pl.DataFrame(
            {
                name: tfidf_matrix[:, i].toarray().flatten()
                for i, name in enumerate(feature_names)
            }
        )

    def compute_semantic_embeddings(self, texts: List[str]) -> pl.DataFrame:
        """
        Generate BERT embeddings to capture semantic meaning in reviews.
        Useful for understanding complex flavor descriptions and overall assessment.
        """
        print("Computing semantic embeddings...")
        embeddings = []
        for text in texts:
            inputs = self.bert_tokenizer(
                text, return_tensors="tf", truncation=True, padding=True
            )
            outputs = self.bert_model(inputs)
            embedding = outputs.last_hidden_state.numpy().mean(axis=1)
            embeddings.append(embedding)

        embeddings = np.vstack(embeddings)
        return pl.DataFrame(
            {f"semantic_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        )

    def extract_review_topics(
        self, texts: List[str], n_topics: int = 10
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Extract review topics using both LDA and NMF.
        Helps identify key themes like:
        - Origin characteristics
        - Processing methods
        - Flavor profiles
        - Brewing recommendations
        """
        print("Extracting review topics...")
        vectorizer = TfidfVectorizer(max_features=1000)
        text_matrix = vectorizer.fit_transform(texts)

        # LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda.fit_transform(text_matrix)

        # NMF for additional topic insights
        nmf = NMF(n_components=n_topics, random_state=42)
        nmf_output = nmf.fit_transform(text_matrix)

        self.topic_terms = {
            "lda": [
                vectorizer.get_feature_names_out()[i]
                for i in lda.components_.argsort()[:, ::-1][:, :10]
            ],
            "nmf": [
                vectorizer.get_feature_names_out()[i]
                for i in nmf.components_.argsort()[:, ::-1][:, :10]
            ],
        }

        return (
            pl.DataFrame({f"topic_lda_{i}": lda_output[:, i] for i in range(n_topics)}),
            pl.DataFrame({f"topic_nmf_{i}": nmf_output[:, i] for i in range(n_topics)}),
        )

    def analyze_sentiment(self, texts: List[str]) -> pl.DataFrame:
        """
        Analyze sentiment in coffee reviews.
        Captures reviewer satisfaction and emotional response to coffee qualities.
        """
        print("Analyzing review sentiment...")
        sentiments = []

        for text in texts:
            if not text or text.strip() == "":
                # Append neutral sentiment for empty or missing text
                sentiments.append((0.5, 0.5))
                continue

            # Tokenize the input text
            inputs = self.bert_tokenizer(
                text, return_tensors="tf", truncation=True, padding=True
            )

            # Get sentiment model outputs
            outputs = self.sentiment_model(inputs)
            sentiment = tf.nn.softmax(outputs.logits, axis=1)

            # Extract positive and negative sentiment scores
            positive_score = float(sentiment[0][1].numpy())
            negative_score = float(sentiment[0][0].numpy())
            sentiments.append((positive_score, negative_score))

        # Convert to Polars DataFrame
        return pl.DataFrame(
            {
                "sentiment_positive": [sent[0] for sent in sentiments],
                "sentiment_negative": [sent[1] for sent in sentiments],
            }
        )

    def process_all_features(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Process all text features for the coffee reviews dataset.
        Returns a dictionary of different feature types for analysis.
        """
        features = {}

        for col in self.desc_columns:
            print(f"\nProcessing features for {col}...")
            texts = df[col].to_list()

            # Extract coffee descriptors
            descriptors_df = self.extract_coffee_descriptors(texts)
            descriptors_df = descriptors_df.rename(
                {name: f"{col}_{name}" for name in descriptors_df.columns}
            )
            features[f"descriptors_{col}"] = descriptors_df

            # Compute semantic embeddings
            embeddings_df = self.compute_semantic_embeddings(texts)
            embeddings_df = embeddings_df.rename(
                {name: f"{col}_{name}" for name in embeddings_df.columns}
            )
            features[f"embeddings_{col}"] = embeddings_df

            # Extract topics
            lda_df, nmf_df = self.extract_review_topics(texts)
            lda_df = lda_df.rename({name: f"{col}_{name}" for name in lda_df.columns})
            nmf_df = nmf_df.rename({name: f"{col}_{name}" for name in nmf_df.columns})
            features[f"topics_lda_{col}"] = lda_df
            features[f"topics_nmf_{col}"] = nmf_df

            # Analyze sentiment
            sentiment_df = self.analyze_sentiment(texts)
            sentiment_df = sentiment_df.rename(
                {name: f"{col}_{name}" for name in sentiment_df.columns}
            )
            features[f"sentiment_{col}"] = sentiment_df

        return features


def load_coffee_data(
    embeddings_path: Path, topic_modeling_path: Path, sentiment_path: Path
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load preprocessed coffee review data from parquet files."""
    print("Loading preprocessed coffee review data...")
    df_embeddings = pl.read_parquet(embeddings_path)
    df_topic = pl.read_parquet(topic_modeling_path)
    df_sentiment = pl.read_parquet(sentiment_path)

    return df_embeddings, df_topic, df_sentiment

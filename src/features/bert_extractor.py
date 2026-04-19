"""
BERT embeddings extractor for coffee review text analysis.

This module implements BERT embeddings extraction following the thesis methodology:
- DistilBERT model for 768-dimensional embeddings
- Efficient batch processing
- Polars DataFrame output for modern data processing
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from .base import BaseVectorExtractor, ExtractorError

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    import torch
    from transformers import DistilBertModel, DistilBertTokenizer

    TRANSFORMERS_AVAILABLE = True
    TorchTensor = torch.Tensor
    logger.info("Transformers available - BERT embeddings enabled")
except ImportError:
    logger.warning("Transformers not installed. BERT features will be limited.")
    TRANSFORMERS_AVAILABLE = False
    TorchTensor = Any  # Fallback type when torch is not available


class BertExtractor(BaseVectorExtractor):
    """
    BERT embeddings extractor following thesis methodology.

    From thesis: "BERT embeddings using DistilBERT (768-dimensional vectors)"

    This extractor produces dense semantic representations using DistilBERT
    and outputs results as Polars DataFrames for efficient processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BERT extractor.

        Args:
            config: Configuration dictionary with parameters:
                - model_name: BERT model name (default: 'distilbert-base-uncased')
                - max_length: Maximum sequence length (default: 512)
                - batch_size: Batch size for processing (default: 16)
                - device: Device to use ('cpu' or 'cuda', default: auto-detect)
                - pooling_strategy: How to pool token embeddings (default: 'mean')
        """
        super().__init__(config)

        # Set default configuration
        default_config = {
            "model_name": "distilbert-base-uncased",
            "max_length": 512,
            "batch_size": 16,
            "pooling_strategy": "mean",  # 'mean', 'cls', 'max'
        }

        if TRANSFORMERS_AVAILABLE:
            default_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            default_config["device"] = "cpu"

        default_config.update(self.config)
        self.config = default_config

        # Initialize models
        self.tokenizer_ = None
        self.model_ = None
        self.vector_dimension = 768  # DistilBERT dimension

        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            logger.warning(
                "Transformers not available. BERT extractor will be limited."
            )

    def _load_models(self) -> None:
        """Load BERT tokenizer and model."""
        if not TRANSFORMERS_AVAILABLE:
            return

        try:
            logger.info(f"Loading BERT models: {self.config['model_name']}")

            self.tokenizer_ = DistilBertTokenizer.from_pretrained(
                self.config["model_name"]
            )
            self.model_ = DistilBertModel.from_pretrained(self.config["model_name"])

            # Move model to device
            self.model_.to(self.config["device"])
            self.model_.eval()  # Set to evaluation mode

            logger.info(
                f"BERT models loaded successfully on device: {self.config['device']}"
            )

        except Exception as e:
            logger.error(f"Failed to load BERT models: {e}")
            raise ExtractorError(f"Failed to load BERT models: {e}")

    def fit(self, texts: List[str]) -> "BertExtractor":
        """
        Fit the BERT extractor (no training needed for pre-trained models).

        Args:
            texts: List of training texts (not used for BERT)

        Returns:
            Self for method chaining
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Transformers not available. BERT extractor cannot be fitted."
            )
            return self

        # BERT is pre-trained, so we just mark as fitted
        self.is_fitted = True
        self.feature_names_ = [f"bert_{i}" for i in range(self.vector_dimension)]

        logger.info("BERT extractor fitted (using pre-trained model)")
        return self

    def extract_features(self, texts: List[str]) -> pl.DataFrame:
        """
        Extract BERT embeddings from texts.

        Args:
            texts: List of texts to process

        Returns:
            Polars DataFrame with BERT embedding features (768 columns)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("BERT not available, returning empty DataFrame")
            return pl.DataFrame()

        if not self.is_fitted:
            raise ExtractorError(
                "BERT extractor must be fitted before feature extraction"
            )

        if not texts or all(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            logger.warning("Empty or invalid text input for BERT extraction")
            return pl.DataFrame()

        logger.info(f"Extracting BERT embeddings for {len(texts)} texts")

        try:
            embeddings = []
            batch_size = self.config["batch_size"]

            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = self._extract_batch_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

            # Convert to Polars DataFrame
            embeddings_array = np.array(embeddings)
            feature_data = {
                f"bert_{i}": embeddings_array[:, i]
                for i in range(embeddings_array.shape[1])
            }

            bert_df = pl.DataFrame(feature_data)
            logger.info(
                f"Created Polars DataFrame with BERT embeddings: {bert_df.shape}"
            )

            return bert_df

        except Exception as e:
            logger.error(f"Error extracting BERT embeddings: {e}")
            raise ExtractorError(f"Failed to extract BERT embeddings: {e}")

    def _extract_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Extract embeddings for a batch of texts.

        Args:
            texts: Batch of texts to process

        Returns:
            List of embedding arrays
        """
        if not TRANSFORMERS_AVAILABLE:
            return []

        try:
            # Tokenize texts
            encoded = self.tokenizer_(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoded["input_ids"].to(self.config["device"])
            attention_mask = encoded["attention_mask"].to(self.config["device"])

            # Get embeddings
            with torch.no_grad():
                outputs = self.model_(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                hidden_states = (
                    outputs.last_hidden_state
                )  # (batch_size, seq_len, hidden_size)

            # Apply pooling strategy
            embeddings = self._apply_pooling(hidden_states, attention_mask)

            # Convert to numpy and return
            return embeddings.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Error in batch embedding extraction: {e}")
            raise ExtractorError(f"Failed to extract batch embeddings: {e}")

    def _apply_pooling(
        self, hidden_states: TorchTensor, attention_mask: TorchTensor
    ) -> TorchTensor:
        """
        Apply pooling strategy to hidden states.

        Args:
            hidden_states: Hidden states from BERT (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ExtractorError("Transformers not available for pooling")

        strategy = self.config["pooling_strategy"]

        if strategy == "cls":
            # Use [CLS] token embedding
            return hidden_states[:, 0, :]

        elif strategy == "mean":
            # Mean pooling with attention mask
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif strategy == "max":
            # Max pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            hidden_states[input_mask_expanded == 0] = (
                -1e9
            )  # Set padding tokens to large negative value
            return torch.max(hidden_states, 1)[0]

        else:
            raise ExtractorError(f"Unknown pooling strategy: {strategy}")

    def get_feature_names(self) -> List[str]:
        """Get BERT feature names."""
        return self.feature_names_.copy()

    def get_vector_dimension(self) -> int:
        """Get the BERT embedding dimension."""
        return self.vector_dimension

    def get_feature_count(self) -> int:
        """Get the number of BERT features."""
        return self.vector_dimension

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the BERT model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config["model_name"],
            "vector_dimension": self.vector_dimension,
            "max_length": self.config["max_length"],
            "device": self.config["device"],
            "pooling_strategy": self.config["pooling_strategy"],
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "is_fitted": self.is_fitted,
        }

    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode a single text to BERT embedding.

        Args:
            text: Text to encode

        Returns:
            BERT embedding array
        """
        if not self.is_fitted:
            raise ExtractorError("BERT extractor must be fitted before encoding")

        embeddings = self._extract_batch_embeddings([text])
        return (
            np.array(embeddings[0]) if embeddings else np.zeros(self.vector_dimension)
        )

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using BERT embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score
        """
        emb1 = self.encode_single_text(text1)
        emb2 = self.encode_single_text(text2)

        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

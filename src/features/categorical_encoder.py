"""
Categorical Feature Encoder for Coffee Analytics

This module implements categorical feature encoding following thesis methodology:
- One-hot encoding for low cardinality variables (roast level)
- Top-K encoding for medium cardinality variables (country_of_origin)
- Frequency-based grouping for high cardinality variables (roaster)

Author: Coffee Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

logger = logging.getLogger(__name__)


class CategoricalFeatureEncoder:
    """
    Categorical feature encoder optimized for coffee analytics dataset.

    Implements thesis methodology for categorical variable encoding:
    - roast: Standard one-hot encoding (5 categories)
    - country_of_origin: Top-K categories + "Other" (reduce dimensionality)
    - roaster: Frequency-based grouping (handle high cardinality)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize categorical encoder with configuration.

        Args:
            config: Configuration dictionary with encoding parameters
        """
        self.config = config or {}

        # Default configuration following thesis methodology
        self.default_config = {
            "roast": {
                "method": "onehot",
                "handle_unknown": "ignore",
                "prefix": "roast",
            },
            "country_of_origin": {
                "method": "top_k",
                "top_k": 15,  # Top 15 countries cover most variance
                "other_label": "Other",
                "prefix": "country",
            },
            "roaster": {
                "method": "frequency_grouping",
                "min_frequency": 3,  # Group roasters with <3 reviews
                "other_label": "Other_Roaster",
                "prefix": "roaster",
            },
        }

        # Merge configurations
        for col, default_config in self.default_config.items():
            if col not in self.config:
                self.config[col] = default_config
            else:
                # Update with defaults for missing keys
                for key, value in default_config.items():
                    if key not in self.config[col]:
                        self.config[col][key] = value

        # Initialize encoders and mappings
        self.encoders = {}
        self.feature_names = {}
        self.category_mappings = {}
        self.is_fitted = False

        logger.info("Initialized CategoricalFeatureEncoder with thesis methodology")
        logger.info(f"Configuration: {self.config}")

    def fit(self, df: pd.DataFrame) -> "CategoricalFeatureEncoder":
        """
        Fit the encoder on the dataset.

        Args:
            df: Input DataFrame with categorical columns

        Returns:
            Self for method chaining
        """
        logger.info("Fitting categorical encoder on dataset")
        logger.info(f"Dataset shape: {df.shape}")

        categorical_columns = ["roaster", "country_of_origin", "roast"]
        available_columns = [col for col in categorical_columns if col in df.columns]

        logger.info(f"Processing categorical columns: {available_columns}")

        for col in available_columns:
            logger.info(f"Fitting encoder for column: {col}")
            config = self.config[col]

            if config["method"] == "onehot":
                self._fit_onehot(df, col, config)
            elif config["method"] == "top_k":
                self._fit_top_k(df, col, config)
            elif config["method"] == "frequency_grouping":
                self._fit_frequency_grouping(df, col, config)
            else:
                raise ValueError(f"Unknown encoding method: {config['method']}")

        self.is_fitted = True
        logger.info("Categorical encoder fitting completed")
        return self

    def _fit_onehot(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit one-hot encoder for a column."""
        # Handle missing values
        values = df[col].fillna("Missing").values.reshape(-1, 1)

        # Create and fit encoder
        encoder = OneHotEncoder(
            drop=None,  # Keep all categories for thesis compliance
            sparse_output=False,
            handle_unknown=config["handle_unknown"],
        )
        encoder.fit(values)

        # Store encoder and feature names
        self.encoders[col] = encoder
        feature_names = [f"{config['prefix']}_{cat}" for cat in encoder.categories_[0]]
        self.feature_names[col] = feature_names

        logger.info(f"One-hot encoder for {col}: {len(feature_names)} features")
        logger.debug(f"Feature names: {feature_names}")

    def _fit_top_k(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit top-K encoder for a column."""
        # Get value counts
        value_counts = df[col].fillna("Missing").value_counts()

        # Select top K categories
        top_k = config["top_k"]
        top_categories = value_counts.head(top_k).index.tolist()

        # Create mapping: top categories keep their names, others become "Other"
        def map_category(value):
            if pd.isna(value):
                return "Missing"
            return value if value in top_categories else config["other_label"]

        # Test the mapping
        mapped_values = df[col].apply(map_category).values.reshape(-1, 1)

        # Create and fit one-hot encoder on mapped values
        encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
        encoder.fit(mapped_values)

        # Store encoder, mapping, and feature names
        self.encoders[col] = encoder
        self.category_mappings[col] = map_category
        feature_names = [f"{config['prefix']}_{cat}" for cat in encoder.categories_[0]]
        self.feature_names[col] = feature_names

        logger.info(
            f"Top-K encoder for {col}: {len(top_categories)} + Other = {len(feature_names)} features"
        )
        logger.debug(f"Top categories: {top_categories}")

    def _fit_frequency_grouping(
        self, df: pd.DataFrame, col: str, config: Dict[str, Any]
    ) -> None:
        """Fit frequency-based grouping encoder for a column."""
        # Get value counts
        value_counts = df[col].fillna("Missing").value_counts()

        # Select categories with minimum frequency
        min_freq = config["min_frequency"]
        frequent_categories = value_counts[value_counts >= min_freq].index.tolist()

        # Create mapping
        def map_category(value):
            if pd.isna(value):
                return "Missing"
            return value if value in frequent_categories else config["other_label"]

        # Test the mapping
        mapped_values = df[col].apply(map_category).values.reshape(-1, 1)

        # Create and fit one-hot encoder on mapped values
        encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
        encoder.fit(mapped_values)

        # Store encoder, mapping, and feature names
        self.encoders[col] = encoder
        self.category_mappings[col] = map_category
        feature_names = [f"{config['prefix']}_{cat}" for cat in encoder.categories_[0]]
        self.feature_names[col] = feature_names

        logger.info(
            f"Frequency grouping for {col}: {len(frequent_categories)} + Other = {len(feature_names)} features"
        )
        logger.debug(f"Frequent categories: {frequent_categories}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset using fitted encoders.

        Args:
            df: Input DataFrame with categorical columns

        Returns:
            DataFrame with encoded categorical features
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transformation")

        logger.info("Transforming categorical features")

        # Start with original DataFrame
        result_df = df.copy()

        # Process each fitted column
        for col in self.encoders.keys():
            if col not in df.columns:
                logger.warning(
                    f"Column {col} not found in DataFrame for transformation"
                )
                continue

            logger.debug(f"Transforming column: {col}")

            # Apply category mapping if needed
            if col in self.category_mappings:
                mapped_values = (
                    df[col].apply(self.category_mappings[col]).values.reshape(-1, 1)
                )
            else:
                mapped_values = df[col].fillna("Missing").values.reshape(-1, 1)

            # Transform using fitted encoder
            encoded_values = self.encoders[col].transform(mapped_values)

            # Create DataFrame with proper feature names
            encoded_df = pd.DataFrame(
                encoded_values, columns=self.feature_names[col], index=df.index
            )

            # Add to result DataFrame
            result_df = pd.concat([result_df, encoded_df], axis=1)

            logger.debug(f"Added {len(self.feature_names[col])} features for {col}")

        logger.info(f"Categorical encoding completed: {result_df.shape}")
        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder and transform the dataset in one step.

        Args:
            df: Input DataFrame with categorical columns

        Returns:
            DataFrame with encoded categorical features
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the feature names for each categorical column.

        Returns:
            Dictionary mapping column names to lists of feature names
        """
        return self.feature_names.copy()

    def get_encoding_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of the encoding for each column.

        Returns:
            Dictionary with encoding summary information
        """
        if not self.is_fitted:
            return {}

        summary = {}
        for col in self.encoders.keys():
            summary[col] = {
                "method": self.config[col]["method"],
                "n_features": len(self.feature_names[col]),
                "feature_names": self.feature_names[col],
                "config": self.config[col],
            }

        return summary

    def save(self, filepath: str) -> None:
        """
        Save the fitted encoder to a file.

        Args:
            filepath: Path to save the encoder
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted encoder")

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "encoders": self.encoders,
                    "feature_names": self.feature_names,
                    "category_mappings": self.category_mappings,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

        logger.info(f"Categorical encoder saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CategoricalFeatureEncoder":
        """
        Load a fitted encoder from a file.

        Args:
            filepath: Path to load the encoder from

        Returns:
            Loaded CategoricalFeatureEncoder instance
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Create new instance
        encoder = cls(data["config"])
        encoder.encoders = data["encoders"]
        encoder.feature_names = data["feature_names"]
        encoder.category_mappings = data["category_mappings"]
        encoder.is_fitted = data["is_fitted"]

        logger.info(f"Categorical encoder loaded from {filepath}")
        return encoder

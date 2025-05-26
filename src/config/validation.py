"""
Configuration Validation Utilities

This module provides validation functions for configuration settings to ensure
they are consistent, valid, and compatible with the project requirements.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

from .settings import Config, PathConfig, ModelConfig, FeatureConfig, DataConfig


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    pass


class ConfigValidator:
    """Validates configuration settings and provides recommendations."""

    def __init__(self, config: Config):
        """
        Initialize validator with configuration instance.

        Args:
            config: Configuration instance to validate
        """
        self.config = config
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all configuration components.

        Returns:
            Tuple of (is_valid, warnings, errors)
        """
        self.warnings.clear()
        self.errors.clear()

        # Validate each component
        self._validate_paths()
        self._validate_models()
        self._validate_features()
        self._validate_data()
        self._validate_environment()

        is_valid = len(self.errors) == 0

        return is_valid, self.warnings.copy(), self.errors.copy()

    def _validate_paths(self):
        """Validate path configuration."""
        paths = self.config.paths

        # Check if root path exists and is accessible
        if not paths.root.exists():
            self.errors.append(f"Root path does not exist: {paths.root}")
        elif not os.access(paths.root, os.R_OK):
            self.errors.append(f"Root path is not readable: {paths.root}")

        # Check data file existence
        raw_data_path = paths.get_raw_data_path()
        if not raw_data_path.exists():
            self.warnings.append(f"Raw data file not found: {raw_data_path}")

        # Check write permissions for output directories
        output_dirs = [paths.processed, paths.models, paths.output, paths.figures]
        for output_dir in output_dirs:
            if output_dir.exists() and not os.access(output_dir, os.W_OK):
                self.errors.append(f"Output directory is not writable: {output_dir}")

    def _validate_models(self):
        """Validate model configuration."""
        models = self.config.models

        # Validate target column
        if not models.target_column:
            self.errors.append("Target column cannot be empty")

        # Validate text columns
        if not models.text_columns:
            self.errors.append("Text columns list cannot be empty")
        elif len(models.text_columns) > 10:
            self.warnings.append(
                f"Large number of text columns ({len(models.text_columns)}) may slow processing"
            )

        # Validate models to train
        valid_models = {
            "linear",
            "random_forest",
            "xgboost",
            "svr",
            "decision_tree",
            "mnir",
        }
        invalid_models = set(models.models_to_train) - valid_models
        if invalid_models:
            self.errors.append(f"Invalid model names: {invalid_models}")

        # Validate cross-validation settings
        if models.cv_folds < 2:
            self.errors.append("CV folds must be at least 2")
        elif models.cv_folds > 20:
            self.warnings.append(
                f"High CV folds ({models.cv_folds}) may be computationally expensive"
            )

        if not 0.1 <= models.test_size <= 0.5:
            self.warnings.append(
                f"Test size ({models.test_size}) outside recommended range [0.1, 0.5]"
            )

        # Validate hyperparameters
        self._validate_model_params("random_forest", models.random_forest_params)
        self._validate_model_params("xgboost", models.xgboost_params)
        self._validate_model_params("linear", models.linear_params)
        self._validate_model_params("svr", models.svr_params)
        self._validate_model_params("decision_tree", models.decision_tree_params)
        self._validate_model_params("mnir", models.mnir_params)

    def _validate_model_params(self, model_name: str, params: Dict[str, Any]):
        """Validate parameters for a specific model."""
        if model_name == "random_forest":
            if params.get("n_estimators", 0) < 10:
                self.warnings.append("Random Forest: Low n_estimators may underfit")
            elif params.get("n_estimators", 0) > 1000:
                self.warnings.append("Random Forest: High n_estimators may be slow")

            if params.get("max_depth") and params.get("max_depth") < 3:
                self.warnings.append("Random Forest: Very shallow trees may underfit")

        elif model_name == "xgboost":
            if (
                params.get("learning_rate", 0) <= 0
                or params.get("learning_rate", 0) > 1
            ):
                self.errors.append("XGBoost: Learning rate must be in (0, 1]")

            if params.get("n_estimators", 0) < 10:
                self.warnings.append("XGBoost: Low n_estimators may underfit")

        elif model_name == "linear":
            if params.get("alpha", 0) < 0:
                self.errors.append("Linear model: Alpha must be non-negative")

        elif model_name == "mnir":
            if params.get("alpha", 0) <= 0:
                self.errors.append("MNIR: Alpha must be positive")

            if params.get("max_iter", 0) < 100:
                self.warnings.append("MNIR: Low max_iter may not converge")

    def _validate_features(self):
        """Validate feature extraction configuration."""
        features = self.config.features

        # Validate TF-IDF parameters
        if features.tfidf_max_features < 100:
            self.warnings.append("TF-IDF: Very low max_features may lose information")
        elif features.tfidf_max_features > 50000:
            self.warnings.append(
                "TF-IDF: Very high max_features may be memory intensive"
            )

        if features.tfidf_min_df < 1:
            self.errors.append("TF-IDF: min_df must be at least 1")

        if not 0 < features.tfidf_max_df <= 1:
            self.errors.append("TF-IDF: max_df must be in (0, 1]")

        # Validate n-gram range
        ngram_min, ngram_max = features.tfidf_ngram_range
        if ngram_min < 1 or ngram_max < ngram_min:
            self.errors.append("TF-IDF: Invalid n-gram range")
        elif ngram_max > 3:
            self.warnings.append("TF-IDF: High n-gram range may create sparse features")

        # Validate topic modeling
        if features.n_topics < 2:
            self.errors.append("Topic modeling: Must have at least 2 topics")
        elif features.n_topics > 50:
            self.warnings.append(
                "Topic modeling: High number of topics may be hard to interpret"
            )

        # Validate BERT parameters
        if features.bert_max_length < 50:
            self.warnings.append(
                "BERT: Very low max_length may truncate important text"
            )
        elif features.bert_max_length > 512:
            self.warnings.append(
                "BERT: max_length > 512 may not be supported by all models"
            )

        if features.bert_batch_size < 1:
            self.errors.append("BERT: Batch size must be at least 1")
        elif features.bert_batch_size > 64:
            self.warnings.append("BERT: Large batch size may cause memory issues")

        # Validate text preprocessing
        if features.min_word_length < 1:
            self.errors.append("Text preprocessing: min_word_length must be at least 1")

        if features.max_word_length < features.min_word_length:
            self.errors.append(
                "Text preprocessing: max_word_length must be >= min_word_length"
            )

    def _validate_data(self):
        """Validate data processing configuration."""
        data = self.config.data

        # Validate rating threshold
        if not 0 <= data.min_rating <= 100:
            self.errors.append("Data: min_rating must be between 0 and 100")
        elif data.min_rating > 95:
            self.warnings.append(
                "Data: Very high min_rating may result in too few samples"
            )

        # Validate missing data threshold
        if not 0 <= data.max_missing_percentage <= 100:
            self.errors.append("Data: max_missing_percentage must be between 0 and 100")

        # Validate text length constraints
        if data.min_text_length < 1:
            self.errors.append("Data: min_text_length must be at least 1")

        if data.max_text_length < data.min_text_length:
            self.errors.append("Data: max_text_length must be >= min_text_length")
        elif data.max_text_length > 10000:
            self.warnings.append("Data: Very high max_text_length may include noise")

    def _validate_environment(self):
        """Validate environment-specific settings."""
        env = self.config.environment

        valid_environments = {"development", "production", "testing"}
        if env not in valid_environments:
            self.warnings.append(
                f"Unknown environment: {env}. Valid options: {valid_environments}"
            )

        # Environment-specific validations
        if env == "production":
            if self.config.logging.level == "DEBUG":
                self.warnings.append("Production: DEBUG logging may impact performance")

            if self.config.models.cv_folds < 5:
                self.warnings.append(
                    "Production: Low CV folds may not be robust enough"
                )

        elif env == "testing":
            if self.config.features.n_topics > 10:
                self.warnings.append("Testing: High n_topics may slow down tests")


def validate_config(config: Config, raise_on_error: bool = False) -> bool:
    """
    Validate configuration and optionally raise on errors.

    Args:
        config: Configuration instance to validate
        raise_on_error: Whether to raise exception on validation errors

    Returns:
        bool: True if configuration is valid

    Raises:
        ConfigValidationError: If raise_on_error=True and validation fails
    """
    validator = ConfigValidator(config)
    is_valid, warnings, errors = validator.validate_all()

    # Log warnings
    for warning in warnings:
        logger.warning(f"Config validation warning: {warning}")

    # Log or raise errors
    for error in errors:
        if raise_on_error:
            raise ConfigValidationError(f"Config validation error: {error}")
        else:
            logger.error(f"Config validation error: {error}")

    if is_valid:
        logger.info("Configuration validation passed")
    else:
        logger.error(f"Configuration validation failed with {len(errors)} errors")

    return is_valid


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if required dependencies are available.

    Returns:
        Tuple of (all_available, missing_packages)
    """
    # Core required packages
    core_packages = [
        "polars",
        "pandas",
        "numpy",
        "sklearn",  # scikit-learn imports as sklearn
        "plotly",
        "nltk",
        "xgboost",
    ]

    # Optional packages (for advanced features)
    optional_packages = [
        "transformers",  # For BERT features
        "torch",  # For BERT/transformers
        "gensim",  # For GloVe embeddings
    ]

    missing_core = []
    missing_optional = []

    for package in core_packages:
        try:
            __import__(package)
        except ImportError:
            missing_core.append(package)

    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)

    all_core_available = len(missing_core) == 0

    if missing_core:
        logger.error(f"Missing required core packages: {missing_core}")
    if missing_optional:
        logger.warning(
            f"Missing optional packages (some features will be limited): {missing_optional}"
        )

    if all_core_available:
        logger.info("All core dependencies are available")

    return all_core_available, missing_core


def get_config_summary(config: Config) -> Dict[str, Any]:
    """
    Get a summary of the current configuration.

    Args:
        config: Configuration instance

    Returns:
        Dictionary with configuration summary
    """
    return {
        "environment": config.environment,
        "data_file": str(config.paths.get_raw_data_path()),
        "target_column": config.models.target_column,
        "text_columns": config.models.text_columns,
        "models_to_train": config.models.models_to_train,
        "n_topics": config.features.n_topics,
        "tfidf_max_features": config.features.tfidf_max_features,
        "min_rating": config.data.min_rating,
        "cv_folds": config.models.cv_folds,
        "test_size": config.models.test_size,
    }


def print_config_summary(config: Config):
    """Print a formatted configuration summary."""
    summary = get_config_summary(config)

    print("\n" + "=" * 60)
    print("COFFEE TEXT ANALYTICS - CONFIGURATION SUMMARY")
    print("=" * 60)

    print(f"Environment: {summary['environment']}")
    print(f"Data File: {summary['data_file']}")
    print(f"Target Column: {summary['target_column']}")
    print(f"Text Columns: {', '.join(summary['text_columns'])}")
    print(f"Models to Train: {', '.join(summary['models_to_train'])}")
    print(f"Number of Topics: {summary['n_topics']}")
    print(f"TF-IDF Max Features: {summary['tfidf_max_features']}")
    print(f"Minimum Rating: {summary['min_rating']}")
    print(f"CV Folds: {summary['cv_folds']}")
    print(f"Test Size: {summary['test_size']}")
    print("=" * 60)

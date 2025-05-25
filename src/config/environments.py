"""
Environment-specific Configuration Presets

This module provides predefined configuration presets for different environments
(development, production, testing) that can be used to override default settings.
"""

from typing import Dict, Any
from .settings import Config


# Development environment configuration
DEVELOPMENT_CONFIG = {
    "data": {
        "min_rating": 80.0,
        "max_missing_percentage": 50.0,
    },
    "features": {
        "n_topics": 10,
        "tfidf_max_features": 5000,
        "bert_batch_size": 8,  # Smaller batch for development
    },
    "models": {
        "cv_folds": 5,
        "test_size": 0.2,
        "random_forest_params": {
            "n_estimators": 50,  # Faster for development
            "max_depth": 8,
            "random_state": 42,
        },
        "xgboost_params": {
            "n_estimators": 50,  # Faster for development
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    },
    "logging": {
        "level": "INFO",
        "console_handler": True,
        "file_handler": False,  # No file logging in development
    },
    "visualization": {
        "figure_width": 800,
        "figure_height": 600,
        "export_dpi": 150,  # Lower DPI for faster rendering
    },
}


# Production environment configuration
PRODUCTION_CONFIG = {
    "data": {
        "min_rating": 85.0,  # Higher quality threshold
        "max_missing_percentage": 30.0,  # Stricter data quality
    },
    "features": {
        "n_topics": 15,  # More topics for better analysis
        "tfidf_max_features": 10000,  # More features for production
        "bert_batch_size": 16,  # Larger batch for efficiency
    },
    "models": {
        "cv_folds": 10,  # More robust validation
        "test_size": 0.15,  # Smaller test set, more training data
        "random_forest_params": {
            "n_estimators": 200,  # More trees for better performance
            "max_depth": 12,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        "xgboost_params": {
            "n_estimators": 200,  # More estimators for production
            "max_depth": 8,
            "learning_rate": 0.05,  # Lower learning rate for stability
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        },
        "mnir_params": {
            "alpha": 0.05,  # More regularization
            "max_iter": 2000,  # More iterations for convergence
            "random_state": 42,
        },
    },
    "logging": {
        "level": "WARNING",  # Less verbose logging
        "console_handler": False,  # No console output in production
        "file_handler": True,
        "log_file": "production_coffee_analytics.log",
    },
    "visualization": {
        "figure_width": 1200,
        "figure_height": 800,
        "export_dpi": 300,  # High quality for production
        "export_width": 1600,
        "export_height": 1200,
    },
}


# Testing environment configuration
TESTING_CONFIG = {
    "data": {
        "min_rating": 70.0,  # Lower threshold for more test data
        "max_missing_percentage": 70.0,  # More lenient for testing
    },
    "features": {
        "n_topics": 5,  # Fewer topics for faster testing
        "tfidf_max_features": 1000,  # Smaller feature set for speed
        "bert_batch_size": 4,  # Small batch for testing
        "bert_max_length": 128,  # Shorter sequences for speed
    },
    "models": {
        "cv_folds": 3,  # Fewer folds for faster testing
        "test_size": 0.3,  # Larger test set for validation
        "random_forest_params": {
            "n_estimators": 10,  # Very few trees for speed
            "max_depth": 5,
            "random_state": 42,
        },
        "xgboost_params": {
            "n_estimators": 10,  # Very few estimators for speed
            "max_depth": 3,
            "learning_rate": 0.3,  # Higher learning rate for speed
            "random_state": 42,
        },
        "mnir_params": {
            "alpha": 0.1,
            "max_iter": 100,  # Fewer iterations for speed
            "random_state": 42,
        },
    },
    "logging": {
        "level": "DEBUG",  # Verbose logging for testing
        "console_handler": True,
        "file_handler": False,
        "log_file": "test_coffee_analytics.log",
    },
    "visualization": {
        "figure_width": 600,
        "figure_height": 400,
        "export_dpi": 100,  # Low DPI for fast testing
    },
}


# CI/CD environment configuration (for automated testing)
CICD_CONFIG = {
    "data": {
        "min_rating": 75.0,
        "max_missing_percentage": 80.0,
    },
    "features": {
        "n_topics": 3,  # Minimal topics for CI speed
        "tfidf_max_features": 500,  # Very small feature set
        "bert_batch_size": 2,  # Minimal batch size
        "bert_max_length": 64,  # Very short sequences
    },
    "models": {
        "cv_folds": 2,  # Minimal validation
        "test_size": 0.4,  # Large test set
        "models_to_train": ["linear"],  # Only fast models
        "random_forest_params": {
            "n_estimators": 5,  # Minimal trees
            "max_depth": 3,
            "random_state": 42,
        },
    },
    "logging": {
        "level": "ERROR",  # Minimal logging for CI
        "console_handler": True,
        "file_handler": False,
    },
    "visualization": {
        "figure_width": 400,
        "figure_height": 300,
        "export_dpi": 72,  # Minimal DPI
    },
}


def apply_environment_config(config: Config, environment: str) -> Config:
    """
    Apply environment-specific configuration overrides.

    Args:
        config: Base configuration instance
        environment: Environment name

    Returns:
        Config: Updated configuration instance
    """
    env_configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG,
        "cicd": CICD_CONFIG,
    }

    if environment not in env_configs:
        return config

    env_config = env_configs[environment]

    # Apply data configuration overrides
    if "data" in env_config:
        for key, value in env_config["data"].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    # Apply feature configuration overrides
    if "features" in env_config:
        for key, value in env_config["features"].items():
            if hasattr(config.features, key):
                setattr(config.features, key, value)

    # Apply model configuration overrides
    if "models" in env_config:
        for key, value in env_config["models"].items():
            if hasattr(config.models, key):
                setattr(config.models, key, value)

    # Apply logging configuration overrides
    if "logging" in env_config:
        for key, value in env_config["logging"].items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)

    # Apply visualization configuration overrides
    if "visualization" in env_config:
        for key, value in env_config["visualization"].items():
            if hasattr(config.visualization, key):
                setattr(config.visualization, key, value)

    return config


def get_environment_config(environment: str) -> Dict[str, Any]:
    """
    Get the configuration dictionary for a specific environment.

    Args:
        environment: Environment name

    Returns:
        Dictionary with environment-specific configuration
    """
    env_configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG,
        "cicd": CICD_CONFIG,
    }

    return env_configs.get(environment, {})


def list_available_environments() -> list:
    """Get list of available environment configurations."""
    return ["development", "production", "testing", "cicd"]


def create_custom_environment(
    base_environment: str = "development", overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a custom environment configuration based on an existing one.

    Args:
        base_environment: Base environment to start from
        overrides: Dictionary of configuration overrides

    Returns:
        Custom environment configuration dictionary
    """
    base_config = get_environment_config(base_environment).copy()

    if overrides:
        # Deep merge overrides
        for section, section_overrides in overrides.items():
            if section in base_config:
                base_config[section].update(section_overrides)
            else:
                base_config[section] = section_overrides

    return base_config


# Example custom environments
RESEARCH_CONFIG = create_custom_environment(
    base_environment="development",
    overrides={
        "features": {
            "n_topics": 20,  # More topics for research
            "tfidf_max_features": 15000,  # More features
        },
        "models": {
            "cv_folds": 10,  # More robust validation
            "models_to_train": [
                "linear",
                "random_forest",
                "xgboost",
                "mnir",
            ],  # All models
        },
        "logging": {
            "level": "DEBUG",  # Detailed logging for research
            "file_handler": True,
            "log_file": "research_coffee_analytics.log",
        },
    },
)


DEMO_CONFIG = create_custom_environment(
    base_environment="development",
    overrides={
        "features": {
            "n_topics": 8,  # Good balance for demo
            "tfidf_max_features": 3000,
        },
        "models": {
            "models_to_train": ["linear", "random_forest"],  # Fast models for demo
            "cv_folds": 3,
        },
        "visualization": {
            "figure_width": 1000,
            "figure_height": 700,
            "export_dpi": 200,  # Good quality for demo
        },
    },
)

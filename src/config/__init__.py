"""
Coffee Text Analytics - Configuration Module

This module provides centralized configuration management for the entire project.
Import the main configuration instance and specific configuration classes as needed.

Usage:
    from config import config
    from config.settings import Config, ModelConfig, FeatureConfig

    # Access configuration
    print(config.paths.raw_data_path)
    print(config.models.target_column)
    print(config.features.n_topics)
"""

from .settings import (
    Config,
    PathConfig,
    ModelConfig,
    FeatureConfig,
    DataConfig,
    VisualizationConfig,
    LoggingConfig,
    config,
    PATHS,
)

# Main configuration instance for easy access
__version__ = "1.0.0"

# Export everything for easy imports
__all__ = [
    "Config",
    "PathConfig",
    "ModelConfig",
    "FeatureConfig",
    "DataConfig",
    "VisualizationConfig",
    "LoggingConfig",
    "config",
    "PATHS",
    "__version__",
]

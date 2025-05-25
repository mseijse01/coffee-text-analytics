"""
Coffee Text Analytics - Centralized Exception Handling

This module provides a comprehensive exception hierarchy for the coffee text analytics project.
All custom exceptions inherit from CoffeeAnalyticsError for consistent error handling.
"""

import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class CoffeeAnalyticsError(Exception):
    """
    Base exception for all coffee analytics project errors.

    Provides consistent error handling with optional context and logging.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.ERROR,
    ):
        """
        Initialize the exception with message and optional context.

        Args:
            message: Error message
            context: Optional context dictionary for debugging
            log_level: Logging level for this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.log_level = log_level

        # Log the error with context
        if self.context:
            logger.log(log_level, f"{message} | Context: {self.context}")
        else:
            logger.log(log_level, message)


# Data-related exceptions
class DataError(CoffeeAnalyticsError):
    """Base exception for data-related errors."""

    pass


class DataLoadingError(DataError):
    """Raised when data loading fails."""

    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""

    pass


class DataPreprocessingError(DataError):
    """Raised when data preprocessing fails."""

    pass


class DataQualityError(DataError):
    """Raised when data quality issues are detected."""

    pass


# Feature extraction exceptions
class FeatureExtractionError(CoffeeAnalyticsError):
    """Base exception for feature extraction errors."""

    pass


class ExtractorNotFittedError(FeatureExtractionError):
    """Raised when trying to use an unfitted extractor."""

    pass


class ExtractorConfigError(FeatureExtractionError):
    """Raised when there's an issue with extractor configuration."""

    pass


class TfidfExtractionError(FeatureExtractionError):
    """Raised when TF-IDF extraction fails."""

    pass


class BertExtractionError(FeatureExtractionError):
    """Raised when BERT extraction fails."""

    pass


class TopicExtractionError(FeatureExtractionError):
    """Raised when topic modeling fails."""

    pass


class SentimentExtractionError(FeatureExtractionError):
    """Raised when sentiment analysis fails."""

    pass


class GloveExtractionError(FeatureExtractionError):
    """Raised when GloVe embedding extraction fails."""

    pass


# Model-related exceptions
class ModelError(CoffeeAnalyticsError):
    """Base exception for model-related errors."""

    pass


class ModelNotFittedError(ModelError):
    """Raised when trying to use an unfitted model."""

    pass


class ModelConfigError(ModelError):
    """Raised when there's an issue with model configuration."""

    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    pass


class ModelEvaluationError(ModelError):
    """Raised when model evaluation fails."""

    pass


class ModelSaveError(ModelError):
    """Raised when model saving fails."""

    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    pass


class MNIRError(ModelError):
    """Raised when MNIR-specific operations fail."""

    pass


# Configuration exceptions
class ConfigError(CoffeeAnalyticsError):
    """Base exception for configuration errors."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    pass


class ConfigLoadError(ConfigError):
    """Raised when configuration loading fails."""

    pass


class EnvironmentConfigError(ConfigError):
    """Raised when environment-specific configuration fails."""

    pass


# Visualization exceptions
class VisualizationError(CoffeeAnalyticsError):
    """Base exception for visualization errors."""

    pass


class PlotCreationError(VisualizationError):
    """Raised when plot creation fails."""

    pass


class PlotSaveError(VisualizationError):
    """Raised when plot saving fails."""

    pass


# File I/O exceptions
class FileError(CoffeeAnalyticsError):
    """Base exception for file operation errors."""

    pass


class FileNotFoundError(FileError):
    """Raised when a required file is not found."""

    pass


class FilePermissionError(FileError):
    """Raised when file permission issues occur."""

    pass


class FileSaveError(FileError):
    """Raised when file saving fails."""

    pass


class FileLoadError(FileError):
    """Raised when file loading fails."""

    pass


# Dependency exceptions
class DependencyError(CoffeeAnalyticsError):
    """Base exception for dependency-related errors."""

    pass


class MissingDependencyError(DependencyError):
    """Raised when a required dependency is missing."""

    pass


class IncompatibleDependencyError(DependencyError):
    """Raised when dependency versions are incompatible."""

    pass


# Utility functions for error handling
def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise_as: Optional[type] = None,
    message: Optional[str] = None,
) -> None:
    """
    Utility function to handle exceptions consistently.

    Args:
        exception: The original exception
        context: Optional context for debugging
        reraise_as: Optional exception class to reraise as
        message: Optional custom message
    """
    original_message = str(exception)

    if message:
        full_message = f"{message}: {original_message}"
    else:
        full_message = original_message

    if reraise_as:
        if issubclass(reraise_as, CoffeeAnalyticsError):
            raise reraise_as(full_message, context=context)
        else:
            raise reraise_as(full_message)
    else:
        # Log the exception with context
        if context:
            logger.error(f"{full_message} | Context: {context}")
        else:
            logger.error(full_message)


def validate_not_none(
    value: Any, name: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate that a value is not None.

    Args:
        value: Value to validate
        name: Name of the value for error message
        context: Optional context for debugging

    Raises:
        DataValidationError: If value is None
    """
    if value is None:
        raise DataValidationError(f"{name} cannot be None", context=context)


def validate_not_empty(
    value: Any, name: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate that a value is not empty.

    Args:
        value: Value to validate
        name: Name of the value for error message
        context: Optional context for debugging

    Raises:
        DataValidationError: If value is empty
    """
    if not value:
        raise DataValidationError(f"{name} cannot be empty", context=context)


def validate_file_exists(
    file_path: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file
        context: Optional context for debugging

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}", context=context)


def validate_directory_exists(
    dir_path: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to the directory
        context: Optional context for debugging

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    import os

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}", context=context)


def require_dependency(
    module_name: str,
    import_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Require that a dependency is available.

    Args:
        module_name: Name of the module to import
        import_name: Optional specific import name
        context: Optional context for debugging

    Raises:
        MissingDependencyError: If dependency is not available
    """
    try:
        if import_name:
            __import__(module_name, fromlist=[import_name])
        else:
            __import__(module_name)
    except ImportError as e:
        raise MissingDependencyError(
            f"Required dependency '{module_name}' is not available: {e}",
            context=context,
        )

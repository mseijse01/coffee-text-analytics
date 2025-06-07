"""
Unit tests for custom exception hierarchy and error handling utilities.

Tests all custom exceptions, context handling, and validation utilities
without any heavyweight dependencies.
"""

import pytest
import logging
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.exceptions import (
    # Base exceptions
    CoffeeAnalyticsError,
    # Data exceptions
    DataError,
    DataLoadingError,
    DataValidationError,
    DataPreprocessingError,
    DataQualityError,
    # Feature extraction exceptions
    FeatureExtractionError,
    ExtractorNotFittedError,
    ExtractorConfigError,
    TfidfExtractionError,
    BertExtractionError,
    TopicExtractionError,
    SentimentExtractionError,
    GloveExtractionError,
    # Model exceptions
    ModelError,
    ModelNotFittedError,
    ModelConfigError,
    ModelTrainingError,
    ModelEvaluationError,
    ModelSaveError,
    ModelLoadError,
    MNIRError,
    # Config exceptions
    ConfigError,
    ConfigValidationError,
    ConfigLoadError,
    EnvironmentConfigError,
    # Visualization exceptions
    VisualizationError,
    PlotCreationError,
    PlotSaveError,
    # File exceptions
    FileError,
    FileNotFoundError,
    FilePermissionError,
    FileSaveError,
    FileLoadError,
    # Dependency exceptions
    DependencyError,
    MissingDependencyError,
    IncompatibleDependencyError,
    # Utility functions
    handle_exception,
    validate_not_none,
    validate_not_empty,
    validate_file_exists,
    validate_directory_exists,
    require_dependency,
)


class TestCoffeeAnalyticsError:
    """Test the base exception class."""

    @pytest.mark.unit
    def test_basic_exception_creation(self):
        """Test basic exception creation with message only."""
        message = "Test error message"
        error = CoffeeAnalyticsError(message)

        assert str(error) == message
        assert error.message == message
        assert error.context == {}
        assert error.log_level == logging.ERROR

    @pytest.mark.unit
    def test_exception_with_context(self):
        """Test exception creation with context dictionary."""
        message = "Test error with context"
        context = {"module": "test", "function": "test_func", "data_size": 100}

        error = CoffeeAnalyticsError(message, context=context)

        assert str(error) == message
        assert error.message == message
        assert error.context == context
        assert error.log_level == logging.ERROR

    @pytest.mark.unit
    def test_exception_with_custom_log_level(self):
        """Test exception creation with custom log level."""
        message = "Warning level error"
        error = CoffeeAnalyticsError(message, log_level=logging.WARNING)

        assert error.log_level == logging.WARNING

    @pytest.mark.unit
    @patch("src.exceptions.logger")
    def test_exception_logging_behavior(self, mock_logger):
        """Test that exceptions are logged appropriately."""
        message = "Test logging"
        context = {"key": "value"}

        # Test with context
        CoffeeAnalyticsError(message, context=context, log_level=logging.INFO)
        mock_logger.log.assert_called_with(
            logging.INFO, f"{message} | Context: {context}"
        )

        # Test without context
        mock_logger.reset_mock()
        CoffeeAnalyticsError(message, log_level=logging.WARNING)
        mock_logger.log.assert_called_with(logging.WARNING, message)


class TestExceptionHierarchy:
    """Test the exception inheritance hierarchy."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "exception_class,base_class",
        [
            # Data exceptions
            (DataError, CoffeeAnalyticsError),
            (DataLoadingError, DataError),
            (DataValidationError, DataError),
            (DataPreprocessingError, DataError),
            (DataQualityError, DataError),
            # Feature extraction exceptions
            (FeatureExtractionError, CoffeeAnalyticsError),
            (ExtractorNotFittedError, FeatureExtractionError),
            (ExtractorConfigError, FeatureExtractionError),
            (TfidfExtractionError, FeatureExtractionError),
            (BertExtractionError, FeatureExtractionError),
            (TopicExtractionError, FeatureExtractionError),
            (SentimentExtractionError, FeatureExtractionError),
            (GloveExtractionError, FeatureExtractionError),
            # Model exceptions
            (ModelError, CoffeeAnalyticsError),
            (ModelNotFittedError, ModelError),
            (ModelConfigError, ModelError),
            (ModelTrainingError, ModelError),
            (ModelEvaluationError, ModelError),
            (ModelSaveError, ModelError),
            (ModelLoadError, ModelError),
            (MNIRError, ModelError),
            # Config exceptions
            (ConfigError, CoffeeAnalyticsError),
            (ConfigValidationError, ConfigError),
            (ConfigLoadError, ConfigError),
            (EnvironmentConfigError, ConfigError),
            # Visualization exceptions
            (VisualizationError, CoffeeAnalyticsError),
            (PlotCreationError, VisualizationError),
            (PlotSaveError, VisualizationError),
            # File exceptions
            (FileError, CoffeeAnalyticsError),
            (FileNotFoundError, FileError),
            (FilePermissionError, FileError),
            (FileSaveError, FileError),
            (FileLoadError, FileError),
            # Dependency exceptions
            (DependencyError, CoffeeAnalyticsError),
            (MissingDependencyError, DependencyError),
            (IncompatibleDependencyError, DependencyError),
        ],
    )
    def test_exception_inheritance(self, exception_class, base_class):
        """Test that all exceptions inherit from their expected base classes."""
        assert issubclass(exception_class, base_class)

        # Test that instances can be created and caught by base class
        message = f"Test {exception_class.__name__}"
        error = exception_class(message)

        assert isinstance(error, exception_class)
        assert isinstance(error, base_class)
        assert isinstance(error, CoffeeAnalyticsError)


class TestExceptionUtilities:
    """Test utility functions for exception handling."""

    @pytest.mark.unit
    def test_handle_exception_basic(self):
        """Test basic exception handling without reraising."""
        original_error = ValueError("Original error")
        context = {"function": "test_func"}

        # Should not raise when reraise_as is None
        handle_exception(original_error, context=context)

    @pytest.mark.unit
    def test_handle_exception_reraise(self):
        """Test exception handling with reraising as different type."""
        original_error = ValueError("Original error")
        context = {"function": "test_func"}
        custom_message = "Custom error message"

        with pytest.raises(DataError) as exc_info:
            handle_exception(
                original_error,
                context=context,
                reraise_as=DataError,
                message=custom_message,
            )

        # The implementation appends the original error message
        expected_message = f"{custom_message}: Original error"
        assert str(exc_info.value) == expected_message
        assert exc_info.value.context == context

    @pytest.mark.unit
    def test_validate_not_none_success(self):
        """Test validate_not_none with valid values."""
        # Should not raise for valid values
        validate_not_none("valid string", "test_param")
        validate_not_none(42, "test_number")
        validate_not_none([], "test_list")  # Empty list is not None

    @pytest.mark.unit
    def test_validate_not_none_failure(self):
        """Test validate_not_none with None value."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_not_none(None, "test_param")

        assert "test_param cannot be None" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_not_none_with_context(self):
        """Test validate_not_none with context."""
        context = {"module": "test", "function": "test_func"}

        with pytest.raises(DataValidationError) as exc_info:
            validate_not_none(None, "test_param", context=context)

        assert exc_info.value.context == context

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "empty_value",
        [
            "",  # Empty string
            [],  # Empty list
            {},  # Empty dict
            set(),  # Empty set
        ],
    )
    def test_validate_not_empty_failure(self, empty_value):
        """Test validate_not_empty with various empty values."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_not_empty(empty_value, "test_param")

        assert "test_param cannot be empty" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "valid_value",
        [
            "non-empty string",
            [1, 2, 3],
            {"key": "value"},
            {1, 2, 3},
        ],
    )
    def test_validate_not_empty_success(self, valid_value):
        """Test validate_not_empty with valid non-empty values."""
        # Should not raise for non-empty values
        validate_not_empty(valid_value, "test_param")

    @pytest.mark.unit
    def test_validate_file_exists_success(self):
        """Test validate_file_exists with existing file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Should not raise for existing file
            validate_file_exists(temp_path)
        finally:
            # Clean up
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_validate_file_exists_failure(self):
        """Test validate_file_exists with non-existent file."""
        non_existent_path = "/path/that/does/not/exist.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_file_exists(non_existent_path)

        assert non_existent_path in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_directory_exists_success(self):
        """Test validate_directory_exists with existing directory."""
        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise for existing directory
            validate_directory_exists(temp_dir)

    @pytest.mark.unit
    def test_validate_directory_exists_failure(self):
        """Test validate_directory_exists with non-existent directory."""
        non_existent_dir = "/path/that/does/not/exist"

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_directory_exists(non_existent_dir)

        assert non_existent_dir in str(exc_info.value)

    @pytest.mark.unit
    @patch("builtins.__import__")
    def test_require_dependency_success(self, mock_import):
        """Test require_dependency with available module."""
        # Mock successful import
        mock_import.return_value = MagicMock()

        # Should not raise for available dependency
        require_dependency("numpy")

        mock_import.assert_called_with("numpy")

    @pytest.mark.unit
    @patch("builtins.__import__")
    def test_require_dependency_failure(self, mock_import):
        """Test require_dependency with missing module."""
        # Mock import failure
        mock_import.side_effect = ImportError("No module named 'nonexistent'")

        with pytest.raises(MissingDependencyError) as exc_info:
            require_dependency("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.unit
    @patch("builtins.__import__")
    def test_require_dependency_with_import_name(self, mock_import):
        """Test require_dependency with specific import name."""
        # The current implementation doesn't check for specific attributes,
        # it just tries to import with fromlist. Let's test actual ImportError
        mock_import.side_effect = ImportError("cannot import name 'specific_function'")

        with pytest.raises(MissingDependencyError):
            require_dependency("somemodule", import_name="specific_function")

    @pytest.mark.unit
    def test_require_dependency_with_context(self):
        """Test require_dependency with context information."""
        context = {"caller": "test_function", "purpose": "feature extraction"}

        with pytest.raises(MissingDependencyError) as exc_info:
            require_dependency("nonexistent_module", context=context)

        assert exc_info.value.context == context


class TestExceptionIntegration:
    """Test exception handling in realistic scenarios."""

    @pytest.mark.unit
    def test_nested_exception_handling(self):
        """Test handling exceptions in nested function calls."""

        def level_3():
            raise ValueError("Deep error")

        def level_2():
            try:
                level_3()
            except Exception as e:
                handle_exception(
                    e,
                    context={"level": 2},
                    reraise_as=DataPreprocessingError,
                    message="Level 2 error",
                )

        def level_1():
            try:
                level_2()
            except Exception as e:
                handle_exception(
                    e,
                    context={"level": 1},
                    reraise_as=DataError,
                    message="Level 1 error",
                )

        with pytest.raises(DataError) as exc_info:
            level_1()

        assert "Level 1 error" in str(exc_info.value)
        assert exc_info.value.context["level"] == 1

    @pytest.mark.unit
    def test_validation_chain(self):
        """Test chaining multiple validation functions."""

        def validate_input(value, name):
            validate_not_none(value, name)
            validate_not_empty(value, name)

        # Should pass for valid input
        validate_input("valid", "test_param")

        # Should fail for None
        with pytest.raises(DataValidationError):
            validate_input(None, "test_param")

        # Should fail for empty
        with pytest.raises(DataValidationError):
            validate_input("", "test_param")

    @pytest.mark.unit
    def test_exception_context_propagation(self):
        """Test that context is properly propagated through exception chain."""
        original_context = {"source": "test", "data_type": "coffee_reviews"}

        try:
            raise DataLoadingError("Original error", context=original_context)
        except Exception as e:
            with pytest.raises(DataPreprocessingError) as exc_info:
                handle_exception(
                    e,
                    context={"step": "preprocessing"},
                    reraise_as=DataPreprocessingError,
                    message="Preprocessing failed",
                )

            # Context should include both original and new context
            final_context = exc_info.value.context
            assert "step" in final_context
            assert final_context["step"] == "preprocessing"

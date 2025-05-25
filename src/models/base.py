"""
Base classes for machine learning models.

This module provides abstract base classes that define the interface for all
models in the coffee text analytics project.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import BaseEstimator

# Import centralized exceptions using absolute import
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import (
    ModelError,
    ModelNotFittedError,
    ModelConfigError,
    ModelTrainingError,
    ModelEvaluationError,
    ModelSaveError,
    ModelLoadError,
    handle_exception,
    validate_not_none,
    validate_not_empty,
)

logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all machine learning models.

    Provides common interface and functionality for model training, prediction,
    and evaluation in the coffee text analytics project.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self.model_ = None
        self.feature_names_ = []
        self.training_metrics_ = {}

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate the model configuration.

        Raises:
            ModelConfigError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ModelConfigError(
                "Configuration must be a dictionary",
                context={"config_type": type(self.config)},
            )

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> "BaseModel":
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self for method chaining

        Raises:
            ModelTrainingError: If training fails
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame, pl.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Features to predict on

        Returns:
            Predictions array

        Raises:
            ModelNotFittedError: If model is not fitted
            ModelError: If prediction fails
        """
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        self._check_fitted()

        # Default implementation - subclasses should override
        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
        elif hasattr(self.model_, "coef_"):
            importances = np.abs(self.model_.coef_)
        else:
            logger.warning(
                f"{self.__class__.__name__} does not support feature importance"
            )
            return {}

        if len(self.feature_names_) == len(importances):
            return dict(zip(self.feature_names_, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def get_training_metrics(self) -> Dict[str, float]:
        """
        Get training metrics.

        Returns:
            Dictionary of training metrics

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        self._check_fitted()
        return self.training_metrics_.copy()

    def _check_fitted(self) -> None:
        """
        Check if the model is fitted.

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"{self.__class__.__name__} must be fitted before use",
                context={"model_type": self.__class__.__name__},
            )

    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series, pl.Series]] = None,
    ) -> None:
        """
        Validate input data.

        Args:
            X: Features to validate
            y: Optional targets to validate

        Raises:
            ModelError: If input is invalid
        """
        validate_not_none(X, "X", context={"model": self.__class__.__name__})

        if isinstance(X, (pd.DataFrame, pl.DataFrame)) and X.is_empty():
            raise ModelError(
                "Input features cannot be empty",
                context={"model": self.__class__.__name__, "X_shape": X.shape},
            )
        elif isinstance(X, np.ndarray) and X.size == 0:
            raise ModelError(
                "Input features cannot be empty",
                context={"model": self.__class__.__name__, "X_shape": X.shape},
            )

        if y is not None:
            validate_not_none(y, "y", context={"model": self.__class__.__name__})


class BaseRegressor(BaseModel):
    """
    Base class for regression models.
    """

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> float:
        """
        Calculate R² score.

        Args:
            X: Test features
            y: True targets

        Returns:
            R² score

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        self._check_fitted()
        self._validate_input(X, y)

        try:
            predictions = self.predict(X)
            from sklearn.metrics import r2_score

            return r2_score(y, predictions)
        except Exception as e:
            handle_exception(
                e,
                context={"model": self.__class__.__name__, "X_shape": X.shape},
                reraise_as=ModelEvaluationError,
                message="Failed to calculate R² score",
            )


class BaseClassifier(BaseModel):
    """
    Base class for classification models.
    """

    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Probability predictions

        Raises:
            ModelNotFittedError: If model is not fitted
            ModelError: If prediction fails
        """
        self._check_fitted()
        self._validate_input(X)

        if hasattr(self.model_, "predict_proba"):
            try:
                return self.model_.predict_proba(X)
            except Exception as e:
                handle_exception(
                    e,
                    context={"model": self.__class__.__name__, "X_shape": X.shape},
                    reraise_as=ModelError,
                    message="Failed to predict probabilities",
                )
        else:
            raise ModelError(
                f"{self.__class__.__name__} does not support probability prediction",
                context={"model": self.__class__.__name__},
            )

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> float:
        """
        Calculate accuracy score.

        Args:
            X: Test features
            y: True targets

        Returns:
            Accuracy score

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        self._check_fitted()
        self._validate_input(X, y)

        try:
            predictions = self.predict(X)
            from sklearn.metrics import accuracy_score

            return accuracy_score(y, predictions)
        except Exception as e:
            handle_exception(
                e,
                context={"model": self.__class__.__name__, "X_shape": X.shape},
                reraise_as=ModelEvaluationError,
                message="Failed to calculate accuracy score",
            )


class BaseEnsembleModel(BaseModel):
    """
    Base class for ensemble models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_models_ = []

    def get_base_models(self) -> List[BaseModel]:
        """
        Get the base models in the ensemble.

        Returns:
            List of base models

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        self._check_fitted()
        return self.base_models_.copy()


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def evaluate_model(
        self,
        model: BaseModel,
        X_test: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y_test: Union[np.ndarray, pd.Series, pl.Series],
    ) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ModelEvaluationError: If evaluation fails
        """
        pass

    @abstractmethod
    def compare_models(
        self,
        models: Dict[str, BaseModel],
        X_test: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y_test: Union[np.ndarray, pd.Series, pl.Series],
    ) -> Dict[str, Any]:
        """
        Compare multiple models.

        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of comparison results

        Raises:
            ModelEvaluationError: If comparison fails
        """
        pass


# Legacy compatibility - keep the old exception names for backward compatibility
# but they now inherit from the centralized exceptions
# (These were already defined correctly, so we just add aliases if needed)

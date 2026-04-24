"""
Model training and evaluation module for coffee rating prediction.

This module provides a comprehensive model training pipeline following
the thesis methodology with component-based architecture.
"""

# Base classes
from .base import (
    BaseClassifier,
    BaseEnsembleModel,
    BaseEvaluator,
    BaseModel,
    BaseRegressor,
    ModelConfigError,
    ModelError,
    ModelEvaluationError,
    ModelNotFittedError,
)

# Evaluation
from .evaluator import CoffeeModelEvaluator

# Individual models
from .mnir import MultinomialInverseRegression
from .regressors import (
    CoffeeDecisionTree,
    CoffeeLassoRegression,
    CoffeeLinearRegression,
    CoffeeRandomForest,
    CoffeeRidgeRegression,
    CoffeeSVR,
    CoffeeXGBoost,
)

# Legacy functions have been removed - use component-based architecture instead

__all__ = [
    # Base classes
    "BaseModel",
    "BaseRegressor",
    "BaseClassifier",
    "BaseEnsembleModel",
    "BaseEvaluator",
    "ModelError",
    "ModelNotFittedError",
    "ModelConfigError",
    "ModelEvaluationError",
    # Individual models
    "MultinomialInverseRegression",
    "CoffeeLinearRegression",
    "CoffeeRidgeRegression",
    "CoffeeLassoRegression",
    "CoffeeRandomForest",
    "CoffeeXGBoost",
    "CoffeeSVR",
    "CoffeeDecisionTree",
    # Evaluation
    "CoffeeModelEvaluator",
]

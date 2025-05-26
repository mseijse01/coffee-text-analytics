"""
Model training and evaluation module for coffee rating prediction.

This module provides a comprehensive model training pipeline following
the thesis methodology with component-based architecture.
"""

# Base classes
from .base import (
    BaseModel,
    BaseRegressor,
    BaseClassifier,
    BaseEnsembleModel,
    BaseEvaluator,
    ModelError,
    ModelNotFittedError,
    ModelConfigError,
    ModelEvaluationError,
)

# Individual models
from .mnir import MultinomialInverseRegression
from .regressors import (
    CoffeeLinearRegression,
    CoffeeRidgeRegression,
    CoffeeLassoRegression,
    CoffeeRandomForest,
    CoffeeXGBoost,
    CoffeeSVR,
    CoffeeDecisionTree,
)

# Evaluation
from .evaluator import CoffeeModelEvaluator

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

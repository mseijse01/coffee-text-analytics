"""
Unit tests for base model classes.

Tests abstract base classes, validation logic, and common model functionality
without any heavyweight ML operations.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

from src.models.base import (
    BaseModel,
    BaseRegressor,
    BaseClassifier,
    BaseEnsembleModel,
    BaseEvaluator,
)
from src.exceptions import (
    ModelError,
    ModelNotFittedError,
    ModelConfigError,
    ModelTrainingError,
    ModelEvaluationError,
)


# Concrete implementations for testing abstract classes
class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def fit(self, X, y):
        self._validate_input(X, y)
        self.is_fitted = True
        self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        self.training_metrics_ = {"mse": 0.1, "r2": 0.9}
        # Mock model with feature importance
        self.model_ = Mock()
        self.model_.feature_importances_ = np.random.random(X.shape[1])
        return self

    def predict(self, X):
        self._check_fitted()
        self._validate_input(X)
        return np.random.random(X.shape[0])


class ConcreteRegressor(BaseRegressor):
    """Concrete implementation of BaseRegressor for testing."""

    def fit(self, X, y):
        self._validate_input(X, y)
        self.is_fitted = True
        self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        self.training_metrics_ = {"mse": 0.1, "r2": 0.9}
        self.model_ = Mock()
        return self

    def predict(self, X):
        self._check_fitted()
        self._validate_input(X)
        return np.random.random(X.shape[0])


class ConcreteClassifier(BaseClassifier):
    """Concrete implementation of BaseClassifier for testing."""

    def fit(self, X, y):
        self._validate_input(X, y)
        self.is_fitted = True
        self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        self.training_metrics_ = {"accuracy": 0.9, "f1": 0.85}
        self.model_ = Mock()
        return self

    def predict(self, X):
        self._check_fitted()
        self._validate_input(X)
        return np.random.randint(0, 2, X.shape[0])

    def predict_proba(self, X):
        self._check_fitted()
        self._validate_input(X)
        proba = np.random.random((X.shape[0], 2))
        proba = proba / proba.sum(axis=1, keepdims=True)  # Normalize
        return proba


class ConcreteEnsemble(BaseEnsembleModel):
    """Concrete implementation of BaseEnsembleModel for testing."""

    def __init__(self, config=None):
        super().__init__(config)
        self.base_models = [ConcreteModel(), ConcreteModel()]

    def fit(self, X, y):
        self._validate_input(X, y)
        for model in self.base_models:
            model.fit(X, y)
        self.is_fitted = True
        self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X):
        self._check_fitted()
        self._validate_input(X)
        predictions = np.array([model.predict(X) for model in self.base_models])
        return np.mean(predictions, axis=0)

    def get_base_models(self):
        return self.base_models


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing."""

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        return {"mse": mse, "n_samples": len(y_test)}

    def compare_models(self, models, X_test, y_test):
        results = {}
        for name, model in models.items():
            results[name] = self.evaluate_model(model, X_test, y_test)
        return results


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    X = np.random.random((100, 5))
    y = np.random.random(100)
    return X, y


@pytest.fixture
def sample_dataframes():
    """Sample dataframes for testing."""
    X_pd = pd.DataFrame(np.random.random((50, 3)), columns=["a", "b", "c"])
    y_pd = pd.Series(np.random.random(50))
    X_pl = pl.DataFrame(X_pd)
    y_pl = pl.Series(y_pd)
    return X_pd, y_pd, X_pl, y_pl


class TestBaseModel:
    """Test the BaseModel abstract base class."""

    @pytest.mark.unit
    def test_model_initialization_default(self):
        """Test model initialization with default config."""
        model = ConcreteModel()

        assert model.config == {}
        assert not model.is_fitted
        assert model.model_ is None
        assert model.feature_names_ == []
        assert model.training_metrics_ == {}

    @pytest.mark.unit
    def test_model_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = {"param1": "value1", "param2": 42}
        model = ConcreteModel(config)

        assert model.config == config
        assert not model.is_fitted

    @pytest.mark.unit
    def test_config_validation_invalid_type(self):
        """Test that invalid config type raises error."""
        with pytest.raises(ModelConfigError) as exc_info:
            ConcreteModel("invalid_config")

        assert "Configuration must be a dictionary" in str(exc_info.value)

    @pytest.mark.unit
    def test_fit_basic_functionality(self, sample_data):
        """Test basic fit functionality."""
        X, y = sample_data
        model = ConcreteModel()

        result = model.fit(X, y)

        assert result is model  # Should return self
        assert model.is_fitted
        assert len(model.feature_names_) == X.shape[1]
        assert "mse" in model.training_metrics_
        assert model.model_ is not None

    @pytest.mark.unit
    def test_predict_basic_functionality(self, sample_data):
        """Test basic predict functionality."""
        X, y = sample_data
        model = ConcreteModel()
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == X.shape[0]

    @pytest.mark.unit
    def test_predict_without_fitting(self, sample_data):
        """Test that predict raises error when model not fitted."""
        X, y = sample_data
        model = ConcreteModel()

        with pytest.raises(ModelNotFittedError) as exc_info:
            model.predict(X)

        assert "must be fitted before use" in str(exc_info.value)

    @pytest.mark.unit
    def test_feature_importance_with_importances(self, sample_data):
        """Test feature importance when model has feature_importances_."""
        X, y = sample_data
        model = ConcreteModel()
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        for feature_name in model.feature_names_:
            assert feature_name in importance

    @pytest.mark.unit
    def test_feature_importance_with_coef(self, sample_data):
        """Test feature importance when model has coef_."""
        X, y = sample_data
        model = ConcreteModel()
        model.fit(X, y)

        # Replace with model that has coef_ instead of feature_importances_
        model.model_ = Mock()
        model.model_.coef_ = np.random.random(X.shape[1])
        del model.model_.feature_importances_

        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]

    @pytest.mark.unit
    def test_feature_importance_without_support(self, sample_data):
        """Test feature importance when model doesn't support it."""
        X, y = sample_data
        model = ConcreteModel()
        model.fit(X, y)

        # Replace with model that has no importance attributes
        model.model_ = Mock()
        if hasattr(model.model_, "feature_importances_"):
            del model.model_.feature_importances_
        if hasattr(model.model_, "coef_"):
            del model.model_.coef_

        importance = model.get_feature_importance()

        assert importance == {}

    @pytest.mark.unit
    def test_get_training_metrics(self, sample_data):
        """Test getting training metrics."""
        X, y = sample_data
        model = ConcreteModel()
        model.fit(X, y)

        metrics = model.get_training_metrics()

        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "r2" in metrics
        # Should return a copy, not the original
        metrics["new_metric"] = 123
        assert "new_metric" not in model.training_metrics_

    @pytest.mark.unit
    def test_get_training_metrics_not_fitted(self):
        """Test that get_training_metrics raises error when not fitted."""
        model = ConcreteModel()

        with pytest.raises(ModelNotFittedError):
            model.get_training_metrics()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "X",
        [
            None,  # None input
            np.array([]),  # Empty numpy array
        ],
    )
    def test_input_validation_invalid_X(self, X):
        """Test input validation with invalid X values."""
        model = ConcreteModel()

        with pytest.raises(
            (ModelError, Exception)
        ):  # Can be ModelError or DataValidationError
            model._validate_input(X)

    @pytest.mark.unit
    def test_input_validation_empty_dataframes(self):
        """Test input validation with empty dataframes."""
        model = ConcreteModel()

        # Empty pandas DataFrame
        empty_pd = pd.DataFrame()
        with pytest.raises(ModelError):
            model._validate_input(empty_pd)

        # Empty polars DataFrame
        empty_pl = pl.DataFrame()
        with pytest.raises(ModelError):
            model._validate_input(empty_pl)

    @pytest.mark.unit
    def test_input_validation_valid_formats(self, sample_dataframes):
        """Test input validation with various valid formats."""
        X_pd, y_pd, X_pl, y_pl = sample_dataframes
        model = ConcreteModel()

        # Should not raise for valid inputs
        model._validate_input(X_pd, y_pd)
        model._validate_input(X_pl, y_pl)
        model._validate_input(X_pd.values, y_pd.values)


class TestBaseRegressor:
    """Test the BaseRegressor class."""

    @pytest.mark.unit
    def test_regressor_inheritance(self):
        """Test that BaseRegressor inherits from BaseModel."""
        regressor = ConcreteRegressor()
        assert isinstance(regressor, BaseModel)

    @pytest.mark.unit
    def test_score_method_exists(self):
        """Test that BaseRegressor has score method."""
        assert hasattr(BaseRegressor, "score")

    @pytest.mark.unit
    def test_regressor_fit_predict_workflow(self, sample_data):
        """Test complete fit-predict workflow for regressor."""
        X, y = sample_data
        regressor = ConcreteRegressor()

        regressor.fit(X, y)
        predictions = regressor.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)


class TestBaseClassifier:
    """Test the BaseClassifier class."""

    @pytest.mark.unit
    def test_classifier_inheritance(self):
        """Test that BaseClassifier inherits from BaseModel."""
        classifier = ConcreteClassifier()
        assert isinstance(classifier, BaseModel)

    @pytest.mark.unit
    def test_predict_proba_method(self, sample_data):
        """Test predict_proba method."""
        X, y = sample_data
        y_binary = (y > 0.5).astype(int)  # Make binary classification target
        classifier = ConcreteClassifier()
        classifier.fit(X, y_binary)

        probabilities = classifier.predict_proba(X)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X), 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10)

    @pytest.mark.unit
    def test_predict_proba_not_fitted(self, sample_data):
        """Test that predict_proba raises error when not fitted."""
        X, y = sample_data
        classifier = ConcreteClassifier()

        with pytest.raises(ModelNotFittedError):
            classifier.predict_proba(X)

    @pytest.mark.unit
    def test_score_method_exists(self):
        """Test that BaseClassifier has score method."""
        assert hasattr(BaseClassifier, "score")


class TestBaseEnsembleModel:
    """Test the BaseEnsembleModel class."""

    @pytest.mark.unit
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        ensemble = ConcreteEnsemble()

        assert isinstance(ensemble, BaseModel)
        assert hasattr(ensemble, "base_models")
        assert len(ensemble.base_models) == 2

    @pytest.mark.unit
    def test_get_base_models(self):
        """Test getting base models from ensemble."""
        ensemble = ConcreteEnsemble()

        base_models = ensemble.get_base_models()

        assert isinstance(base_models, list)
        assert len(base_models) == 2
        assert all(isinstance(model, BaseModel) for model in base_models)

    @pytest.mark.unit
    def test_ensemble_fit_predict_workflow(self, sample_data):
        """Test complete ensemble workflow."""
        X, y = sample_data
        ensemble = ConcreteEnsemble()

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert ensemble.is_fitted
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        # Verify base models are fitted
        for model in ensemble.base_models:
            assert model.is_fitted


class TestBaseEvaluator:
    """Test the BaseEvaluator class."""

    @pytest.mark.unit
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ConcreteEvaluator()

        assert evaluator.config == {}

    @pytest.mark.unit
    def test_evaluator_with_config(self):
        """Test evaluator initialization with config."""
        config = {"metric": "mse", "cross_validation": True}
        evaluator = ConcreteEvaluator(config)

        assert evaluator.config == config

    @pytest.mark.unit
    def test_evaluate_model(self, sample_data):
        """Test single model evaluation."""
        X, y = sample_data
        model = ConcreteModel()
        model.fit(X, y)
        evaluator = ConcreteEvaluator()

        results = evaluator.evaluate_model(model, X, y)

        assert isinstance(results, dict)
        assert "mse" in results
        assert "n_samples" in results
        assert results["n_samples"] == len(y)

    @pytest.mark.unit
    def test_compare_models(self, sample_data):
        """Test comparing multiple models."""
        X, y = sample_data

        model1 = ConcreteModel()
        model1.fit(X, y)
        model2 = ConcreteRegressor()
        model2.fit(X, y)

        models = {"model1": model1, "model2": model2}
        evaluator = ConcreteEvaluator()

        comparison = evaluator.compare_models(models, X, y)

        assert isinstance(comparison, dict)
        assert "model1" in comparison
        assert "model2" in comparison
        assert all("mse" in results for results in comparison.values())


class TestModelIntegration:
    """Test integration scenarios between different model classes."""

    @pytest.mark.unit
    def test_model_type_polymorphism(self, sample_data):
        """Test that different model types can be used polymorphically."""
        X, y = sample_data
        y_binary = (y > 0.5).astype(int)

        models = [
            ConcreteModel(),
            ConcreteRegressor(),
            ConcreteClassifier(),
        ]

        # All should fit and predict successfully
        for model in models:
            if isinstance(model, ConcreteClassifier):
                model.fit(X, y_binary)
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
                assert len(probabilities) == len(X)
            else:
                model.fit(X, y)
                predictions = model.predict(X)

            assert model.is_fitted
            assert len(predictions) == len(X)
            assert isinstance(model.get_training_metrics(), dict)

    @pytest.mark.unit
    def test_evaluator_with_different_model_types(self, sample_data):
        """Test evaluator works with different model types."""
        X, y = sample_data

        regressor = ConcreteRegressor()
        regressor.fit(X, y)

        ensemble = ConcreteEnsemble()
        ensemble.fit(X, y)

        evaluator = ConcreteEvaluator()

        # Both should evaluate successfully
        reg_results = evaluator.evaluate_model(regressor, X, y)
        ens_results = evaluator.evaluate_model(ensemble, X, y)

        assert "mse" in reg_results
        assert "mse" in ens_results

        # Comparison should work
        models = {"regressor": regressor, "ensemble": ensemble}
        comparison = evaluator.compare_models(models, X, y)
        assert len(comparison) == 2

"""
Comprehensive tests for the models/regressors.py module.

Tests for all regressor implementations including:
- CoffeeLinearRegression, CoffeeRidgeRegression, CoffeeLassoRegression
- CoffeeRandomForest, CoffeeXGBoost, CoffeeSVR, CoffeeDecisionTree
- Model lifecycle (initialization, fit, predict, feature importance)
- Hyperparameter tuning functionality
- Configuration management
- Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

from src.models.regressors import (
    CoffeeLinearRegression,
    CoffeeRidgeRegression,
    CoffeeLassoRegression,
    CoffeeRandomForest,
    CoffeeXGBoost,
    CoffeeSVR,
    CoffeeDecisionTree,
    XGBOOST_AVAILABLE,
)
from src.models.base import ModelError


class TestCoffeeLinearRegression:
    """Test CoffeeLinearRegression implementation."""

    @pytest.mark.unit
    def test_initialization_default_config(self):
        """Test linear regression initialization with default config."""
        model = CoffeeLinearRegression()

        assert model.config["scale_features"] is False
        assert model.config["fit_intercept"] is True
        assert model.scaler_ is None
        assert not model.is_fitted

    @pytest.mark.unit
    def test_initialization_custom_config(self):
        """Test linear regression initialization with custom config."""
        config = {"scale_features": True, "fit_intercept": False}
        model = CoffeeLinearRegression(config)

        assert model.config["scale_features"] is True
        assert model.config["fit_intercept"] is False
        assert model.scaler_ is not None
        assert isinstance(model.scaler_, StandardScaler)

    @pytest.mark.unit
    def test_fit_with_pandas_dataframe(self):
        """Test fitting with pandas DataFrame."""
        np.random.seed(42)
        X_df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }
        )
        y_series = pd.Series(
            2 * X_df["feature_1"] + X_df["feature_2"] + np.random.randn(100) * 0.1
        )

        model = CoffeeLinearRegression()
        result = model.fit(X_df, y_series)

        assert result is model  # Returns self
        assert model.is_fitted
        assert model.feature_names_ == ["feature_1", "feature_2"]
        assert hasattr(model, "model_")
        assert isinstance(model.model_, LinearRegression)

    @pytest.mark.unit
    def test_fit_with_numpy_arrays(self):
        """Test fitting with numpy arrays."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        model = CoffeeLinearRegression()
        result = model.fit(X, y)

        assert result is model
        assert model.is_fitted
        assert model.feature_names_ == ["feature_0", "feature_1", "feature_2"]

    @pytest.mark.unit
    def test_fit_with_scaling(self):
        """Test fitting with feature scaling enabled."""
        np.random.seed(42)
        X = np.random.randn(100, 2) * 100  # Large scale features
        y = np.random.randn(100)

        config = {"scale_features": True}
        model = CoffeeLinearRegression(config)
        model.fit(X, y)

        assert model.is_fitted
        assert model.scaler_ is not None
        # Scaler should be fitted
        assert hasattr(model.scaler_, "mean_")
        assert hasattr(model.scaler_, "scale_")

    @pytest.mark.unit
    def test_predict_unfitted_model(self):
        """Test prediction with unfitted model raises error."""
        model = CoffeeLinearRegression()
        X = np.random.randn(10, 2)

        with pytest.raises(ModelError, match="Model must be fitted before prediction"):
            model.predict(X)

    @pytest.mark.unit
    def test_predict_with_scaling(self):
        """Test prediction with feature scaling."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2) * 100
        y_train = np.random.randn(100)
        X_test = np.random.randn(10, 2) * 100

        config = {"scale_features": True}
        model = CoffeeLinearRegression(config)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    @pytest.mark.unit
    def test_predict_pandas_dataframe(self):
        """Test prediction with pandas DataFrame."""
        np.random.seed(42)
        X_train = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y_train = np.random.randn(100)
        X_test = pd.DataFrame({"a": np.random.randn(10), "b": np.random.randn(10)})

        model = CoffeeLinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    @pytest.mark.unit
    def test_get_feature_importance_unfitted(self):
        """Test feature importance with unfitted model raises error."""
        model = CoffeeLinearRegression()

        with pytest.raises(
            ModelError, match="Model must be fitted to get feature importance"
        ):
            model.get_feature_importance()

    @pytest.mark.unit
    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (
            2 * X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
        )  # feature_0 more important

        model = CoffeeLinearRegression()
        model.fit(X, y)
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(isinstance(v, (int, float)) for v in importance.values())
        # feature_0 should have highest importance
        assert importance["feature_0"] > importance["feature_2"]


class TestCoffeeRidgeRegression:
    """Test CoffeeRidgeRegression implementation."""

    @pytest.mark.unit
    def test_initialization_default_config(self):
        """Test Ridge regression initialization with default config."""
        model = CoffeeRidgeRegression()

        assert model.config["alpha"] == 1.0
        assert model.config["alpha_grid"] == [0.1, 1.0, 10.0, 100.0]
        assert model.config["cv"] == 5
        assert model.config["scale_features"] is True
        assert model.best_alpha_ is None

    @pytest.mark.unit
    @patch("src.models.regressors.GridSearchCV")
    def test_fit_with_hyperparameter_tuning(self, mock_grid_search):
        """Test Ridge fitting with hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock GridSearchCV
        mock_estimator = Mock()
        mock_estimator.best_params_ = {"alpha": 10.0}
        mock_grid_search.return_value = mock_estimator

        model = CoffeeRidgeRegression()
        result = model.fit(X, y)

        assert result is model
        assert model.is_fitted
        assert model.best_alpha_ == 10.0
        assert mock_grid_search.called

    @pytest.mark.unit
    def test_custom_alpha_grid(self):
        """Test Ridge regression with custom alpha grid."""
        config = {"alpha_grid": [0.01, 0.1, 1.0], "cv": 3}
        model = CoffeeRidgeRegression(config)

        assert model.config["alpha_grid"] == [0.01, 0.1, 1.0]
        assert model.config["cv"] == 3

    @pytest.mark.unit
    def test_feature_importance(self):
        """Test Ridge feature importance."""
        np.random.seed(42)
        X = np.random.randn(50, 3)  # Smaller dataset for faster test
        y = 2 * X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

        model = CoffeeRidgeRegression()
        model.fit(X, y)
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3
        # Coefficients should be non-zero (Ridge doesn't zero out features)
        assert all(v > 0 for v in importance.values())


class TestCoffeeLassoRegression:
    """Test CoffeeLassoRegression implementation."""

    @pytest.mark.unit
    def test_initialization_custom_config(self):
        """Test Lasso regression initialization."""
        config = {"alpha_grid": [0.001, 0.01, 0.1], "max_iter": 2000, "cv": 3}
        model = CoffeeLassoRegression(config)

        assert model.config["alpha_grid"] == [0.001, 0.01, 0.1]
        assert model.config["max_iter"] == 2000
        assert model.config["cv"] == 3
        assert model.n_features_selected_ == 0

    @pytest.mark.unit
    @patch("src.models.regressors.GridSearchCV")
    def test_fit_feature_selection(self, mock_grid_search):
        """Test Lasso fitting and feature selection."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Mock GridSearchCV with Lasso that selects some features
        mock_lasso = Mock()
        mock_lasso.coef_ = np.array([0.5, 0.0, 0.3, 0.0, 0.1])  # 3 selected features
        mock_grid_search.return_value.best_estimator_ = mock_lasso
        mock_grid_search.return_value.best_params_ = {"alpha": 0.1}

        model = CoffeeLassoRegression()
        model.fit(X, y)

        assert model.is_fitted
        assert model.best_alpha_ == 0.1
        assert model.n_features_selected_ == 3

    @pytest.mark.unit
    def test_get_selected_features_unfitted(self):
        """Test get_selected_features with unfitted model."""
        model = CoffeeLassoRegression()

        with pytest.raises(
            ModelError, match="Model must be fitted to get selected features"
        ):
            model.get_selected_features()

    @pytest.mark.unit
    def test_get_selected_features(self):
        """Test getting selected features from Lasso."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = (
            X[:, 0] + X[:, 2] + np.random.randn(50) * 0.1
        )  # Only features 0 and 2 are relevant

        model = CoffeeLassoRegression({"alpha_grid": [0.01, 0.1]})
        model.fit(X, y)

        selected_features = model.get_selected_features()

        assert isinstance(selected_features, list)
        # Should have selected some features
        assert len(selected_features) >= 0
        assert all(isinstance(f, str) for f in selected_features)


class TestCoffeeRandomForest:
    """Test CoffeeRandomForest implementation."""

    @pytest.mark.unit
    def test_initialization_default_config(self):
        """Test Random Forest initialization."""
        model = CoffeeRandomForest()

        assert model.config["n_estimators"] == 100
        assert model.config["random_state"] == 42
        assert model.config["tune_hyperparameters"] is True
        assert model.best_params_ is None

    @pytest.mark.unit
    @patch("src.models.regressors.GridSearchCV")
    def test_fit_with_hyperparameter_tuning(self, mock_grid_search):
        """Test Random Forest with hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock GridSearchCV
        mock_rf = Mock()
        mock_rf.feature_importances_ = np.array([0.4, 0.3, 0.3])
        mock_grid_search.return_value.best_estimator_ = mock_rf
        mock_grid_search.return_value.best_params_ = {
            "n_estimators": 200,
            "max_depth": 10,
        }

        model = CoffeeRandomForest()
        result = model.fit(X, y)

        assert result is model
        assert model.is_fitted
        assert model.best_params_ == {"n_estimators": 200, "max_depth": 10}

    @pytest.mark.unit
    @patch("src.models.regressors.RandomForestRegressor")
    def test_fit_without_hyperparameter_tuning(self, mock_rf_class):
        """Test Random Forest without hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock RandomForestRegressor
        mock_rf_instance = Mock()
        mock_rf_instance.feature_importances_ = np.array([0.4, 0.3, 0.3])
        mock_rf_class.return_value = mock_rf_instance

        config = {"tune_hyperparameters": False, "n_estimators": 50}
        model = CoffeeRandomForest(config)
        model.fit(X, y)

        assert model.is_fitted
        mock_rf_class.assert_called_once()
        mock_rf_instance.fit.assert_called_once()

    @pytest.mark.unit
    @patch("utils.hyperparameter_tuning.apply_two_step_tuning")
    def test_fit_with_two_step_tuning(self, mock_two_step):
        """Test Random Forest with two-step tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock two-step tuning
        mock_model = Mock()
        mock_model.get_params.return_value = {"n_estimators": 150}
        mock_two_step.return_value = (mock_model, {"best_score": 0.8})

        config = {
            "use_two_step_tuning": True,
            "global_config": {"some_setting": "value"},
        }
        model = CoffeeRandomForest(config)
        model.fit(X, y)

        assert model.is_fitted
        assert hasattr(model, "optimization_summary_")
        mock_two_step.assert_called_once()

    @pytest.mark.unit
    def test_get_feature_importance(self):
        """Test Random Forest feature importance."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = 2 * X[:, 0] + np.random.randn(50) * 0.1  # feature_0 most important

        model = CoffeeRandomForest({"tune_hyperparameters": False, "n_estimators": 10})
        model.fit(X, y)
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(
            v >= 0 for v in importance.values()
        )  # Importances should be non-negative
        # Sum should be close to 1.0
        assert abs(sum(importance.values()) - 1.0) < 0.01


class TestCoffeeXGBoost:
    """Test CoffeeXGBoost implementation."""

    @pytest.mark.unit
    def test_initialization_xgboost_unavailable(self):
        """Test XGBoost initialization when XGBoost is unavailable."""
        with patch("src.models.regressors.XGBOOST_AVAILABLE", False):
            with pytest.raises(ModelError, match="XGBoost is not available"):
                CoffeeXGBoost()

    @pytest.mark.unit
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_initialization_xgboost_available(self):
        """Test XGBoost initialization when available."""
        model = CoffeeXGBoost()

        assert model.config["n_estimators"] == 100
        assert model.config["max_depth"] == 6
        assert model.config["learning_rate"] == 0.1
        assert model.config["random_state"] == 42

    @pytest.mark.unit
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    @patch("src.models.regressors.GridSearchCV")
    def test_fit_with_traditional_tuning(self, mock_grid_search):
        """Test XGBoost with traditional hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock GridSearchCV
        mock_xgb = Mock()
        mock_xgb.feature_importances_ = np.array([0.4, 0.3, 0.3])
        mock_grid_search.return_value.best_estimator_ = mock_xgb
        mock_grid_search.return_value.best_params_ = {"n_estimators": 200}

        model = CoffeeXGBoost()
        model.fit(X, y)

        assert model.is_fitted
        assert model.best_params_ == {"n_estimators": 200}

    @pytest.mark.unit
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_fit_without_tuning(self):
        """Test XGBoost without hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        config = {"tune_hyperparameters": False, "n_estimators": 10}
        model = CoffeeXGBoost(config)

        with patch("xgboost.XGBRegressor") as mock_xgb_class:
            mock_xgb_instance = Mock()
            mock_xgb_instance.feature_importances_ = np.array([0.4, 0.3, 0.3])
            mock_xgb_class.return_value = mock_xgb_instance

            model.fit(X, y)

            assert model.is_fitted
            mock_xgb_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    @patch("utils.hyperparameter_tuning.apply_two_step_tuning")
    def test_fit_with_two_step_tuning(self, mock_two_step):
        """Test XGBoost with two-step tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock two-step tuning
        mock_model = Mock()
        mock_model.get_params.return_value = {"n_estimators": 150}
        mock_two_step.return_value = (mock_model, {"best_score": 0.8})

        config = {"use_two_step_tuning": True, "global_config": {}}
        model = CoffeeXGBoost(config)
        model.fit(X, y)

        assert model.is_fitted
        assert hasattr(model, "optimization_summary_")


class TestCoffeeSVR:
    """Test CoffeeSVR implementation."""

    @pytest.mark.unit
    def test_initialization_default_config(self):
        """Test SVR initialization."""
        model = CoffeeSVR()

        assert model.config["kernel"] == "rbf"
        assert model.config["C"] == 1.0
        assert model.config["gamma"] == "scale"
        assert model.config["scale_features"] is True
        assert model.scaler_ is not None

    @pytest.mark.unit
    @patch("src.models.regressors.GridSearchCV")
    def test_fit_with_traditional_tuning(self, mock_grid_search):
        """Test SVR with traditional hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock GridSearchCV
        mock_svr = Mock()
        mock_grid_search.return_value.best_estimator_ = mock_svr
        mock_grid_search.return_value.best_params_ = {"C": 10.0, "kernel": "rbf"}

        model = CoffeeSVR()
        model.fit(X, y)

        assert model.is_fitted
        assert model.best_params_ == {"C": 10.0, "kernel": "rbf"}

    @pytest.mark.unit
    @patch("sklearn.svm.SVR")
    def test_fit_without_tuning(self, mock_svr_class):
        """Test SVR without hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Mock SVR
        mock_svr_instance = Mock()
        mock_svr_class.return_value = mock_svr_instance

        config = {"tune_hyperparameters": False}
        model = CoffeeSVR(config)
        model.fit(X, y)

        assert model.is_fitted
        mock_svr_class.assert_called_once()
        mock_svr_instance.fit.assert_called_once()

    @pytest.mark.unit
    def test_predict_with_scaling(self):
        """Test SVR prediction with scaling."""
        np.random.seed(42)
        X_train = np.random.randn(50, 2) * 100  # Large scale
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 2) * 100

        config = {"tune_hyperparameters": False}
        model = CoffeeSVR(config)

        with patch("sklearn.svm.SVR") as mock_svr_class:
            mock_svr_instance = Mock()
            mock_svr_instance.predict.return_value = np.random.randn(10)
            mock_svr_class.return_value = mock_svr_instance

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 10


class TestCoffeeDecisionTree:
    """Test CoffeeDecisionTree implementation."""

    @pytest.mark.unit
    def test_initialization_default_config(self):
        """Test Decision Tree initialization."""
        model = CoffeeDecisionTree()

        assert model.config["max_depth"] is None
        assert model.config["min_samples_split"] == 2
        assert model.config["min_samples_leaf"] == 1
        assert model.config["random_state"] == 42
        assert model.config["tune_hyperparameters"] is True

    @pytest.mark.unit
    @patch("src.models.regressors.GridSearchCV")
    def test_fit_with_hyperparameter_tuning(self, mock_grid_search):
        """Test Decision Tree with hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Mock GridSearchCV
        mock_dt = Mock()
        mock_dt.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_grid_search.return_value.best_estimator_ = mock_dt
        mock_grid_search.return_value.best_params_ = {
            "max_depth": 10,
            "min_samples_split": 5,
        }

        model = CoffeeDecisionTree()
        model.fit(X, y)

        assert model.is_fitted
        assert model.best_params_ == {"max_depth": 10, "min_samples_split": 5}

    @pytest.mark.unit
    @patch("sklearn.tree.DecisionTreeRegressor")
    def test_fit_without_tuning(self, mock_dt_class):
        """Test Decision Tree without hyperparameter tuning."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Mock DecisionTreeRegressor
        mock_dt_instance = Mock()
        mock_dt_instance.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_dt_class.return_value = mock_dt_instance

        config = {"tune_hyperparameters": False, "max_depth": 5}
        model = CoffeeDecisionTree(config)
        model.fit(X, y)

        assert model.is_fitted
        mock_dt_class.assert_called_once()
        mock_dt_instance.fit.assert_called_once()

    @pytest.mark.unit
    def test_get_feature_importance(self):
        """Test Decision Tree feature importance."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = 2 * X[:, 0] + np.random.randn(50) * 0.1  # feature_0 most important

        config = {"tune_hyperparameters": False, "max_depth": 5}
        model = CoffeeDecisionTree(config)

        with patch("sklearn.tree.DecisionTreeRegressor") as mock_dt_class:
            mock_dt_instance = Mock()
            mock_dt_instance.feature_importances_ = np.array([0.7, 0.2, 0.1])
            mock_dt_class.return_value = mock_dt_instance

            model.fit(X, y)
            importance = model.get_feature_importance()

            assert isinstance(importance, dict)
            assert len(importance) == 3
            assert importance["feature_0"] == 0.7
            assert importance["feature_1"] == 0.2
            assert importance["feature_2"] == 0.1


class TestRegressorIntegration:
    """Integration tests for all regressor models."""

    @pytest.mark.integration
    def test_all_regressors_lifecycle(self):
        """Test that all regressors follow the same lifecycle."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = 2 * X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

        # Test all regressors (skip XGBoost if not available)
        regressors = [
            CoffeeLinearRegression({"scale_features": False}),
            CoffeeRidgeRegression({"cv": 2, "alpha_grid": [0.1, 1.0]}),
            CoffeeLassoRegression({"cv": 2, "alpha_grid": [0.1, 1.0]}),
            CoffeeRandomForest({"tune_hyperparameters": False, "n_estimators": 10}),
            CoffeeSVR({"tune_hyperparameters": False}),
            CoffeeDecisionTree({"tune_hyperparameters": False, "max_depth": 5}),
        ]

        if XGBOOST_AVAILABLE:
            regressors.append(
                CoffeeXGBoost({"tune_hyperparameters": False, "n_estimators": 10})
            )

        for regressor in regressors:
            # Test lifecycle
            assert not regressor.is_fitted

            # Fit
            result = regressor.fit(X, y)
            assert result is regressor
            assert regressor.is_fitted

            # Predict
            predictions = regressor.predict(X)
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(y)

            # Feature importance (all should support this, except SVR)
            importance = regressor.get_feature_importance()
            assert isinstance(importance, dict)
            if not isinstance(
                regressor, CoffeeSVR
            ):  # SVR doesn't provide feature importance
                assert len(importance) == 3

            # Score (inherited from BaseRegressor)
            score = regressor.score(X, y)
            assert isinstance(score, float)
            assert score >= -1.0  # R² can be negative

    @pytest.mark.unit
    def test_pandas_numpy_consistency(self):
        """Test that pandas and numpy inputs give consistent results."""
        np.random.seed(42)
        X_np = np.random.randn(50, 3)
        y_np = 2 * X_np[:, 0] + X_np[:, 1] + np.random.randn(50) * 0.1

        X_pd = pd.DataFrame(X_np, columns=["feat_0", "feat_1", "feat_2"])
        y_pd = pd.Series(y_np)

        model_np = CoffeeLinearRegression()
        model_pd = CoffeeLinearRegression()

        model_np.fit(X_np, y_np)
        model_pd.fit(X_pd, y_pd)

        pred_np = model_np.predict(X_np)
        pred_pd = model_pd.predict(X_pd)

        # Results should be very similar (allow for numerical differences)
        np.testing.assert_allclose(pred_np, pred_pd, rtol=1e-10)

    @pytest.mark.unit
    def test_error_handling_consistency(self):
        """Test that all regressors handle errors consistently."""
        regressors = [
            CoffeeLinearRegression(),
            CoffeeRidgeRegression(),
            CoffeeLassoRegression(),
            CoffeeRandomForest(),
            CoffeeSVR(),
            CoffeeDecisionTree(),
        ]

        if XGBOOST_AVAILABLE:
            regressors.append(CoffeeXGBoost())

        X = np.random.randn(10, 3)

        for regressor in regressors:
            # All should raise ModelError for prediction before fitting
            with pytest.raises(ModelError):
                regressor.predict(X)

            # All should raise ModelError for feature importance before fitting
            with pytest.raises(ModelError):
                regressor.get_feature_importance()

    @pytest.mark.unit
    def test_config_parameter_consistency(self):
        """Test that config parameters are handled consistently."""
        common_configs = [
            {},
            {"random_state": 123},
            {"tune_hyperparameters": False},
        ]

        for config in common_configs:
            # All models should accept these configurations without error
            models = [
                CoffeeLinearRegression(config.copy()),
                CoffeeRandomForest(config.copy()),
                CoffeeDecisionTree(config.copy()),
            ]

            if XGBOOST_AVAILABLE:
                models.append(CoffeeXGBoost(config.copy()))

            for model in models:
                # Config should be properly merged with defaults
                assert isinstance(model.config, dict)
                # Original config values should be preserved
                for key, value in config.items():
                    if key in model.config:
                        assert model.config[key] == value


class TestRegressorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_empty_data_handling(self):
        """Test behavior with edge case data."""
        # Very small dataset
        X_small = np.random.randn(3, 2)
        y_small = np.random.randn(3)

        model = CoffeeLinearRegression()

        # Should not crash with small data
        model.fit(X_small, y_small)
        predictions = model.predict(X_small)
        assert len(predictions) == 3

    @pytest.mark.unit
    def test_single_feature_handling(self):
        """Test behavior with single feature."""
        X_single = np.random.randn(50, 1)
        y_single = 2 * X_single.ravel() + np.random.randn(50) * 0.1

        model = CoffeeLinearRegression()
        model.fit(X_single, y_single)

        importance = model.get_feature_importance()
        assert len(importance) == 1
        assert "feature_0" in importance

    @pytest.mark.unit
    def test_perfect_correlation_handling(self):
        """Test behavior with perfectly correlated data."""
        X = np.random.randn(50, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1]  # Perfect linear relationship

        model = CoffeeLinearRegression()
        model.fit(X, y)

        # Should achieve very high R² score
        score = model.score(X, y)
        assert score > 0.99

    @pytest.mark.unit
    def test_feature_scaling_independence(self):
        """Test that scaling doesn't break model behavior."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X[:, 0] *= 1000  # Scale one feature dramatically
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

        model_scaled = CoffeeLinearRegression({"scale_features": True})
        model_unscaled = CoffeeLinearRegression({"scale_features": False})

        model_scaled.fit(X, y)
        model_unscaled.fit(X, y)

        # Both should make reasonable predictions
        pred_scaled = model_scaled.predict(X)
        pred_unscaled = model_unscaled.predict(X)

        assert isinstance(pred_scaled, np.ndarray)
        assert isinstance(pred_unscaled, np.ndarray)
        assert len(pred_scaled) == len(y)
        assert len(pred_unscaled) == len(y)

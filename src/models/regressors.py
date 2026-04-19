"""
Individual regressor implementations for coffee rating prediction.

This module provides wrapped implementations of standard regression models
that integrate with the configuration system and follow the base model interface.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, Union, cast
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .base import BaseRegressor, ModelError

logger = logging.getLogger(__name__)

# Check for XGBoost availability
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
    logger.info("XGBoost available")
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models will not be available.")
    XGBOOST_AVAILABLE = False


class CoffeeLinearRegression(BaseRegressor):
    """
    Linear Regression wrapper for coffee rating prediction.

    Simple linear regression model with optional feature scaling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Linear Regression model.

        Args:
            config: Configuration dictionary with parameters:
                - scale_features: Whether to scale features (default: False)
                - fit_intercept: Whether to fit intercept (default: True)
        """
        super().__init__(config)

        defaults: Dict[str, Any] = {"scale_features": False, "fit_intercept": True}
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.scaler_ = None
        if self.config["scale_features"]:
            self.scaler_ = StandardScaler()

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeLinearRegression":
        """Fit Linear Regression model."""
        logger.info("Fitting Linear Regression model")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Scale features if requested
        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)

        # Fit model
        self.model_ = LinearRegression(fit_intercept=self.config["fit_intercept"])
        self.model_.fit(X, y)

        self.is_fitted = True
        logger.info("Linear Regression model fitted successfully")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features if scaler was used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (coefficients)."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        coefficients = np.abs(self.model_.coef_)
        return dict(zip(self.feature_names_, coefficients))


class CoffeeRidgeRegression(BaseRegressor):
    """
    Ridge Regression wrapper with hyperparameter tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Ridge Regression model.

        Args:
            config: Configuration dictionary with parameters:
                - alpha: Regularization strength (default: 1.0)
                - alpha_grid: Grid of alpha values for tuning (default: [0.1, 1.0, 10.0])
                - cv: Cross-validation folds for tuning (default: 5)
                - scale_features: Whether to scale features (default: True)
        """
        super().__init__(config)

        defaults: Dict[str, Any] = {
            "alpha": 1.0,
            "alpha_grid": [0.1, 1.0, 10.0, 100.0],
            "cv": 5,
            "scale_features": True,
            "fit_intercept": True,
        }
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.scaler_ = None
        if self.config["scale_features"]:
            self.scaler_ = StandardScaler()

        self.best_alpha_ = None

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeRidgeRegression":
        """Fit Ridge Regression with hyperparameter tuning."""
        logger.info("Fitting Ridge Regression with hyperparameter tuning")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Scale features if requested
        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)

        # Hyperparameter tuning
        ridge = Ridge(fit_intercept=self.config["fit_intercept"])
        param_grid = {"alpha": self.config["alpha_grid"]}

        grid_search = GridSearchCV(
            ridge, param_grid, cv=self.config["cv"], scoring="r2", n_jobs=-1
        )
        grid_search.fit(X, y)

        self.model_ = grid_search.best_estimator_
        self.best_alpha_ = grid_search.best_params_["alpha"]

        self.is_fitted = True
        logger.info(f"Ridge Regression fitted with best alpha: {self.best_alpha_}")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (coefficients)."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        coefficients = np.abs(self.model_.coef_)
        return dict(zip(self.feature_names_, coefficients))


class CoffeeLassoRegression(BaseRegressor):
    """
    Lasso Regression wrapper with hyperparameter tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Lasso Regression model.

        Args:
            config: Configuration dictionary with parameters:
                - alpha: Regularization strength (default: 1.0)
                - alpha_grid: Grid of alpha values for tuning (default: [0.01, 0.1, 1.0])
                - cv: Cross-validation folds for tuning (default: 5)
                - scale_features: Whether to scale features (default: True)
                - max_iter: Maximum iterations (default: 1000)
        """
        super().__init__(config)

        defaults: Dict[str, Any] = {
            "alpha": 1.0,
            "alpha_grid": [0.01, 0.1, 1.0, 10.0],
            "cv": 5,
            "scale_features": True,
            "fit_intercept": True,
            "max_iter": 1000,
        }
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.scaler_ = None
        if self.config["scale_features"]:
            self.scaler_ = StandardScaler()

        self.best_alpha_ = None
        self.n_features_selected_ = 0

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeLassoRegression":
        """Fit Lasso Regression with hyperparameter tuning."""
        logger.info("Fitting Lasso Regression with hyperparameter tuning")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Scale features if requested
        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)

        # Hyperparameter tuning
        lasso = Lasso(
            fit_intercept=self.config["fit_intercept"], max_iter=self.config["max_iter"]
        )
        param_grid = {"alpha": self.config["alpha_grid"]}

        grid_search = GridSearchCV(
            lasso, param_grid, cv=self.config["cv"], scoring="r2", n_jobs=-1
        )
        grid_search.fit(X, y)

        self.model_ = grid_search.best_estimator_
        self.best_alpha_ = grid_search.best_params_["alpha"]
        self.n_features_selected_ = np.sum(self.model_.coef_ != 0)

        self.is_fitted = True
        logger.info(f"Lasso Regression fitted with best alpha: {self.best_alpha_}")
        logger.info(
            f"Features selected: {self.n_features_selected_}/{len(self.feature_names_)}"
        )
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (non-zero coefficients)."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        coefficients = np.abs(self.model_.coef_)
        return dict(zip(self.feature_names_, coefficients))

    def get_selected_features(self) -> List[str]:
        """Get names of features selected by Lasso."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get selected features")

        selected_mask = self.model_.coef_ != 0
        return [
            name
            for name, selected in zip(self.feature_names_, selected_mask)
            if selected
        ]


class CoffeeRandomForest(BaseRegressor):
    """
    Random Forest Regressor wrapper with hyperparameter tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model.

        Args:
            config: Configuration dictionary with parameters:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum depth (default: None)
                - min_samples_split: Minimum samples to split (default: 2)
                - min_samples_leaf: Minimum samples in leaf (default: 1)
                - random_state: Random state (default: 42)
                - tune_hyperparameters: Whether to tune hyperparameters (default: True)
        """
        super().__init__(config)

        defaults: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "tune_hyperparameters": True,
            "cv": 5,
        }
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.best_params_ = None

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeRandomForest":
        """Fit Random Forest with optional two-step hyperparameter tuning."""
        logger.info("Fitting Random Forest model")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        if self.config["tune_hyperparameters"]:
            # Check if two-step tuning is enabled
            if self.config.get("use_two_step_tuning", False):
                # Two-step hyperparameter tuning (signature approach)
                from utils.hyperparameter_tuning import apply_two_step_tuning

                rf = RandomForestRegressor(random_state=self.config["random_state"])

                # Apply two-step tuning
                self.model_, self.optimization_summary_ = apply_two_step_tuning(
                    estimator=rf,
                    X=X,
                    y=y,
                    model_name="random_forest",
                    config=self.config.get("global_config"),  # Pass global config
                )

                self.best_params_ = self.model_.get_params()
                logger.info(f"Random Forest fitted with two-step tuning")

            else:
                # Traditional single-step hyperparameter tuning
                rf = RandomForestRegressor(random_state=self.config["random_state"])

                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }

                grid_search = GridSearchCV(
                    rf, param_grid, cv=self.config["cv"], scoring="r2", n_jobs=-1
                )
                grid_search.fit(X, y)

                self.model_ = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_

                logger.info(
                    f"Random Forest fitted with traditional tuning: {self.best_params_}"
                )
        else:
            # Use default parameters
            self.model_ = RandomForestRegressor(
                n_estimators=self.config["n_estimators"],
                max_depth=self.config["max_depth"],
                min_samples_split=self.config["min_samples_split"],
                min_samples_leaf=self.config["min_samples_leaf"],
                random_state=self.config["random_state"],
            )
            self.model_.fit(X, y)

            logger.info("Random Forest fitted with default parameters")

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        importances = self.model_.feature_importances_
        return dict(zip(self.feature_names_, importances))


class CoffeeXGBoost(BaseRegressor):
    """
    XGBoost Regressor wrapper with hyperparameter tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.

        Args:
            config: Configuration dictionary with parameters:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum depth (default: 6)
                - learning_rate: Learning rate (default: 0.1)
                - random_state: Random state (default: 42)
                - tune_hyperparameters: Whether to tune hyperparameters (default: True)
        """
        super().__init__(config)

        if not XGBOOST_AVAILABLE:
            raise ModelError(
                "XGBoost is not available. Please install xgboost package."
            )

        defaults: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "tune_hyperparameters": True,
            "cv": 5,
        }
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.best_params_ = None

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeXGBoost":
        """Fit XGBoost with optional two-step hyperparameter tuning."""
        logger.info("Fitting XGBoost model")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        if self.config["tune_hyperparameters"]:
            # Check if two-step tuning is enabled
            if self.config.get("use_two_step_tuning", False):
                # Two-step hyperparameter tuning (signature approach)
                from utils.hyperparameter_tuning import apply_two_step_tuning

                xgb_model = xgb.XGBRegressor(random_state=self.config["random_state"])

                # Apply two-step tuning
                self.model_, self.optimization_summary_ = apply_two_step_tuning(
                    estimator=xgb_model,
                    X=X,
                    y=y,
                    model_name="xgboost",
                    config=self.config.get("global_config"),  # Pass global config
                )

                self.best_params_ = self.model_.get_params()
                logger.info(f"XGBoost fitted with two-step tuning")

            else:
                # Traditional single-step hyperparameter tuning
                xgb_model = xgb.XGBRegressor(random_state=self.config["random_state"])

                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                }

                grid_search = GridSearchCV(
                    xgb_model, param_grid, cv=self.config["cv"], scoring="r2", n_jobs=-1
                )
                grid_search.fit(X, y)

                self.model_ = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_

                logger.info(
                    f"XGBoost fitted with traditional tuning: {self.best_params_}"
                )
        else:
            # Use default parameters
            self.model_ = xgb.XGBRegressor(
                n_estimators=self.config["n_estimators"],
                max_depth=self.config["max_depth"],
                learning_rate=self.config["learning_rate"],
                random_state=self.config["random_state"],
            )
            self.model_.fit(X, y)

            logger.info("XGBoost fitted with default parameters")

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        importances = self.model_.feature_importances_
        return dict(zip(self.feature_names_, importances))


class CoffeeSVR(BaseRegressor):
    """
    Support Vector Regressor wrapper following thesis methodology.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SVR model.

        Args:
            config: Configuration dictionary with parameters:
                - kernel: Kernel type (default: 'rbf')
                - C: Regularization parameter (default: 1.0)
                - gamma: Kernel coefficient (default: 'scale')
                - epsilon: Epsilon parameter (default: 0.1)
                - scale_features: Whether to scale features (default: True)
                - tune_hyperparameters: Whether to tune hyperparameters (default: True)
                - cv: Cross-validation folds (default: 5)
        """
        super().__init__(config)

        defaults: Dict[str, Any] = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "epsilon": 0.1,
            "scale_features": True,
            "tune_hyperparameters": True,
            "cv": 5,
        }
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.scaler_ = None
        if self.config["scale_features"]:
            self.scaler_ = StandardScaler()

        self.best_params_ = None

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeSVR":
        """Fit SVR with optional two-step hyperparameter tuning."""
        logger.info("Fitting SVR model")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Scale features if requested
        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)

        if self.config["tune_hyperparameters"]:
            # Check if two-step tuning is enabled
            if self.config.get("use_two_step_tuning", False):
                # Two-step hyperparameter tuning (signature approach)
                from utils.hyperparameter_tuning import apply_two_step_tuning
                from sklearn.svm import SVR

                svr = SVR()

                # Apply two-step tuning
                self.model_, self.optimization_summary_ = apply_two_step_tuning(
                    estimator=svr,
                    X=X,
                    y=y,
                    model_name="svr",
                    config=self.config.get("global_config"),  # Pass global config
                )

                self.best_params_ = self.model_.get_params()
                logger.info(f"SVR fitted with two-step tuning")

            else:
                # Traditional single-step hyperparameter tuning
                from sklearn.svm import SVR

                svr = SVR()

                param_grid = {
                    "kernel": ["rbf", "linear", "poly"],
                    "C": [0.1, 1.0, 10.0],
                    "gamma": ["scale", "auto", 0.001, 0.01],
                    "epsilon": [0.01, 0.1, 0.2],
                }

                grid_search = GridSearchCV(
                    svr, param_grid, cv=self.config["cv"], scoring="r2", n_jobs=-1
                )
                grid_search.fit(X, y)

                self.model_ = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_

                logger.info(f"SVR fitted with traditional tuning: {self.best_params_}")
        else:
            # Import SVR
            from sklearn.svm import SVR

            # Use default parameters
            self.model_ = SVR(
                kernel=self.config["kernel"],
                C=self.config["C"],
                gamma=self.config["gamma"],
                epsilon=self.config["epsilon"],
            )
            self.model_.fit(X, y)

            logger.info("SVR fitted with default parameters")

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (not available for SVR)."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        logger.warning("SVR does not provide feature importance scores")
        return {}


class CoffeeDecisionTree(BaseRegressor):
    """
    Decision Tree Regressor wrapper following thesis methodology.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Decision Tree model.

        Args:
            config: Configuration dictionary with parameters:
                - max_depth: Maximum depth (default: None)
                - min_samples_split: Minimum samples to split (default: 2)
                - min_samples_leaf: Minimum samples in leaf (default: 1)
                - random_state: Random state (default: 42)
                - tune_hyperparameters: Whether to tune hyperparameters (default: True)
                - cv: Cross-validation folds (default: 5)
        """
        super().__init__(config)

        defaults: Dict[str, Any] = {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "tune_hyperparameters": True,
            "cv": 5,
        }
        defaults.update(self.config)  # type: ignore[has-type]
        self.config = cast(Dict[str, Any], defaults)

        self.best_params_ = None

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "CoffeeDecisionTree":
        """Fit Decision Tree with optional hyperparameter tuning."""
        logger.info("Fitting Decision Tree model")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        if self.config["tune_hyperparameters"]:
            # Import DecisionTreeRegressor
            from sklearn.tree import DecisionTreeRegressor

            # Hyperparameter tuning
            dt = DecisionTreeRegressor(random_state=self.config["random_state"])

            param_grid = {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

            grid_search = GridSearchCV(
                dt, param_grid, cv=self.config["cv"], scoring="r2", n_jobs=-1
            )
            grid_search.fit(X, y)

            self.model_ = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_

            logger.info(f"Decision Tree fitted with best params: {self.best_params_}")
        else:
            # Import DecisionTreeRegressor
            from sklearn.tree import DecisionTreeRegressor

            # Use default parameters
            self.model_ = DecisionTreeRegressor(
                max_depth=self.config["max_depth"],
                min_samples_split=self.config["min_samples_split"],
                min_samples_leaf=self.config["min_samples_leaf"],
                random_state=self.config["random_state"],
            )
            self.model_.fit(X, y)

            logger.info("Decision Tree fitted with default parameters")

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model_.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Decision Tree."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance")

        importances = self.model_.feature_importances_
        return dict(zip(self.feature_names_, importances))

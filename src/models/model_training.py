"""
Model training utilities for coffee rating prediction.
"""

import os
import logging
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models will not be available.")
    XGBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    logger.warning(
        "SHAP not installed. Feature importance explanations will be limited."
    )
    SHAP_AVAILABLE = False


def prepare_features(
    df: pd.DataFrame, target_column: str, exclude_columns: Optional[List[str]] = None
):
    """
    Prepare features for model training.

    Args:
        df: DataFrame containing features
        target_column: Name of target column
        exclude_columns: Columns to exclude from features

    Returns:
        Tuple: (X, y, feature_names)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Default columns to exclude
    if exclude_columns is None:
        exclude_columns = [
            "id",
            "name",
            "location",
            "origin",
            "country",
            "roaster",
            "url",
            "processed_text",
            "merged_text",
        ]

    # Remove target from features
    exclude_columns = exclude_columns + [target_column]

    # Filter to only exclude columns that exist
    exclude_columns = [col for col in exclude_columns if col in df.columns]

    # Select features
    feature_cols = [col for col in df.columns if col not in exclude_columns]

    # Handle non-numeric columns
    non_numeric_cols = df[feature_cols].select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Removing non-numeric columns: {list(non_numeric_cols)}")
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]

    if not feature_cols:
        raise ValueError("No valid features found for model training")

    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[target_column].copy()

    # Handle missing values in features
    X = X.fillna(X.mean())

    logger.info(f"Prepared {X.shape[1]} features for model training")
    return X, y, feature_cols


def train_linear_regression(X_train, y_train):
    """
    Train a simple linear regression model.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained model
    """
    logger.info("Training linear regression model")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train):
    """
    Train a ridge regression model with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained model
    """
    logger.info("Training ridge regression model with hyperparameter tuning")

    param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    model = Ridge(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f"Best ridge parameters: {grid_search.best_params_}")

    return best_model


def train_lasso_regression(X_train, y_train):
    """
    Train a lasso regression model with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained model
    """
    logger.info("Training lasso regression model with hyperparameter tuning")

    param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    model = Lasso(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f"Best lasso parameters: {grid_search.best_params_}")

    return best_model


def train_random_forest(X_train, y_train):
    """
    Train a random forest regression model.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained model
    """
    logger.info("Training random forest regression model")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost regression model.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained model or None if XGBoost is not available
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available. Skipping XGBoost model.")
        return None

    logger.info("Training XGBoost regression model")

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values

    Returns:
        Dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    return metrics


def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        Dict: Feature importances
    """
    importances = {}

    try:
        # For models with feature_importances_ attribute (tree-based models)
        if hasattr(model, "feature_importances_"):
            for feature, importance in zip(feature_names, model.feature_importances_):
                importances[feature] = float(importance)

        # For linear models
        elif hasattr(model, "coef_"):
            for feature, coef in zip(feature_names, model.coef_):
                importances[feature] = float(abs(coef))

        return importances
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return {}


def plot_feature_importance(importances, title="Feature Importance", n_top=15):
    """
    Plot feature importance.

    Args:
        importances: Dict mapping feature names to importance values
        title: Plot title
        n_top: Number of top features to show

    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Limit to top n features
    if n_top > 0:
        sorted_features = sorted_features[:n_top]

    # Unpack features and values
    features, values = zip(*sorted_features)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(features))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(feature) for feature in features])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_title(title)
    ax.set_xlabel("Importance")

    fig.tight_layout()
    return fig


def save_model(model, model_name, models_dir):
    """
    Save a trained model to disk.

    Args:
        model: Trained model
        model_name: Name of the model
        models_dir: Directory to save the model
    """
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    with open(os.path.join(models_dir, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model '{model_name}' saved to {models_dir}")


def train_and_evaluate_models(
    input_file, target_column, models_to_train=None, models_dir="models"
):
    """
    Train and evaluate multiple models.

    Args:
        input_file: Path to input features file
        target_column: Name of target column
        models_to_train: List of models to train
        models_dir: Directory to save models
    """
    # Default models to train
    if models_to_train is None:
        models_to_train = ["linear", "ridge", "lasso", "random_forest", "xgboost"]

    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    if df.empty:
        raise ValueError(f"No data found in {input_file}")

    # Prepare features and target
    X, y, feature_names = prepare_features(df, target_column)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(
        f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples"
    )

    # Results dictionary
    results = {}

    # Train models
    for model_name in models_to_train:
        try:
            if model_name == "linear":
                model = train_linear_regression(X_train, y_train)
            elif model_name == "ridge":
                model = train_ridge_regression(X_train, y_train)
            elif model_name == "lasso":
                model = train_lasso_regression(X_train, y_train)
            elif model_name == "random_forest":
                model = train_random_forest(X_train, y_train)
            elif model_name == "xgboost":
                model = train_xgboost(X_train, y_train)
                if model is None:
                    continue
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)
            results[model_name] = metrics

            # Get feature importance
            importances = get_feature_importance(model, feature_names)

            # Plot feature importance
            if importances:
                fig = plot_feature_importance(
                    importances, f"{model_name.capitalize()} Feature Importance"
                )

                # Save figure
                output_dir = "output/figures"
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(output_dir, f"{model_name}_importance.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

            # Save model
            save_model(model, model_name, models_dir)

        except Exception as e:
            logger.error(f"Error training {model_name} model: {e}")

    # Save results
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "model_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    logger.info(
        f"Model training and evaluation completed. Results saved to 'output/model_results.json'"
    )
    return results

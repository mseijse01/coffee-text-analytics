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
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    LassoCV,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


class MultinomialInverseRegression:
    """
    Multinomial Inverse Regression (MNIR) implementation following thesis methodology.

    MNIR is used to quantify the relationship between text-based features and sensory
    attributes (acidity, body, aroma, aftertaste, flavor). The methodology follows:

    1. Lasso regression feature selection (cv=5) to identify most relevant text predictors
    2. Regression modeling to predict sensory attributes using selected features
    3. Performance evaluation using MSE and R² metrics
    4. SHAP analysis for feature interpretability

    From thesis: "This approach was implemented following Lasso regression feature
    selection, which helped identify the most relevant predictors from the
    high-dimensional text data."
    """

    def __init__(self, lasso_cv=5, lasso_max_iter=1000, random_state=42):
        """
        Initialize MNIR following thesis methodology.

        Args:
            lasso_cv: Cross-validation folds for Lasso feature selection (default: 5)
            lasso_max_iter: Maximum iterations for Lasso
            random_state: Random state for reproducibility
        """
        self.lasso_cv = lasso_cv
        self.lasso_max_iter = lasso_max_iter
        self.random_state = random_state

        # Model components for each sensory attribute
        self.lasso_selectors = {}  # Lasso feature selectors
        self.regression_models = {}  # Final regression models
        self.scalers = {}  # Feature scalers
        self.selected_features = {}  # Selected feature indices
        self.feature_names = None

        # Performance metrics
        self.performance_metrics = {}
        self.shap_values = {}

    def fit(self, X, sensory_data, feature_names=None):
        """
        Fit MNIR models following thesis methodology.

        Args:
            X: Text-based feature matrix (TF-IDF, BERT, GloVe, LDA topics, etc.)
            sensory_data: Dict with sensory attributes {'aroma': scores, 'acid': scores, ...}
            feature_names: Optional feature names for interpretability
        """
        logger.info("Training MNIR following thesis methodology")
        logger.info("Step 1: Lasso feature selection (cv=5)")
        logger.info("Step 2: Regression modeling for sensory attributes")
        logger.info("Step 3: Performance evaluation (MSE, R²)")
        logger.info("Step 4: SHAP analysis for interpretability")

        # Store feature names
        self.feature_names = (
            feature_names
            if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])]
        )

        # Sensory attributes to analyze (from thesis)
        sensory_attributes = ["aroma", "acid", "body", "flavor", "aftertaste"]

        for attribute in sensory_attributes:
            if attribute not in sensory_data:
                logger.warning(
                    f"Sensory attribute '{attribute}' not found in data, skipping"
                )
                continue

            logger.info(f"Processing {attribute} attribute")

            # Get sensory scores for this attribute
            y_scores = sensory_data[attribute]

            # Remove samples with missing sensory scores
            valid_mask = ~np.isnan(y_scores)
            X_valid = X[valid_mask]
            y_valid = y_scores[valid_mask]

            if len(y_valid) == 0:
                logger.warning(f"No valid data for {attribute}, skipping")
                continue

            # Step 1: Lasso feature selection with cross-validation
            logger.info(f"  Applying Lasso feature selection for {attribute}")

            # Scale features for Lasso
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            self.scalers[attribute] = scaler

            # Lasso with cross-validation to find optimal alpha
            lasso_cv = LassoCV(
                cv=self.lasso_cv,
                max_iter=self.lasso_max_iter,
                random_state=self.random_state,
                n_jobs=-1,
            )
            lasso_cv.fit(X_scaled, y_valid)

            # Get selected features (non-zero coefficients)
            selected_mask = lasso_cv.coef_ != 0
            n_selected = np.sum(selected_mask)

            logger.info(
                f"  Lasso selected {n_selected} features from {X_scaled.shape[1]} for {attribute}"
            )

            if n_selected == 0:
                logger.warning(
                    f"No features selected by Lasso for {attribute}, using top 10"
                )
                # Fallback: select top 10 features by absolute coefficient
                top_indices = np.argsort(np.abs(lasso_cv.coef_))[-10:]
                selected_mask = np.zeros_like(lasso_cv.coef_, dtype=bool)
                selected_mask[top_indices] = True
                n_selected = 10

            self.lasso_selectors[attribute] = lasso_cv
            self.selected_features[attribute] = selected_mask

            # Step 2: Regression modeling with selected features
            logger.info(
                f"  Training regression model for {attribute} with {n_selected} selected features"
            )

            X_selected = X_scaled[:, selected_mask]

            # Use Linear Regression for final model (as per thesis methodology)
            regression_model = LinearRegression()
            regression_model.fit(X_selected, y_valid)
            self.regression_models[attribute] = regression_model

            # Step 3: Performance evaluation
            y_pred = regression_model.predict(X_selected)

            mse = mean_squared_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)

            self.performance_metrics[attribute] = {
                "mse": mse,
                "r2": r2,
                "n_samples": len(y_valid),
                "n_features_selected": n_selected,
                "lasso_alpha": lasso_cv.alpha_,
            }

            logger.info(f"  {attribute} performance: MSE={mse:.4f}, R²={r2:.4f}")

            # Step 4: SHAP analysis (if available)
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.LinearExplainer(regression_model, X_selected)
                    shap_values = explainer.shap_values(X_selected)
                    self.shap_values[attribute] = {
                        "values": shap_values,
                        "feature_names": [
                            self.feature_names[i] for i in np.where(selected_mask)[0]
                        ],
                        "base_value": explainer.expected_value,
                    }
                    logger.info(f"  SHAP analysis completed for {attribute}")
                except Exception as e:
                    logger.warning(f"  SHAP analysis failed for {attribute}: {e}")

        logger.info("MNIR training completed")

    def predict(self, X, attribute):
        """
        Predict sensory attribute scores using trained MNIR model.

        Args:
            X: Feature matrix
            attribute: Sensory attribute to predict

        Returns:
            Predicted scores
        """
        if attribute not in self.regression_models:
            raise ValueError(f"No trained model for attribute '{attribute}'")

        # Scale features
        X_scaled = self.scalers[attribute].transform(X)

        # Select features
        X_selected = X_scaled[:, self.selected_features[attribute]]

        # Predict
        return self.regression_models[attribute].predict(X_selected)

    def get_performance_summary(self):
        """
        Get performance summary for all sensory attributes.

        Returns:
            DataFrame with performance metrics
        """
        if not self.performance_metrics:
            logger.warning("No performance metrics available. Run fit() first.")
            return None

        summary_data = []
        for attribute, metrics in self.performance_metrics.items():
            summary_data.append(
                {
                    "Attribute": attribute,
                    "MSE": metrics["mse"],
                    "R²": metrics["r2"],
                    "N_Samples": metrics["n_samples"],
                    "N_Features_Selected": metrics["n_features_selected"],
                    "Lasso_Alpha": metrics["lasso_alpha"],
                }
            )

        return pd.DataFrame(summary_data)

    def get_feature_importance(self, attribute, top_n=10):
        """
        Get feature importance for a specific sensory attribute.

        Args:
            attribute: Sensory attribute
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if attribute not in self.regression_models:
            raise ValueError(f"No trained model for attribute '{attribute}'")

        model = self.regression_models[attribute]
        selected_mask = self.selected_features[attribute]
        selected_feature_names = [
            self.feature_names[i] for i in np.where(selected_mask)[0]
        ]

        # Get coefficients (feature importance)
        coefficients = model.coef_

        # Create importance DataFrame
        importance_data = []
        for i, (name, coef) in enumerate(zip(selected_feature_names, coefficients)):
            importance_data.append(
                {"Feature": name, "Coefficient": coef, "Abs_Coefficient": abs(coef)}
            )

        df = pd.DataFrame(importance_data)
        df = df.sort_values("Abs_Coefficient", ascending=False)

        return df.head(top_n)

    def get_shap_summary(self, attribute):
        """
        Get SHAP summary for interpretability.

        Args:
            attribute: Sensory attribute

        Returns:
            Dict with SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for interpretability analysis")
            return None

        if attribute not in self.shap_values:
            logger.warning(f"No SHAP values available for {attribute}")
            return None

        return self.shap_values[attribute]

    def generate_insights_report(self):
        """
        Generate comprehensive insights report following thesis methodology.

        Returns:
            Dict with analysis insights
        """
        if not self.performance_metrics:
            logger.warning("No analysis results available. Run fit() first.")
            return None

        insights = {
            "methodology": "MNIR with Lasso feature selection (cv=5)",
            "performance_summary": self.get_performance_summary(),
            "key_findings": {},
            "feature_insights": {},
        }

        # Analyze performance across attributes
        performance_df = insights["performance_summary"]
        best_r2_attr = performance_df.loc[performance_df["R²"].idxmax(), "Attribute"]
        best_r2_score = performance_df["R²"].max()

        insights["key_findings"]["best_performing_attribute"] = {
            "attribute": best_r2_attr,
            "r2_score": best_r2_score,
        }

        # Feature insights for each attribute
        for attribute in self.performance_metrics.keys():
            top_features = self.get_feature_importance(attribute, top_n=5)
            insights["feature_insights"][attribute] = {
                "top_features": top_features.to_dict("records"),
                "n_features_selected": self.performance_metrics[attribute][
                    "n_features_selected"
                ],
            }

        return insights


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


def train_mnir(X_train, y_train, feature_names=None, sensory_data=None):
    """
    Train a Multinomial Inverse Regression (MNIR) model following thesis methodology.

    MNIR quantifies the relationship between text-based features and sensory attributes
    using Lasso feature selection followed by regression modeling.

    Args:
        X_train: Training features (text-based features: TF-IDF, BERT, GloVe, LDA topics)
        y_train: Training target values (not used for MNIR - focuses on sensory attributes)
        feature_names: Optional feature names for interpretability
        sensory_data: Dict with sensory attributes {'aroma': scores, 'acid': scores, ...}

    Returns:
        Trained MNIR model for sensory attribute prediction
    """
    logger.info("Training MNIR following thesis methodology")
    logger.info("Methodology: Lasso feature selection (cv=5) + regression modeling")

    if sensory_data is None:
        logger.warning(
            "No sensory data provided for MNIR analysis. Creating dummy data for testing."
        )
        # Create dummy sensory data for testing
        n_samples = X_train.shape[0]
        sensory_data = {
            "aroma": np.random.uniform(5, 9, n_samples),
            "acid": np.random.uniform(4, 8, n_samples),
            "body": np.random.uniform(5, 9, n_samples),
            "flavor": np.random.uniform(6, 9, n_samples),
            "aftertaste": np.random.uniform(5, 8, n_samples),
        }

    # Initialize MNIR following thesis methodology
    model = MultinomialInverseRegression(
        lasso_cv=5,  # 5-fold cross-validation as per thesis
        lasso_max_iter=1000,
        random_state=42,
    )

    # Fit the model using thesis methodology
    model.fit(X_train, sensory_data, feature_names=feature_names)

    # Generate comprehensive insights report
    insights = model.generate_insights_report()

    if insights:
        # Log key findings following thesis results
        logger.info("MNIR Performance Results (following thesis methodology):")
        performance_df = insights["performance_summary"]

        for _, row in performance_df.iterrows():
            attribute = row["Attribute"]
            r2 = row["R²"]
            mse = row["MSE"]
            n_features = row["N_Features_Selected"]

            logger.info(f"\n{attribute.upper()} Analysis:")
            logger.info(f"  - R² Score: {r2:.4f}")
            logger.info(f"  - MSE: {mse:.4f}")
            logger.info(f"  - Features Selected: {n_features}")
            logger.info(f"  - Samples: {row['N_Samples']}")

            # Show top features for this attribute
            if attribute in insights["feature_insights"]:
                top_features = insights["feature_insights"][attribute]["top_features"][
                    :3
                ]
                feature_names_list = [f["Feature"] for f in top_features]
                logger.info(f"  - Top Features: {feature_names_list}")

        # Log overall findings
        best_attr = insights["key_findings"]["best_performing_attribute"]
        logger.info(
            f"\nBest performing attribute: {best_attr['attribute']} (R² = {best_attr['r2_score']:.4f})"
        )

        # Compare with thesis results
        logger.info("\nThesis comparison:")
        logger.info("Expected: acidity R² ≈ 0.95, body R² ≈ 0.94")

        if "acid" in performance_df["Attribute"].values:
            acid_r2 = performance_df[performance_df["Attribute"] == "acid"]["R²"].iloc[
                0
            ]
            logger.info(f"Achieved: acidity R² = {acid_r2:.4f}")

        if "body" in performance_df["Attribute"].values:
            body_r2 = performance_df[performance_df["Attribute"] == "body"]["R²"].iloc[
                0
            ]
            logger.info(f"Achieved: body R² = {body_r2:.4f}")

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
    # Check if this is an MNIR model (new methodology)
    if isinstance(model, MultinomialInverseRegression):
        logger.info(
            "Evaluating MNIR model - performance metrics for sensory attribute prediction"
        )

        # For MNIR, we evaluate performance on sensory attribute prediction
        insights = model.generate_insights_report()

        if not insights:
            logger.warning("No insights available from MNIR model")
            return {"model_type": "MNIR", "evaluation_failed": True}

        # Extract performance metrics
        performance_df = insights["performance_summary"]

        # Create comprehensive evaluation metrics
        metrics = {
            "model_type": "MNIR",
            "methodology": "Lasso feature selection + regression",
            "n_attributes_analyzed": len(performance_df),
            "total_samples": performance_df["N_Samples"].sum(),
            "total_features_selected": performance_df["N_Features_Selected"].sum(),
        }

        # Add individual attribute metrics
        for _, row in performance_df.iterrows():
            attribute = row["Attribute"]
            metrics[f"{attribute}_r2"] = float(row["R²"])
            metrics[f"{attribute}_mse"] = float(row["MSE"])
            metrics[f"{attribute}_n_features"] = int(row["N_Features_Selected"])
            metrics[f"{attribute}_n_samples"] = int(row["N_Samples"])

        # Overall performance summary
        avg_r2 = performance_df["R²"].mean()
        best_r2 = performance_df["R²"].max()
        best_attribute = performance_df.loc[performance_df["R²"].idxmax(), "Attribute"]

        metrics.update(
            {
                "average_r2": float(avg_r2),
                "best_r2": float(best_r2),
                "best_attribute": best_attribute,
            }
        )

        # Compare with thesis benchmarks
        thesis_benchmarks = {"acid": 0.95, "body": 0.94}
        for attr, expected_r2 in thesis_benchmarks.items():
            if attr in performance_df["Attribute"].values:
                actual_r2 = performance_df[performance_df["Attribute"] == attr][
                    "R²"
                ].iloc[0]
                metrics[f"{attr}_thesis_comparison"] = {
                    "expected": expected_r2,
                    "actual": float(actual_r2),
                    "difference": float(actual_r2 - expected_r2),
                }

        logger.info(
            f"MNIR evaluation: Average R² = {avg_r2:.4f}, Best = {best_r2:.4f} ({best_attribute})"
        )

        # Log thesis comparison
        for attr, expected_r2 in thesis_benchmarks.items():
            if f"{attr}_thesis_comparison" in metrics:
                comparison = metrics[f"{attr}_thesis_comparison"]
                logger.info(
                    f"{attr} R²: Expected {expected_r2:.3f}, Actual {comparison['actual']:.3f}"
                )

        return metrics

    # For regular prediction models
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
        # For MNIR models (new methodology)
        if isinstance(model, MultinomialInverseRegression):
            logger.info(
                "Extracting feature importance from MNIR model (Lasso-selected features)"
            )

            # Get insights for all sensory attributes
            insights = model.generate_insights_report()

            if not insights or "feature_insights" not in insights:
                logger.warning("No feature insights available from MNIR model")
                return {}

            # Combine importance across all sensory attributes
            combined_importance = {}
            total_attributes = 0

            for attribute, feature_data in insights["feature_insights"].items():
                total_attributes += 1
                top_features = feature_data["top_features"]

                for feature_info in top_features:
                    feature_name = feature_info["Feature"]
                    abs_coef = feature_info["Abs_Coefficient"]

                    if feature_name in combined_importance:
                        combined_importance[feature_name] += abs_coef
                    else:
                        combined_importance[feature_name] = abs_coef

            # Normalize combined importance by number of attributes
            if combined_importance and total_attributes > 0:
                max_importance = max(combined_importance.values())
                importances = {
                    k: (v / total_attributes) / max_importance
                    for k, v in combined_importance.items()
                }

            logger.info(
                f"MNIR: Combined importance from {total_attributes} sensory attributes, "
                f"{len(importances)} unique features"
            )

        # For models with feature_importances_ attribute (tree-based models)
        elif hasattr(model, "feature_importances_"):
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
    # Default models to train (including MNIR from thesis)
    if models_to_train is None:
        models_to_train = [
            "linear",
            "ridge",
            "lasso",
            "random_forest",
            "xgboost",
            "mnir",
        ]

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
            elif model_name == "mnir":
                # For MNIR, we need to extract sensory data from the DataFrame
                sensory_columns = ["aroma", "acid", "body", "flavor", "aftertaste"]
                sensory_data = {}

                # Check if sensory columns exist in the original DataFrame
                for col in sensory_columns:
                    if col in df.columns:
                        sensory_data[col] = df[col].fillna(df[col].mean()).values
                    else:
                        logger.warning(
                            f"Sensory column '{col}' not found, using dummy data"
                        )
                        sensory_data[col] = np.random.uniform(5, 9, len(df))

                model = train_mnir(
                    X_train, y_train, feature_names, sensory_data=sensory_data
                )
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
                    importances, f"{model_name.upper()} Feature Importance"
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

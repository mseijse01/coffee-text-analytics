"""
Multinomial Inverse Regression (MNIR) implementation for coffee review analysis.

This module implements MNIR following the thesis methodology for analyzing
the relationship between text-based features and sensory attributes.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, Union
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from .base import BaseRegressor, ModelError

logger = logging.getLogger(__name__)

# Check for SHAP availability
try:
    import shap

    SHAP_AVAILABLE = True
    logger.info("SHAP available - feature importance explanations enabled")
except ImportError:
    logger.warning(
        "SHAP not installed. Feature importance explanations will be limited."
    )
    SHAP_AVAILABLE = False


class MultinomialInverseRegression(BaseRegressor):
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MNIR following thesis methodology.

        Args:
            config: Configuration dictionary with parameters:
                - lasso_cv: Cross-validation folds for Lasso feature selection (default: 5)
                - lasso_max_iter: Maximum iterations for Lasso (default: 1000)
                - random_state: Random state for reproducibility (default: 42)
                - sensory_attributes: List of sensory attributes to analyze
        """
        super().__init__(config)

        # Set default configuration
        default_config = {
            "lasso_cv": 5,
            "lasso_max_iter": 1000,
            "random_state": 42,
            "sensory_attributes": ["aroma", "acid", "body", "flavor", "aftertaste"],
        }
        default_config.update(self.config)
        self.config = default_config

        # Model components for each sensory attribute
        self.lasso_selectors = {}  # Lasso feature selectors
        self.regression_models = {}  # Final regression models
        self.scalers = {}  # Feature scalers
        self.selected_features = {}  # Selected feature indices
        self.feature_names = None

        # Performance metrics
        self.performance_metrics = {}
        self.shap_values = {}
        self.shap_explainers = {}

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], sensory_data: Dict[str, np.ndarray]
    ) -> "MultinomialInverseRegression":
        """
        Fit MNIR models following thesis methodology.

        Args:
            X: Text-based feature matrix (TF-IDF, BERT, GloVe, LDA topics, etc.)
            sensory_data: Dict with sensory attributes {'aroma': scores, 'acid': scores, ...}

        Returns:
            Self for method chaining
        """
        logger.info("Training MNIR following thesis methodology")
        logger.info("Step 1: Lasso feature selection (cv=5)")
        logger.info("Step 2: Regression modeling for sensory attributes")
        logger.info("Step 3: Performance evaluation (MSE, R²)")
        logger.info("Step 4: SHAP analysis for interpretability")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Process each sensory attribute
        sensory_attributes = self.config["sensory_attributes"]

        for attribute in sensory_attributes:
            if attribute not in sensory_data:
                logger.warning(
                    f"Sensory attribute '{attribute}' not found in data, skipping"
                )
                continue

            logger.info(f"Processing {attribute} attribute")
            self._fit_attribute_model(X, sensory_data[attribute], attribute)

        self.is_fitted = True
        logger.info("MNIR models fitted successfully")
        return self

    def _fit_attribute_model(
        self, X: np.ndarray, y_scores: np.ndarray, attribute: str
    ) -> None:
        """
        Fit MNIR model for a specific sensory attribute.

        Args:
            X: Feature matrix
            y_scores: Sensory scores for the attribute
            attribute: Name of the sensory attribute
        """
        # Remove samples with missing sensory scores
        valid_mask = ~np.isnan(y_scores)
        X_valid = X[valid_mask]
        y_valid = y_scores[valid_mask]

        if len(y_valid) == 0:
            logger.warning(f"No valid data for {attribute}, skipping")
            return

        # Step 1: Lasso feature selection with cross-validation
        logger.info(f"  Applying Lasso feature selection for {attribute}")

        # Scale features for Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        self.scalers[attribute] = scaler

        # Lasso with cross-validation to find optimal alpha
        lasso_cv = LassoCV(
            cv=self.config["lasso_cv"],
            max_iter=self.config["lasso_max_iter"],
            random_state=self.config["random_state"],
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
            "rmse": np.sqrt(mse),
            "r2": r2,
            "n_samples": len(y_valid),
            "n_features_selected": n_selected,
            "lasso_alpha": lasso_cv.alpha_,
        }

        logger.info(f"  {attribute} - MSE: {mse:.4f}, R²: {r2:.4f}")

        # Step 4: SHAP analysis for interpretability
        if SHAP_AVAILABLE:
            self._compute_shap_values(X_selected, regression_model, attribute)

    def _compute_shap_values(
        self, X_selected: np.ndarray, model: LinearRegression, attribute: str
    ) -> None:
        """
        Compute SHAP values for feature interpretability.

        Args:
            X_selected: Selected features matrix
            model: Fitted regression model
            attribute: Sensory attribute name
        """
        try:
            logger.info(f"  Computing SHAP values for {attribute}")

            # Create SHAP explainer
            explainer = shap.LinearExplainer(model, X_selected)
            shap_values = explainer.shap_values(X_selected)

            self.shap_explainers[attribute] = explainer
            self.shap_values[attribute] = shap_values

            logger.info(f"  SHAP values computed for {attribute}")

        except Exception as e:
            logger.warning(f"Failed to compute SHAP values for {attribute}: {e}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame], attribute: str) -> np.ndarray:
        """
        Make predictions for a specific sensory attribute.

        Args:
            X: Feature matrix
            attribute: Sensory attribute to predict

        Returns:
            Predictions for the attribute
        """
        if not self.is_fitted:
            raise ModelError("MNIR model must be fitted before prediction")

        if attribute not in self.regression_models:
            raise ModelError(f"No model fitted for attribute: {attribute}")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scalers[attribute].transform(X)

        # Select features
        X_selected = X_scaled[:, self.selected_features[attribute]]

        # Make predictions
        return self.regression_models[attribute].predict(X_selected)

    def predict_all_attributes(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for all fitted sensory attributes.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping attributes to predictions
        """
        predictions = {}
        for attribute in self.regression_models.keys():
            predictions[attribute] = self.predict(X, attribute)
        return predictions

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary for all attributes.

        Returns:
            Dictionary with performance metrics for each attribute
        """
        return self.performance_metrics.copy()

    def get_feature_importance(self, attribute: str, top_n: int = 10) -> List[tuple]:
        """
        Get feature importance for a specific attribute.

        Args:
            attribute: Sensory attribute
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if attribute not in self.regression_models:
            raise ModelError(f"No model fitted for attribute: {attribute}")

        # Get selected feature indices
        selected_mask = self.selected_features[attribute]
        selected_feature_names = [
            self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]
        ]

        # Get coefficients from the regression model
        coefficients = self.regression_models[attribute].coef_

        # Create feature importance list
        feature_importance = list(zip(selected_feature_names, np.abs(coefficients)))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance[:top_n]

    def get_shap_summary(self, attribute: str) -> Optional[Dict[str, Any]]:
        """
        Get SHAP summary for a specific attribute.

        Args:
            attribute: Sensory attribute

        Returns:
            Dictionary with SHAP analysis results
        """
        if not SHAP_AVAILABLE or attribute not in self.shap_values:
            return None

        shap_values = self.shap_values[attribute]

        # Get selected feature names
        selected_mask = self.selected_features[attribute]
        selected_feature_names = [
            self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]
        ]

        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance from SHAP
        shap_importance = list(zip(selected_feature_names, mean_shap))
        shap_importance.sort(key=lambda x: x[1], reverse=True)

        return {
            "shap_values": shap_values,
            "feature_names": selected_feature_names,
            "importance_ranking": shap_importance,
            "mean_abs_shap": mean_shap,
        }

    def generate_insights_report(self) -> str:
        """
        Generate a comprehensive insights report for all attributes.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=== MNIR Analysis Report ===\n")

        # Overall summary
        report.append(f"Analyzed {len(self.regression_models)} sensory attributes")
        report.append(
            f"Feature selection method: Lasso CV (cv={self.config['lasso_cv']})"
        )
        report.append(f"Total features available: {len(self.feature_names)}\n")

        # Performance summary
        report.append("=== Performance Summary ===")
        for attribute, metrics in self.performance_metrics.items():
            report.append(f"\n{attribute.upper()}:")
            report.append(f"  R² Score: {metrics['r2']:.4f}")
            report.append(f"  RMSE: {metrics['rmse']:.4f}")
            report.append(f"  Features Selected: {metrics['n_features_selected']}")
            report.append(f"  Lasso Alpha: {metrics['lasso_alpha']:.6f}")

        # Feature importance
        report.append("\n=== Top Features by Attribute ===")
        for attribute in self.regression_models.keys():
            report.append(f"\n{attribute.upper()} - Top 5 Features:")
            top_features = self.get_feature_importance(attribute, top_n=5)
            for i, (feature, importance) in enumerate(top_features, 1):
                report.append(f"  {i}. {feature}: {importance:.4f}")

        # SHAP insights
        if SHAP_AVAILABLE and self.shap_values:
            report.append("\n=== SHAP Analysis Available ===")
            for attribute in self.shap_values.keys():
                report.append(f"  {attribute}: SHAP values computed")

        return "\n".join(report)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get overall feature importance across all attributes.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Aggregate importance across all attributes
        feature_importance = {}

        for attribute in self.regression_models.keys():
            attr_importance = self.get_feature_importance(
                attribute, top_n=len(self.feature_names)
            )

            for feature_name, importance in attr_importance:
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = 0
                feature_importance[feature_name] += importance

        # Normalize by number of attributes
        n_attributes = len(self.regression_models)
        for feature_name in feature_importance:
            feature_importance[feature_name] /= n_attributes

        return feature_importance

    def save_model(self, filepath: str) -> None:
        """
        Save the fitted MNIR model.

        Args:
            filepath: Path to save the model
        """
        import pickle

        model_data = {
            "config": self.config,
            "lasso_selectors": self.lasso_selectors,
            "regression_models": self.regression_models,
            "scalers": self.scalers,
            "selected_features": self.selected_features,
            "feature_names": self.feature_names,
            "performance_metrics": self.performance_metrics,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"MNIR model saved to {filepath}")

    def load_model(self, filepath: str) -> "MultinomialInverseRegression":
        """
        Load a previously saved MNIR model.

        Args:
            filepath: Path to the saved model

        Returns:
            Self for method chaining
        """
        import pickle

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.config = model_data["config"]
        self.lasso_selectors = model_data["lasso_selectors"]
        self.regression_models = model_data["regression_models"]
        self.scalers = model_data["scalers"]
        self.selected_features = model_data["selected_features"]
        self.feature_names = model_data["feature_names"]
        self.performance_metrics = model_data["performance_metrics"]
        self.is_fitted = model_data["is_fitted"]

        logger.info(f"MNIR model loaded from {filepath}")
        return self

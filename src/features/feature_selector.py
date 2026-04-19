"""
Corrected LASSO-based Feature Selection for Coffee Text Analytics

This module implements the CORRECT LASSO-based feature selection following
exact thesis methodology:
1. Combine ALL text features from all desc columns
2. Apply single LASSO with CV to select best text features
3. Keep sensory and categorical features separate
4. Final feature set: selected_text + sensory + categorical
"""

import numpy as np
import pandas as pd
import polars as pl
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class LassoFeatureSelector:
    """
    Corrected LASSO-based feature selector following exact thesis methodology.

    Supports Polars-first architecture with pandas/numpy fallback support.

    Thesis approach:
    1. Identify text features (all TF-IDF, BERT, GloVe, topics, sentiment from all desc columns)
    2. Combine all text features into single matrix
    3. Apply LASSO with CV to select ~500-1000 most predictive text features
    4. Keep sensory features separate (aroma, acid, body, flavor, aftertaste)
    5. Keep categorical features separate (origin, roast, etc.)
    6. Final feature set: selected_text + sensory + categorical
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize corrected LASSO feature selector.

        Args:
            config: Configuration dictionary with parameters:
                - alpha_range: List of alpha values for CV (default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
                - cv_folds: Number of CV folds (default: 5)
                - target_text_features: Target number of text features to select (default: 500)
                - min_text_features: Minimum text features to select (default: 100)
                - max_text_features: Maximum text features to select (default: 1000)
                - selection_threshold: Threshold for feature selection (default: 'mean')
                - random_state: Random state for reproducibility (default: 57)
                - scale_features: Whether to scale features (default: True)
        """
        self.config = config or {}

        # Set default configuration following thesis methodology
        default_config = {
            "alpha_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "cv_folds": 5,
            "target_text_features": 500,  # Target ~500 text features (thesis range)
            "min_text_features": 100,  # Minimum text features
            "max_text_features": 1000,  # Maximum text features
            "selection_threshold": "mean",
            "random_state": 57,
            "scale_features": True,
        }

        # Update with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

        # Initialize components
        self.text_features_ = []
        self.sensory_features_ = []
        self.categorical_features_ = []
        self.selected_text_features_ = []
        self.final_features_ = []
        self.text_selector_ = None
        self.text_scaler_ = None
        self.feature_importance_ = {}
        self.selection_stats_ = {}
        self.is_fitted_ = False

        logger.info(
            f"LassoFeatureSelector initialized with config: {self.config}"
        )
        logger.info(
            "Following thesis methodology: combine all text features, then apply LASSO"
        )

    def _identify_feature_types(
        self, feature_names: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify feature types following thesis methodology.

        Args:
            feature_names: List of all feature names

        Returns:
            Tuple of (text_features, sensory_features, categorical_features)
        """
        text_features = []
        sensory_features = []
        categorical_features = []

        # Define sensory features (thesis methodology)
        sensory_keywords = ["aroma", "acid", "body", "flavor", "aftertaste"]

        # Define categorical features (thesis methodology)
        categorical_keywords = [
            "origin",
            "roast",
            "roaster",
            "country",
            "location",
            "price",
        ]

        for feature in feature_names:
            feature_lower = feature.lower()

            # Check if it's a text feature (from any desc column)
            if any(
                prefix in feature
                for prefix in [
                    "tfidf_desc_",
                    "bert_desc_",
                    "glove_desc_",
                    "lda_topic_",
                    "nmf_topic_",
                    "sentiment_desc_",
                ]
            ):
                text_features.append(feature)
            # Check if it's a sensory feature
            elif any(keyword in feature_lower for keyword in sensory_keywords):
                sensory_features.append(feature)
            # Check if it's a categorical feature
            elif any(keyword in feature_lower for keyword in categorical_keywords):
                categorical_features.append(feature)
            else:
                # Default: treat as categorical if not clearly text or sensory
                categorical_features.append(feature)

        logger.info(f"Feature type identification (thesis methodology):")
        logger.info(
            f"  Text features: {len(text_features)} (all desc columns combined)"
        )
        logger.info(
            f"  Sensory features: {len(sensory_features)} (aroma, acid, body, flavor, aftertaste)"
        )
        logger.info(
            f"  Categorical features: {len(categorical_features)} (origin, roast, etc.)"
        )

        return text_features, sensory_features, categorical_features

    def fit_select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> "LassoFeatureSelector":
        """
        Fit LASSO feature selector following thesis methodology.

        Args:
            X: Feature matrix (Polars DataFrame preferred, pandas DataFrame or numpy array supported)
            y: Target variable (Polars Series preferred, pandas Series or numpy array supported)

        Returns:
            Self for method chaining
        """
        logger.info("Fitting corrected LASSO feature selector (thesis methodology)")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        elif isinstance(X, pl.DataFrame):
            # For Polars, convert to pandas for sklearn compatibility but preserve original
            feature_names = X.columns
            X_df = X.to_pandas()  # Convert to pandas for sklearn operations
        else:
            # For pandas, use copy
            X_df = X.copy()
            feature_names = X_df.columns.tolist()

        # Identify feature types following thesis methodology
        self.text_features_, self.sensory_features_, self.categorical_features_ = (
            self._identify_feature_types(feature_names)
        )

        if not self.text_features_:
            logger.warning("No text features found! Check feature naming convention.")
            self.selected_text_features_ = []
            self.final_features_ = self.sensory_features_ + self.categorical_features_
            self.is_fitted_ = True
            return self

        # Extract text features matrix (thesis step 1: combine all text features)
        X_text = X_df[self.text_features_].values
        logger.info(f"Combined text features matrix: {X_text.shape}")

        # Scale text features if requested
        if self.config["scale_features"]:
            self.text_scaler_ = StandardScaler()
            X_text_scaled = self.text_scaler_.fit_transform(X_text)
            logger.info("Text features scaled using StandardScaler")
        else:
            X_text_scaled = X_text

        # Apply LASSO with cross-validation (thesis step 2: single LASSO selection)
        logger.info("Applying LASSO with cross-validation to combined text features")
        self.text_selector_ = LassoCV(
            alphas=self.config["alpha_range"],
            cv=self.config["cv_folds"],
            random_state=self.config["random_state"],
            max_iter=2000,
            n_jobs=-1,
        )

        self.text_selector_.fit(X_text_scaled, y)

        # Apply LASSO feature selection to text features (thesis step 2)
        # Adapt max_features to actual number of text features available
        max_text_features_adapted = min(
            self.config["max_text_features"], len(self.text_features_)
        )

        selector = SelectFromModel(
            self.text_selector_,
            threshold=self.config["selection_threshold"],
            max_features=max_text_features_adapted,
        )

        selector.fit(X_text_scaled, y)
        selected_mask = selector.get_support()
        selected_indices = np.where(selected_mask)[0]

        # Ensure we have reasonable number of features
        if len(selected_indices) < self.config["min_text_features"]:
            # Select top features by absolute coefficient value
            abs_coefs = np.abs(self.text_selector_.coef_)
            n_select = min(self.config["min_text_features"], len(abs_coefs))
            top_indices = np.argsort(abs_coefs)[-n_select:]
            selected_mask = np.zeros(len(abs_coefs), dtype=bool)
            selected_mask[top_indices] = True
            selected_indices = top_indices
            logger.info(f"Enforced minimum {n_select} text features")

        # Store selected text features
        self.selected_text_features_ = [
            self.text_features_[i] for i in selected_indices
        ]

        # Create final feature list (thesis step 3: combine with sensory + categorical)
        self.final_features_ = (
            self.selected_text_features_
            + self.sensory_features_
            + self.categorical_features_
        )

        # Calculate feature importance
        self.feature_importance_ = {}
        for i, feature in enumerate(self.selected_text_features_):
            original_idx = self.text_features_.index(feature)
            self.feature_importance_[feature] = abs(
                self.text_selector_.coef_[original_idx]
            )

        # Store selection statistics
        self.selection_stats_ = {
            "original_text_features": len(self.text_features_),
            "selected_text_features": len(self.selected_text_features_),
            "sensory_features": len(self.sensory_features_),
            "categorical_features": len(self.categorical_features_),
            "total_final_features": len(self.final_features_),
            "text_reduction_ratio": 1
            - (len(self.selected_text_features_) / len(self.text_features_)),
            "lasso_alpha": self.text_selector_.alpha_,
            "lasso_score": self.text_selector_.score(X_text_scaled, y),
        }

        self.is_fitted_ = True

        logger.info("LASSO feature selection completed (thesis methodology)")
        logger.info(
            f"  Original text features: {self.selection_stats_['original_text_features']}"
        )
        logger.info(
            f"  Selected text features: {self.selection_stats_['selected_text_features']}"
        )
        logger.info(f"  Sensory features: {self.selection_stats_['sensory_features']}")
        logger.info(
            f"  Categorical features: {self.selection_stats_['categorical_features']}"
        )
        logger.info(
            f"  Total final features: {self.selection_stats_['total_final_features']}"
        )
        logger.info(
            f"  Text reduction ratio: {self.selection_stats_['text_reduction_ratio']:.1%}"
        )
        logger.info(f"  Optimal alpha: {self.selection_stats_['lasso_alpha']:.6f}")

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Transform features using fitted selector.

        Args:
            X: Feature matrix to transform (Polars DataFrame preferred, pandas DataFrame or numpy array supported)

        Returns:
            Transformed feature matrix with selected features only (maintains input type when possible)
        """
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted before transformation")

        # Remember input type for output
        input_type = type(X)

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        elif isinstance(X, pl.DataFrame):
            # For Polars, convert to pandas for processing
            X_df = X.to_pandas()
        else:
            # For pandas, use copy
            X_df = X.copy()

        # Select final features
        available_features = [f for f in self.final_features_ if f in X_df.columns]
        if len(available_features) != len(self.final_features_):
            missing = set(self.final_features_) - set(available_features)
            logger.warning(f"Missing features in transform: {missing}")

        X_selected = X_df[available_features]

        # Convert back to original type if needed
        if input_type == pl.DataFrame:
            X_selected = pl.from_pandas(X_selected)
        elif input_type == np.ndarray:
            X_selected = X_selected.values

        logger.info(f"Transformed features: {X_selected.shape}")
        return X_selected

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> Union[np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Fit selector and transform features in one step.

        Args:
            X: Feature matrix (Polars DataFrame preferred, pandas DataFrame or numpy array supported)
            y: Target variable (Polars Series preferred, pandas Series or numpy array supported)

        Returns:
            Transformed feature matrix
        """
        return self.fit_select_features(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if not self.is_fitted_:
            return []
        return self.final_features_.copy()

    def get_text_features(self) -> List[str]:
        """Get list of selected text feature names."""
        if not self.is_fitted_:
            return []
        return self.selected_text_features_.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores for selected text features."""
        if not self.is_fitted_:
            return {}
        return self.feature_importance_.copy()

    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection process."""
        if not self.is_fitted_:
            return {}
        return self.selection_stats_.copy()

    def print_summary(self) -> None:
        """Print detailed summary of feature selection."""
        if not self.is_fitted_:
            print("Selector not fitted yet")
            return

        print("\n" + "=" * 60)
        print("CORRECTED LASSO FEATURE SELECTION SUMMARY")
        print("Following Thesis Methodology")
        print("=" * 60)

        stats = self.selection_stats_
        print(f"Original text features: {stats['original_text_features']:,}")
        print(f"Selected text features: {stats['selected_text_features']:,}")
        print(f"Sensory features: {stats['sensory_features']:,}")
        print(f"Categorical features: {stats['categorical_features']:,}")
        print(f"Total final features: {stats['total_final_features']:,}")
        print(f"Text reduction ratio: {stats['text_reduction_ratio']:.1%}")
        print(f"Optimal LASSO alpha: {stats['lasso_alpha']:.6f}")
        print(f"LASSO R² score: {stats['lasso_score']:.4f}")

        # Show top text features by importance
        if self.feature_importance_:
            print(f"\nTop 10 Selected Text Features:")
            sorted_features = sorted(
                self.feature_importance_.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feature}: {importance:.4f}")

        print("=" * 60)

    def save_selector(self, filepath: Union[str, Path]) -> None:
        """Save fitted selector to file."""
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted selector")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Selector saved to {filepath}")

    @classmethod
    def load_selector(
        cls, filepath: Union[str, Path]
    ) -> "LassoFeatureSelector":
        """Load fitted selector from file."""
        with open(filepath, "rb") as f:
            selector = pickle.load(f)
        logger.info(f"Selector loaded from {filepath}")
        return selector

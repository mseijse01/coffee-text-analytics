"""
LASSO-based Feature Selection for Coffee Text Analytics

This module implements group-wise LASSO feature selection optimized for coffee
text analytics data with support for Polars-first architecture.
"""

import numpy as np
import pandas as pd
import polars as pl
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class LassoFeatureSelector:
    """
    Group-wise LASSO feature selector with Polars-first support.

    Performs LASSO feature selection on feature groups (TF-IDF, BERT, sensory, etc.)
    with cross-validation for optimal regularization. Supports Polars DataFrames
    as the preferred input type with pandas/numpy fallback support.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LASSO feature selector.

        Args:
            config: Configuration dictionary with parameters:
                - alpha_range: List of alpha values for CV (default: from config)
                - cv_folds: Number of CV folds (default: 5)
                - max_features_per_group: Maximum features per group (default: 200)
                - min_features_per_group: Minimum features per group (default: 10)
                - selection_threshold: Threshold for feature selection (default: 'mean')
                - random_state: Random state for reproducibility (default: 57)
                - scale_features: Whether to scale features (default: True)
        """
        self.config = config or {}

        # Set default configuration
        default_config = {
            "alpha_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "cv_folds": 5,
            "max_features_per_group": 200,
            "min_features_per_group": 10,
            "selection_threshold": "mean",
            "random_state": 57,
            "scale_features": True,
        }

        # Update with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

        # Initialize components
        self.feature_groups_ = {}
        self.group_selectors_ = {}
        self.group_scalers_ = {}
        self.selected_features_ = []
        self.feature_importance_ = {}
        self.selection_stats_ = {}
        self.is_fitted_ = False

        logger.info(f"LassoFeatureSelector initialized with config: {self.config}")

    def _identify_feature_groups(
        self, feature_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Identify feature groups based on feature name patterns.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping group names to feature lists
        """
        groups = {
            "tfidf": [],
            "bert": [],
            "topics_lda": [],
            "topics_nmf": [],
            "sentiment": [],
            "glove": [],
            "sensory": [],
            "metadata": [],
        }

        for feature in feature_names:
            feature_lower = feature.lower()

            if feature.startswith("tfidf_"):
                groups["tfidf"].append(feature)
            elif feature.startswith("bert_"):
                groups["bert"].append(feature)
            elif feature.startswith("lda_topic_"):
                groups["topics_lda"].append(feature)
            elif feature.startswith("nmf_topic_"):
                groups["topics_nmf"].append(feature)
            elif "sentiment" in feature_lower:
                groups["sentiment"].append(feature)
            elif feature.startswith("glove_"):
                groups["glove"].append(feature)
            elif feature_lower in ["aroma", "acid", "body", "flavor", "aftertaste"]:
                groups["sensory"].append(feature)
            else:
                groups["metadata"].append(feature)

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}

        logger.info(f"Identified {len(groups)} feature groups:")
        for group_name, features in groups.items():
            logger.info(f"  {group_name}: {len(features)} features")

        return groups

    def _select_features_for_group(
        self, X_group: np.ndarray, y: np.ndarray, group_name: str
    ) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """
        Select features for a specific group using LASSO with cross-validation.

        Args:
            X_group: Feature matrix for the group
            y: Target variable
            group_name: Name of the feature group

        Returns:
            Tuple of (selected_features_mask, selected_indices, selection_stats)
        """
        logger.info(
            f"Selecting features for group '{group_name}' ({X_group.shape[1]} features)"
        )

        # Scale features if requested
        scaler = None
        if self.config["scale_features"]:
            scaler = StandardScaler()
            X_group_scaled = scaler.fit_transform(X_group)
        else:
            X_group_scaled = X_group

        # Perform LASSO with cross-validation
        lasso_cv = LassoCV(
            alphas=self.config["alpha_range"],
            cv=self.config["cv_folds"],
            random_state=self.config["random_state"],
            max_iter=2000,
            n_jobs=-1,
        )

        lasso_cv.fit(X_group_scaled, y)

        # Create feature selector based on LASSO coefficients
        # Adapt max_features to group size
        max_features_for_group = min(
            self.config["max_features_per_group"], X_group.shape[1]
        )

        selector = SelectFromModel(
            lasso_cv,
            threshold=self.config["selection_threshold"],
            max_features=max_features_for_group,
        )

        # Fit selector and get selected features
        selector.fit(X_group_scaled, y)
        selected_mask = selector.get_support()
        selected_indices = np.where(selected_mask)[0]

        # Ensure minimum number of features (but not more than available)
        min_features_for_group = min(
            self.config["min_features_per_group"], X_group.shape[1]
        )

        if len(selected_indices) < min_features_for_group:
            # Select top features by absolute coefficient value
            abs_coefs = np.abs(lasso_cv.coef_)
            top_indices = np.argsort(abs_coefs)[-min_features_for_group:]
            selected_mask = np.zeros(len(abs_coefs), dtype=bool)
            selected_mask[top_indices] = True
            selected_indices = top_indices
            logger.info(
                f"Enforced minimum {min_features_for_group} features for group '{group_name}' (adapted from {self.config['min_features_per_group']})"
            )

        # Calculate selection statistics
        selection_stats = {
            "original_features": X_group.shape[1],
            "selected_features": len(selected_indices),
            "selection_ratio": len(selected_indices) / X_group.shape[1],
            "best_alpha": lasso_cv.alpha_,
            "cv_score": lasso_cv.score(X_group_scaled, y),
            "mean_cv_score": np.mean(lasso_cv.mse_path_).item(),
            "feature_importance": dict(
                zip(selected_indices, np.abs(lasso_cv.coef_[selected_indices]))
            ),
        }

        # Store group components
        self.group_selectors_[group_name] = selector
        if scaler is not None:
            self.group_scalers_[group_name] = scaler

        logger.info(
            f"Group '{group_name}': {selection_stats['selected_features']}/{selection_stats['original_features']} features selected (α={selection_stats['best_alpha']:.4f})"
        )

        return selected_mask, selected_indices, selection_stats

    def fit_select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> "LassoFeatureSelector":
        """
        Fit the feature selector and select features using group-wise LASSO.

        Args:
            X: Feature matrix (Polars DataFrame preferred, pandas DataFrame or numpy array supported)
            y: Target variable (Polars Series preferred, pandas Series or numpy array supported)

        Returns:
            Self for method chaining
        """
        logger.info("Starting LASSO-based feature selection")

        # Convert inputs to appropriate format for sklearn
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_array = X.values
        elif isinstance(X, pl.DataFrame):
            feature_names = list(X.columns)
            X_array = X.to_numpy()  # Convert Polars to numpy for sklearn
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        elif isinstance(y, pl.Series):
            y_array = y.to_numpy()  # Convert Polars to numpy for sklearn
        else:
            y_array = y

        logger.info(f"Input shape: {X_array.shape}, Target shape: {y_array.shape}")

        # Validate inputs
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("Input feature matrix cannot be empty")
        if y_array.shape[0] == 0:
            raise ValueError("Target variable cannot be empty")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X_array.shape[0]}) must match number of samples in y ({y_array.shape[0]})"
            )

        # Identify feature groups
        self.feature_groups_ = self._identify_feature_groups(feature_names)

        # Validate that we have feature groups
        if not self.feature_groups_:
            raise ValueError(
                "No feature groups identified - check feature naming conventions"
            )

        # Select features for each group
        all_selected_indices = []
        all_selected_names = []

        for group_name, group_features in self.feature_groups_.items():
            # Get indices for this group
            group_indices = [feature_names.index(feat) for feat in group_features]
            X_group = X_array[:, group_indices]

            # Select features for this group
            selected_mask, selected_indices, stats = self._select_features_for_group(
                X_group, y_array, group_name
            )

            # Convert local indices to global indices
            global_selected_indices = [group_indices[i] for i in selected_indices]
            selected_feature_names = [group_features[i] for i in selected_indices]

            # Store results
            all_selected_indices.extend(global_selected_indices)
            all_selected_names.extend(selected_feature_names)
            self.selection_stats_[group_name] = stats

            # Store feature importance
            for local_idx, global_idx in zip(selected_indices, global_selected_indices):
                feature_name = feature_names[global_idx]
                importance = stats["feature_importance"][local_idx]
                self.feature_importance_[feature_name] = importance

        # Store final selected features
        self.selected_features_ = sorted(all_selected_indices)
        self.selected_feature_names_ = [
            feature_names[i] for i in self.selected_features_
        ]

        self.is_fitted_ = True

        # Log summary
        total_original = len(feature_names)
        total_selected = len(self.selected_features_)
        reduction_ratio = (
            (total_original - total_selected) / total_original
            if total_original > 0
            else 0.0
        )

        logger.info(f"Feature selection completed:")
        logger.info(f"  Original features: {total_original}")
        logger.info(f"  Selected features: {total_selected}")
        logger.info(f"  Reduction ratio: {reduction_ratio:.2%}")

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Transform feature matrix by selecting only the chosen features.

        Args:
            X: Feature matrix to transform (Polars DataFrame preferred, pandas DataFrame or numpy array supported)

        Returns:
            Transformed feature matrix with selected features only (maintains input type when possible)
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        elif isinstance(X, pl.DataFrame):
            # For Polars, select columns by index (convert to column names first)
            all_columns = X.columns
            selected_columns = [all_columns[i] for i in self.selected_features_]
            return X.select(selected_columns)
        else:
            return X[:, self.selected_features_]

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        y: Union[np.ndarray, pd.Series, pl.Series],
    ) -> Union[np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Fit the selector and transform the data in one step.

        Args:
            X: Feature matrix (Polars DataFrame preferred, pandas DataFrame or numpy array supported)
            y: Target variable (Polars Series preferred, pandas Series or numpy array supported)

        Returns:
            Transformed feature matrix (maintains input type when possible)
        """
        return self.fit_select_features(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """
        Get the names of selected features.

        Returns:
            List of selected feature names
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted first")

        return self.selected_feature_names_.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores for selected features.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted first")

        return self.feature_importance_.copy()

    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the feature selection process.

        Returns:
            Dictionary with selection statistics and summaries
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted first")

        summary = {
            "total_original_features": sum(
                stats["original_features"] for stats in self.selection_stats_.values()
            ),
            "total_selected_features": len(self.selected_features_),
            "overall_reduction_ratio": 1
            - len(self.selected_features_)
            / sum(
                stats["original_features"] for stats in self.selection_stats_.values()
            ),
            "group_statistics": self.selection_stats_.copy(),
            "selected_features_by_group": {},
        }

        # Count selected features by group
        for group_name, group_features in self.feature_groups_.items():
            selected_in_group = [
                feat for feat in group_features if feat in self.selected_feature_names_
            ]
            summary["selected_features_by_group"][group_name] = {
                "count": len(selected_in_group),
                "features": selected_in_group,
            }

        return summary

    def save_selector(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted feature selector to disk.

        Args:
            filepath: Path to save the selector
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Feature selector saved to {filepath}")

    @classmethod
    def load_selector(cls, filepath: Union[str, Path]) -> "LassoFeatureSelector":
        """
        Load a fitted feature selector from disk.

        Args:
            filepath: Path to the saved selector

        Returns:
            Loaded feature selector
        """
        with open(filepath, "rb") as f:
            selector = pickle.load(f)

        logger.info(f"Feature selector loaded from {filepath}")
        return selector

    def print_summary(self) -> None:
        """Print a detailed summary of the feature selection results."""
        if not self.is_fitted_:
            print("Feature selector not fitted yet.")
            return

        summary = self.get_selection_summary()

        print("\n" + "=" * 60)
        print("LASSO FEATURE SELECTION SUMMARY")
        print("=" * 60)

        print(f"Total original features: {summary['total_original_features']:,}")
        print(f"Total selected features: {summary['total_selected_features']:,}")
        print(f"Overall reduction: {summary['overall_reduction_ratio']:.1%}")

        print(f"\nFeature selection by group:")
        print("-" * 40)

        for group_name, group_stats in summary["group_statistics"].items():
            selected_count = summary["selected_features_by_group"][group_name]["count"]
            print(
                f"{group_name:15s}: {selected_count:3d}/{group_stats['original_features']:3d} "
                f"({selected_count / group_stats['original_features']:.1%}) "
                f"α={group_stats['best_alpha']:.4f}"
            )

        print(f"\nTop 10 most important features:")
        print("-" * 40)

        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance_.items(), key=lambda x: x[1], reverse=True
        )

        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"{i + 1:2d}. {feature:30s}: {importance:.4f}")

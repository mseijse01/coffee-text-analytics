"""Pipeline step: LASSO-based feature selection."""

import logging
import pickle
import shutil
import traceback

import pandas as pd
import polars as pl

from features import LassoFeatureSelector
from src.pipeline.constants import EXCLUDE_COLUMNS

logger = logging.getLogger(__name__)


def run_feature_selection(args, config) -> bool:
    """
    Reduce the feature space using LASSO with cross-validation.

    Follows thesis methodology: combine all text features, apply single
    LASSO pass, keep sensory and categorical features untouched.

    Args:
        args: Parsed CLI arguments.
        config: Application configuration object.

    Returns:
        True on success, False on any error.
    """
    logger.info("Starting LASSO-based feature selection step")
    try:
        if not config.models.feature_selection_enabled:
            logger.info("Feature selection disabled — copying features file unchanged")
            features_path = config.paths.get_features_data_path()
            selected_path = config.paths.processed / "coffee_features_selected.csv"
            shutil.copy2(features_path, selected_path)
            return True

        features_data_path = config.paths.get_features_data_path()
        logger.info(f"Loading features from {features_data_path}")
        df = pl.read_csv(features_data_path).to_pandas()
        logger.info(f"Loaded features shape: {df.shape}")

        target_column = config.models.target_column
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return False

        exclude_cols = list(
            set(EXCLUDE_COLUMNS) | set(config.models.text_columns) | {target_column}
        )
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        feature_columns = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_columns]
        y = df[target_column]
        logger.info(f"Features shape before selection: {X.shape}")

        selector_config = config.models.feature_selection_config.copy()
        selector = LassoFeatureSelector(selector_config)

        logger.info("Fitting LASSO feature selector...")
        X_selected = selector.fit_transform(X, y)
        logger.info(f"Features shape after selection: {X_selected.shape}")
        selector.print_summary()

        if isinstance(X_selected, pd.DataFrame):
            selected_feature_names = X_selected.columns.tolist()
        else:
            selected_feature_names = selector.get_selected_features()
            X_selected = pd.DataFrame(
                X_selected, columns=selected_feature_names, index=X.index
            )

        result_df = pd.concat([df[exclude_cols], X_selected], axis=1)

        selected_path = config.paths.processed / "coffee_features_selected.csv"
        logger.info(f"Saving selected features to {selected_path}")
        result_df.to_csv(selected_path, index=False)

        selector.save_selector(config.paths.models / "lasso_feature_selector.pkl")

        summary = selector.get_selection_summary()
        with open(config.paths.output / "feature_selection_summary.pkl", "wb") as f:
            pickle.dump(summary, f)

        logger.info(f"Feature selection completed. Final shape: {result_df.shape}")
        return True

    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        logger.error(traceback.format_exc())
        return False

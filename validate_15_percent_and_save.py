#!/usr/bin/env python3
"""
15% Sample Validation + Model Persistence Script

Combines quick validation (15% sample) with model artifact saving.
Trains all models and persists them to disk for serving layer use.

Usage: python validate_15_percent_and_save.py
Expected runtime: ~4-5 minutes
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import polars as pl
from sklearn.metrics import r2_score

# Add src to path
sys.path.append("src")

from features.feature_manager import CoffeeFeatureManager
from features.feature_selector import LassoFeatureSelector
from models.mnir import MultinomialInverseRegression
from models.regressors import (
    CoffeeLassoRegression,
    CoffeeLinearRegression,
    CoffeeRandomForest,
    CoffeeRidgeRegression,
    CoffeeXGBoost,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
DATA_FILE = "data/raw/coffee_clean.csv"


def load_and_sample(sample_fraction: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and create stratified sample."""
    logger.info("📖 Loading coffee dataset")
    df = pd.read_csv(DATA_FILE)
    logger.info(f"✅ Loaded {len(df)} rows")

    logger.info(f"🔍 Creating {sample_fraction:.0%} stratified sample")
    sample_size = max(50, int(len(df) * sample_fraction))
    sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    logger.info(f"✅ Sample size: {len(sample_df)} rows")

    return df, sample_df


def extract_features(
    sample_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LassoFeatureSelector, Any]:
    """Extract and select features."""
    logger.info("🔧 Initializing feature extraction")

    # Preserve target before feature extraction
    y = sample_df["rating"].values

    # Convert to polars for feature extraction
    df_polars = pl.from_pandas(sample_df)

    # Feature manager
    fm = CoffeeFeatureManager(
        {
            "extractors": {
                "tfidf": True,
                "bert": True,
                "topics": True,
                "sentiment": True,
                "glove": False,  # Skip GloVe to save time
            }
        }
    )

    logger.info("📊 Fitting feature extractors")
    text_cols = ["desc_1", "desc_2", "desc_3"]
    all_texts = []
    for col in text_cols:
        if col in df_polars.columns:
            all_texts.extend(df_polars[col].to_list())

    fm.fit(df_polars, text_cols)
    logger.info("✅ Feature extractors fitted")

    logger.info("🎨 Extracting all features")
    features_df = fm.extract_all_features(df_polars, text_columns=text_cols)
    logger.info(f"✅ Features extracted: {features_df.shape}")

    # Convert to pandas for sklearn
    X = features_df.to_pandas()
    # Reset index to match y
    X = X.reset_index(drop=True)

    # Drop non-numeric columns (URLs, text, etc. from original dataframe)
    # Keep only numeric feature columns
    numeric_cols = X.select_dtypes(
        include=["float64", "float32", "int64", "int32"]
    ).columns.tolist()
    logger.info(
        f"Selecting {len(numeric_cols)} numeric columns from {X.shape[1]} total columns"
    )
    X = X[numeric_cols]
    logger.info(f"✅ Numeric features only: {X.shape}")

    # Feature selection
    logger.info("🔍 Applying LASSO feature selection")
    selector_config = {
        "alpha_range": [0.001, 0.01, 0.1, 1.0],
        "cv_folds": 5,
        "target_text_features": 500,
    }
    selector = LassoFeatureSelector(selector_config)

    # Fit and select features
    selector.fit_select_features(X, y)
    X_selected = selector.transform(X)
    logger.info(f"✅ Features selected: {X_selected.shape[1]} features")

    # Handle NaN values from feature extraction using mean imputation
    from sklearn.impute import SimpleImputer

    nan_count = X_selected.isna().sum().sum()
    if nan_count > 0:
        logger.warning(
            f"⚠️  Found {nan_count} NaN values in features, imputing with column means"
        )
        imputer = SimpleImputer(strategy="mean")
        X_array = imputer.fit_transform(X_selected)
        X_selected = pd.DataFrame(X_array, columns=X_selected.columns)
        logger.info("✅ NaN values imputed using column means")

    return X_selected, selector, y


def train_and_save_models(
    X: pd.DataFrame, y: pd.DataFrame
) -> Dict[str, Tuple[float, Any]]:
    """Train all models and save to disk."""
    from sklearn.model_selection import train_test_split

    logger.info("🚀 Splitting data for training")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}

    # Linear Regression
    logger.info("🔧 Training Linear Regression")
    linear = CoffeeLinearRegression({})
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"✅ LINEAR - R²: {r2:.4f}")
    with open(MODELS_DIR / "linear_model.pkl", "wb") as f:
        pickle.dump(linear, f)
    results["linear"] = (r2, linear)

    # Ridge Regression
    logger.info("🔧 Training Ridge Regression")
    ridge = CoffeeRidgeRegression({})
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"✅ RIDGE - R²: {r2:.4f}")
    with open(MODELS_DIR / "ridge_model.pkl", "wb") as f:
        pickle.dump(ridge, f)
    results["ridge"] = (r2, ridge)

    # LASSO Regression
    logger.info("🔧 Training LASSO Regression")
    lasso = CoffeeLassoRegression({})
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"✅ LASSO - R²: {r2:.4f}")
    with open(MODELS_DIR / "lasso_model.pkl", "wb") as f:
        pickle.dump(lasso, f)
    results["lasso"] = (r2, lasso)

    # Random Forest
    logger.info("🔧 Training Random Forest")
    rf = CoffeeRandomForest({})
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"✅ RANDOM_FOREST - R²: {r2:.4f}")
    with open(MODELS_DIR / "random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    results["random_forest"] = (r2, rf)

    # XGBoost
    logger.info("🔧 Training XGBoost")
    xgb = CoffeeXGBoost({})
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"✅ XGBOOST - R²: {r2:.4f}")
    with open(MODELS_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    results["xgboost"] = (r2, xgb)

    # MNIR
    logger.info("🔧 Training MNIR")
    mnir = MultinomialInverseRegression({})
    mnir.fit(X_train, y_train)
    logger.info("✅ MNIR trained")
    mnir.save_model(str(MODELS_DIR / "mnir_model.pkl"))
    results["mnir"] = (0.0, mnir)

    return results


def main():
    """Run validation and save models."""
    logger.info("=" * 80)
    logger.info("🎯 15% Sample Validation + Model Persistence")
    logger.info("=" * 80)

    try:
        # Load data
        full_df, sample_df = load_and_sample(sample_fraction=0.15)

        # Extract features
        X, selector, y = extract_features(sample_df)

        # Save feature selector
        logger.info("💾 Saving LASSO feature selector")
        selector.save_selector(MODELS_DIR / "lasso_feature_selector.pkl")
        logger.info("✅ Selector saved")

        # Train and save models
        results = train_and_save_models(X, y)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("🏆 RESULTS SUMMARY")
        logger.info("=" * 80)
        for model_name, (r2, _) in results.items():
            logger.info(f"  {model_name:15s} R²: {r2:.4f}")

        logger.info("\n💾 MODEL ARTIFACTS SAVED")
        logger.info("=" * 80)
        for pkl_file in MODELS_DIR.glob("*_model.pkl"):
            size = pkl_file.stat().st_size / 1024
            logger.info(f"  ✅ {pkl_file.name:40s} ({size:.1f} KB)")

        logger.info("\n✅ Pipeline complete!")
        logger.info("Ready for serving layer to load artifacts from models/")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

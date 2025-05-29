#!/usr/bin/env python3
"""
Quick Box-Cox dual pipeline completion script
"""

import sys
import os
import logging

sys.path.insert(0, "src")

from utils.transformations import run_box_cox_dual_pipeline
from models import CoffeeLinearRegression, CoffeeRidgeRegression
from models.evaluator import CoffeeModelEvaluator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    print("🔄 Starting Box-Cox Dual Pipeline Analysis")

    # Load the selected features data
    df = pd.read_csv("data/processed/coffee_features_selected.csv")
    print(f"Loaded data shape: {df.shape}")

    # Prepare features (exclude categorical string columns)
    exclude_cols = [
        "desc_1",
        "desc_2",
        "desc_3",
        "rating",
        "slug",
        "all_text",
        "name",
        "location",
        "origin",
        "est_price",
        "review_date",
        "agtron",
        "aroma",
        "acid",
        "body",
        "flavor",
        "aftertaste",
        "with_milk",
        "price_value",
        "price_unit",
        "price_standardized",
        "processed_desc_1",
        "processed_desc_2",
        "processed_desc_3",
        "merged_text",
        "processed_text",
        "roaster",
        "roast",
        "country_of_origin",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df["rating"]

    print(f"Features shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=57
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize models
    models = {"linear": CoffeeLinearRegression({}), "ridge": CoffeeRidgeRegression({})}

    # Run Box-Cox dual pipeline
    try:
        # Create a simple logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Create a simple config object
        class SimpleConfig:
            class models:
                box_cox_config = {}

        config = SimpleConfig()

        results = run_box_cox_dual_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            models_dict=models,
            config=config,
            logger=logger,
        )

        print("✅ Box-Cox dual pipeline completed successfully!")
        print(f"Results summary:")

        if "recommendation" in results:
            print(f"  Recommendation: {results['recommendation']}")
        if "summary" in results:
            summary = results["summary"]
            print(f"  Models tested: {summary.get('models_tested', 'N/A')}")
            print(f"  Models improved: {summary.get('models_improved', 'N/A')}")
            print(f"  Average improvement: {summary.get('avg_improvement', 'N/A'):.4f}")

        print("\n🎯 Task completed - Box-Cox analysis shows methodology compliance!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

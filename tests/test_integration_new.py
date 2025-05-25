#!/usr/bin/env python3
"""
Integration tests for the new component-based coffee analytics architecture.
"""

import unittest
import tempfile
import os
import sys
import pandas as pd
import polars as pl
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.preprocessing import preprocess_text, clean_text
from features import CoffeeFeatureManager, TfidfExtractor
from models import (
    CoffeeLinearRegression,
    CoffeeRandomForest,
    MultinomialInverseRegression,
    CoffeeModelEvaluator,
)
from utils.cleaning import clean_dataset as clean_coffee_data


class TestNewArchitectureIntegration(unittest.TestCase):
    """Test the new component-based architecture integration."""

    def setUp(self):
        """Set up test data for integration testing."""
        # Create comprehensive test dataset
        self.sample_data = {
            "rating": [85, 90, 88, 92, 87, 89, 91, 86, 93, 84],
            "desc_1": [
                "Great coffee with fruity notes and bright acidity",
                "Excellent balance with smooth finish",
                "Good body with complex flavor profile",
                "Outstanding coffee with exceptional aroma",
                "Nice coffee with clean taste",
                "Rich texture with chocolate undertones",
                "Bright and vibrant with citrus notes",
                "Smooth and creamy with nutty flavors",
                "Complex and layered with wine-like qualities",
                "Simple but pleasant with mild characteristics",
            ],
            "aroma": [8.0, 8.5, 7.5, 9.0, 8.0, 8.2, 8.7, 7.8, 9.2, 7.3],
            "acid": [7.5, 8.0, 7.0, 8.5, 7.5, 7.2, 8.8, 7.1, 8.3, 6.9],
            "body": [8.0, 8.5, 8.0, 9.0, 8.0, 8.3, 7.9, 8.4, 8.8, 7.6],
            "flavor": [8.5, 9.0, 8.0, 9.5, 8.5, 8.7, 9.1, 8.2, 9.3, 7.9],
            "aftertaste": [8.0, 8.5, 7.5, 9.0, 8.0, 8.1, 8.6, 7.7, 8.9, 7.4],
            "origin": [
                "Ethiopia",
                "Colombia",
                "Kenya",
                "Jamaica",
                "Guatemala",
                "Brazil",
                "Costa Rica",
                "Panama",
                "Yemen",
                "Peru",
            ],
        }

        # Create DataFrame
        self.df = pl.DataFrame(self.sample_data)

    def test_feature_manager_integration(self):
        """Test the new CoffeeFeatureManager."""
        # Initialize feature manager with limited extractors for testing
        config = {
            "extractors": {
                "tfidf": True,
                "bert": False,
                "glove": False,
                "topics": False,
                "sentiment": False,
            },
            "tfidf": {"max_features": 50},
        }
        feature_manager = CoffeeFeatureManager(config)

        # Fit and extract all features
        combined_texts = self.df["desc_1"].to_list()
        feature_manager.fit(combined_texts)
        features_df = feature_manager.extract_all_features(
            self.df, text_columns=["desc_1"]
        )

        # Check output
        self.assertIsInstance(features_df, pl.DataFrame)
        self.assertEqual(features_df.shape[0], self.df.shape[0])
        self.assertGreater(features_df.shape[1], 0)

    def test_model_training_integration(self):
        """Test the new model classes."""
        # Prepare simple features
        X = self.df.select(
            ["aroma", "acid", "body", "flavor", "aftertaste"]
        ).to_pandas()
        y = self.df["rating"].to_pandas()

        # Test Linear Regression
        linear_model = CoffeeLinearRegression()
        linear_model.fit(X, y)
        predictions = linear_model.predict(X)
        self.assertEqual(len(predictions), len(y))

        # Test Random Forest
        rf_model = CoffeeRandomForest()
        rf_model.fit(X, y)
        rf_predictions = rf_model.predict(X)
        self.assertEqual(len(rf_predictions), len(y))

    def test_mnir_integration(self):
        """Test MNIR model integration."""
        # Prepare features and sensory data
        X = self.df.select(
            ["aroma", "acid", "body", "flavor", "aftertaste"]
        ).to_pandas()
        y = self.df["rating"].to_pandas()

        sensory_data = {
            "aroma": self.df["aroma"].to_pandas().values,
            "acid": self.df["acid"].to_pandas().values,
            "body": self.df["body"].to_pandas().values,
            "flavor": self.df["flavor"].to_pandas().values,
            "aftertaste": self.df["aftertaste"].to_pandas().values,
        }

        # Test MNIR
        mnir_model = MultinomialInverseRegression()
        mnir_model.fit(X, sensory_data)
        mnir_predictions = mnir_model.predict_all_attributes(X)
        self.assertIsInstance(mnir_predictions, dict)

    def test_model_evaluation_integration(self):
        """Test model evaluation."""
        # Prepare data
        X = self.df.select(
            ["aroma", "acid", "body", "flavor", "aftertaste"]
        ).to_pandas()
        y = self.df["rating"].to_pandas()

        # Train a model
        model = CoffeeLinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # Evaluate
        evaluator = CoffeeModelEvaluator()
        evaluation_results = evaluator.evaluate(model, X, y)

        # Check metrics
        self.assertIn("metrics", evaluation_results)
        metrics = evaluation_results["metrics"]
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

    def test_text_preprocessing_integration(self):
        """Test text preprocessing functions."""
        sample_text = "Great coffee! Very smooth and delicious."

        # Test clean_text
        cleaned = clean_text(sample_text)
        self.assertIsInstance(cleaned, str)

        # Test preprocess_text
        processed = preprocess_text(sample_text)
        self.assertIsInstance(processed, str)

    def test_end_to_end_pipeline(self):
        """Test a simplified end-to-end pipeline."""
        # 1. Data cleaning
        cleaned_df, stats = clean_coffee_data(self.df)
        self.assertIsInstance(cleaned_df, pl.DataFrame)
        self.assertIsInstance(stats, dict)

        # 2. Feature extraction
        config = {
            "extractors": {
                "tfidf": True,
                "bert": False,
                "glove": False,
                "topics": False,
                "sentiment": False,
            },
            "tfidf": {"max_features": 20},
        }
        feature_manager = CoffeeFeatureManager(config)
        combined_texts = cleaned_df["desc_1"].to_list()
        feature_manager.fit(combined_texts)
        features_df = feature_manager.extract_all_features(
            cleaned_df, text_columns=["desc_1"]
        )

        # 3. Model training
        # Combine sensory and text features
        sensory_features = cleaned_df.select(
            ["aroma", "acid", "body", "flavor", "aftertaste"]
        )
        combined_features = sensory_features.hstack(
            features_df.select(
                [col for col in features_df.columns if col.startswith("tfidf_")]
            )
        )

        X = combined_features.to_pandas()
        y = cleaned_df["rating"].to_pandas()

        # Train model
        model = CoffeeLinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # 4. Evaluation
        evaluator = CoffeeModelEvaluator()
        evaluation_results = evaluator.evaluate(model, X, y)

        # Verify pipeline worked
        self.assertEqual(len(predictions), len(y))
        self.assertIn("metrics", evaluation_results)
        self.assertIn("r2", evaluation_results["metrics"])


if __name__ == "__main__":
    unittest.main()

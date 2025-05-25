#!/usr/bin/env python3
"""
Integration test suite for coffee text analytics project.

Tests end-to-end functionality and component integration.
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.preprocessing import load_csv_for_preprocessing, preprocess_text, clean_text
from features.feature_extraction import CoffeeFeatureExtractor
from models.model_training import (
    MultinomialInverseRegression,
    train_linear_regression,
    train_random_forest,
    train_mnir,
    evaluate_model,
    prepare_features,
)
from utils.cleaning import clean_dataset as clean_coffee_data
from utils.utils import convert_pandas_to_polars, convert_polars_to_pandas


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline functionality."""

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
            "desc_2": [
                "Bright acidity",
                "Smooth finish",
                "Rich texture",
                "Complex profile",
                "Clean taste",
                "Chocolate notes",
                "Citrus brightness",
                "Nutty undertones",
                "Wine-like",
                "Mild character",
            ],
            "desc_3": [
                "Recommended",
                "Must try",
                "Solid choice",
                "Exceptional",
                "Good value",
                "Premium quality",
                "Unique profile",
                "Crowd pleaser",
                "Sophisticated",
                "Entry level",
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
            "roast": [
                "Light",
                "Medium",
                "Medium",
                "Light",
                "Medium",
                "Dark",
                "Light",
                "Medium",
                "Light",
                "Medium",
            ],
            "est_price": [25.0, 30.0, 28.0, 45.0, 22.0, 35.0, 40.0, 26.0, 50.0, 20.0],
            "roaster": [
                "Roaster A",
                "Roaster B",
                "Roaster C",
                "Roaster D",
                "Roaster E",
                "Roaster F",
                "Roaster G",
                "Roaster H",
                "Roaster I",
                "Roaster J",
            ],
        }

        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        pd.DataFrame(self.sample_data).to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_complete_data_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        # Step 1: Load data (returns Pandas DataFrame)
        df = load_csv_for_preprocessing(self.temp_file.name)
        self.assertIsInstance(df, pd.DataFrame)  # Changed from pl.DataFrame
        self.assertEqual(df.shape[0], 10)

        # Convert to Polars for cleaning
        df_polars = pl.from_pandas(df)

        # Step 2: Clean data
        cleaned_df, stats = clean_coffee_data(df_polars)  # Unpack tuple
        self.assertIsInstance(cleaned_df, pl.DataFrame)
        self.assertIsInstance(stats, dict)
        self.assertGreater(cleaned_df.shape[1], 0)

        # Step 3: Text preprocessing
        for col in ["desc_1", "desc_2", "desc_3"]:
            if col in cleaned_df.columns:
                sample_text = cleaned_df[col].to_list()[0]
                if sample_text:
                    processed = preprocess_text(sample_text)
                    self.assertIsInstance(processed, str)

    def test_complete_feature_extraction_pipeline(self):
        """Test complete feature extraction pipeline."""
        # Load and prepare data (returns Pandas DataFrame)
        df = load_csv_for_preprocessing(self.temp_file.name)

        # Convert to Polars for feature extraction
        df_polars = pl.from_pandas(df)

        # Initialize feature extractor
        extractor = CoffeeFeatureExtractor()

        # Extract features (limited for testing)
        features_df = extractor.extract_all_features(
            df_polars,  # Use Polars DataFrame
            text_columns=["desc_1"],  # Single column for speed
            n_topics=3,  # Reduced for testing
        )

        # Check output
        self.assertIsInstance(features_df, pl.DataFrame)
        if not features_df.is_empty():
            self.assertEqual(features_df.shape[0], df_polars.shape[0])
            self.assertGreater(features_df.shape[1], 0)

    def test_complete_model_training_pipeline(self):
        """Test complete model training pipeline."""
        # Load and prepare data (returns Pandas DataFrame)
        df = load_csv_for_preprocessing(self.temp_file.name)
        df_pandas = df  # Already a Pandas DataFrame, no need to convert

        # Prepare features
        X, y, feature_names = prepare_features(df_pandas, "rating")

        # Train models
        models = {}

        # Linear regression
        models["linear"] = train_linear_regression(X, y)
        self.assertIsNotNone(models["linear"])

        # Random forest
        models["rf"] = train_random_forest(X, y)
        self.assertIsNotNone(models["rf"])

        # MNIR with sensory data
        sensory_data = {
            "aroma": df_pandas["aroma"].values,
            "acid": df_pandas["acid"].values,
            "body": df_pandas["body"].values,
            "flavor": df_pandas["flavor"].values,
            "aftertaste": df_pandas["aftertaste"].values,
        }

        models["mnir"] = train_mnir(X, y, feature_names, sensory_data)
        self.assertIsInstance(models["mnir"], MultinomialInverseRegression)

        # Evaluate models
        for model_name, model in models.items():
            if model is not None:
                metrics = evaluate_model(
                    model, X, y
                )  # Using training data for simplicity
                self.assertIsInstance(metrics, dict)

                if model_name != "mnir":
                    self.assertIn("rmse", metrics)
                    self.assertIn("r2", metrics)
                else:
                    self.assertIn("model_type", metrics)
                    self.assertEqual(metrics["model_type"], "MNIR")

    def test_polars_pandas_integration(self):
        """Test integration between Polars and Pandas components."""
        # Load data (returns Pandas DataFrame)
        df_pandas = load_csv_for_preprocessing(self.temp_file.name)
        self.assertIsInstance(df_pandas, pd.DataFrame)  # Changed expectation

        # Convert to Polars
        df_polars = pl.from_pandas(df_pandas)
        self.assertIsInstance(df_polars, pl.DataFrame)

        # Check data consistency
        self.assertEqual(df_polars.shape, df_pandas.shape)
        self.assertEqual(list(df_polars.columns), list(df_pandas.columns))

        # Test feature preparation (Pandas-based)
        X, y, feature_names = prepare_features(df_pandas, "rating")
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

        # Convert back to Polars if needed
        X_polars = pl.from_pandas(X)
        self.assertIsInstance(X_polars, pl.DataFrame)
        self.assertEqual(X.shape, X_polars.shape)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components."""

    def setUp(self):
        """Set up test data."""
        self.sample_texts = [
            "Great coffee with fruity notes and bright acidity",
            "Excellent balance with smooth finish and rich texture",
            "Good body with complex flavor profile and clean taste",
        ]

        self.sample_df = pl.DataFrame(
            {
                "desc_1": self.sample_texts,
                "desc_2": ["Bright", "Smooth", "Rich"],
                "rating": [85, 90, 88],
                "aroma": [8.0, 8.5, 7.5],
                "acid": [7.5, 8.0, 7.0],
                "body": [8.0, 8.5, 8.0],
                "flavor": [8.5, 9.0, 8.0],
                "aftertaste": [8.0, 8.5, 7.5],
            }
        )

    def test_preprocessing_feature_extraction_integration(self):
        """Test integration between preprocessing and feature extraction."""
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in self.sample_texts]

        # Extract features
        extractor = CoffeeFeatureExtractor()
        tfidf_df, vectorizer = extractor.extract_tfidf_features(
            processed_texts, max_features=20
        )

        # Check integration
        self.assertIsInstance(tfidf_df, pl.DataFrame)
        if not tfidf_df.is_empty():
            self.assertEqual(tfidf_df.shape[0], len(processed_texts))

    def test_feature_extraction_model_training_integration(self):
        """Test integration between feature extraction and model training."""
        # Extract features
        extractor = CoffeeFeatureExtractor()
        features_df = extractor.extract_all_features(
            self.sample_df, text_columns=["desc_1"], n_topics=2
        )

        if not features_df.is_empty():
            # Convert to format suitable for model training
            features_pandas = features_df.to_pandas()

            # Add target variable
            features_pandas["rating"] = self.sample_df["rating"].to_pandas()

            # Prepare for training
            X, y, feature_names = prepare_features(features_pandas, "rating")

            # Train model
            model = train_linear_regression(X, y)

            # Check integration
            self.assertIsNotNone(model)
            self.assertEqual(len(feature_names), X.shape[1])

    def test_data_cleaning_feature_extraction_integration(self):
        """Test integration between data cleaning and feature extraction."""
        # Add some messy data
        messy_df = self.sample_df.clone()
        messy_df = messy_df.with_columns(
            [
                pl.col("desc_1").str.replace_all(r"[^\w\s]", ""),  # Remove punctuation
            ]
        )

        # Clean data
        cleaned_df, _ = clean_coffee_data(messy_df)  # Unpack tuple, ignore stats

        # Extract features
        extractor = CoffeeFeatureExtractor()
        features_df = extractor.extract_all_features(
            cleaned_df, text_columns=["desc_1"], n_topics=2
        )

        # Check integration
        self.assertIsInstance(features_df, pl.DataFrame)

    def test_mnir_evaluation_integration(self):
        """Test integration between MNIR training and evaluation."""
        # Create feature matrix
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.random.uniform(80, 95, 20)
        feature_names = [f"feature_{i}" for i in range(10)]

        # Create sensory data
        sensory_data = {
            "aroma": np.random.uniform(6, 9, 20),
            "acid": np.random.uniform(5, 8, 20),
            "body": np.random.uniform(6, 9, 20),
        }

        # Train MNIR
        mnir = train_mnir(X, y, feature_names, sensory_data)

        # Evaluate
        metrics = evaluate_model(mnir, X, y)

        # Check integration
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics["model_type"], "MNIR")
        self.assertIn("n_attributes_analyzed", metrics)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across integrated components."""

    def test_missing_data_handling(self):
        """Test handling of missing data across pipeline."""
        # Create data with missing values
        df_with_missing = pl.DataFrame(
            {
                "desc_1": ["Great coffee", None, "Good coffee"],
                "desc_2": ["Bright", "Smooth", None],
                "rating": [85, None, 88],
                "aroma": [8.0, None, 7.5],
            }
        )

        # Test feature extraction with missing data
        extractor = CoffeeFeatureExtractor()
        features_df = extractor.extract_all_features(
            df_with_missing, text_columns=["desc_1", "desc_2"], n_topics=2
        )

        # Should handle gracefully
        self.assertIsInstance(features_df, pl.DataFrame)

    def test_empty_data_handling(self):
        """Test handling of empty data across pipeline."""
        # Empty DataFrame
        empty_df = pl.DataFrame()

        # Test feature extraction
        extractor = CoffeeFeatureExtractor()
        features_df = extractor.extract_all_features(empty_df)

        # Should handle gracefully
        self.assertIsInstance(features_df, pl.DataFrame)

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs across pipeline."""
        # Invalid text data
        invalid_df = pl.DataFrame(
            {
                "desc_1": [123, 456, 789],  # Numeric instead of text
                "rating": [85, 90, 88],
            }
        )

        # Test feature extraction
        extractor = CoffeeFeatureExtractor()
        features_df = extractor.extract_all_features(
            invalid_df, text_columns=["desc_1"], n_topics=2
        )

        # Should handle gracefully
        self.assertIsInstance(features_df, pl.DataFrame)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance characteristics of integrated pipeline."""

    def test_pipeline_performance(self):
        """Test that complete pipeline runs in reasonable time."""
        import time

        # Create larger dataset
        n_samples = 100
        large_data = {
            "rating": np.random.uniform(80, 95, n_samples),
            "desc_1": ["Great coffee with excellent flavor"] * n_samples,
            "aroma": np.random.uniform(6, 9, n_samples),
            "acid": np.random.uniform(5, 8, n_samples),
            "body": np.random.uniform(6, 9, n_samples),
            "flavor": np.random.uniform(7, 10, n_samples),
            "aftertaste": np.random.uniform(6, 9, n_samples),
        }

        df = pl.DataFrame(large_data)

        # Time the pipeline
        start_time = time.time()

        # Feature extraction
        extractor = CoffeeFeatureExtractor()
        features_df = extractor.extract_all_features(
            df, text_columns=["desc_1"], n_topics=3
        )

        # Model training
        if not features_df.is_empty():
            features_pandas = features_df.to_pandas()
            features_pandas["rating"] = df["rating"].to_pandas()

            X, y, feature_names = prepare_features(features_pandas, "rating")
            model = train_linear_regression(X, y)

        end_time = time.time()

        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 60, "Pipeline took too long")

    def test_memory_efficiency(self):
        """Test memory efficiency of pipeline."""
        # This is a basic test - in practice you'd use memory profiling tools

        # Create moderately sized dataset
        n_samples = 200
        data = {
            "desc_1": ["Coffee with notes"] * n_samples,
            "rating": list(range(n_samples)),
        }

        df = pl.DataFrame(data)

        # Test that operations complete without memory errors
        try:
            extractor = CoffeeFeatureExtractor()
            features_df = extractor.extract_all_features(
                df, text_columns=["desc_1"], n_topics=2
            )

            # Should complete successfully
            self.assertIsInstance(features_df, pl.DataFrame)

        except MemoryError:
            self.fail("Pipeline ran out of memory")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestEndToEndPipeline,
        TestComponentIntegration,
        TestErrorHandlingIntegration,
        TestPerformanceIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

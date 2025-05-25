#!/usr/bin/env python3
"""
Test suite for data processing components.

Tests data loading, preprocessing, and quality analysis functions.
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.preprocessing import (
    load_csv_for_preprocessing,
    preprocess_text,
    tokenize_text,
    remove_stopwords,
    lemmatize_text,
    clean_text,
    extract_country_info,
    standardize_prices,
)

# Import consolidated data quality functions
from utils.data_quality import analyze_data_quality
from utils.utils import (
    convert_pandas_to_polars,
    convert_polars_to_pandas,
)
from utils.cleaning import (
    clean_dataset as clean_coffee_data,
    extract_and_correct_country as standardize_country_names,
    drop_irrelevant_columns,
)


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def setUp(self):
        """Set up test data."""
        # Create sample CSV data
        self.sample_data = {
            "rating": [85, 90, 88, 92, 87],
            "desc_1": [
                "Great coffee with fruity notes",
                "Excellent balance",
                "Good body",
                "Outstanding flavor",
                "Nice aroma",
            ],
            "desc_2": [
                "Bright acidity",
                "Smooth finish",
                "Rich texture",
                "Complex profile",
                "Clean taste",
            ],
            "desc_3": [
                "Recommended",
                "Must try",
                "Solid choice",
                "Exceptional",
                "Good value",
            ],
            "aroma": [8.0, 8.5, 7.5, 9.0, 8.0],
            "acid": [7.5, 8.0, 7.0, 8.5, 7.5],
            "body": [8.0, 8.5, 8.0, 9.0, 8.0],
            "flavor": [8.5, 9.0, 8.0, 9.5, 8.5],
            "aftertaste": [8.0, 8.5, 7.5, 9.0, 8.0],
            "origin": ["Ethiopia", "Colombia", "Kenya", "Jamaica", "Guatemala"],
            "roast": ["Light", "Medium", "Medium", "Light", "Medium"],
            "est_price": [25.0, 30.0, 28.0, 45.0, 22.0],
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

    def test_load_coffee_data_basic(self):
        """Test basic data loading functionality."""
        df = load_csv_for_preprocessing(self.temp_file.name)

        # Check that data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 5)
        self.assertGreater(df.shape[1], 0)

        # Check that essential columns exist
        self.assertIn("rating", df.columns)
        self.assertIn("desc_1", df.columns)

    def test_load_coffee_data_missing_file(self):
        """Test handling of missing file."""
        df = load_csv_for_preprocessing("nonexistent_file.csv")

        # Function returns empty DataFrame instead of raising exception
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_analyze_data_quality(self):
        """Test data quality analysis."""
        df = load_csv_for_preprocessing(self.temp_file.name)

        # Should run without errors
        try:
            analyze_data_quality(df)
        except Exception as e:
            self.fail(f"analyze_data_quality raised {e} unexpectedly")


class TestTextPreprocessing(unittest.TestCase):
    """Test text preprocessing functionality."""

    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "This is a GREAT coffee with amazing FLAVOR!"
        result = preprocess_text(text)

        # Should be lowercase and cleaned
        self.assertIsInstance(result, str)
        self.assertNotIn("!", result)
        self.assertTrue(result.islower() or not result.isalpha())

    def test_preprocess_text_empty(self):
        """Test preprocessing empty text."""
        result = preprocess_text("")
        self.assertEqual(result, "")

        result = preprocess_text(None)
        self.assertEqual(result, "")

    def test_tokenize_text(self):
        """Test text tokenization."""
        text = "Great coffee with fruity notes"
        tokens = tokenize_text(text)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn("coffee", [t.lower() for t in tokens])

    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["this", "is", "great", "coffee", "with", "amazing", "flavor"]
        filtered = remove_stopwords(tokens)

        self.assertIsInstance(filtered, list)
        # Should remove common stopwords
        self.assertNotIn("this", filtered)
        self.assertNotIn("is", filtered)
        # Should keep content words
        self.assertIn("coffee", filtered)
        self.assertIn("flavor", filtered)

    def test_lemmatize_text(self):
        """Test text lemmatization."""
        tokens = ["running", "flies", "better"]
        lemmatized = lemmatize_text(tokens)

        self.assertIsInstance(lemmatized, list)
        self.assertEqual(len(lemmatized), len(tokens))

    def test_clean_text_pipeline(self):
        """Test complete text cleaning pipeline."""
        text = "This is AMAZING coffee with great FLAVOR and excellent AROMA!!!"
        result = clean_text(text)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_extract_country_from_origin(self):
        """Test country extraction from origin."""
        test_cases = [
            ("Ethiopia Yirgacheffe", "Ethiopia"),
            ("Colombia Huila", "Colombia"),
            ("Kenya AA", "Kenya"),
            ("Jamaica Blue Mountain", "Jamaica"),
            (
                "Unknown Region",
                "Unknown",
            ),  # extract_country_info returns first capitalized word
        ]

        for origin, expected in test_cases:
            result = extract_country_info(origin)
            self.assertEqual(result, expected)

    def test_standardize_prices(self):
        """Test price standardization."""
        df = pl.DataFrame(
            {"est_price": [25.0, 30.0, None, 45.0], "rating": [85, 90, 88, 92]}
        )

        result = standardize_prices(df)

        # Should handle missing values
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result.shape[0], df.shape[0])


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functionality."""

    def setUp(self):
        """Set up test data."""
        self.sample_df = pl.DataFrame(
            {
                "rating": [85, 90, 88, 92, 87],
                "desc_1": ["Great coffee", "Excellent", "Good", "Outstanding", "Nice"],
                "aroma": [8.0, 8.5, 7.5, 9.0, 8.0],
                "acid": [7.5, 8.0, 7.0, 8.5, 7.5],
                "origin": ["Ethiopia", "Colombia", "Kenya", "Jamaica", "Guatemala"],
                "slug": ["url1", "url2", "url3", "url4", "url5"],
                "all_text": ["text1", "text2", "text3", "text4", "text5"],
            }
        )

    def test_clean_coffee_data(self):
        """Test complete data cleaning pipeline."""
        result, stats = clean_coffee_data(self.sample_df)  # Unpack tuple

        self.assertIsInstance(result, pl.DataFrame)
        self.assertIsInstance(stats, dict)  # Check stats is returned
        self.assertGreater(result.shape[0], 0)

        # Should have key columns
        self.assertIn("rating", result.columns)
        self.assertIn("desc_1", result.columns)

    def test_standardize_country_names(self):
        """Test country name standardization."""
        df = pl.DataFrame(
            {"origin": ["USA", "United States", "US", "Ethiopia", "Colombia"]}
        )

        result = standardize_country_names(df)

        # Should add country_of_origin column
        self.assertIn("country_of_origin", result.columns)

        # Check that some standardization occurred
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result.shape[0], df.shape[0])

    def test_drop_irrelevant_columns(self):
        """Test dropping irrelevant columns."""
        result = drop_irrelevant_columns(self.sample_df)

        # Should remove specified columns
        self.assertNotIn("slug", result.columns)
        self.assertNotIn("all_text", result.columns)

        # Should keep important columns
        self.assertIn("rating", result.columns)
        self.assertIn("desc_1", result.columns)


class TestDataQualityAnalysis(unittest.TestCase):
    """Test data quality analysis functions."""

    def setUp(self):
        """Set up test data with quality issues."""
        self.df_with_issues = pl.DataFrame(
            {
                "rating": [85, 90, None, 92, 87],
                "desc_1": ["Great", "Excellent", "", "Outstanding", "Nice"],
                "aroma": [8.0, None, 7.5, 9.0, 8.0],
                "duplicate_col": [1, 2, 3, 4, 5],
            }
        )

        # Add duplicate row
        self.df_with_issues = pl.concat(
            [self.df_with_issues, self.df_with_issues.slice(0, 1)]
        )

    def test_analyze_data_quality_consolidated(self):
        """Test consolidated data quality analysis function."""
        try:
            analyze_data_quality(self.df_with_issues)
        except Exception as e:
            self.fail(f"analyze_data_quality raised {e} unexpectedly")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency."""

    def setUp(self):
        """Set up test data."""
        self.sample_df = pl.DataFrame(
            {
                "rating": [85, 90, 88, 92, 87],
                "desc_1": ["Great coffee", "Excellent", "Good", "Outstanding", "Nice"],
                "aroma": [8.0, 8.5, 7.5, 9.0, 8.0],
                "acid": [7.5, 8.0, 7.0, 8.5, 7.5],
                "body": [8.0, 8.5, 8.0, 9.0, 8.0],
                "flavor": [8.5, 9.0, 8.0, 9.5, 8.5],
                "origin": ["Ethiopia", "Colombia", "Kenya", "Jamaica", "Guatemala"],
            }
        )

    def test_column_consistency(self):
        """Test that required columns are maintained through processing."""
        essential_columns = ["rating", "desc_1", "aroma", "acid", "body", "flavor"]

        # Test data cleaning
        cleaned, stats = clean_coffee_data(self.sample_df)  # Unpack tuple
        for col in essential_columns:
            if col in self.sample_df.columns:
                self.assertIn(
                    col, cleaned.columns, f"Essential column {col} was removed"
                )

    def test_data_type_consistency(self):
        """Test that data types are consistent."""
        df = pl.DataFrame(
            {
                "rating": [85.0, 90.0, 88.0],
                "aroma": [8.0, 8.5, 7.5],
                "desc_1": ["text1", "text2", "text3"],
            }
        )

        # Numeric columns should remain numeric
        self.assertTrue(
            df["rating"].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        )
        self.assertTrue(
            df["aroma"].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        )

        # Text columns should be string
        self.assertEqual(df["desc_1"].dtype, pl.Utf8)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDataLoading,
        TestTextPreprocessing,
        TestDataCleaning,
        TestDataQualityAnalysis,
        TestDataIntegrity,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

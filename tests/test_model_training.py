#!/usr/bin/env python3
"""
Test suite for model training components.

Tests MNIR, traditional ML models, and training pipeline functionality.
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import pandas as pd
import polars as pl
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.model_training import (
    MultinomialInverseRegression,
    train_linear_regression,
    train_ridge_regression,
    train_lasso_regression,
    train_random_forest,
    train_xgboost,
    train_mnir,
    evaluate_model,
    get_feature_importance,
    prepare_features,
)


class TestMultinomialInverseRegression(unittest.TestCase):
    """Test the MNIR implementation."""

    def setUp(self):
        """Set up test data for MNIR."""
        np.random.seed(42)

        # Create synthetic feature matrix (text features)
        self.n_samples = 50
        self.n_features = 20
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Create synthetic sensory data with some correlation to features
        self.sensory_data = {
            "aroma": 7.0 + 0.5 * self.X[:, 0] + 0.3 * np.random.randn(self.n_samples),
            "acid": 6.5 + 0.4 * self.X[:, 1] + 0.3 * np.random.randn(self.n_samples),
            "body": 7.5 + 0.3 * self.X[:, 2] + 0.3 * np.random.randn(self.n_samples),
            "flavor": 8.0 + 0.6 * self.X[:, 3] + 0.3 * np.random.randn(self.n_samples),
            "aftertaste": 7.2
            + 0.4 * self.X[:, 4]
            + 0.3 * np.random.randn(self.n_samples),
        }

        # Ensure sensory scores are in realistic range
        for attr in self.sensory_data:
            self.sensory_data[attr] = np.clip(self.sensory_data[attr], 5.0, 10.0)

        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def test_mnir_initialization(self):
        """Test MNIR initialization."""
        mnir = MultinomialInverseRegression()

        self.assertIsInstance(mnir, MultinomialInverseRegression)
        self.assertEqual(mnir.lasso_cv, 5)
        self.assertEqual(mnir.random_state, 42)
        self.assertEqual(len(mnir.lasso_selectors), 0)
        self.assertEqual(len(mnir.regression_models), 0)

    def test_mnir_fit_basic(self):
        """Test basic MNIR fitting."""
        mnir = MultinomialInverseRegression(lasso_cv=3)  # Faster for testing

        # Fit the model
        mnir.fit(self.X, self.sensory_data, self.feature_names)

        # Check that models were trained
        self.assertGreater(len(mnir.lasso_selectors), 0)
        self.assertGreater(len(mnir.regression_models), 0)
        self.assertGreater(len(mnir.performance_metrics), 0)

        # Check that all sensory attributes were processed
        expected_attributes = set(self.sensory_data.keys())
        actual_attributes = set(mnir.performance_metrics.keys())
        self.assertEqual(expected_attributes, actual_attributes)

    def test_mnir_fit_missing_sensory_data(self):
        """Test MNIR fitting with missing sensory data."""
        # Create sensory data with missing values
        sensory_with_missing = self.sensory_data.copy()
        sensory_with_missing["aroma"][10:15] = np.nan

        mnir = MultinomialInverseRegression(lasso_cv=3)
        mnir.fit(self.X, sensory_with_missing, self.feature_names)

        # Should still work
        self.assertIn("aroma", mnir.performance_metrics)
        self.assertGreater(mnir.performance_metrics["aroma"]["n_samples"], 0)

    def test_mnir_predict(self):
        """Test MNIR prediction."""
        mnir = MultinomialInverseRegression(lasso_cv=3)
        mnir.fit(self.X, self.sensory_data, self.feature_names)

        # Test prediction
        X_test = np.random.randn(5, self.n_features)

        for attribute in self.sensory_data.keys():
            predictions = mnir.predict(X_test, attribute)

            self.assertIsInstance(predictions, np.ndarray)
            self.assertEqual(len(predictions), 5)
            self.assertTrue(np.all(np.isfinite(predictions)))

    def test_mnir_predict_invalid_attribute(self):
        """Test MNIR prediction with invalid attribute."""
        mnir = MultinomialInverseRegression(lasso_cv=3)
        mnir.fit(self.X, self.sensory_data, self.feature_names)

        X_test = np.random.randn(5, self.n_features)

        with self.assertRaises(ValueError):
            mnir.predict(X_test, "invalid_attribute")

    def test_mnir_performance_summary(self):
        """Test MNIR performance summary."""
        mnir = MultinomialInverseRegression(lasso_cv=3)
        mnir.fit(self.X, self.sensory_data, self.feature_names)

        summary = mnir.get_performance_summary()

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), len(self.sensory_data))

        # Check required columns
        required_columns = [
            "Attribute",
            "MSE",
            "R²",
            "N_Samples",
            "N_Features_Selected",
        ]
        for col in required_columns:
            self.assertIn(col, summary.columns)

        # Check that R² values are reasonable
        r2_values = summary["R²"].values
        self.assertTrue(np.all(r2_values >= -1))  # R² can be negative for bad models
        self.assertTrue(np.all(r2_values <= 1))  # R² cannot exceed 1

    def test_mnir_feature_importance(self):
        """Test MNIR feature importance extraction."""
        mnir = MultinomialInverseRegression(lasso_cv=3)
        mnir.fit(self.X, self.sensory_data, self.feature_names)

        for attribute in self.sensory_data.keys():
            importance = mnir.get_feature_importance(attribute, top_n=5)

            self.assertIsInstance(importance, pd.DataFrame)
            self.assertLessEqual(len(importance), 5)

            if len(importance) > 0:
                # Check required columns
                self.assertIn("Feature", importance.columns)
                self.assertIn("Coefficient", importance.columns)
                self.assertIn("Abs_Coefficient", importance.columns)

    def test_mnir_insights_report(self):
        """Test MNIR insights report generation."""
        mnir = MultinomialInverseRegression(lasso_cv=3)
        mnir.fit(self.X, self.sensory_data, self.feature_names)

        insights = mnir.generate_insights_report()

        self.assertIsInstance(insights, dict)
        self.assertIn("methodology", insights)
        self.assertIn("performance_summary", insights)
        self.assertIn("key_findings", insights)
        self.assertIn("feature_insights", insights)

        # Check key findings
        self.assertIn("best_performing_attribute", insights["key_findings"])

    def test_mnir_empty_sensory_data(self):
        """Test MNIR with empty sensory data."""
        mnir = MultinomialInverseRegression(lasso_cv=3)

        # Empty sensory data
        empty_sensory = {}
        mnir.fit(self.X, empty_sensory, self.feature_names)

        # Should handle gracefully
        self.assertEqual(len(mnir.performance_metrics), 0)

        summary = mnir.get_performance_summary()
        self.assertIsNone(summary)


class TestTraditionalModels(unittest.TestCase):
    """Test traditional ML model training functions."""

    def setUp(self):
        """Set up test data for traditional models."""
        np.random.seed(42)

        # Create synthetic regression data
        self.n_samples = 100
        self.n_features = 10
        self.X_train = np.random.randn(self.n_samples, self.n_features)

        # Create target with some relationship to features
        self.y_train = (
            2 * self.X_train[:, 0]
            + 1.5 * self.X_train[:, 1]
            + 0.5 * np.random.randn(self.n_samples)
            + 85  # Base rating
        )

    def test_train_linear_regression(self):
        """Test linear regression training."""
        model = train_linear_regression(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "coef_"))

        # Test prediction
        predictions = model.predict(self.X_train[:5])
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_train_ridge_regression(self):
        """Test ridge regression training."""
        model = train_ridge_regression(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "coef_"))
        self.assertTrue(hasattr(model, "alpha"))

    def test_train_lasso_regression(self):
        """Test lasso regression training."""
        model = train_lasso_regression(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "coef_"))
        self.assertTrue(hasattr(model, "alpha"))

    def test_train_random_forest(self):
        """Test random forest training."""
        model = train_random_forest(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "feature_importances_"))

        # Check feature importances
        importances = model.feature_importances_
        self.assertEqual(len(importances), self.n_features)
        self.assertTrue(np.all(importances >= 0))
        self.assertAlmostEqual(np.sum(importances), 1.0, places=5)

    @patch("models.model_training.XGBOOST_AVAILABLE", True)
    @patch("models.model_training.xgb", create=True)
    def test_train_xgboost_mocked(self, mock_xgb):
        """Test XGBoost training with mocked library."""
        # Mock XGBoost
        mock_model = MagicMock()
        mock_xgb.XGBRegressor.return_value = mock_model

        model = train_xgboost(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        mock_xgb.XGBRegressor.assert_called_once()
        mock_model.fit.assert_called_once()

    @patch("models.model_training.XGBOOST_AVAILABLE", False)
    def test_train_xgboost_unavailable(self):
        """Test XGBoost training when unavailable."""
        model = train_xgboost(self.X_train, self.y_train)

        self.assertIsNone(model)


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation functionality."""

    def setUp(self):
        """Set up test data for evaluation."""
        np.random.seed(42)

        self.n_samples = 50
        self.n_features = 10
        self.X_test = np.random.randn(self.n_samples, self.n_features)
        self.y_test = 2 * self.X_test[:, 0] + 1.5 * self.X_test[:, 1] + 85

        # Train a simple model for testing
        self.X_train = np.random.randn(100, self.n_features)
        self.y_train = 2 * self.X_train[:, 0] + 1.5 * self.X_train[:, 1] + 85
        self.model = train_linear_regression(self.X_train, self.y_train)

    def test_evaluate_traditional_model(self):
        """Test evaluation of traditional ML models."""
        metrics = evaluate_model(self.model, self.X_test, self.y_test)

        self.assertIsInstance(metrics, dict)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

        # Check that metrics are reasonable
        self.assertGreater(metrics["rmse"], 0)
        self.assertGreater(metrics["mae"], 0)
        self.assertLessEqual(metrics["r2"], 1)

    def test_evaluate_mnir_model(self):
        """Test evaluation of MNIR models."""
        # Create and train MNIR model
        mnir = MultinomialInverseRegression(lasso_cv=3)

        sensory_data = {
            "aroma": 7.0 + 0.5 * self.X_train[:, 0] + 0.3 * np.random.randn(100),
            "acid": 6.5 + 0.4 * self.X_train[:, 1] + 0.3 * np.random.randn(100),
        }

        feature_names = [f"feature_{i}" for i in range(self.n_features)]
        mnir.fit(self.X_train, sensory_data, feature_names)

        # Evaluate
        metrics = evaluate_model(mnir, self.X_test, self.y_test)

        self.assertIsInstance(metrics, dict)
        self.assertIn("model_type", metrics)
        self.assertEqual(metrics["model_type"], "MNIR")
        self.assertIn("methodology", metrics)
        self.assertIn("n_attributes_analyzed", metrics)


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance extraction."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

        self.n_features = 10
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Train models for testing
        X_train = np.random.randn(100, self.n_features)
        y_train = 2 * X_train[:, 0] + 1.5 * X_train[:, 1] + 85

        self.linear_model = train_linear_regression(X_train, y_train)
        self.rf_model = train_random_forest(X_train, y_train)

    def test_get_feature_importance_linear(self):
        """Test feature importance for linear models."""
        importances = get_feature_importance(self.linear_model, self.feature_names)

        self.assertIsInstance(importances, dict)
        self.assertEqual(len(importances), self.n_features)

        # Check that all importances are non-negative (absolute values)
        for importance in importances.values():
            self.assertGreaterEqual(importance, 0)

    def test_get_feature_importance_random_forest(self):
        """Test feature importance for tree-based models."""
        importances = get_feature_importance(self.rf_model, self.feature_names)

        self.assertIsInstance(importances, dict)
        self.assertEqual(len(importances), self.n_features)

        # Check that all importances are non-negative
        for importance in importances.values():
            self.assertGreaterEqual(importance, 0)

    def test_get_feature_importance_mnir(self):
        """Test feature importance for MNIR models."""
        # Create and train MNIR model
        mnir = MultinomialInverseRegression(lasso_cv=3)

        X_train = np.random.randn(50, self.n_features)
        sensory_data = {
            "aroma": 7.0 + 0.5 * X_train[:, 0] + 0.3 * np.random.randn(50),
            "acid": 6.5 + 0.4 * X_train[:, 1] + 0.3 * np.random.randn(50),
        }

        mnir.fit(X_train, sensory_data, self.feature_names)

        # Get feature importance
        importances = get_feature_importance(mnir, self.feature_names)

        self.assertIsInstance(importances, dict)
        # MNIR may have fewer features due to Lasso selection
        self.assertLessEqual(len(importances), self.n_features)


class TestPrepareFeatures(unittest.TestCase):
    """Test feature preparation functionality."""

    def setUp(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "rating": [85, 90, 88, 92, 87],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5],
                "text_col": ["text1", "text2", "text3", "text4", "text5"],
                "id": [1, 2, 3, 4, 5],
                "name": ["coffee1", "coffee2", "coffee3", "coffee4", "coffee5"],
            }
        )

    def test_prepare_features_basic(self):
        """Test basic feature preparation."""
        X, y, feature_names = prepare_features(self.sample_df, "rating")

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(feature_names, list)

        # Check dimensions
        self.assertEqual(len(X), len(self.sample_df))
        self.assertEqual(len(y), len(self.sample_df))
        self.assertEqual(len(feature_names), X.shape[1])

        # Check that target is excluded from features
        self.assertNotIn("rating", X.columns)

        # Check that excluded columns are removed
        self.assertNotIn("id", X.columns)
        self.assertNotIn("name", X.columns)
        self.assertNotIn("text_col", X.columns)  # Non-numeric

    def test_prepare_features_custom_exclude(self):
        """Test feature preparation with custom exclusions."""
        exclude_cols = ["feature_1", "text_col"]
        X, y, feature_names = prepare_features(
            self.sample_df, "rating", exclude_columns=exclude_cols
        )

        # Check that custom exclusions are applied
        self.assertNotIn("feature_1", X.columns)
        self.assertIn("feature_2", X.columns)

    def test_prepare_features_missing_target(self):
        """Test feature preparation with missing target column."""
        with self.assertRaises(ValueError):
            prepare_features(self.sample_df, "missing_column")

    def test_prepare_features_no_valid_features(self):
        """Test feature preparation with no valid features."""
        # DataFrame with only non-numeric columns and target
        df = pd.DataFrame(
            {"rating": [85, 90, 88], "text1": ["a", "b", "c"], "text2": ["x", "y", "z"]}
        )

        with self.assertRaises(ValueError):
            prepare_features(df, "rating")


class TestTrainMNIR(unittest.TestCase):
    """Test the train_mnir function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

        self.n_samples = 50
        self.n_features = 15
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.uniform(80, 95, self.n_samples)
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Create sensory data
        self.sensory_data = {
            "aroma": np.random.uniform(6, 9, self.n_samples),
            "acid": np.random.uniform(5, 8, self.n_samples),
            "body": np.random.uniform(6, 9, self.n_samples),
            "flavor": np.random.uniform(7, 10, self.n_samples),
            "aftertaste": np.random.uniform(6, 9, self.n_samples),
        }

    def test_train_mnir_with_sensory_data(self):
        """Test train_mnir with provided sensory data."""
        model = train_mnir(
            self.X_train, self.y_train, self.feature_names, self.sensory_data
        )

        self.assertIsInstance(model, MultinomialInverseRegression)
        self.assertGreater(len(model.performance_metrics), 0)

    def test_train_mnir_without_sensory_data(self):
        """Test train_mnir without sensory data (should create dummy data)."""
        model = train_mnir(self.X_train, self.y_train, self.feature_names)

        self.assertIsInstance(model, MultinomialInverseRegression)
        # Should create dummy data and train successfully
        self.assertGreater(len(model.performance_metrics), 0)

    def test_train_mnir_insights_generation(self):
        """Test that train_mnir generates insights."""
        model = train_mnir(
            self.X_train, self.y_train, self.feature_names, self.sensory_data
        )

        insights = model.generate_insights_report()
        self.assertIsNotNone(insights)
        self.assertIn("performance_summary", insights)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestMultinomialInverseRegression,
        TestTraditionalModels,
        TestModelEvaluation,
        TestFeatureImportance,
        TestPrepareFeatures,
        TestTrainMNIR,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

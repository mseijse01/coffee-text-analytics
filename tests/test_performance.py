#!/usr/bin/env python3
"""
Performance tests for coffee text analytics optimizations.

This module tests the performance improvements from Phase 5 optimizations.
"""

import unittest
import time
import sys
import os
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.performance import (
    PerformanceProfiler,
    DataFrameBenchmark,
    FeatureExtractionBenchmark,
    benchmark_function,
)
from utils.polars_utils import (
    PolarsOptimizer,
    DataTypeOptimizer,
    efficient_pandas_apply,
    prepare_for_sklearn,
    optimize_memory,
)
from utils.cache import CacheManager, FeatureCache, clear_all_cache


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimization utilities."""

    def setUp(self):
        """Set up test data."""
        # Create test DataFrame
        self.test_data = {
            "text": [
                "Great coffee with fruity notes",
                "Excellent balance and smooth finish",
                "Good body with complex flavor",
                "Outstanding aroma and taste",
                "Nice clean coffee",
            ]
            * 20,  # 100 rows
            "rating": np.random.uniform(80, 95, 100),
            "price": np.random.uniform(10, 50, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }

        self.df_polars = pl.DataFrame(self.test_data)
        self.df_pandas = self.df_polars.to_pandas()

    def test_polars_optimizer_efficient_apply(self):
        """Test efficient apply function."""

        def uppercase_func(text):
            return text.upper() if isinstance(text, str) else ""

        # Test efficient apply
        start_time = time.time()
        result_df = PolarsOptimizer.efficient_apply(
            self.df_polars, "text", uppercase_func, "text_upper"
        )
        efficient_time = time.time() - start_time

        # Verify result
        self.assertIn("text_upper", result_df.columns)
        self.assertEqual(len(result_df), len(self.df_polars))

        # Check that transformation worked
        first_result = result_df["text_upper"][0]
        first_original = result_df["text"][0]
        self.assertEqual(first_result, first_original.upper())

    def test_polars_optimizer_sklearn_conversion(self):
        """Test efficient sklearn conversion."""
        feature_cols = ["rating", "price"]
        target_col = "rating"

        X, y = PolarsOptimizer.batch_convert_for_sklearn(
            self.df_polars, feature_cols, target_col
        )

        # Verify types and shapes
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape[0], len(self.df_polars))
        self.assertEqual(len(y), len(self.df_polars))
        self.assertEqual(list(X.columns), feature_cols)

    def test_data_type_optimizer(self):
        """Test data type optimization."""
        # Create DataFrame with suboptimal types
        data = {
            "small_int": [1, 2, 3, 4, 5],  # Could be Int8
            "large_float": [1.0, 2.0, 3.0, 4.0, 5.0],  # Could be Float32
            "text": ["a", "b", "c", "d", "e"],
        }
        df = pl.DataFrame(data)

        # Get memory info before optimization
        memory_before = DataTypeOptimizer.analyze_memory_usage(df)

        # Optimize
        optimized_df = DataTypeOptimizer.optimize_dtypes(df)

        # Get memory info after optimization
        memory_after = DataTypeOptimizer.analyze_memory_usage(optimized_df)

        # Verify optimization occurred
        self.assertLessEqual(
            memory_after["total_memory_mb"], memory_before["total_memory_mb"]
        )

    def test_performance_profiler(self):
        """Test performance profiler functionality."""
        profiler = PerformanceProfiler()

        # Test measurement
        with profiler.measure("test_operation", {"test": True}):
            time.sleep(0.1)  # Simulate work

        summary = profiler.get_summary()

        # Verify measurement was recorded
        self.assertEqual(summary["total_measurements"], 1)
        self.assertGreater(summary["total_duration_seconds"], 0.09)
        self.assertIn("test_operation", summary["measurements"][0]["operation"])

    def test_dataframe_benchmark(self):
        """Test DataFrame benchmarking utilities."""

        # Define simple operations
        def polars_mean(df):
            return df["rating"].mean()

        def pandas_mean(df):
            return df["rating"].mean()

        # Benchmark comparison
        result = DataFrameBenchmark.compare_polars_pandas(
            "mean_calculation",
            polars_mean,
            pandas_mean,
            self.df_polars,
            self.df_pandas,
            iterations=3,
        )

        # Verify benchmark results
        self.assertIn("polars_avg_time", result)
        self.assertIn("pandas_avg_time", result)
        self.assertIn("speedup_factor", result)
        self.assertIsInstance(result["polars_faster"], bool)

    def test_conversion_overhead_benchmark(self):
        """Test conversion overhead benchmarking."""
        result = DataFrameBenchmark.benchmark_conversion_overhead(
            self.df_polars, iterations=3
        )

        # Verify benchmark results
        self.assertIn("to_pandas_avg_time", result)
        self.assertIn("to_polars_avg_time", result)
        self.assertIn("total_conversion_overhead", result)
        self.assertEqual(result["dataframe_shape"], self.df_polars.shape)


class TestCachingSystem(unittest.TestCase):
    """Test caching system performance."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing cache
        clear_all_cache()

        self.test_texts = [
            "Great coffee with excellent flavor",
            "Smooth and balanced taste profile",
            "Rich aroma with fruity notes",
            "Complex flavor with good body",
            "Clean finish with bright acidity",
        ] * 10  # 50 texts

    def tearDown(self):
        """Clean up after tests."""
        clear_all_cache()

    def test_cache_manager_basic_operations(self):
        """Test basic cache manager operations."""
        cache_manager = CacheManager()

        # Test set and get
        test_data = {"test": "value"}
        cache_manager.set("test_key", test_data, "features")

        retrieved_data = cache_manager.get("test_key", "features")
        self.assertEqual(retrieved_data, test_data)

        # Test cache info
        info = cache_manager.cache_info()
        self.assertIn("cache_types", info)
        self.assertIn("features", info["cache_types"])

    def test_cache_get_or_compute(self):
        """Test get_or_compute functionality."""
        cache_manager = CacheManager()

        def expensive_computation(x):
            time.sleep(0.1)  # Simulate expensive operation
            return x * 2

        # First call should compute
        start_time = time.time()
        result1 = cache_manager.get_or_compute(
            "test_compute", expensive_computation, "features", 5
        )
        first_call_time = time.time() - start_time

        # Second call should use cache
        start_time = time.time()
        result2 = cache_manager.get_or_compute(
            "test_compute", expensive_computation, "features", 5
        )
        second_call_time = time.time() - start_time

        # Verify results and performance
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)
        self.assertLess(second_call_time, first_call_time)

    def test_feature_cache(self):
        """Test feature-specific caching."""
        cache_manager = CacheManager()
        feature_cache = FeatureCache(cache_manager)

        def mock_tfidf_computation(texts, config):
            # Simulate TF-IDF computation
            time.sleep(0.05)
            return {"features": len(texts), "config": config}

        config = {"max_features": 100}

        # First call
        start_time = time.time()
        result1 = feature_cache.get_tfidf_features(
            self.test_texts, config, mock_tfidf_computation
        )
        first_time = time.time() - start_time

        # Second call (should be cached)
        start_time = time.time()
        result2 = feature_cache.get_tfidf_features(
            self.test_texts, config, mock_tfidf_computation
        )
        second_time = time.time() - start_time

        # Verify caching worked
        self.assertEqual(result1, result2)
        self.assertLess(second_time, first_time)


class TestOptimizedOperations(unittest.TestCase):
    """Test optimized operations vs original implementations."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pl.DataFrame(
            {
                "text_col": [
                    "This is a test text for processing",
                    "Another sample text with different content",
                    "Third text sample for comparison",
                    "Fourth text with various words",
                    "Final text sample for testing",
                ]
                * 20,  # 100 rows
                "numeric_col": np.random.uniform(0, 100, 100),
                "category_col": np.random.choice(["A", "B", "C", "D"], 100),
            }
        )

    def test_optimized_vs_original_apply(self):
        """Compare optimized apply vs original to_pandas().apply()."""

        def test_function(text):
            return len(text.split()) if isinstance(text, str) else 0

        # Original method
        start_time = time.time()
        original_result = self.test_df.with_columns(
            pl.Series(
                "word_count_original",
                self.test_df["text_col"].to_pandas().apply(test_function),
            )
        )
        original_time = time.time() - start_time

        # Optimized method
        start_time = time.time()
        optimized_result = efficient_pandas_apply(
            self.test_df, "text_col", test_function, "word_count_optimized"
        )
        optimized_time = time.time() - start_time

        # Verify results are equivalent
        original_values = original_result["word_count_original"].to_list()
        optimized_values = optimized_result["word_count_optimized"].to_list()
        self.assertEqual(original_values, optimized_values)

        # Performance comparison (optimized should be similar or better)
        print(
            f"Original time: {original_time:.4f}s, Optimized time: {optimized_time:.4f}s"
        )

    def test_memory_optimization_impact(self):
        """Test memory optimization impact."""
        # Create DataFrame with large integers that can be optimized
        large_df = pl.DataFrame(
            {
                "small_ints": list(range(1000)),  # Can be optimized to smaller int type
                "floats": [float(i) for i in range(1000)],  # Might be optimizable
                "text": [f"text_{i}" for i in range(1000)],
            }
        )

        # Analyze memory before optimization
        memory_before = DataTypeOptimizer.analyze_memory_usage(large_df)

        # Optimize
        optimized_df = optimize_memory(large_df)

        # Analyze memory after optimization
        memory_after = DataTypeOptimizer.analyze_memory_usage(optimized_df)

        # Verify optimization
        self.assertLessEqual(
            memory_after["total_memory_mb"], memory_before["total_memory_mb"]
        )

        print(f"Memory before: {memory_before['total_memory_mb']:.2f}MB")
        print(f"Memory after: {memory_after['total_memory_mb']:.2f}MB")
        print(
            f"Memory saved: {memory_before['total_memory_mb'] - memory_after['total_memory_mb']:.2f}MB"
        )


class TestIntegrationPerformance(unittest.TestCase):
    """Test performance of integrated optimizations."""

    def setUp(self):
        """Set up integration test data."""
        # Create larger dataset for meaningful performance testing
        np.random.seed(42)
        self.large_df = pl.DataFrame(
            {
                "desc_1": [
                    f"Coffee review {i} with various descriptive words about flavor, aroma, and body characteristics"
                    for i in range(500)
                ],
                "rating": np.random.uniform(80, 95, 500),
                "aroma": np.random.uniform(7, 9, 500),
                "acid": np.random.uniform(6, 9, 500),
                "body": np.random.uniform(7, 9, 500),
                "flavor": np.random.uniform(7, 9, 500),
                "aftertaste": np.random.uniform(6, 9, 500),
            }
        )

    def test_end_to_end_performance_with_optimizations(self):
        """Test end-to-end performance with all optimizations enabled."""
        profiler = PerformanceProfiler()

        # Test data preprocessing with optimizations
        with profiler.measure("optimized_preprocessing"):
            from data.preprocessing import preprocess_text

            # Use optimized apply
            processed_df = efficient_pandas_apply(
                self.large_df,
                "desc_1",
                lambda text: preprocess_text(text, remove_stop=True),
                "processed_desc_1",
            )

        # Test feature extraction with caching
        with profiler.measure("optimized_feature_extraction"):
            from features import TfidfExtractor

            config = {"max_features": 100, "ngram_range": (1, 2)}
            extractor = TfidfExtractor(config)
            texts = processed_df["processed_desc_1"].to_list()
            extractor.fit(texts)
            features = extractor.extract_features(texts)

        # Test sklearn preparation optimization
        with profiler.measure("optimized_sklearn_prep"):
            feature_cols = [col for col in features.columns if col.startswith("tfidf_")]
            X, y = prepare_for_sklearn(
                features.hstack(self.large_df.select(["rating"])),
                feature_cols,
                "rating",
            )

        # Test model training
        with profiler.measure("model_training"):
            from models import CoffeeLinearRegression

            model = CoffeeLinearRegression()
            model.fit(X, y)

        summary = profiler.get_summary()

        # Verify all operations completed successfully
        self.assertEqual(summary["total_measurements"], 4)
        self.assertGreater(len(features.columns), 0)
        self.assertEqual(X.shape[0], len(self.large_df))

        # Print performance summary
        print("\n=== Performance Summary ===")
        for measurement in summary["measurements"]:
            print(f"{measurement['operation']}: {measurement['duration_seconds']:.3f}s")
        print(f"Total time: {summary['total_duration_seconds']:.3f}s")


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("\n" + "=" * 60)
    print("🚀 COFFEE ANALYTICS - PERFORMANCE BENCHMARKS")
    print("=" * 60)

    # Create test data
    test_df = pl.DataFrame(
        {
            "text": [f"Sample text {i} for benchmarking purposes" for i in range(1000)],
            "values": np.random.uniform(0, 100, 1000),
        }
    )

    # Benchmark 1: Polars vs Pandas operations
    print("\n📊 Benchmark 1: Polars vs Pandas Operations")
    print("-" * 40)

    def polars_groupby(df):
        return df.group_by("text").agg(pl.col("values").mean())

    def pandas_groupby(df):
        return df.groupby("text")["values"].mean()

    result = DataFrameBenchmark.compare_polars_pandas(
        "groupby_mean",
        polars_groupby,
        pandas_groupby,
        test_df,
        test_df.to_pandas(),
        iterations=5,
    )

    print(f"Polars time: {result['polars_avg_time']:.4f}s")
    print(f"Pandas time: {result['pandas_avg_time']:.4f}s")
    print(f"Speedup: {result['speedup_factor']:.2f}x")
    print(f"Polars faster: {result['polars_faster']}")

    # Benchmark 2: Conversion overhead
    print("\n📊 Benchmark 2: Conversion Overhead")
    print("-" * 40)

    conversion_result = DataFrameBenchmark.benchmark_conversion_overhead(
        test_df, iterations=5
    )

    print(f"Polars → Pandas: {conversion_result['to_pandas_avg_time']:.4f}s")
    print(f"Pandas → Polars: {conversion_result['to_polars_avg_time']:.4f}s")
    print(f"Total overhead: {conversion_result['total_conversion_overhead']:.4f}s")

    # Benchmark 3: Memory optimization
    print("\n📊 Benchmark 3: Memory Optimization")
    print("-" * 40)

    memory_before = DataTypeOptimizer.analyze_memory_usage(test_df)
    optimized_df = DataTypeOptimizer.optimize_dtypes(test_df)
    memory_after = DataTypeOptimizer.analyze_memory_usage(optimized_df)

    print(f"Memory before: {memory_before['total_memory_mb']:.2f}MB")
    print(f"Memory after: {memory_after['total_memory_mb']:.2f}MB")
    print(
        f"Memory saved: {memory_before['total_memory_mb'] - memory_after['total_memory_mb']:.2f}MB"
    )
    print(
        f"Reduction: {((memory_before['total_memory_mb'] - memory_after['total_memory_mb']) / memory_before['total_memory_mb'] * 100):.1f}%"
    )


if __name__ == "__main__":
    # Run benchmarks first
    run_performance_benchmarks()

    # Then run unit tests
    print("\n" + "=" * 60)
    print("🧪 RUNNING PERFORMANCE TESTS")
    print("=" * 60)

    unittest.main(verbosity=2)

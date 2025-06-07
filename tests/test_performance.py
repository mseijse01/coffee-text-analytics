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

    def test_performance_monitor_edge_cases(self):
        """Test PerformanceMonitor edge cases and error handling."""
        from utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Test with exception in context manager
        try:
            with monitor.time_operation("failing_operation"):
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Verify metrics were still recorded despite exception
        metrics = monitor.get_metrics()
        self.assertIn("failing_operation", metrics)
        self.assertEqual(metrics["failing_operation"]["count"], 1)

        # Test multiple operations with same name
        with monitor.time_operation("repeated_op"):
            time.sleep(0.01)

        with monitor.time_operation("repeated_op"):
            time.sleep(0.02)

        metrics = monitor.get_metrics()
        repeated_metrics = metrics["repeated_op"]
        self.assertEqual(repeated_metrics["count"], 2)
        self.assertGreater(repeated_metrics["avg_time"], 0)
        self.assertGreater(repeated_metrics["max_time"], repeated_metrics["min_time"])

        # Test clear_metrics
        monitor.clear_metrics()
        self.assertEqual(len(monitor.get_metrics()), 0)

    def test_performance_profiler_edge_cases(self):
        """Test PerformanceProfiler edge cases."""
        profiler = PerformanceProfiler()

        # Test empty profiler summary
        empty_summary = profiler.get_summary()
        self.assertEqual(empty_summary["total_measurements"], 0)

        # Test with context information
        context = {"data_size": 1000, "algorithm": "test"}
        with profiler.measure("context_operation", context):
            time.sleep(0.01)

        summary = profiler.get_summary()
        measurement = summary["measurements"][0]
        self.assertEqual(measurement["context"], context)
        self.assertGreater(measurement["duration_seconds"], 0)
        self.assertIsInstance(measurement["memory_delta_mb"], float)

        # Test clear functionality
        profiler.clear()
        self.assertEqual(len(profiler.measurements), 0)

    def test_benchmark_function_comprehensive(self):
        """Test benchmark_function with various scenarios."""

        def simple_function(x, y, multiplier=1):
            return (x + y) * multiplier

        def function_with_exception():
            raise RuntimeError("Test error")

        # Test successful function benchmark
        result = benchmark_function(simple_function, 5, 10, multiplier=2)

        self.assertIn("total_measurements", result)
        self.assertIn("result", result)
        self.assertEqual(result["result"], 30)  # (5 + 10) * 2
        self.assertGreater(result["total_duration_seconds"], 0)

        # Test function that raises exception
        with self.assertRaises(RuntimeError):
            benchmark_function(function_with_exception)

    def test_dataframe_benchmark_comprehensive(self):
        """Test comprehensive DataFrame benchmarking functionality."""
        # Test with different data sizes
        small_df_polars = pl.DataFrame({"values": range(10)})
        small_df_pandas = small_df_polars.to_pandas()

        large_df_polars = pl.DataFrame({"values": range(1000)})
        large_df_pandas = large_df_polars.to_pandas()

        def sum_operation_polars(df):
            return df["values"].sum()

        def sum_operation_pandas(df):
            return df["values"].sum()

        # Test small data comparison
        small_result = DataFrameBenchmark.compare_polars_pandas(
            "sum_small",
            sum_operation_polars,
            sum_operation_pandas,
            small_df_polars,
            small_df_pandas,
            iterations=2,
        )

        self.assertIn("polars_avg_time", small_result)
        self.assertIn("pandas_avg_time", small_result)
        self.assertIn("speedup_factor", small_result)

        # Test large data comparison
        large_result = DataFrameBenchmark.compare_polars_pandas(
            "sum_large",
            sum_operation_polars,
            sum_operation_pandas,
            large_df_polars,
            large_df_pandas,
            iterations=2,
        )

        self.assertIn("polars_avg_time", large_result)
        self.assertIn("pandas_avg_time", large_result)

    def test_feature_extraction_benchmark_tfidf(self):
        """Test TF-IDF extraction benchmarking."""
        texts = [
            "coffee with great flavor",
            "excellent taste and aroma",
            "smooth balanced profile",
            "rich complex notes",
        ] * 5  # 20 texts

        configs = [
            {"max_features": 10, "ngram_range": (1, 1)},
            {"max_features": 20, "ngram_range": (1, 2)},
        ]

        result = FeatureExtractionBenchmark.benchmark_tfidf_extraction(
            texts, configs, iterations=2
        )

        self.assertIn("config_results", result)
        self.assertIn("best_config", result)
        self.assertIn("worst_config", result)
        self.assertEqual(len(result["config_results"]), 2)

        # Verify each config result has required fields
        for config_result in result["config_results"]:
            self.assertIn("config", config_result)
            self.assertIn("avg_time", config_result)
            self.assertIn("feature_count", config_result)

    def test_feature_extraction_benchmark_caching(self):
        """Test caching impact benchmarking."""
        texts = ["test text"] * 10
        config = {"max_features": 5}

        result = FeatureExtractionBenchmark.benchmark_caching_impact(
            texts, config, iterations=2
        )

        self.assertIn("without_cache", result)
        self.assertIn("with_cache", result)
        self.assertIn("cache_speedup", result)
        self.assertIn("cache_hit_ratio", result)

        # Cache should provide some speedup
        self.assertGreaterEqual(result["cache_speedup"], 1.0)

    def test_pipeline_performance_profiling(self):
        """Test pipeline performance profiling."""
        # Create a temporary test data file
        import tempfile
        import csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["desc_1", "rating"])
            for i in range(50):
                writer.writerow([f"coffee description {i}", 80 + i % 15])
            temp_path = f.name

        try:
            from utils.performance import profile_pipeline_performance

            # Test with small sample
            result = profile_pipeline_performance(temp_path, sample_size=20)

            self.assertIn("data_loading", result)
            self.assertIn("preprocessing", result)
            self.assertIn("feature_extraction", result)
            self.assertIn("total_pipeline_time", result)
            self.assertIn("sample_size", result)
            self.assertEqual(result["sample_size"], 20)

        finally:
            # Clean up temp file
            import os

            os.unlink(temp_path)

    def test_performance_report_generation(self):
        """Test performance report generation."""
        from utils.performance import generate_performance_report

        # Create mock benchmark results
        benchmark_results = {
            "polars_vs_pandas": {
                "operation": "groupby_mean",
                "polars_avg_time": 0.05,
                "pandas_avg_time": 0.15,
                "speedup_factor": 3.0,
                "polars_faster": True,
            },
            "caching_impact": {
                "without_cache": 0.5,
                "with_cache": 0.1,
                "cache_speedup": 5.0,
            },
            "memory_optimization": {
                "before_mb": 100.0,
                "after_mb": 60.0,
                "reduction_percent": 40.0,
            },
        }

        # Test report generation without file output
        report = generate_performance_report(benchmark_results)

        self.assertIsInstance(report, str)
        self.assertIn("PERFORMANCE BENCHMARK REPORT", report)
        self.assertIn("Polars vs Pandas", report)
        self.assertIn("Caching Impact", report)
        self.assertIn("Memory Optimization", report)
        self.assertIn("3.0x faster", report)

        # Test with file output
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            report_with_file = generate_performance_report(benchmark_results, temp_path)

            # Verify file was created and contains report
            with open(temp_path, "r") as f:
                file_content = f.read()

            self.assertEqual(report_with_file, file_content)
            self.assertIn("PERFORMANCE BENCHMARK REPORT", file_content)

        finally:
            import os

            os.unlink(temp_path)

    def test_performance_decorator(self):
        """Test performance measurement decorator."""
        from utils.performance import measure_performance, get_profiler

        # Clear any existing measurements
        profiler = get_profiler()
        profiler.clear()

        @measure_performance("decorated_function", {"test": True})
        def test_function(x, y):
            time.sleep(0.01)
            return x + y

        # Call decorated function
        result = test_function(5, 10)
        self.assertEqual(result, 15)

        # Verify measurement was recorded
        summary = profiler.get_summary()
        self.assertEqual(summary["total_measurements"], 1)

        measurement = summary["measurements"][0]
        self.assertEqual(measurement["operation"], "decorated_function")
        self.assertEqual(measurement["context"], {"test": True})
        self.assertGreater(measurement["duration_seconds"], 0.009)

    def test_global_profiler_singleton(self):
        """Test global profiler singleton behavior."""
        from utils.performance import get_profiler

        profiler1 = get_profiler()
        profiler2 = get_profiler()

        # Should return same instance
        self.assertIs(profiler1, profiler2)

        # Test that measurements persist across calls
        with profiler1.measure("test_singleton"):
            time.sleep(0.01)

        summary = profiler2.get_summary()
        self.assertEqual(summary["total_measurements"], 1)

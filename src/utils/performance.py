"""
Performance benchmarking and profiling utilities for coffee text analytics.

This module provides tools to measure and optimize performance of various
components in the pipeline.
"""

import time
import psutil
import logging
import functools
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitor for tracking operation timing and metrics.

    Simplified interface for performance measurement used in integration tests.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.current_operations = {}

    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations.

        Args:
            operation_name: Name of the operation to time
        """
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            if operation_name not in self.metrics:
                self.metrics[operation_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                }

            metrics = self.metrics[operation_name]
            metrics["count"] += 1
            metrics["total_time"] += duration
            metrics["avg_time"] = metrics["total_time"] / metrics["count"]
            metrics["min_time"] = min(metrics["min_time"], duration)
            metrics["max_time"] = max(metrics["max_time"], duration)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dictionary with performance metrics for all operations
        """
        return self.metrics.copy()

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.current_operations.clear()


class PerformanceProfiler:
    """
    Performance profiler for measuring execution time and memory usage.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.measurements = []
        self.current_measurement = None

    @contextmanager
    def measure(self, operation_name: str, context: Dict[str, Any] = None):
        """
        Context manager for measuring operation performance.

        Args:
            operation_name: Name of the operation being measured
            context: Additional context information
        """
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            measurement = {
                "operation": operation_name,
                "duration_seconds": end_time - start_time,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_delta_mb": final_memory - initial_memory,
                "context": context or {},
            }

            self.measurements.append(measurement)
            logger.info(
                f"{operation_name}: {measurement['duration_seconds']:.2f}s, "
                f"Memory: {measurement['memory_delta_mb']:+.1f}MB"
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all measurements.

        Returns:
            Dictionary with performance summary
        """
        if not self.measurements:
            return {"total_measurements": 0}

        durations = [m["duration_seconds"] for m in self.measurements]
        memory_deltas = [m["memory_delta_mb"] for m in self.measurements]

        return {
            "total_measurements": len(self.measurements),
            "total_duration_seconds": sum(durations),
            "average_duration_seconds": np.mean(durations),
            "max_duration_seconds": max(durations),
            "total_memory_delta_mb": sum(memory_deltas),
            "max_memory_delta_mb": max(memory_deltas),
            "measurements": self.measurements,
        }

    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()


def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a single function call.

    Args:
        func: Function to benchmark
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Dictionary with benchmark results
    """
    profiler = PerformanceProfiler()

    with profiler.measure(
        func.__name__, {"args_count": len(args), "kwargs_count": len(kwargs)}
    ):
        result = func(*args, **kwargs)

    summary = profiler.get_summary()
    summary["result"] = result

    return summary


class DataFrameBenchmark:
    """
    Specialized benchmarking for DataFrame operations.
    """

    @staticmethod
    def compare_polars_pandas(
        operation_name: str,
        polars_func: Callable,
        pandas_func: Callable,
        data_polars: pl.DataFrame,
        data_pandas: pd.DataFrame,
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare performance between Polars and Pandas operations.

        Args:
            operation_name: Name of the operation
            polars_func: Function that operates on Polars DataFrame
            pandas_func: Function that operates on Pandas DataFrame
            data_polars: Polars DataFrame
            data_pandas: Pandas DataFrame
            iterations: Number of iterations to run

        Returns:
            Comparison results
        """
        profiler = PerformanceProfiler()

        # Benchmark Polars
        polars_times = []
        for i in range(iterations):
            with profiler.measure(f"{operation_name}_polars_iter_{i}"):
                polars_result = polars_func(data_polars)

        # Benchmark Pandas
        pandas_times = []
        for i in range(iterations):
            with profiler.measure(f"{operation_name}_pandas_iter_{i}"):
                pandas_result = pandas_func(data_pandas)

        summary = profiler.get_summary()

        # Separate Polars and Pandas measurements
        polars_measurements = [
            m for m in summary["measurements"] if "polars" in m["operation"]
        ]
        pandas_measurements = [
            m for m in summary["measurements"] if "pandas" in m["operation"]
        ]

        polars_avg_time = np.mean([m["duration_seconds"] for m in polars_measurements])
        pandas_avg_time = np.mean([m["duration_seconds"] for m in pandas_measurements])

        speedup = (
            pandas_avg_time / polars_avg_time if polars_avg_time > 0 else float("inf")
        )

        return {
            "operation": operation_name,
            "polars_avg_time": polars_avg_time,
            "pandas_avg_time": pandas_avg_time,
            "speedup_factor": speedup,
            "polars_faster": speedup > 1,
            "detailed_measurements": summary["measurements"],
        }

    @staticmethod
    def benchmark_conversion_overhead(
        df: pl.DataFrame, iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark the overhead of Polars ↔ Pandas conversions.

        Args:
            df: Polars DataFrame to test
            iterations: Number of iterations

        Returns:
            Conversion benchmark results
        """
        profiler = PerformanceProfiler()

        # Benchmark Polars → Pandas conversion
        for i in range(iterations):
            with profiler.measure(f"polars_to_pandas_iter_{i}"):
                pandas_df = df.to_pandas()

        # Benchmark Pandas → Polars conversion
        for i in range(iterations):
            with profiler.measure(f"pandas_to_polars_iter_{i}"):
                polars_df = pl.from_pandas(pandas_df)

        summary = profiler.get_summary()

        to_pandas_times = [
            m["duration_seconds"]
            for m in summary["measurements"]
            if "to_pandas" in m["operation"]
        ]
        to_polars_times = [
            m["duration_seconds"]
            for m in summary["measurements"]
            if "to_polars" in m["operation"]
        ]

        return {
            "dataframe_shape": df.shape,
            "to_pandas_avg_time": np.mean(to_pandas_times),
            "to_polars_avg_time": np.mean(to_polars_times),
            "total_conversion_overhead": np.mean(to_pandas_times)
            + np.mean(to_polars_times),
            "detailed_measurements": summary["measurements"],
        }


class FeatureExtractionBenchmark:
    """
    Specialized benchmarking for feature extraction operations.
    """

    @staticmethod
    def benchmark_tfidf_extraction(
        texts: List[str], configs: List[Dict[str, Any]], iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark TF-IDF extraction with different configurations.

        Args:
            texts: List of texts to process
            configs: List of TF-IDF configurations to test
            iterations: Number of iterations per configuration

        Returns:
            Benchmark results
        """
        from ..features import TfidfExtractor

        profiler = PerformanceProfiler()
        results = {}

        for config_idx, config in enumerate(configs):
            config_name = f"config_{config_idx}"
            config_times = []

            for iteration in range(iterations):
                with profiler.measure(
                    f"tfidf_{config_name}_iter_{iteration}", {"config": config}
                ):
                    extractor = TfidfExtractor(config)
                    extractor.fit(texts)
                    features = extractor.extract_features(texts)

            # Get measurements for this config
            config_measurements = [
                m
                for m in profiler.measurements
                if f"tfidf_{config_name}" in m["operation"]
            ]

            avg_time = np.mean([m["duration_seconds"] for m in config_measurements])
            avg_memory = np.mean([m["memory_delta_mb"] for m in config_measurements])

            results[config_name] = {
                "config": config,
                "avg_time_seconds": avg_time,
                "avg_memory_mb": avg_memory,
                "feature_count": len(features.columns) if features.shape[0] > 0 else 0,
                "measurements": config_measurements,
            }

        return {
            "text_count": len(texts),
            "configurations": results,
            "fastest_config": min(
                results.keys(), key=lambda k: results[k]["avg_time_seconds"]
            ),
            "most_memory_efficient": min(
                results.keys(), key=lambda k: results[k]["avg_memory_mb"]
            ),
        }

    @staticmethod
    def benchmark_caching_impact(
        texts: List[str], config: Dict[str, Any], iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark the impact of caching on feature extraction.

        Args:
            texts: List of texts to process
            config: TF-IDF configuration
            iterations: Number of iterations

        Returns:
            Caching benchmark results
        """
        from ..features import TfidfExtractor
        from ..utils.cache import clear_all_cache

        profiler = PerformanceProfiler()

        # Clear cache first
        clear_all_cache()

        # Benchmark without cache (first run)
        with profiler.measure("tfidf_no_cache"):
            extractor = TfidfExtractor(config)
            extractor.fit(texts)
            features = extractor.extract_features(texts)

        # Benchmark with cache (subsequent runs)
        cache_times = []
        for i in range(iterations):
            with profiler.measure(f"tfidf_with_cache_iter_{i}"):
                extractor = TfidfExtractor(config)
                extractor.fit(texts)
                features = extractor.extract_features(texts)

        summary = profiler.get_summary()

        no_cache_time = next(
            m["duration_seconds"]
            for m in summary["measurements"]
            if "no_cache" in m["operation"]
        )
        cache_times = [
            m["duration_seconds"]
            for m in summary["measurements"]
            if "with_cache" in m["operation"]
        ]
        avg_cache_time = np.mean(cache_times)

        speedup = no_cache_time / avg_cache_time if avg_cache_time > 0 else float("inf")

        return {
            "no_cache_time": no_cache_time,
            "avg_cache_time": avg_cache_time,
            "speedup_factor": speedup,
            "cache_effective": speedup > 1.1,  # At least 10% improvement
            "detailed_measurements": summary["measurements"],
        }


def profile_pipeline_performance(
    data_path: str, sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Profile the performance of the entire pipeline.

    Args:
        data_path: Path to data file
        sample_size: Optional sample size for testing

    Returns:
        Pipeline performance profile
    """
    profiler = PerformanceProfiler()

    # Data loading
    with profiler.measure("data_loading"):
        df = pl.read_csv(data_path)
        if sample_size:
            df = df.sample(sample_size)

    # Data preprocessing
    with profiler.measure("data_preprocessing"):
        from ..data.preprocessing import preprocess_text

        # Simulate text preprocessing
        if "desc_1" in df.columns:
            processed_texts = [
                preprocess_text(text) for text in df["desc_1"].to_list()[:100]
            ]

    # Feature extraction
    with profiler.measure("feature_extraction"):
        from ..features import CoffeeFeatureManager

        config = {
            "extractors": {
                "tfidf": True,
                "bert": False,
                "glove": False,
                "topics": False,
                "sentiment": False,
            },
            "tfidf": {"max_features": 100},
        }
        feature_manager = CoffeeFeatureManager(config)
        if "desc_1" in df.columns:
            texts = df["desc_1"].to_list()[:100]
            feature_manager.fit(texts)
            features = feature_manager.extract_all_features(
                df.head(100), text_columns=["desc_1"]
            )

    # Model training
    with profiler.measure("model_training"):
        from ..models import CoffeeLinearRegression

        if "rating" in df.columns and len(df) > 10:
            X = (
                df.select(["aroma", "acid", "body", "flavor", "aftertaste"])
                .head(100)
                .to_pandas()
            )
            y = df["rating"].head(100).to_pandas()
            model = CoffeeLinearRegression()
            model.fit(X, y)

    return profiler.get_summary()


def generate_performance_report(
    benchmark_results: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive performance report.

    Args:
        benchmark_results: Results from various benchmarks
        output_path: Optional path to save the report

    Returns:
        Formatted report string
    """
    report_lines = [
        "# Coffee Text Analytics - Performance Report",
        "=" * 50,
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"Total operations measured: {benchmark_results.get('total_measurements', 0)}",
        f"Total execution time: {benchmark_results.get('total_duration_seconds', 0):.2f}s",
        f"Peak memory usage: {benchmark_results.get('max_memory_delta_mb', 0):.1f}MB",
        "",
        "## Detailed Measurements",
    ]

    for measurement in benchmark_results.get("measurements", []):
        report_lines.extend(
            [
                f"### {measurement['operation']}",
                f"- Duration: {measurement['duration_seconds']:.3f}s",
                f"- Memory delta: {measurement['memory_delta_mb']:+.1f}MB",
                f"- Context: {measurement['context']}",
                "",
            ]
        )

    report = "\n".join(report_lines)

    if output_path:
        Path(output_path).write_text(report)
        logger.info(f"Performance report saved to {output_path}")

    return report


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def measure_performance(operation_name: str, context: Dict[str, Any] = None):
    """Decorator for measuring function performance."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _global_profiler.measure(operation_name or func.__name__, context):
                return func(*args, **kwargs)

        return wrapper

    return decorator

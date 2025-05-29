"""
Integration tests for utility functions and caching system.

Tests the complete utility ecosystem including caching, Polars optimizations,
and data processing utilities using real sample data.
"""

import pytest
import tempfile
import time
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
import pandas as pd
import numpy as np

# Import modules under test
from src.utils.cache import CacheManager, FeatureCache, ModelCache, cached_function
from src.utils.polars_utils import PolarsOptimizer, DataTypeOptimizer
from src.utils.data_quality import DataQualityChecker
from src.utils.performance import PerformanceMonitor


class TestCachingSystemIntegration:
    """Integration tests for the complete caching system."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for caching tests."""
        return pl.DataFrame(
            {
                "text": [
                    "Great coffee with floral notes",
                    "Rich and bold flavor",
                    "Smooth finish",
                ],
                "rating": [92.5, 88.0, 90.5],
                "price": [15.99, 12.50, 18.75],
                "country": ["Ethiopia", "Brazil", "Colombia"],
            }
        )

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create cache manager with temporary directory."""
        return CacheManager(cache_dir=temp_cache_dir, max_age_hours=1)

    def test_cache_manager_initialization(self, temp_cache_dir):
        """Test cache manager initialization and directory structure."""
        cache_manager = CacheManager(cache_dir=temp_cache_dir)

        # Validate cache directory structure
        assert cache_manager.cache_dir.exists()
        assert (cache_manager.cache_dir / "features").exists()
        assert (cache_manager.cache_dir / "models").exists()
        assert (cache_manager.cache_dir / "data").exists()
        assert (cache_manager.cache_dir / "preprocessing").exists()

        # Validate configuration
        assert cache_manager.max_age_hours == 24  # default value

    def test_basic_cache_operations(self, cache_manager, sample_data):
        """Test basic cache set/get operations."""
        key = "test_data"
        cache_type = "data"

        # Test cache miss
        result = cache_manager.get(key, cache_type)
        assert result is None

        # Test cache set
        cache_manager.set(key, sample_data, cache_type)

        # Test cache hit
        cached_data = cache_manager.get(key, cache_type)
        assert cached_data is not None
        assert isinstance(cached_data, pl.DataFrame)
        assert cached_data.shape == sample_data.shape

    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation for different argument combinations."""
        # Test with different argument combinations
        key1 = cache_manager._generate_key("arg1", "arg2", param1="value1")
        key2 = cache_manager._generate_key("arg1", "arg2", param1="value1")
        key3 = cache_manager._generate_key("arg1", "arg2", param1="value2")

        # Same arguments should generate same key
        assert key1 == key2

        # Different arguments should generate different keys
        assert key1 != key3

        # Keys should be valid MD5 hashes
        assert len(key1) == 32
        assert all(c in "0123456789abcdef" for c in key1)

    def test_cache_expiration(self, temp_cache_dir):
        """Test cache expiration based on age."""
        # Create cache manager with very short expiration
        cache_manager = CacheManager(
            cache_dir=temp_cache_dir,
            max_age_hours=0.00001,  # ~0.036 seconds
        )

        key = "expiring_data"
        test_data = {"test": "value"}

        # Cache the data
        cache_manager.set(key, test_data)

        # Should be available immediately
        result = cache_manager.get(key)
        assert result == test_data

        # Wait for expiration
        time.sleep(0.1)  # Wait longer than expiration time

        # Should be expired now
        result = cache_manager.get(key)
        assert result is None

    def test_get_or_compute_functionality(self, cache_manager):
        """Test get_or_compute method with real computation."""

        def expensive_computation(x, y):
            """Simulate expensive computation."""
            time.sleep(0.01)  # Small delay to simulate work
            return x * y + 42

        key = "computation_result"

        # First call should compute
        start_time = time.time()
        result1 = cache_manager.get_or_compute(
            key, expensive_computation, "general", 5, 10
        )
        first_call_time = time.time() - start_time

        assert result1 == 92  # 5 * 10 + 42

        # Second call should use cache (much faster)
        start_time = time.time()
        result2 = cache_manager.get_or_compute(
            key, expensive_computation, "general", 5, 10
        )
        second_call_time = time.time() - start_time

        assert result2 == 92
        assert second_call_time < first_call_time  # Cache should be faster

    def test_feature_cache_integration(self, cache_manager, sample_data):
        """Test feature cache with different feature types."""
        feature_cache = FeatureCache(cache_manager)

        def mock_tfidf_extractor(texts, config):
            """Mock TF-IDF feature extraction."""
            return pl.DataFrame(
                {"tfidf_feature_1": [0.5, 0.3, 0.8], "tfidf_feature_2": [0.2, 0.7, 0.1]}
            )

        texts = sample_data["text"].to_list()
        config = {"max_features": 100, "ngram_range": (1, 2)}

        # First call should compute
        features1 = feature_cache.get_tfidf_features(
            texts, config, mock_tfidf_extractor
        )

        # Second call should use cache
        features2 = feature_cache.get_tfidf_features(
            texts, config, mock_tfidf_extractor
        )

        # Results should be identical
        assert features1.equals(features2)
        assert features1.shape == (3, 2)

    def test_model_cache_integration(self, cache_manager):
        """Test model cache with training scenarios."""
        model_cache = ModelCache(cache_manager)

        def mock_model_trainer(**kwargs):
            """Mock model training."""
            X = kwargs.get("X", [])
            y = kwargs.get("y", [])
            config = kwargs.get("config", {})

            return {
                "model_type": "random_forest",
                "n_estimators": config.get("n_estimators", 100),
                "trained": True,
                "feature_count": X.shape[1] if hasattr(X, "shape") else len(X),
            }

        # Create mock training data
        X_hash = "feature_hash_123"
        y_hash = "target_hash_456"
        config = {"n_estimators": 50, "random_state": 42}

        # First training should compute
        model1 = model_cache.get_trained_model(
            "random_forest",
            X_hash,
            y_hash,
            config,
            mock_model_trainer,
            X=[1, 2, 3],
            y=[0, 1, 0],
        )

        # Second training should use cache
        model2 = model_cache.get_trained_model(
            "random_forest",
            X_hash,
            y_hash,
            config,
            mock_model_trainer,
            X=[1, 2, 3],
            y=[0, 1, 0],
        )

        # Results should be identical
        assert model1 == model2
        assert model1["trained"] is True

    def test_cached_function_decorator(self, temp_cache_dir):
        """Test the cached_function decorator."""
        call_count = 0

        # Create a decorator that uses the temp cache directory
        def cached_function_with_temp_dir(
            cache_type: str = "general", max_age_hours: int = 24
        ):
            def decorator(func):
                cache_manager = CacheManager(
                    cache_dir=temp_cache_dir, max_age_hours=max_age_hours
                )

                def wrapper(*args, **kwargs):
                    cache_key = f"{func.__name__}_{cache_manager._generate_key(*args, **kwargs)}"
                    return cache_manager.get_or_compute(
                        cache_key, func, cache_type, *args, **kwargs
                    )

                return wrapper

            return decorator

        @cached_function_with_temp_dir(cache_type="general", max_age_hours=1)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x**2 + y**2

        # First call
        result1 = expensive_function(3, 4)
        assert result1 == 25
        assert call_count == 1

        # Second call with same arguments should use cache
        result2 = expensive_function(3, 4)
        assert result2 == 25
        assert call_count == 1  # Should not increment

        # Different arguments should compute again
        result3 = expensive_function(5, 6)
        assert result3 == 61
        assert call_count == 2

    def test_cache_management_operations(self, cache_manager, sample_data):
        """Test cache management operations like clearing and info."""
        # Add some data to different cache types
        cache_manager.set("data1", sample_data, "data")
        cache_manager.set("features1", {"feature": "value"}, "features")
        cache_manager.set("model1", {"model": "trained"}, "models")

        # Test cache info
        info = cache_manager.cache_info()
        assert "cache_dir" in info
        assert "cache_types" in info
        assert "data" in info["cache_types"]
        assert info["cache_types"]["data"]["file_count"] >= 1

        # Test selective cache clearing
        cache_manager.clear_cache("data")

        # Data cache should be empty, others should remain
        assert cache_manager.get("data1", "data") is None
        assert cache_manager.get("features1", "features") is not None

        # Test clearing all cache
        cache_manager.clear_cache()
        assert cache_manager.get("features1", "features") is None
        assert cache_manager.get("model1", "models") is None


class TestPolarsOptimizationIntegration:
    """Integration tests for Polars optimization utilities."""

    @pytest.fixture
    def sample_polars_data(self):
        """Create sample Polars DataFrame for testing."""
        return pl.DataFrame(
            {
                "text_col": [
                    "Coffee review one",
                    "Coffee review two",
                    "Coffee review three",
                ],
                "rating": [92.5, 88.0, 90.5],
                "price": [15.99, 12.50, 18.75],
                "country": ["Ethiopia", "Brazil", "Colombia"],
                "roaster": ["Roaster A", "Roaster B", "Roaster A"],
                "large_int": [1000000, 2000000, 3000000],
                "small_int": [1, 2, 3],
            }
        )

    def test_efficient_apply_operations(self, sample_polars_data):
        """Test efficient apply operations without full conversion."""

        def text_length(text):
            return len(text) if text else 0

        # Apply function efficiently
        result = PolarsOptimizer.efficient_apply(
            sample_polars_data, "text_col", text_length, "text_length"
        )

        # Validate results
        assert "text_length" in result.columns
        assert result["text_length"].to_list() == [17, 17, 19]
        assert result.shape[0] == sample_polars_data.shape[0]

    def test_batch_convert_for_sklearn(self, sample_polars_data):
        """Test efficient conversion for sklearn operations."""
        feature_columns = ["rating", "price"]
        target_column = "small_int"

        X, y = PolarsOptimizer.batch_convert_for_sklearn(
            sample_polars_data, feature_columns, target_column
        )

        # Validate conversion
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (3, 2)
        assert len(y) == 3
        assert list(X.columns) == feature_columns

    def test_efficient_groupby_operations(self, sample_polars_data):
        """Test efficient groupby operations using Polars native functions."""
        result = PolarsOptimizer.efficient_groupby_stats(
            sample_polars_data,
            group_col="roaster",
            agg_cols=["rating", "price"],
            stats=["mean", "count"],
        )

        # Validate groupby results
        assert result.shape[0] == 2  # Two unique roasters
        assert "rating_mean" in result.columns
        assert "price_count" in result.columns

        # Check specific values
        roaster_a_data = result.filter(pl.col("roaster") == "Roaster A")
        assert roaster_a_data["rating_count"].item() == 2

    def test_lazy_text_processing(self, sample_polars_data):
        """Test lazy text processing operations."""
        result = PolarsOptimizer.lazy_text_processing(
            sample_polars_data,
            text_columns=["text_col", "country"],
            operations=["lowercase", "strip"],
        )

        # Validate text processing
        assert result["text_col"].to_list()[0] == "coffee review one"
        assert result["country"].to_list()[0] == "ethiopia"
        assert result.shape == sample_polars_data.shape

    def test_memory_efficient_join(self, sample_polars_data):
        """Test memory-efficient join operations."""
        # Create second DataFrame for joining
        roaster_info = pl.DataFrame(
            {
                "roaster": ["Roaster A", "Roaster B"],
                "location": ["City A", "City B"],
                "founded": [2010, 2015],
            }
        )

        result = PolarsOptimizer.memory_efficient_join(
            sample_polars_data, roaster_info, on="roaster", how="left"
        )

        # Validate join results
        assert "location" in result.columns
        assert "founded" in result.columns
        assert result.shape[0] == sample_polars_data.shape[0]

        # Check join correctness
        roaster_a_rows = result.filter(pl.col("roaster") == "Roaster A")
        assert roaster_a_rows["location"].unique().to_list() == ["City A"]

    def test_data_type_optimization(self, sample_polars_data):
        """Test data type optimization for memory efficiency."""
        # Add some data that can be optimized
        test_data = sample_polars_data.with_columns(
            [
                pl.col("small_int").cast(pl.Int64),  # Can be optimized to smaller int
                pl.lit(100).alias("tiny_int").cast(pl.Int64),  # Can be UInt8
            ]
        )

        optimized = DataTypeOptimizer.optimize_dtypes(test_data)

        # Validate optimization (exact types depend on data ranges)
        assert optimized.shape == test_data.shape
        assert optimized.columns == test_data.columns

        # Memory analysis should show improvement
        original_memory = DataTypeOptimizer.analyze_memory_usage(test_data)
        optimized_memory = DataTypeOptimizer.analyze_memory_usage(optimized)

        assert "total_memory_mb" in original_memory
        assert "total_memory_mb" in optimized_memory

    def test_memory_usage_analysis(self, sample_polars_data):
        """Test memory usage analysis functionality."""
        memory_info = DataTypeOptimizer.analyze_memory_usage(sample_polars_data)

        # Validate memory analysis structure
        assert "total_memory_mb" in memory_info
        assert "column_info" in memory_info
        assert "recommendations" in memory_info

        # Check column-specific information
        for col in sample_polars_data.columns:
            assert col in memory_info["column_info"]
            col_info = memory_info["column_info"][col]
            assert "dtype" in col_info
            assert "memory_mb" in col_info


class TestUtilityIntegrationScenarios:
    """Integration tests for complete utility scenarios."""

    @pytest.fixture
    def coffee_sample_data(self):
        """Load real coffee sample data for integration testing."""
        try:
            return pl.read_csv("tests/data/coffee_sample.csv")
        except:
            # Fallback to synthetic data if sample not available
            return pl.DataFrame(
                {
                    "desc_1": [
                        "Floral and bright",
                        "Rich chocolate notes",
                        "Citrus and berry",
                    ],
                    "rating": [92.5, 88.0, 90.5],
                    "price": [15.99, 12.50, 18.75],
                    "country_of_origin": ["Ethiopia", "Brazil", "Colombia"],
                    "roaster": ["Blue Bottle", "Stumptown", "Intelligentsia"],
                }
            )

    def test_end_to_end_caching_with_polars_optimization(
        self, coffee_sample_data, tmp_path
    ):
        """Test complete workflow: data processing with caching and Polars optimization."""
        cache_manager = CacheManager(cache_dir=tmp_path)

        def expensive_data_processing(df):
            """Simulate expensive data processing."""
            # Optimize data types
            optimized_df = DataTypeOptimizer.optimize_dtypes(df)

            # Apply text processing
            if "desc_1" in optimized_df.columns:
                processed_df = PolarsOptimizer.lazy_text_processing(
                    optimized_df, ["desc_1"], ["lowercase", "strip"]
                )
            else:
                processed_df = optimized_df

            # Add computed features
            if "rating" in processed_df.columns:
                processed_df = processed_df.with_columns(
                    [
                        (pl.col("rating") > 90).alias("high_rating"),
                        pl.col("rating").rank().alias("rating_rank"),
                    ]
                )

            return processed_df

        # First processing should compute and cache
        cache_key = cache_manager._generate_key(
            "data_processing", coffee_sample_data.shape
        )

        start_time = time.time()
        result1 = cache_manager.get_or_compute(
            cache_key, expensive_data_processing, "data", coffee_sample_data
        )
        first_time = time.time() - start_time

        # Second processing should use cache
        start_time = time.time()
        result2 = cache_manager.get_or_compute(
            cache_key, expensive_data_processing, "data", coffee_sample_data
        )
        second_time = time.time() - start_time

        # Validate results
        assert result1.equals(result2)
        assert second_time < first_time  # Cache should be faster

        # Validate processing results
        if "rating" in result1.columns:
            assert "high_rating" in result1.columns
            assert "rating_rank" in result1.columns

    def test_performance_monitoring_integration(self, coffee_sample_data):
        """Test performance monitoring with real operations."""
        monitor = PerformanceMonitor()

        with monitor.time_operation("data_processing"):
            # Simulate data processing operations
            processed = PolarsOptimizer.lazy_text_processing(
                coffee_sample_data,
                [
                    col
                    for col in coffee_sample_data.columns
                    if coffee_sample_data[col].dtype == pl.Utf8
                ],
                ["lowercase"],
            )

            # Memory optimization
            optimized = DataTypeOptimizer.optimize_dtypes(processed)

        # Get performance metrics
        metrics = monitor.get_metrics()

        # Validate monitoring
        assert "data_processing" in metrics
        assert metrics["data_processing"]["count"] == 1
        assert metrics["data_processing"]["total_time"] > 0

    def test_data_quality_with_caching(self, coffee_sample_data, tmp_path):
        """Test data quality checks with caching integration."""
        cache_manager = CacheManager(cache_dir=tmp_path)

        def quality_check_with_cache(df):
            """Perform data quality checks with caching."""
            checker = DataQualityChecker()

            # Check for missing values
            missing_report = checker.check_missing_values(df)

            # Check data types
            dtype_report = checker.check_data_types(df)

            return {
                "missing_values": missing_report,
                "data_types": dtype_report,
                "shape": df.shape,
            }

        cache_key = "quality_check"

        # First check should compute
        quality_report = cache_manager.get_or_compute(
            cache_key, quality_check_with_cache, "data", coffee_sample_data
        )

        # Validate quality report structure
        assert "missing_values" in quality_report
        assert "data_types" in quality_report
        assert "shape" in quality_report
        assert quality_report["shape"] == coffee_sample_data.shape

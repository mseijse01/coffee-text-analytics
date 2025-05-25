"""
Caching utilities for expensive operations in coffee text analytics.

This module provides caching capabilities for feature extraction, model training,
and other computationally expensive operations.
"""

import pickle
import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Comprehensive cache manager for expensive operations.
    """

    def __init__(self, cache_dir: Union[str, Path] = "cache", max_age_hours: int = 24):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age_hours = max_age_hours

        # Create subdirectories for different cache types
        (self.cache_dir / "features").mkdir(exist_ok=True)
        (self.cache_dir / "models").mkdir(exist_ok=True)
        (self.cache_dir / "data").mkdir(exist_ok=True)
        (self.cache_dir / "preprocessing").mkdir(exist_ok=True)

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Unique cache key
        """
        # Create a string representation of all arguments
        key_data = {
            "args": str(args),
            "kwargs": sorted(kwargs.items()) if kwargs else {},
        }

        # Convert to JSON string and hash
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """
        Check if cache file is still valid based on age.

        Args:
            cache_file: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False

        # Check file age
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return file_age_hours < self.max_age_hours

    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key
            cache_type: Type of cache (features, models, data, preprocessing)

        Returns:
            Cached item or None if not found/expired
        """
        cache_file = self.cache_dir / cache_type / f"{key}.pkl"

        if not self._is_cache_valid(cache_file):
            return None

        try:
            with open(cache_file, "rb") as f:
                cached_item = pickle.load(f)
            logger.info(f"Cache hit for key: {key[:8]}...")
            return cached_item
        except Exception as e:
            logger.warning(f"Failed to load cache for key {key[:8]}...: {e}")
            return None

    def set(self, key: str, value: Any, cache_type: str = "general") -> None:
        """
        Store item in cache.

        Args:
            key: Cache key
            value: Item to cache
            cache_type: Type of cache (features, models, data, preprocessing)
        """
        cache_file = self.cache_dir / cache_type / f"{key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
            logger.info(f"Cached item with key: {key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to cache item with key {key[:8]}...: {e}")

    def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        cache_type: str = "general",
        *args,
        **kwargs,
    ) -> Any:
        """
        Get item from cache or compute and cache it.

        Args:
            key: Cache key
            compute_func: Function to compute the value
            cache_type: Type of cache
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function

        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        cached_value = self.get(key, cache_type)
        if cached_value is not None:
            return cached_value

        # Compute and cache
        logger.info(f"Computing and caching for key: {key[:8]}...")
        start_time = time.time()

        computed_value = compute_func(*args, **kwargs)

        compute_time = time.time() - start_time
        logger.info(f"Computation completed in {compute_time:.2f}s")

        self.set(key, computed_value, cache_type)
        return computed_value

    def clear_cache(self, cache_type: str = None) -> None:
        """
        Clear cache files.

        Args:
            cache_type: Specific cache type to clear, or None for all
        """
        if cache_type:
            cache_path = self.cache_dir / cache_type
            for cache_file in cache_path.glob("*.pkl"):
                cache_file.unlink()
            logger.info(f"Cleared {cache_type} cache")
        else:
            for cache_file in self.cache_dir.rglob("*.pkl"):
                cache_file.unlink()
            logger.info("Cleared all cache")

    def cache_info(self) -> Dict[str, Any]:
        """
        Get information about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "max_age_hours": self.max_age_hours,
            "cache_types": {},
        }

        for cache_type in ["features", "models", "data", "preprocessing"]:
            cache_path = self.cache_dir / cache_type
            if cache_path.exists():
                cache_files = list(cache_path.glob("*.pkl"))
                total_size = sum(f.stat().st_size for f in cache_files)

                info["cache_types"][cache_type] = {
                    "file_count": len(cache_files),
                    "total_size_mb": total_size / (1024 * 1024),
                    "files": [f.name for f in cache_files],
                }

        return info


class FeatureCache:
    """
    Specialized cache for feature extraction operations.
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize feature cache.

        Args:
            cache_manager: CacheManager instance
        """
        self.cache_manager = cache_manager

    def get_tfidf_features(
        self, texts: list, config: dict, compute_func: Callable
    ) -> Any:
        """
        Get or compute TF-IDF features.

        Args:
            texts: List of texts
            config: TF-IDF configuration
            compute_func: Function to compute features

        Returns:
            TF-IDF features
        """
        # Create cache key from texts hash and config
        texts_hash = hashlib.md5(str(texts).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8]
        cache_key = f"tfidf_{texts_hash}_{config_hash}"

        return self.cache_manager.get_or_compute(
            cache_key, compute_func, "features", texts, config
        )

    def get_bert_features(
        self, texts: list, model_name: str, compute_func: Callable
    ) -> Any:
        """
        Get or compute BERT features.

        Args:
            texts: List of texts
            model_name: BERT model name
            compute_func: Function to compute features

        Returns:
            BERT features
        """
        texts_hash = hashlib.md5(str(texts).encode()).hexdigest()[:8]
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        cache_key = f"bert_{texts_hash}_{model_hash}"

        return self.cache_manager.get_or_compute(
            cache_key, compute_func, "features", texts, model_name
        )

    def get_topic_features(
        self, texts: list, n_topics: int, compute_func: Callable
    ) -> Any:
        """
        Get or compute topic modeling features.

        Args:
            texts: List of texts
            n_topics: Number of topics
            compute_func: Function to compute features

        Returns:
            Topic features
        """
        texts_hash = hashlib.md5(str(texts).encode()).hexdigest()[:8]
        cache_key = f"topics_{texts_hash}_{n_topics}"

        return self.cache_manager.get_or_compute(
            cache_key, compute_func, "features", texts, n_topics
        )


class ModelCache:
    """
    Specialized cache for model training operations.
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize model cache.

        Args:
            cache_manager: CacheManager instance
        """
        self.cache_manager = cache_manager

    def get_trained_model(
        self,
        model_type: str,
        X_hash: str,
        y_hash: str,
        config: dict,
        compute_func: Callable,
    ) -> Any:
        """
        Get or compute trained model.

        Args:
            model_type: Type of model
            X_hash: Hash of training features
            y_hash: Hash of training targets
            config: Model configuration
            compute_func: Function to train model

        Returns:
            Trained model
        """
        config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8]
        cache_key = f"{model_type}_{X_hash}_{y_hash}_{config_hash}"

        return self.cache_manager.get_or_compute(
            cache_key, compute_func, "models", config
        )


def cached_function(cache_type: str = "general", max_age_hours: int = 24):
    """
    Decorator for caching function results.

    Args:
        cache_type: Type of cache to use
        max_age_hours: Maximum age of cache entries

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        cache_manager = CacheManager(max_age_hours=max_age_hours)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = (
                f"{func.__name__}_{cache_manager._generate_key(*args, **kwargs)}"
            )

            return cache_manager.get_or_compute(
                cache_key, func, cache_type, *args, **kwargs
            )

        return wrapper

    return decorator


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    Get global cache manager instance.

    Returns:
        CacheManager instance
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def clear_all_cache():
    """Clear all cached data."""
    cache_manager = get_cache_manager()
    cache_manager.clear_cache()


def cache_info() -> Dict[str, Any]:
    """Get cache information."""
    cache_manager = get_cache_manager()
    return cache_manager.cache_info()

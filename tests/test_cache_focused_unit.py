"""
Focused unit tests for cache system fixes.

Tests the critical fix: nested cache directory creation (parents=True).
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.cache import CacheManager


class TestCacheNestedDirectories:
    """Test the critical fix: mkdir(parents=True) for nested cache types."""

    @pytest.mark.unit
    def test_simple_cache_type_directory_creation(self):
        """Cache manager should create simple cache type directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Simple flat cache type
            cache_manager.set("key1", {"data": "value1"}, "features")

            # Verify directory was created
            assert (Path(temp_dir) / "features").exists()

            # Verify data can be retrieved
            assert cache_manager.get("key1", "features") == {"data": "value1"}

    @pytest.mark.unit
    def test_nested_cache_type_single_level(self):
        """Cache manager should create single-level nested cache type directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Single-level nested cache type
            cache_manager.set("key1", {"data": "value1"}, "features/tfidf")

            # Verify nested directory was created
            assert (Path(temp_dir) / "features" / "tfidf").exists()

            # Verify data can be retrieved
            assert cache_manager.get("key1", "features/tfidf") == {"data": "value1"}

    @pytest.mark.unit
    def test_nested_cache_type_multiple_levels(self):
        """Cache manager should create multi-level nested cache type directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Multi-level nested cache type (this is the fix)
            cache_manager.set("key1", {"data": "value1"}, "features/tfidf/extracted")
            cache_manager.set("key2", {"data": "value2"}, "models/lasso/selected")
            cache_manager.set(
                "key3", {"data": "value3"}, "preprocessing/categorical/encoded"
            )

            # Verify all nested directories were created
            assert (Path(temp_dir) / "features" / "tfidf" / "extracted").exists()
            assert (Path(temp_dir) / "models" / "lasso" / "selected").exists()
            assert (
                Path(temp_dir) / "preprocessing" / "categorical" / "encoded"
            ).exists()

            # Verify data can be retrieved from all levels
            assert cache_manager.get("key1", "features/tfidf/extracted") == {
                "data": "value1"
            }
            assert cache_manager.get("key2", "models/lasso/selected") == {
                "data": "value2"
            }
            assert cache_manager.get("key3", "preprocessing/categorical/encoded") == {
                "data": "value3"
            }

    @pytest.mark.unit
    def test_multiple_keys_same_nested_type(self):
        """Multiple keys in same nested cache type should work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Multiple keys in same nested cache type
            cache_manager.set("key1", {"data": "value1"}, "features/tfidf/extracted")
            cache_manager.set("key2", {"data": "value2"}, "features/tfidf/extracted")
            cache_manager.set("key3", {"data": "value3"}, "features/tfidf/extracted")

            # Verify directory exists only once
            nested_dir = Path(temp_dir) / "features" / "tfidf" / "extracted"
            assert nested_dir.exists()

            # Verify all keys can be retrieved
            assert cache_manager.get("key1", "features/tfidf/extracted") == {
                "data": "value1"
            }
            assert cache_manager.get("key2", "features/tfidf/extracted") == {
                "data": "value2"
            }
            assert cache_manager.get("key3", "features/tfidf/extracted") == {
                "data": "value3"
            }

    @pytest.mark.unit
    def test_cache_persistence_across_instances(self):
        """Cache should persist across different CacheManager instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write with first instance
            cm1 = CacheManager(cache_dir=temp_dir)
            cm1.set("key1", {"data": "persisted"}, "features/tfidf/extracted")

            # Read with second instance (same cache dir)
            cm2 = CacheManager(cache_dir=temp_dir)
            result = cm2.get("key1", "features/tfidf/extracted")

            assert result == {"data": "persisted"}

    @pytest.mark.unit
    def test_deeply_nested_cache_type(self):
        """Cache manager should handle very deep nesting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Very deep nesting (5 levels)
            deep_type = "a/b/c/d/e"
            cache_manager.set("deep_key", {"deep": "data"}, deep_type)

            # Verify deep directory was created
            deep_path = Path(temp_dir)
            for part in deep_type.split("/"):
                deep_path = deep_path / part
            assert deep_path.exists()

            # Verify data can be retrieved
            assert cache_manager.get("deep_key", deep_type) == {"deep": "data"}

"""
Utility functions and classes for the Coffee Text Analytics project.
"""

# Data quality and analysis utilities
from .data_quality import analyze_data_quality

# Cleaning and preprocessing utilities
from .cleaning import (
    clean_price,
    standardize_prices,
    extract_country,
    extract_and_correct_country,
    apply_text_preprocessing,
    clean_dataset,
    profile_dataset,
    analyze_numerical_columns,
    drop_irrelevant_columns,
)

# Performance optimization utilities
from .polars_utils import (
    PolarsOptimizer,
    DataTypeOptimizer,
    efficient_pandas_apply,
    optimize_memory,
    analyze_memory,
)

# Caching utilities
from .cache import CacheManager, FeatureCache, ModelCache

# Performance profiling utilities (optional)
try:
    from .performance import PerformanceProfiler, DataFrameBenchmark

    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# Documentation generation
from .doc_generator import generate_api_docs

# General utilities
from .utils import (
    load_dataset_from_utils,
    convert_pandas_to_polars,
    convert_polars_to_pandas,
)

__all__ = [
    # Data quality
    "analyze_data_quality",
    # Cleaning
    "clean_price",
    "standardize_prices",
    "extract_country",
    "extract_and_correct_country",
    "apply_text_preprocessing",
    "clean_dataset",
    "profile_dataset",
    "analyze_numerical_columns",
    "drop_irrelevant_columns",
    # Performance
    "PolarsOptimizer",
    "DataTypeOptimizer",
    "efficient_pandas_apply",
    "optimize_memory",
    "analyze_memory",
    # Caching
    "CacheManager",
    "FeatureCache",
    "ModelCache",
    # Profiling (if available)
    *(
        [
            "PerformanceProfiler",
            "DataFrameBenchmark",
        ]
        if PERFORMANCE_AVAILABLE
        else []
    ),
    # Documentation
    "generate_api_docs",
    # General
    "load_dataset_from_utils",
    "convert_pandas_to_polars",
    "convert_polars_to_pandas",
]

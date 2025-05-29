"""
Polars optimization utilities for efficient data processing.

This module provides utilities to minimize Polars ↔ Pandas conversions
and optimize data processing performance.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PolarsOptimizer:
    """
    Utility class for optimizing Polars operations and minimizing conversions.
    """

    @staticmethod
    def efficient_apply(
        df: pl.DataFrame, column: str, func: callable, new_column: str = None
    ) -> pl.DataFrame:
        """
        Apply a function to a Polars column efficiently without full conversion.

        Args:
            df: Polars DataFrame
            column: Column name to apply function to
            func: Function to apply
            new_column: Name for new column (defaults to original column)

        Returns:
            DataFrame with function applied
        """
        if new_column is None:
            new_column = column

        # Convert only the specific column to pandas, apply function, convert back
        series_pandas = df[column].to_pandas()
        result_pandas = series_pandas.apply(func)

        return df.with_columns(pl.Series(new_column, result_pandas))

    @staticmethod
    def batch_convert_for_sklearn(
        df: pl.DataFrame, feature_columns: List[str], target_column: str = None
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Efficiently convert Polars DataFrame to format needed for sklearn.

        Args:
            df: Polars DataFrame
            feature_columns: List of feature column names
            target_column: Target column name (optional)

        Returns:
            Tuple of (features_df, target_series)
        """
        # Convert only needed columns
        X = df.select(feature_columns).to_pandas()

        y = None
        if target_column:
            y = df[target_column].to_pandas()

        return X, y

    @staticmethod
    def efficient_groupby_stats(
        df: pl.DataFrame,
        group_col: str,
        agg_cols: List[str],
        stats: List[str] = ["mean", "count"],
    ) -> pl.DataFrame:
        """
        Perform efficient groupby operations using Polars native functions.

        Args:
            df: Polars DataFrame
            group_col: Column to group by
            agg_cols: Columns to aggregate
            stats: Statistics to compute

        Returns:
            Aggregated DataFrame
        """
        agg_exprs = []

        for col in agg_cols:
            for stat in stats:
                if stat == "mean":
                    agg_exprs.append(pl.col(col).mean().alias(f"{col}_{stat}"))
                elif stat == "count":
                    agg_exprs.append(pl.col(col).count().alias(f"{col}_{stat}"))
                elif stat == "std":
                    agg_exprs.append(pl.col(col).std().alias(f"{col}_{stat}"))
                elif stat == "min":
                    agg_exprs.append(pl.col(col).min().alias(f"{col}_{stat}"))
                elif stat == "max":
                    agg_exprs.append(pl.col(col).max().alias(f"{col}_{stat}"))

        return df.group_by(group_col).agg(agg_exprs)

    @staticmethod
    def lazy_text_processing(
        df: pl.DataFrame,
        text_columns: List[str],
        operations: List[str] = ["lowercase", "strip"],
    ) -> pl.DataFrame:
        """
        Apply text processing operations using Polars lazy evaluation.

        Args:
            df: Polars DataFrame
            text_columns: Text columns to process
            operations: List of operations to apply

        Returns:
            DataFrame with processed text
        """
        lazy_df = df.lazy()

        for col in text_columns:
            for op in operations:
                if op == "lowercase":
                    lazy_df = lazy_df.with_columns(pl.col(col).str.to_lowercase())
                elif op == "strip":
                    lazy_df = lazy_df.with_columns(pl.col(col).str.strip_chars())
                elif op == "replace_nulls":
                    lazy_df = lazy_df.with_columns(pl.col(col).fill_null(""))

        return lazy_df.collect()

    @staticmethod
    def memory_efficient_join(
        left: pl.DataFrame,
        right: pl.DataFrame,
        on: Union[str, List[str]],
        how: str = "inner",
    ) -> pl.DataFrame:
        """
        Perform memory-efficient joins using Polars.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Column(s) to join on
            how: Join type

        Returns:
            Joined DataFrame
        """
        return left.lazy().join(right.lazy(), on=on, how=how).collect()


class DataTypeOptimizer:
    """
    Utility class for optimizing data types and memory usage.
    """

    @staticmethod
    def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize data types for memory efficiency.

        Args:
            df: Polars DataFrame

        Returns:
            DataFrame with optimized data types
        """
        optimized_df = df.clone()

        for col in df.columns:
            dtype = df[col].dtype

            # Optimize integer types
            if dtype == pl.Int64:
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min >= 0:  # Unsigned integers
                    if col_max <= 255:
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.UInt8)
                        )
                    elif col_max <= 65535:
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.UInt16)
                        )
                    elif col_max <= 4294967295:
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.UInt32)
                        )
                else:  # Signed integers
                    if col_min >= -128 and col_max <= 127:
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.Int8)
                        )
                    elif col_min >= -32768 and col_max <= 32767:
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.Int16)
                        )
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.Int32)
                        )

            # Optimize float types
            elif dtype == pl.Float64:
                # Check if we can use Float32 without losing precision
                try:
                    float32_series = df[col].cast(pl.Float32)
                    if float32_series.equals(df[col].cast(pl.Float32).cast(pl.Float64)):
                        optimized_df = optimized_df.with_columns(
                            pl.col(col).cast(pl.Float32)
                        )
                except:
                    pass  # Keep as Float64 if conversion fails

        return optimized_df

    @staticmethod
    def analyze_memory_usage(df: pl.DataFrame) -> Dict[str, Any]:
        """
        Analyze memory usage of DataFrame.

        Args:
            df: Polars DataFrame

        Returns:
            Dictionary with memory usage statistics
        """
        memory_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_info": {},
            "total_memory_mb": 0,
            "recommendations": [],
        }

        for col in df.columns:
            # Estimate memory usage (approximate)
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.UInt8]:
                bytes_per_value = 1
            elif dtype in [pl.Int16, pl.UInt16]:
                bytes_per_value = 2
            elif dtype in [pl.Int32, pl.UInt32, pl.Float32]:
                bytes_per_value = 4
            elif dtype in [pl.Int64, pl.UInt64, pl.Float64]:
                bytes_per_value = 8
            elif dtype == pl.Utf8:
                # Estimate string memory (rough approximation)
                avg_length = df[col].str.len_chars().mean() or 0
                bytes_per_value = avg_length * 1.5  # UTF-8 overhead
            else:
                bytes_per_value = 8  # Default estimate

            col_memory_mb = (len(df) * bytes_per_value) / (1024 * 1024)
            memory_info["column_info"][col] = {
                "dtype": str(dtype),
                "memory_mb": col_memory_mb,
            }
            memory_info["total_memory_mb"] += col_memory_mb

            # Add recommendations for optimization
            if dtype == pl.Int64:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min >= 0 and col_max <= 255:
                    memory_info["recommendations"].append(
                        f"Column '{col}' can be optimized from Int64 to UInt8"
                    )

        return memory_info


# Convenience functions for common operations
def efficient_pandas_apply(
    df: pl.DataFrame, column: str, func: callable
) -> pl.DataFrame:
    """Convenience function for efficient apply operations."""
    return PolarsOptimizer.efficient_apply(df, column, func)


def prepare_for_sklearn(df: pl.DataFrame, features: List[str], target: str = None):
    """Convenience function for sklearn preparation."""
    return PolarsOptimizer.batch_convert_for_sklearn(df, features, target)


def optimize_memory(df: pl.DataFrame) -> pl.DataFrame:
    """Convenience function for memory optimization."""
    return DataTypeOptimizer.optimize_dtypes(df)


def analyze_memory(df: pl.DataFrame) -> Dict[str, Any]:
    """Convenience function for memory analysis."""
    return DataTypeOptimizer.analyze_memory_usage(df)

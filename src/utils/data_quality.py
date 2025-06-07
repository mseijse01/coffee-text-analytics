"""
Data quality analysis utilities for coffee review data.

This module provides comprehensive data quality analysis functions
that work with both Pandas and Polars DataFrames.
"""

import polars as pl
import pandas as pd
from typing import Union, Dict, Any


class DataQualityChecker:
    """
    Class-based interface for data quality analysis.

    Provides object-oriented wrapper around data quality functions
    for use in integration tests and more complex workflows.
    """

    def check_missing_values(
        self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Dict[str, Any]:
        """
        Check for missing values and return detailed report.

        Args:
            df: Input DataFrame (Pandas or Polars)

        Returns:
            Dictionary with missing values report
        """
        # Convert to Polars if it's a Pandas DataFrame
        if hasattr(df, "isnull"):  # Pandas DataFrame
            df_polars = pl.from_pandas(df)
        else:  # Already Polars DataFrame
            df_polars = df

        missing_values = df_polars.null_count()
        total_rows = len(df_polars)

        report = {
            "total_rows": total_rows,
            "columns_with_missing": {},
            "total_missing_values": 0,
        }

        for col, count in zip(df_polars.columns, missing_values.row(0)):
            if count > 0:
                percentage = (count / total_rows) * 100
                report["columns_with_missing"][col] = {
                    "count": count,
                    "percentage": percentage,
                }
                report["total_missing_values"] += count

        return report

    def check_data_types(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """
        Check data types and return detailed report.

        Args:
            df: Input DataFrame (Pandas or Polars)

        Returns:
            Dictionary with data types report
        """
        # Convert to Polars if it's a Pandas DataFrame
        if hasattr(df, "isnull"):  # Pandas DataFrame
            df_polars = pl.from_pandas(df)
        else:  # Already Polars DataFrame
            df_polars = df

        report = {
            "column_types": {},
            "numeric_columns": [],
            "text_columns": [],
            "categorical_columns": [],
        }

        for col in df_polars.columns:
            dtype = df_polars[col].dtype
            report["column_types"][col] = str(dtype)

            if dtype in [
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
                pl.UInt64,
                pl.UInt32,
            ]:
                report["numeric_columns"].append(col)
            elif dtype in [pl.Utf8, pl.Categorical]:
                if dtype == pl.Categorical:
                    report["categorical_columns"].append(col)
                else:
                    report["text_columns"].append(col)

        return report


def analyze_data_quality(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    """
    Analyze data quality including missing values, duplicates, and value ranges.

    This function provides comprehensive data quality analysis that works with
    both Pandas and Polars DataFrames. It automatically converts Pandas to Polars
    for consistent analysis.

    Args:
        df: Input DataFrame (Pandas or Polars)

    Displays:
        - Missing values summary with counts and percentages
        - Duplicate rows count
        - Numerical columns range and statistics
    """
    # Convert to Polars if it's a Pandas DataFrame
    if hasattr(df, "isnull"):  # Pandas DataFrame
        df_polars = pl.from_pandas(df)
    else:  # Already Polars DataFrame
        df_polars = df

    # Missing values analysis
    missing_values = df_polars.null_count()

    # Check if there are any missing values
    total_missing = missing_values.sum_horizontal().item()

    if total_missing > 0:
        print("\nMissing Values Summary:")
        for col, count in zip(df_polars.columns, missing_values.row(0)):
            if count > 0:
                percentage = (count / len(df_polars)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
    else:
        print("\nNo missing values found.")

    # Duplicate analysis
    n_duplicates = len(df_polars) - df_polars.unique().height
    print(f"\nNumber of duplicate rows: {n_duplicates}")

    # Value ranges for numerical columns
    print("\nNumerical Columns Range:")

    # Extended list of numerical columns that might be present
    numerical_cols = [
        "rating",
        "aroma",
        "acid",
        "body",
        "flavor",
        "aftertaste",
        "est_price",
        "agtron",
        "price_per_kg",
    ]

    for col in numerical_cols:
        if col in df_polars.columns:
            try:
                stats = df_polars.select(
                    [
                        pl.col(col).min().alias("min"),
                        pl.col(col).mean().alias("mean"),
                        pl.col(col).max().alias("max"),
                    ]
                )
                min_val = stats.item(0, "min")
                mean_val = stats.item(0, "mean")
                max_val = stats.item(0, "max")
                print(f"\n{col}:")
                print(f"Range: {min_val:.2f} to {max_val:.2f}")
                print(f"Mean: {mean_val:.2f}")
            except Exception as e:
                print(f"\n{col}: Unable to calculate statistics ({e})")


def get_data_overview(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    """
    Display comprehensive overview of the dataset.

    Works with both Pandas and Polars DataFrames, providing detailed
    information about columns, data types, and sample values.

    Args:
        df: Input DataFrame (Pandas or Polars)

    Displays:
        - Dataset shape
        - Column types and unique value counts
        - Sample values for categorical columns
    """
    # Convert to Polars if it's a Pandas DataFrame
    if hasattr(df, "isnull"):  # Pandas DataFrame
        df_polars = pl.from_pandas(df)
    else:  # Already Polars DataFrame
        df_polars = df

    print(f"Dataset Shape: {df_polars.shape}")
    print("\nColumn Descriptions:")

    for col in df_polars.columns:
        unique_count = df_polars[col].n_unique()
        print(f"\n{col}:")
        print(f"- Type: {df_polars[col].dtype}")
        print(f"- Unique values: {unique_count}")

        # Show sample of unique values for categorical columns
        if df_polars[col].dtype in [pl.Utf8, pl.Categorical]:
            try:
                if unique_count <= 10:
                    # Show all values if 10 or fewer
                    unique_vals = df_polars[col].unique().sort().to_list()
                    print(f"- Values: {unique_vals}")
                else:
                    # Show sample of values if more than 10
                    sample_values = (
                        df_polars[col].unique().sample(min(5, unique_count), seed=42)
                    )
                    print(f"- Sample values: {sample_values.to_list()}")
            except Exception as e:
                print(f"- Unable to display sample values ({e})")


def calculate_sensory_stats(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """
    Calculate summary statistics for sensory attributes.

    Computes mean, median, and standard deviation for all sensory rating columns.

    Args:
        df: Input DataFrame (Pandas or Polars)

    Returns:
        pl.DataFrame: DataFrame with statistical summaries for sensory attributes
    """
    # Convert to Polars if it's a Pandas DataFrame
    if hasattr(df, "isnull"):  # Pandas DataFrame
        df_polars = pl.from_pandas(df)
    else:  # Already Polars DataFrame
        df_polars = df

    sensory_cols = ["rating", "aroma", "acid", "body", "flavor", "aftertaste"]

    # Filter to only include columns that exist in the DataFrame
    existing_cols = [col for col in sensory_cols if col in df_polars.columns]

    if not existing_cols:
        print("Warning: No sensory columns found in DataFrame")
        return pl.DataFrame()

    return df_polars.select(
        [pl.col(col).mean().alias(f"{col}_mean") for col in existing_cols]
        + [pl.col(col).median().alias(f"{col}_median") for col in existing_cols]
        + [pl.col(col).std().alias(f"{col}_std") for col in existing_cols]
    )

"""Utility functions for coffee review analysis."""

from pathlib import Path
import polars as pl
import pandas as pd
from typing import Union

# Import consolidated data quality functions
from .data_quality import (
    analyze_data_quality,
    get_data_overview,
    calculate_sensory_stats,
)


def load_dataset_from_utils() -> pl.DataFrame:
    """
    Load coffee review data from hardcoded utility path.

    This is a utility function that loads data from a hardcoded path.
    Consider using load_main_dataset() from data.loader for main pipeline.

    Returns:
        pl.DataFrame: Coffee review dataset

    Note:
        This function uses a hardcoded path and may not work in all environments.
        Prefer load_main_dataset() for production use.
    """
    data_path = Path().absolute().parent / "data" / "raw" / "coffee_clean.csv"
    return pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)


def convert_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert pandas DataFrame to Polars with proper type handling.

    Args:
        df (pd.DataFrame): Input pandas DataFrame

    Returns:
        pl.DataFrame: Converted Polars DataFrame

    Note:
        Handles common type conversions and null value representations.
    """
    try:
        return pl.from_pandas(df)
    except Exception as e:
        # Fallback: convert via CSV if direct conversion fails
        import io

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return pl.read_csv(csv_buffer)


def convert_polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert Polars DataFrame to pandas with proper type handling.

    Args:
        df (pl.DataFrame): Input Polars DataFrame

    Returns:
        pd.DataFrame: Converted pandas DataFrame

    Note:
        Preserves data types where possible and handles Polars-specific types.
    """
    try:
        return df.to_pandas()
    except Exception as e:
        # Fallback: convert via CSV if direct conversion fails
        import io

        csv_buffer = io.StringIO()
        df.write_csv(csv_buffer)
        csv_buffer.seek(0)
        return pd.read_csv(csv_buffer)

"""Functions for loading and validating coffee review data."""

from pathlib import Path
import polars as pl

from config.settings import PATHS


def load_main_dataset() -> pl.DataFrame:
    """
    Load the main coffee review dataset for analysis pipeline.

    This function loads the primary dataset used throughout the analysis pipeline.
    Returns a Polars DataFrame optimized for large-scale data operations.

    Returns:
        pl.DataFrame: Main dataset optimized for Polars operations

    Raises:
        FileNotFoundError: If the data file doesn't exist
        pl.ComputeError: If there are issues reading the CSV
    """
    data_path = PATHS["raw"] / "coffee_clean.csv"
    df = pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)
    return df


def analyze_data_quality(df) -> None:
    """
    Analyze data quality including missing values, duplicates, and value ranges.

    Args:
        df: Input DataFrame (Pandas or Polars)
    """
    # Convert to Polars if it's a Pandas DataFrame
    if hasattr(df, "isnull"):  # Pandas DataFrame
        import pandas as pd

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
    numerical_cols = [
        "rating",
        "aroma",
        "acid",
        "body",
        "flavor",
        "aftertaste",
        "est_price",
        "agtron",
    ]

    for col in numerical_cols:
        if col in df_polars.columns:
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


def get_data_overview(df: pl.DataFrame) -> None:
    """
    Display comprehensive overview of the dataset.

    Args:
        df: Input DataFrame
    """
    print("\nColumn Descriptions:")
    for col in df.columns:
        unique_count = df[col].n_unique()
        print(f"\n{col}:")
        print(f"- Type: {df[col].dtype}")
        print(f"- Unique values: {unique_count}")

        # Show sample of unique values for categorical columns
        if df[col].dtype in [pl.Utf8, pl.Categorical]:
            sample_values = df[col].unique().sample(min(5, unique_count), seed=42)
            print(f"- Sample values: {sample_values.to_list()}")

"""Utility functions for coffee review analysis."""

from pathlib import Path
import polars as pl


def load_coffee_data() -> pl.DataFrame:
    """Load coffee review data."""
    data_path = Path().absolute().parent / "data" / "raw" / "coffee_clean.csv"
    return pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)


def analyze_data_quality(df: pl.DataFrame) -> None:
    """Analyze and display data quality metrics."""
    # Missing values analysis
    missing_values = df.null_count().row(0)  # Get the first row as values
    missing_cols = [
        (col, count) for col, count in zip(df.columns, missing_values) if count > 0
    ]

    if missing_cols:
        print("\nMissing Values Summary:")
        for col, count in missing_cols:
            percentage = (count / len(df)) * 100
            print(f"{col}: {count} ({percentage:.2f}%)")
    else:
        print("\nNo missing values found.")

    # Duplicate analysis
    n_duplicates = len(df) - df.unique().height
    print(f"\nNumber of duplicate rows: {n_duplicates}")

    # Value ranges for numerical columns
    print("\nNumerical Columns Range:")
    numerical_cols = ["rating", "aroma", "acid", "body", "flavor", "aftertaste"]

    for col in numerical_cols:
        if col in df.columns:
            stats = df.select(
                [
                    pl.col(col).min().alias("min"),
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).max().alias("max"),
                ]
            ).row(0)
            print(f"\n{col}:")
            print(f"Range: {stats[0]:.2f} to {stats[2]:.2f}")
            print(f"Mean: {stats[1]:.2f}")


def get_data_overview(df: pl.DataFrame) -> None:
    """Display basic information about the dataset."""
    print(f"Dataset Shape: {df.shape}")
    print("\nColumn Descriptions:")
    for col in df.columns:
        n_unique = df[col].n_unique()
        print(f"\n{col}:")
        print(f"- Type: {df[col].dtype}")
        print(f"- Unique values: {n_unique}")

        if df[col].dtype == pl.Utf8 and n_unique < 10:
            unique_vals = df[col].unique().sort().to_list()
            print(f"- Values: {unique_vals}")


def calculate_sensory_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate summary statistics for sensory attributes."""
    sensory_cols = ["rating", "aroma", "acid", "body", "flavor", "aftertaste"]

    return df.select(
        [pl.col(col).mean().alias(f"{col}_mean") for col in sensory_cols]
        + [pl.col(col).median().alias(f"{col}_median") for col in sensory_cols]
        + [pl.col(col).std().alias(f"{col}_std") for col in sensory_cols]
    )

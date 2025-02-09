"""Functions for loading and validating coffee review data."""

from pathlib import Path
import polars as pl

from src.config.settings import PATHS


def load_coffee_data() -> pl.DataFrame:
    """
    Load coffee review data and perform initial quality checks.

    Returns:
        Polars DataFrame containing coffee reviews
    """
    data_path = PATHS["raw"] / "coffee_clean.csv"
    df = pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)
    return df


def analyze_data_quality(df: pl.DataFrame) -> None:
    """
    Analyze data quality including missing values, duplicates, and value ranges.

    Args:
        df: Input DataFrame
    """
    # Missing values analysis
    missing_values = df.null_count()

    if missing_values.sum() > 0:
        print("\nMissing Values Summary:")
        for col, count in zip(df.columns, missing_values.row(0)):
            if count > 0:
                percentage = (count / len(df)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
    else:
        print("\nNo missing values found.")

    # Duplicate analysis
    n_duplicates = len(df) - df.unique().height
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
        if col in df.columns:
            stats = df.select(
                [
                    pl.col(col).min().alias("min"),
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).max().alias("max"),
                ]
            )
            print(f"\n{col}:")
            print(f"Range: {stats[0]['min']:.2f} to {stats[0]['max']:.2f}")
            print(f"Mean: {stats[0]['mean']:.2f}")


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

"""Functions for loading and validating coffee review data."""

from pathlib import Path
import polars as pl
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import from the new configuration system
try:
    from config import config

    PATHS = {
        "raw": config.paths.raw,
        "processed": config.paths.processed,
        "root": config.paths.root,
    }
except ImportError:
    # Fallback for when config is not available
    PATHS = {"raw": Path("data/raw"), "processed": Path("data/processed")}

# Import consolidated data quality functions
try:
    from utils.data_quality import analyze_data_quality, get_data_overview
except ImportError:
    # Fallback - define simple versions
    def analyze_data_quality(df):
        print("Data quality analysis not available")

    def get_data_overview(df):
        print(f"Dataset shape: {df.shape}")


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
    try:
        # Use configuration system if available
        data_path = config.paths.get_raw_data_path()
    except NameError:
        # Fallback path
        data_path = PATHS["raw"] / "coffee_clean.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)
    return df


def load_processed_dataset() -> pl.DataFrame:
    """
    Load the processed coffee review dataset.

    Returns:
        pl.DataFrame: Processed dataset

    Raises:
        FileNotFoundError: If the processed data file doesn't exist
    """
    try:
        # Use configuration system if available
        data_path = config.paths.get_processed_data_path()
    except NameError:
        # Fallback path
        data_path = PATHS["processed"] / "coffee_processed.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")

    df = pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)
    return df


def load_features_dataset() -> pl.DataFrame:
    """
    Load the features dataset.

    Returns:
        pl.DataFrame: Features dataset

    Raises:
        FileNotFoundError: If the features data file doesn't exist
    """
    try:
        # Use configuration system if available
        data_path = config.paths.get_features_data_path()
    except NameError:
        # Fallback path
        data_path = PATHS["processed"] / "coffee_features.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Features data file not found: {data_path}")

    df = pl.read_csv(data_path, null_values="NA", infer_schema_length=10000)
    return df


def validate_dataset(df: pl.DataFrame, dataset_name: str = "Dataset") -> bool:
    """
    Validate a dataset and provide quality analysis.

    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset for logging

    Returns:
        bool: True if dataset passes basic validation
    """
    print(f"\n{dataset_name} Validation:")
    print("-" * 50)

    # Basic checks
    if df.is_empty():
        print(f"ERROR: {dataset_name} is empty")
        return False

    print(f"✓ Dataset loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Perform quality analysis
    try:
        analyze_data_quality(df)
        get_data_overview(df)
        print(f"✓ {dataset_name} validation completed")
        return True
    except Exception as e:
        print(f"ERROR: Validation failed for {dataset_name}: {e}")
        return False

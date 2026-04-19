"""
Dual-mode data loading utility for Coffee Text Analytics.

This module provides data loading functionality that can work in two modes:
1. Local development: Load from local CSV files
2. Container/production: Load from MinIO S3-compatible storage

The mode is controlled by the ENVIRONMENT environment variable.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import boto3
import polars as pl
from botocore.exceptions import ClientError, NoCredentialsError

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# Import from the configuration system
try:
    # Try importing from the correct location
    from config import config

    # Verify the config is working correctly by checking if the path exists
    if config and not config.paths.get_raw_data_path().exists():
        logger.warning(
            "Config imported but path doesn't exist, falling back to manual resolution"
        )
        config = None
except ImportError:
    try:
        # Try importing from parent directory
        from ..config import config

        if config and not config.paths.get_raw_data_path().exists():
            logger.warning(
                "Config imported but path doesn't exist, falling back to manual resolution"
            )
            config = None
    except ImportError:
        # Fallback for when config is not available
        config = None


class CoffeeDataLoader:
    """
    Dual-mode data loader for coffee text analytics.

    Supports loading data from:
    - Local filesystem (development mode)
    - MinIO S3-compatible storage (container/production mode)
    """

    def __init__(self):
        """Initialize the data loader with environment-based configuration."""
        self.environment = os.getenv("ENVIRONMENT", "local").lower()
        self.s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:9000")
        self.s3_access_key = os.getenv("S3_ACCESS_KEY", "minio_access_key")
        self.s3_secret_key = os.getenv("S3_SECRET_KEY", "minio_secret_key")
        self.s3_bucket = os.getenv("S3_BUCKET", "mlflow")
        self.dataset_s3_key = os.getenv("DATASET_S3_KEY", "datasets/coffee_clean.csv")

        # Local paths - resolve relative to current working directory
        if config:
            self.local_data_path = config.paths.get_raw_data_path()
        else:
            # Use current working directory as project root
            # Check if we're running from project root or from src/data
            cwd = Path.cwd()
            if cwd.name == "coffee-text-analytics":
                # Running from project root
                self.local_data_path = cwd / "data" / "raw" / "coffee_clean.csv"
            else:
                # Try to find project root
                project_root = cwd
                while project_root.parent != project_root:
                    if (project_root / "data" / "raw" / "coffee_clean.csv").exists():
                        break
                    project_root = project_root.parent
                self.local_data_path = (
                    project_root / "data" / "raw" / "coffee_clean.csv"
                )

        logger.info(f"CoffeeDataLoader initialized in {self.environment} mode")

    def _create_s3_client(self):
        """Create and return an S3 client for MinIO."""
        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=self.s3_endpoint,
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name="us-east-1",  # MinIO doesn't require a specific region
            )
            return s3_client
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            raise

    def _load_from_local(self) -> pl.DataFrame:
        """Load data from local filesystem."""
        if not self.local_data_path.exists():
            raise FileNotFoundError(
                f"Local data file not found: {self.local_data_path}"
            )

        logger.info(f"Loading data from local file: {self.local_data_path}")
        df = pl.read_csv(
            self.local_data_path, null_values="NA", infer_schema_length=10000
        )
        logger.info(f"Successfully loaded {df.shape[0]:,} rows from local file")
        return df

    def _load_from_minio(self) -> pl.DataFrame:
        """Load data from MinIO S3-compatible storage."""
        try:
            s3_client = self._create_s3_client()

            # Test connection by checking if bucket exists
            try:
                s3_client.head_bucket(Bucket=self.s3_bucket)
                logger.info(f"Successfully connected to MinIO bucket: {self.s3_bucket}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise FileNotFoundError(f"MinIO bucket not found: {self.s3_bucket}")
                else:
                    raise

            # Check if the dataset exists
            try:
                s3_client.head_object(Bucket=self.s3_bucket, Key=self.dataset_s3_key)
                logger.info(f"Dataset found in MinIO: {self.dataset_s3_key}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise FileNotFoundError(
                        f"Dataset not found in MinIO: s3://{self.s3_bucket}/{self.dataset_s3_key}"
                    )
                else:
                    raise

            # Download to temporary file and load
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".csv", delete=False
            ) as tmp_file:
                try:
                    logger.info(
                        f"Downloading dataset from MinIO: s3://{self.s3_bucket}/{self.dataset_s3_key}"
                    )
                    s3_client.download_file(
                        self.s3_bucket, self.dataset_s3_key, tmp_file.name
                    )

                    # Load with Polars
                    df = pl.read_csv(
                        tmp_file.name, null_values="NA", infer_schema_length=10000
                    )
                    logger.info(f"Successfully loaded {df.shape[0]:,} rows from MinIO")
                    return df

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass

        except NoCredentialsError:
            raise RuntimeError(
                "MinIO credentials not configured. Please set S3_ACCESS_KEY and S3_SECRET_KEY environment variables."
            )
        except Exception as e:
            logger.error(f"Failed to load data from MinIO: {e}")
            raise

    def load_coffee_data(self) -> pl.DataFrame:
        """
        Load coffee dataset based on the current environment mode.

        Returns:
            pl.DataFrame: Coffee dataset

        Raises:
            FileNotFoundError: If dataset is not found in the specified location
            RuntimeError: If there are configuration or connection issues
        """
        try:
            if self.environment in ["local", "development"]:
                return self._load_from_local()
            elif self.environment in ["container", "production"]:
                return self._load_from_minio()
            else:
                logger.warning(
                    f"Unknown environment '{self.environment}', defaulting to local mode"
                )
                return self._load_from_local()

        except Exception as e:
            logger.error(f"Failed to load coffee data: {e}")
            raise


def load_coffee_data() -> pl.DataFrame:
    """
    Convenience function to load coffee dataset using environment-based configuration.

    This function creates a CoffeeDataLoader instance and loads the dataset
    based on the current environment (local vs container).

    Returns:
        pl.DataFrame: Coffee dataset

    Example:
        # In development (ENVIRONMENT=local)
        df = load_coffee_data()  # Loads from ./data/raw/coffee_clean.csv

        # In containers (ENVIRONMENT=container)
        df = load_coffee_data()  # Loads from MinIO s3://mlflow/datasets/coffee_clean.csv
    """
    loader = CoffeeDataLoader()
    return loader.load_coffee_data()


def upload_dataset_to_minio(local_path: Optional[str] = None) -> bool:
    """
    Utility function to upload the local dataset to MinIO.

    This is a helper function for setting up the container environment.

    Args:
        local_path: Path to local dataset file (optional, uses default if not provided)

    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        loader = CoffeeDataLoader()
        s3_client = loader._create_s3_client()

        # Use provided path or default
        if local_path:
            upload_path = Path(local_path)
        else:
            upload_path = loader.local_data_path

        if not upload_path.exists():
            logger.error(f"Local dataset not found: {upload_path}")
            return False

        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=loader.s3_bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(f"Creating MinIO bucket: {loader.s3_bucket}")
                s3_client.create_bucket(Bucket=loader.s3_bucket)
            else:
                raise

        # Upload the file
        logger.info(
            f"Uploading {upload_path} to MinIO: s3://{loader.s3_bucket}/{loader.dataset_s3_key}"
        )
        s3_client.upload_file(str(upload_path), loader.s3_bucket, loader.dataset_s3_key)

        logger.info("Dataset uploaded successfully to MinIO")
        return True

    except Exception as e:
        logger.error(f"Failed to upload dataset to MinIO: {e}")
        return False


if __name__ == "__main__":
    """
    Command-line interface for testing the data loader.

    Examples:
        python -m src.data.load_data                    # Load data in current environment
        ENVIRONMENT=local python -m src.data.load_data  # Force local mode
        ENVIRONMENT=container python -m src.data.load_data  # Force container mode
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the dual-mode coffee data loader"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload local dataset to MinIO"
    )
    parser.add_argument("--test-load", action="store_true", help="Test loading dataset")
    parser.add_argument(
        "--environment",
        choices=["local", "container"],
        help="Override environment mode",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Override environment if specified
    if args.environment:
        os.environ["ENVIRONMENT"] = args.environment

    if args.upload:
        print("Uploading dataset to MinIO...")
        success = upload_dataset_to_minio()
        if success:
            print("✓ Upload completed successfully")
        else:
            print("✗ Upload failed")
            sys.exit(1)

    if args.test_load or not (args.upload):
        print(f"Testing data loading in {os.getenv('ENVIRONMENT', 'local')} mode...")
        try:
            df = load_coffee_data()
            print(
                f"✓ Successfully loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns"
            )
            print(
                f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}"
            )
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            sys.exit(1)

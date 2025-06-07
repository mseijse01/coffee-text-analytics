"""
Production MLflow Configuration
Provides environment-based configuration for local development vs production deployment
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowConfig:
    """
    Production-grade MLflow configuration management

    Supports:
    - Local development (file-based tracking)
    - Remote production (PostgreSQL + S3)
    - Environment-based switching
    - Connection validation
    - Model registry management
    """

    def __init__(self, environment: str = "local"):
        """
        Initialize MLflow configuration

        Args:
            environment: "local", "production", or "docker"
        """
        self.environment = environment
        self.config = self._get_config()
        self.client: Optional[MlflowClient] = None

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration based on environment"""

        if self.environment == "local":
            return {
                "tracking_uri": "file:./mlruns",
                "artifact_location": "./mlruns",
                "backend_type": "file",
                "description": "Local file-based tracking for development",
            }

        elif self.environment == "docker":
            return {
                "tracking_uri": "http://localhost:5555",
                "artifact_location": "s3://mlflow-artifacts/",
                "backend_type": "remote",
                "s3_endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "minio_access_key",
                "aws_secret_access_key": "minio_secret_key",
                "description": "Docker-based remote tracking with PostgreSQL + MinIO",
            }

        elif self.environment == "production":
            # Production environment (cloud deployment)
            return {
                "tracking_uri": os.getenv(
                    "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
                ),
                "artifact_location": os.getenv(
                    "MLFLOW_ARTIFACT_ROOT", "s3://mlflow-production-artifacts/"
                ),
                "backend_type": "remote",
                "s3_endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "description": "Production cloud deployment",
            }

        else:
            raise ValueError(f"Unknown environment: {self.environment}")

    def setup(self) -> bool:
        """
        Setup MLflow with current configuration

        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config["tracking_uri"])
            logger.info(f"Set MLflow tracking URI: {self.config['tracking_uri']}")

            # Configure S3 for remote environments
            if self.config["backend_type"] == "remote":
                self._setup_s3_config()

            # Initialize client and validate connection
            self.client = MlflowClient(tracking_uri=self.config["tracking_uri"])

            # Test connection
            if self._validate_connection():
                logger.info(f"✅ MLflow setup successful ({self.environment})")
                logger.info(f"   Description: {self.config['description']}")
                return True
            else:
                logger.error(f"❌ MLflow connection validation failed")
                return False

        except Exception as e:
            logger.error(f"❌ MLflow setup failed: {e}")
            return False

    def _setup_s3_config(self):
        """Configure S3 settings for artifact storage"""
        s3_config = {
            "MLFLOW_S3_ENDPOINT_URL": self.config.get("s3_endpoint_url"),
            "AWS_ACCESS_KEY_ID": self.config.get("aws_access_key_id"),
            "AWS_SECRET_ACCESS_KEY": self.config.get("aws_secret_access_key"),
            "MLFLOW_S3_IGNORE_TLS": "true",  # For local MinIO
        }

        # Set environment variables for MLflow S3 integration
        for key, value in s3_config.items():
            if value:
                os.environ[key] = value
                logger.debug(f"Set {key}")

    def _validate_connection(self) -> bool:
        """Validate MLflow connection"""
        try:
            # Try to list experiments
            experiments = self.client.search_experiments()
            logger.info(f"Found {len(experiments)} existing experiments")
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def create_experiment_if_not_exists(self, experiment_name: str) -> str:
        """
        Create experiment if it doesn't exist

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment ID
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=f"{self.config['artifact_location']}/{experiment_name}",
                )
                logger.info(
                    f"Created experiment: {experiment_name} (ID: {experiment_id})"
                )
                return experiment_id
            else:
                logger.info(
                    f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})"
                )
                return experiment.experiment_id
        except Exception as e:
            logger.error(f"Failed to create/get experiment {experiment_name}: {e}")
            raise

    def get_model_registry_info(self) -> Dict[str, Any]:
        """Get model registry information and statistics"""
        try:
            models = self.client.search_registered_models()

            model_info = {"total_models": len(models), "models": []}

            for model in models:
                model_versions = self.client.search_model_versions(
                    f"name='{model.name}'"
                )

                # Get latest versions by stage
                stages = {}
                for version in model_versions:
                    stage = version.current_stage
                    if stage not in stages or int(version.version) > int(
                        stages[stage].version
                    ):
                        stages[stage] = version

                model_info["models"].append(
                    {
                        "name": model.name,
                        "description": model.description,
                        "total_versions": len(model_versions),
                        "stages": {
                            stage: version.version for stage, version in stages.items()
                        },
                    }
                )

            return model_info

        except Exception as e:
            logger.error(f"Failed to get model registry info: {e}")
            return {"error": str(e)}

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """
        Transition model to different stage (None/Staging/Production/Archived)

        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage

        Returns:
            True if successful
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False

    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            "environment": self.environment,
            "tracking_uri": self.config["tracking_uri"],
            "backend_type": self.config["backend_type"],
            "description": self.config["description"],
            "artifact_location": self.config["artifact_location"],
            "client_available": self.client is not None,
        }


def setup_production_mlflow(environment: str = "local") -> MLflowConfig:
    """
    Quick setup for production MLflow with environment detection

    Args:
        environment: "local", "docker", or "production"

    Returns:
        Configured MLflowConfig instance
    """
    # Auto-detect environment if not specified
    if environment == "auto":
        if os.getenv("DOCKER_ENV"):
            environment = "docker"
        elif os.getenv("PROD_ENV"):
            environment = "production"
        else:
            environment = "local"

    config = MLflowConfig(environment)

    if config.setup():
        logger.info(f"🚀 Production MLflow ready ({environment})")
        return config
    else:
        logger.error(f"❌ Failed to setup MLflow ({environment})")
        raise RuntimeError(f"MLflow setup failed for environment: {environment}")


# Environment detection helpers
def get_environment() -> str:
    """Auto-detect MLflow environment"""
    if os.getenv("DOCKER_ENV") or os.path.exists("/.dockerenv"):
        return "docker"
    elif os.getenv("KUBERNETES_SERVICE_HOST"):
        return "production"
    else:
        return "local"


def is_remote_available() -> bool:
    """Check if remote MLflow is available"""
    try:
        import requests

        response = requests.get("http://localhost:5000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

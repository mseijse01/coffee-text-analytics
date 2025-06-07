#!/usr/bin/env python3
"""
Production MLflow Setup Script
Easy one-command setup for production-grade MLflow infrastructure
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import Optional
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mlflow_setup.mlflow_config import setup_production_mlflow, is_remote_available

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionMLflowSetup:
    """Automated setup for production MLflow infrastructure"""

    def __init__(self, setup_type: str = "docker"):
        """
        Initialize setup

        Args:
            setup_type: "docker" for local Docker setup, "cloud" for cloud deployment
        """
        self.setup_type = setup_type
        self.setup_dir = Path(__file__).parent
        self.project_root = self.setup_dir.parent

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        logger.info("🔍 Checking prerequisites...")

        requirements = []

        # Check Docker
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"✅ Docker: {result.stdout.strip()}")
            else:
                requirements.append("Docker")
        except FileNotFoundError:
            requirements.append("Docker")

        # Check Docker Compose
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"✅ Docker Compose: {result.stdout.strip()}")
            else:
                requirements.append("Docker Compose")
        except FileNotFoundError:
            requirements.append("Docker Compose")

        # Check Python dependencies
        try:
            import mlflow
            import psycopg2
            import boto3

            logger.info("✅ Python dependencies available")
        except ImportError as e:
            logger.warning(f"⚠️ Missing Python dependency: {e}")
            logger.info("Install with: pip install mlflow psycopg2-binary boto3")

        if requirements:
            logger.error(f"❌ Missing requirements: {', '.join(requirements)}")
            logger.info("Please install the missing components and try again.")
            return False

        logger.info("✅ All prerequisites satisfied!")
        return True

    def setup_docker_environment(self) -> bool:
        """Setup Docker-based MLflow environment"""
        logger.info("🐳 Setting up Docker MLflow environment...")

        try:
            # Change to setup directory
            os.chdir(self.setup_dir)

            # Create necessary directories
            artifacts_dir = self.setup_dir / "mlflow_artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            # Stop any existing containers
            logger.info("Stopping existing containers...")
            subprocess.run(["docker-compose", "down"], capture_output=True)

            # Start services
            logger.info("Starting MLflow services...")
            result = subprocess.run(
                ["docker-compose", "up", "-d"], capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Failed to start services: {result.stderr}")
                return False

            logger.info("✅ Docker services started successfully!")

            # Wait for services to be ready
            return self._wait_for_services()

        except Exception as e:
            logger.error(f"❌ Docker setup failed: {e}")
            return False

    def _wait_for_services(self, max_wait: int = 120) -> bool:
        """Wait for all services to be ready"""
        logger.info("⏳ Waiting for services to be ready...")

        services = {"MLflow": "http://localhost:5555", "MinIO": "http://localhost:9001"}

        start_time = time.time()
        ready_services = set()

        while time.time() - start_time < max_wait:
            for service_name, url in services.items():
                if service_name not in ready_services:
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code in [
                            200,
                            403,
                        ]:  # 403 is OK for MinIO console
                            logger.info(f"✅ {service_name} is ready")
                            ready_services.add(service_name)
                    except:
                        pass

            if len(ready_services) == len(services):
                logger.info("🎉 All services are ready!")
                return True

            time.sleep(5)

        missing = set(services.keys()) - ready_services
        logger.error(f"❌ Timeout waiting for services: {', '.join(missing)}")
        return False

    def initialize_mlflow(self) -> bool:
        """Initialize MLflow with production configuration"""
        logger.info("🔧 Initializing MLflow configuration...")

        try:
            # Setup production MLflow config
            mlflow_config = setup_production_mlflow("docker")

            # Create default experiment
            experiment_id = mlflow_config.create_experiment_if_not_exists(
                "coffee-text-analytics-production"
            )

            # Setup MinIO bucket
            self._setup_minio_bucket()

            logger.info("✅ MLflow initialized successfully!")
            logger.info(f"   Experiment ID: {experiment_id}")

            # Print access information
            self._print_access_info()

            return True

        except Exception as e:
            logger.error(f"❌ MLflow initialization failed: {e}")
            return False

    def _setup_minio_bucket(self):
        """Setup MinIO bucket for artifacts"""
        try:
            from minio import Minio

            client = Minio(
                "localhost:9000",
                access_key="minio_access_key",
                secret_key="minio_secret_key",
                secure=False,
            )

            bucket_name = "mlflow-artifacts"
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"✅ Created MinIO bucket: {bucket_name}")
            else:
                logger.info(f"✅ MinIO bucket exists: {bucket_name}")

        except Exception as e:
            logger.warning(f"⚠️ MinIO bucket setup warning: {e}")
            logger.info("MinIO bucket will be created automatically when needed")

    def _print_access_info(self):
        """Print access information for services"""
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PRODUCTION MLFLOW SETUP COMPLETE!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📊 Access Points:")
        logger.info("   MLflow UI:      http://localhost:5555")
        logger.info("   Model Registry: http://localhost:5556")
        logger.info("   MinIO Console:  http://localhost:9001")
        logger.info("")
        logger.info("🔑 Credentials:")
        logger.info("   MinIO Access Key: minio_access_key")
        logger.info("   MinIO Secret Key: minio_secret_key")
        logger.info("")
        logger.info("🐳 Docker Management:")
        logger.info("   View logs:    docker-compose logs -f")
        logger.info("   Stop services: docker-compose down")
        logger.info("   Restart:      docker-compose restart")
        logger.info("")
        logger.info("💻 Python Usage:")
        logger.info("   from mlflow_setup.mlflow_config import setup_production_mlflow")
        logger.info("   config = setup_production_mlflow('docker')")
        logger.info("")
        logger.info("=" * 60)

    def run_health_check(self) -> bool:
        """Run comprehensive health check"""
        logger.info("🏥 Running health check...")

        checks = {
            "MLflow Server": self._check_mlflow_health,
            "Model Registry": self._check_model_registry,
            "MinIO Storage": self._check_minio_health,
            "Database Connection": self._check_database_health,
        }

        results = {}
        for check_name, check_func in checks.items():
            try:
                results[check_name] = check_func()
                status = "✅" if results[check_name] else "❌"
                logger.info(f"   {status} {check_name}")
            except Exception as e:
                results[check_name] = False
                logger.error(f"   ❌ {check_name}: {e}")

        success_count = sum(results.values())
        logger.info(f"\n🏥 Health Check: {success_count}/{len(checks)} checks passed")

        return success_count == len(checks)

    def _check_mlflow_health(self) -> bool:
        """Check MLflow server health"""
        response = requests.get("http://localhost:5000/health", timeout=10)
        return response.status_code == 200

    def _check_model_registry(self) -> bool:
        """Check model registry functionality"""
        config = setup_production_mlflow("docker")
        model_info = config.get_model_registry_info()
        return "error" not in model_info

    def _check_minio_health(self) -> bool:
        """Check MinIO health"""
        response = requests.get("http://localhost:9000/minio/health/live", timeout=10)
        return response.status_code == 200

    def _check_database_health(self) -> bool:
        """Check PostgreSQL database health"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host="localhost",
                port="5432",
                database="mlflow",
                user="mlflow",
                password="mlflow_password",
            )
            conn.close()
            return True
        except:
            return False


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Production MLflow")
    parser.add_argument(
        "--type",
        choices=["docker", "cloud"],
        default="docker",
        help="Setup type (default: docker)",
    )
    parser.add_argument(
        "--skip-prereq", action="store_true", help="Skip prerequisite checks"
    )
    parser.add_argument(
        "--health-check-only", action="store_true", help="Only run health check"
    )

    args = parser.parse_args()

    setup = ProductionMLflowSetup(args.type)

    if args.health_check_only:
        success = setup.run_health_check()
        sys.exit(0 if success else 1)

    # Check prerequisites
    if not args.skip_prereq and not setup.check_prerequisites():
        sys.exit(1)

    # Setup environment
    if args.type == "docker":
        if not setup.setup_docker_environment():
            sys.exit(1)

    # Initialize MLflow
    if not setup.initialize_mlflow():
        sys.exit(1)

    # Final health check
    if not setup.run_health_check():
        logger.warning("⚠️ Some health checks failed, but setup may still be functional")

    logger.info("🎉 Production MLflow setup completed successfully!")


if __name__ == "__main__":
    main()

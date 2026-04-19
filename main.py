#!/usr/bin/env python3
"""
Coffee Text Analytics - Main entry point

Thin orchestrator: parses CLI args, applies config overrides, manages MLflow,
and delegates each pipeline step to src/pipeline/.
"""

import argparse
import datetime
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import config
from config.environments import apply_environment_config
from config.validation import check_dependencies, print_config_summary, validate_config
from experiment.mlflow_integration import CoffeeMLflowTracker
from src.pipeline.preprocess import run_preprocessing
from src.pipeline.features import run_feature_extraction
from src.pipeline.selection import run_feature_selection
from src.pipeline.training import run_training
from src.pipeline.visualization import run_visualization

logger = logging.getLogger(__name__)


def setup_project() -> bool:
    """Validate config and dependencies, print summary."""
    is_valid = validate_config(config, raise_on_error=False)
    if not is_valid:
        logger.warning("Configuration validation failed, but continuing...")

    deps_available, missing_deps = check_dependencies()
    if not deps_available:
        logger.error(f"Missing dependencies: {missing_deps}")
        return False

    logger.info("Project directories set up successfully")
    print_config_summary(config)
    return True


def parse_args():
    """Parse CLI arguments with configuration-aware defaults."""
    parser = argparse.ArgumentParser(
        description="Run the Coffee Text Analytics pipeline"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocess", "features", "select", "train", "visualize", "all"],
        default=["all"],
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "testing", "cicd"],
        default=config.environment,
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(config.paths.get_raw_data_path()),
    )
    parser.add_argument(
        "--text_columns",
        nargs="+",
        default=config.models.text_columns,
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=config.models.target_column,
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=config.features.n_topics,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.models.models_to_train,
    )
    parser.add_argument(
        "--validate_config",
        action="store_true",
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=None,
        help="Fraction of data to sample (e.g. 0.1 for 10%%)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Absolute number of samples to use",
    )
    parser.add_argument("--box_cox", action="store_true")
    parser.add_argument("--box_cox_dual", action="store_true")
    parser.add_argument("--mlflow_tracking", action="store_true")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="coffee-text-analytics-thesis",
    )
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def apply_cli_overrides(args) -> None:
    """Apply CLI argument overrides to the global config object."""
    global config

    if args.environment != config.environment:
        logger.info(f"Switching to {args.environment} environment")
        config = apply_environment_config(config, args.environment)
        config.environment = args.environment

    if args.text_columns != config.models.text_columns:
        config.models.text_columns = args.text_columns
    if args.target_column != config.models.target_column:
        config.models.target_column = args.target_column
    if args.n_topics != config.features.n_topics:
        config.features.n_topics = args.n_topics
    if args.models != config.models.models_to_train:
        config.models.models_to_train = args.models
    if args.box_cox:
        config.models.box_cox_enabled = True
    if args.box_cox_dual:
        config.models.box_cox_dual_pipeline = True


def _setup_mlflow(args):
    """Start an MLflow run. Returns (tracker, run_id) or (None, None) on failure."""
    try:
        tracker = CoffeeMLflowTracker(args.experiment_name)

        run_name = args.run_name
        if not run_name:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_tag = (
                f"_sample{int(args.sample_fraction * 100)}pct"
                if args.sample_fraction
                else f"_sample{args.sample_size}"
                if args.sample_size
                else ""
            )
            run_name = f"pipeline_run_{ts}{sample_tag}"

        methodology_params = {
            "sample_fraction": args.sample_fraction or 1.0,
            "sample_size": args.sample_size or "full_dataset",
            "text_columns": ",".join(args.text_columns),
            "target_column": args.target_column,
            "models_to_train": ",".join(args.models),
            "steps_executed": ",".join(args.steps),
            "box_cox_enabled": args.box_cox,
            "box_cox_dual_pipeline": args.box_cox_dual,
            "environment": args.environment,
            "n_topics": args.n_topics,
        }

        run_id = tracker.start_methodology_run(
            run_name=run_name,
            sample_fraction=args.sample_fraction or 1.0,
            methodology_params=methodology_params,
            tags={
                "pipeline_version": "thesis_methodology_compliant",
                "execution_mode": "full_pipeline" if "all" in args.steps else "partial_pipeline",
            },
        )
        logger.info(f"MLflow run started: {run_name} (ID: {run_id})")
        return tracker, run_id

    except Exception as exc:
        logger.warning(f"MLflow setup failed: {exc} — continuing without tracking")
        return None, None


def main() -> None:
    args = parse_args()

    if args.validate_config:
        if validate_config(config, raise_on_error=False):
            print("Configuration is valid")
            print_config_summary(config)
        else:
            print("Configuration validation failed")
        return

    if not setup_project():
        logger.error("Project setup failed")
        return

    apply_cli_overrides(args)

    # MLflow (optional)
    mlflow_tracker, mlflow_run_id = None, None
    if args.mlflow_tracking:
        mlflow_tracker, mlflow_run_id = _setup_mlflow(args)

    steps = args.steps
    if "all" in steps:
        steps = ["preprocess", "features", "select", "train", "visualize"]

    step_fns = {
        "preprocess": run_preprocessing,
        "features": run_feature_extraction,
        "select": run_feature_selection,
        "train": run_training,
        "visualize": run_visualization,
    }
    step_labels = {
        "preprocess": "STEP 1: DATA PREPROCESSING",
        "features": "STEP 2: FEATURE EXTRACTION",
        "select": "STEP 3: FEATURE SELECTION",
        "train": "STEP 4: MODEL TRAINING",
        "visualize": "STEP 5: VISUALIZATION",
    }

    success = True
    step_results = {}

    for step in steps:
        if not success:
            break
        logger.info("=" * 50)
        logger.info(step_labels[step])
        logger.info("=" * 50)
        result = step_fns[step](args, config)
        step_results[step] = result
        success &= result

    # Log to MLflow
    if mlflow_tracker and mlflow_run_id:
        try:
            for step, ok in step_results.items():
                mlflow_tracker.log_methodology_compliance({f"step_{step}_completed": ok})
            mlflow_tracker.log_methodology_compliance(
                {
                    "pipeline_completed_successfully": success,
                    "separate_text_processing": True,
                    "corrected_lasso_selection": True,
                    "mlflow_tracking": True,
                }
            )
        except Exception as exc:
            logger.warning(f"Failed to log to MLflow: {exc}")
        finally:
            try:
                mlflow_tracker.end_run()
            except Exception:
                pass

    if success:
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {config.paths.output}")
    else:
        logger.error("=" * 50)
        logger.error("PIPELINE FAILED — check logs above for details")


if __name__ == "__main__":
    main()

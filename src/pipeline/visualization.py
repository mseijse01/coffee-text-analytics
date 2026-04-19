"""Pipeline step: results visualization."""

import logging
import pickle
import traceback

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from models import CoffeeModelEvaluator
from src.pipeline.constants import EXCLUDE_COLUMNS

logger = logging.getLogger(__name__)

# Must match the filename used by training.py
EVAL_RESULTS_FILENAME = "comprehensive_model_evaluation.pkl"


def run_visualization(args, config) -> bool:
    """
    Generate model comparison and feature importance plots.

    Loads evaluation results written by run_training and produces figures
    in config.paths.output/figures/.

    Args:
        args: Parsed CLI arguments.
        config: Application configuration object.

    Returns:
        True on success, False on any error.
    """
    logger.info("Starting visualization step")
    try:
        results_path = config.paths.output / EVAL_RESULTS_FILENAME
        if not results_path.exists():
            logger.error(
                f"No evaluation results found at {results_path}. Run training step first."
            )
            return False

        with open(results_path, "rb") as f:
            comparison_results = pickle.load(f)

        evaluator = CoffeeModelEvaluator()

        figures_dir = config.paths.output / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Overall model comparison
        evaluator.plot_model_comparison(
            comparison_results,
            metric="r2",
            save_path=str(figures_dir / "model_comparison_r2.png"),
        )

        # Per-model feature importance
        for model_name, results in comparison_results.get("individual_results", {}).items():
            if "feature_importance" in results:
                evaluator.plot_feature_importance(
                    results["feature_importance"],
                    model_name=model_name,
                    save_path=str(figures_dir / f"feature_importance_{model_name}.png"),
                )

        # Predicted vs actual for the best model
        best_model_name = comparison_results["best_models"]["r2"]
        best_results = comparison_results.get("individual_results", {}).get(best_model_name, {})

        if "predictions" in best_results:
            features_path = config.paths.get_features_data_path()
            df = pl.read_csv(features_path).to_pandas()

            target_column = config.models.target_column
            exclude_cols = list(
                set(EXCLUDE_COLUMNS) | set(config.models.text_columns) | {target_column}
            )
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_columns]
            y = df[target_column]

            try:
                y_binned = pd.cut(y, bins=5, labels=False)
                _, _, _, y_test = train_test_split(
                    X,
                    y,
                    test_size=config.models.test_size,
                    random_state=config.models.random_state,
                    stratify=y_binned,
                )
            except Exception:
                _, _, _, y_test = train_test_split(
                    X,
                    y,
                    test_size=config.models.test_size,
                    random_state=config.models.random_state,
                )

            evaluator.plot_predictions(
                y_test,
                best_results["predictions"],
                model_name=best_model_name,
                save_path=str(figures_dir / f"predictions_{best_model_name}.png"),
            )

        logger.info(f"Visualizations saved to {figures_dir}")
        return True

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.error(traceback.format_exc())
        return False

"""
Two-Step Hyperparameter Tuning Utilities

This module implements the signature two-step hyperparameter optimization approach:
Phase 1: Randomized Search (wide exploration)
Phase 2: Grid Search (fine-tuning around best parameters)

Following thesis methodology for robust hyperparameter optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator
import time
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class TwoStepHyperparameterTuner:
    """
    Two-step hyperparameter tuning implementation.

    Phase 1: Randomized Search for wide parameter space exploration
    Phase 2: Grid Search for fine-tuning around best parameters

    This approach balances computational efficiency with thorough optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize two-step tuner.

        Args:
            config: Configuration dictionary with tuning parameters
        """
        self.config = config
        self.randomized_search_config = config.get("randomized_search_config", {})
        self.grid_search_config = config.get("grid_search_config", {})

        # Results storage
        self.randomized_search_results_ = None
        self.grid_search_results_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []

        # Timing
        self.phase1_time_ = None
        self.phase2_time_ = None
        self.total_time_ = None

    def fit(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_distributions: Dict[str, List],
        grid_refinement_factor: int = 3,
    ) -> "TwoStepHyperparameterTuner":
        """
        Perform two-step hyperparameter tuning.

        Args:
            estimator: Base estimator to tune
            X: Training features
            y: Training targets
            param_distributions: Parameter distributions for randomized search
            grid_refinement_factor: How many values around best to test in grid search

        Returns:
            Self for method chaining
        """
        logger.info("🔍 Starting Two-Step Hyperparameter Tuning (Signature Approach)")
        logger.info("Phase 1: Randomized Search → Phase 2: Grid Search")

        start_time = time.time()

        # Phase 1: Randomized Search
        logger.info("📊 PHASE 1: Randomized Search (Wide Exploration)")
        phase1_start = time.time()

        randomized_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            **self.randomized_search_config,
        )

        logger.info(
            f"Exploring {self.randomized_search_config.get('n_iter', 50)} parameter combinations..."
        )
        randomized_search.fit(X, y)

        self.randomized_search_results_ = randomized_search
        self.phase1_time_ = time.time() - phase1_start

        logger.info(f"✅ Phase 1 completed in {self.phase1_time_:.2f}s")
        logger.info(
            f"Best randomized search score: {randomized_search.best_score_:.4f}"
        )
        logger.info(f"Best parameters from Phase 1: {randomized_search.best_params_}")

        # Phase 2: Grid Search around best parameters
        logger.info("🎯 PHASE 2: Grid Search (Fine-tuning)")
        phase2_start = time.time()

        # Create refined parameter grid around best parameters
        refined_param_grid = self._create_refined_grid(
            randomized_search.best_params_, param_distributions, grid_refinement_factor
        )

        logger.info(f"Fine-tuning with refined parameter grid:")
        for param, values in refined_param_grid.items():
            logger.info(f"  {param}: {values}")

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=refined_param_grid,
            **self.grid_search_config,
        )

        grid_search.fit(X, y)

        self.grid_search_results_ = grid_search
        self.phase2_time_ = time.time() - phase2_start
        self.total_time_ = time.time() - start_time

        # Store final results
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_

        # Store optimization history
        self._store_optimization_history()

        logger.info(f"✅ Phase 2 completed in {self.phase2_time_:.2f}s")
        logger.info(f"🎯 Two-Step Optimization Summary:")
        logger.info(f"  Total time: {self.total_time_:.2f}s")
        logger.info(f"  Phase 1 time: {self.phase1_time_:.2f}s")
        logger.info(f"  Phase 2 time: {self.phase2_time_:.2f}s")
        logger.info(f"  Final best score: {self.best_score_:.4f}")
        logger.info(
            f"  Score improvement: {self.best_score_ - randomized_search.best_score_:.4f}"
        )
        logger.info(f"  Final best parameters: {self.best_params_}")

        return self

    def _create_refined_grid(
        self,
        best_params: Dict[str, Any],
        param_distributions: Dict[str, List],
        refinement_factor: int,
    ) -> Dict[str, List]:
        """
        Create refined parameter grid around best parameters from randomized search.

        Args:
            best_params: Best parameters from randomized search
            param_distributions: Original parameter distributions
            refinement_factor: Number of values to test around best

        Returns:
            Refined parameter grid for grid search
        """
        refined_grid = {}

        for param, best_value in best_params.items():
            if param not in param_distributions:
                # If parameter not in original distributions, use best value
                refined_grid[param] = [best_value]
                continue

            original_values = param_distributions[param]

            if isinstance(best_value, (int, float)):
                # Numerical parameter - create range around best value
                refined_grid[param] = self._create_numerical_range(
                    best_value, original_values, refinement_factor
                )
            else:
                # Categorical parameter - include best and nearby options
                refined_grid[param] = self._create_categorical_range(
                    best_value, original_values, refinement_factor
                )

        return refined_grid

    def _create_numerical_range(
        self,
        best_value: Union[int, float],
        original_values: List,
        refinement_factor: int,
    ) -> List:
        """Create numerical range around best value."""
        # Sort original values
        sorted_values = sorted(
            [v for v in original_values if isinstance(v, (int, float))]
        )

        if not sorted_values:
            return [best_value]

        # Find position of best value
        try:
            best_idx = sorted_values.index(best_value)
        except ValueError:
            # Best value not in original list, find closest
            best_idx = min(
                range(len(sorted_values)),
                key=lambda i: abs(sorted_values[i] - best_value),
            )

        # Create range around best value
        start_idx = max(0, best_idx - refinement_factor // 2)
        end_idx = min(len(sorted_values), best_idx + refinement_factor // 2 + 1)

        refined_values = sorted_values[start_idx:end_idx]

        # Ensure best value is included
        if best_value not in refined_values:
            refined_values.append(best_value)
            refined_values.sort()

        return refined_values

    def _create_categorical_range(
        self, best_value: Any, original_values: List, refinement_factor: int
    ) -> List:
        """Create categorical range around best value."""
        # For categorical values, include best value and some others
        refined_values = [best_value]

        # Add other values up to refinement_factor
        other_values = [v for v in original_values if v != best_value]
        refined_values.extend(other_values[: refinement_factor - 1])

        return refined_values

    def _store_optimization_history(self):
        """Store optimization history for analysis."""
        self.optimization_history_ = {
            "phase1_results": {
                "best_score": self.randomized_search_results_.best_score_,
                "best_params": self.randomized_search_results_.best_params_,
                "cv_results": self.randomized_search_results_.cv_results_,
                "time": self.phase1_time_,
            },
            "phase2_results": {
                "best_score": self.grid_search_results_.best_score_,
                "best_params": self.grid_search_results_.best_params_,
                "cv_results": self.grid_search_results_.cv_results_,
                "time": self.phase2_time_,
            },
            "improvement": {
                "score_improvement": self.best_score_
                - self.randomized_search_results_.best_score_,
                "total_time": self.total_time_,
                "efficiency": (
                    self.best_score_ - self.randomized_search_results_.best_score_
                )
                / self.total_time_,
            },
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.

        Returns:
            Dictionary with optimization results and analysis
        """
        if not self.best_estimator_:
            return {"error": "Tuner not fitted yet"}

        return {
            "methodology": "Two-Step Hyperparameter Tuning (Signature Approach)",
            "phases": {
                "phase1": {
                    "method": "Randomized Search",
                    "purpose": "Wide parameter space exploration",
                    "iterations": self.randomized_search_config.get("n_iter", 50),
                    "cv_folds": self.randomized_search_config.get("cv", 3),
                    "time": self.phase1_time_,
                    "best_score": self.randomized_search_results_.best_score_,
                },
                "phase2": {
                    "method": "Grid Search",
                    "purpose": "Fine-tuning around best parameters",
                    "cv_folds": self.grid_search_config.get("cv", 5),
                    "time": self.phase2_time_,
                    "best_score": self.best_score_,
                },
            },
            "final_results": {
                "best_score": self.best_score_,
                "best_params": self.best_params_,
                "score_improvement": self.best_score_
                - self.randomized_search_results_.best_score_,
                "total_time": self.total_time_,
                "efficiency_score": (
                    self.best_score_ - self.randomized_search_results_.best_score_
                )
                / self.total_time_,
            },
            "performance_analysis": {
                "phase1_contribution": f"{(self.randomized_search_results_.best_score_ / self.best_score_) * 100:.1f}%",
                "phase2_improvement": f"{((self.best_score_ - self.randomized_search_results_.best_score_) / self.randomized_search_results_.best_score_) * 100:.2f}%",
                "time_distribution": {
                    "phase1_pct": f"{(self.phase1_time_ / self.total_time_) * 100:.1f}%",
                    "phase2_pct": f"{(self.phase2_time_ / self.total_time_) * 100:.1f}%",
                },
            },
        }

    def save_results(self, filepath: Union[str, Path]):
        """Save optimization results to file."""
        results = {
            "tuner_config": self.config,
            "optimization_summary": self.get_optimization_summary(),
            "optimization_history": self.optimization_history_,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
        }

        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(results, f)

        logger.info(f"Two-step tuning results saved to {filepath}")

    def print_summary(self):
        """Print comprehensive optimization summary."""
        summary = self.get_optimization_summary()

        print("\n" + "=" * 60)
        print("🎯 TWO-STEP HYPERPARAMETER TUNING SUMMARY")
        print("=" * 60)
        print(f"Methodology: {summary['methodology']}")
        print()

        print("📊 PHASE 1: Randomized Search")
        phase1 = summary["phases"]["phase1"]
        print(f"  Purpose: {phase1['purpose']}")
        print(f"  Iterations: {phase1['iterations']}")
        print(f"  CV Folds: {phase1['cv_folds']}")
        print(f"  Time: {phase1['time']:.2f}s")
        print(f"  Best Score: {phase1['best_score']:.4f}")
        print()

        print("🎯 PHASE 2: Grid Search")
        phase2 = summary["phases"]["phase2"]
        print(f"  Purpose: {phase2['purpose']}")
        print(f"  CV Folds: {phase2['cv_folds']}")
        print(f"  Time: {phase2['time']:.2f}s")
        print(f"  Best Score: {phase2['best_score']:.4f}")
        print()

        print("🏆 FINAL RESULTS")
        final = summary["final_results"]
        print(f"  Best Score: {final['best_score']:.4f}")
        print(f"  Score Improvement: {final['score_improvement']:.4f}")
        print(f"  Total Time: {final['total_time']:.2f}s")
        print(f"  Efficiency Score: {final['efficiency_score']:.6f}")
        print()

        print("📈 PERFORMANCE ANALYSIS")
        perf = summary["performance_analysis"]
        print(f"  Phase 1 Contribution: {perf['phase1_contribution']}")
        print(f"  Phase 2 Improvement: {perf['phase2_improvement']}")
        print(
            f"  Time Distribution: Phase 1 ({perf['time_distribution']['phase1_pct']}), Phase 2 ({perf['time_distribution']['phase2_pct']})"
        )
        print()

        print("🔧 BEST PARAMETERS")
        for param, value in self.best_params_.items():
            print(f"  {param}: {value}")
        print("=" * 60)


def apply_two_step_tuning(
    estimator: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model_name: str,
    config: Dict[str, Any],
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Apply two-step hyperparameter tuning to a model.

    Args:
        estimator: Base estimator to tune
        X: Training features
        y: Training targets
        model_name: Name of the model for configuration lookup
        config: Configuration object with tuning parameters

    Returns:
        Tuple of (best_estimator, optimization_summary)
    """
    # Check if two-step tuning is enabled
    if not config.models.two_step_tuning_enabled:
        logger.info(
            f"Two-step tuning disabled for {model_name}, using default parameters"
        )
        estimator.fit(X, y)
        return estimator, {"method": "default", "tuning_disabled": True}

    # Get model-specific configuration
    model_config_key = f"{model_name}_two_step"
    if not hasattr(config.models, model_config_key):
        logger.warning(
            f"No two-step configuration found for {model_name}, using default parameters"
        )
        estimator.fit(X, y)
        return estimator, {"method": "default", "no_config": True}

    model_two_step_config = getattr(config.models, model_config_key)

    # Initialize tuner
    tuner_config = {
        "randomized_search_config": config.models.randomized_search_config,
        "grid_search_config": config.models.grid_search_config,
    }

    tuner = TwoStepHyperparameterTuner(tuner_config)

    # Perform two-step tuning
    logger.info(f"🚀 Applying two-step tuning to {model_name}")
    tuner.fit(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions=model_two_step_config["randomized_params"],
        grid_refinement_factor=model_two_step_config["grid_refinement_factor"],
    )

    # Print summary
    tuner.print_summary()

    return tuner.best_estimator_, tuner.get_optimization_summary()

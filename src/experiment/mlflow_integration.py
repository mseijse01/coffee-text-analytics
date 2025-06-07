"""
MLflow Integration for Coffee Text Analytics
Enhanced experiment tracking for paper reimplementation with comprehensive research capabilities
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Import SHAP if available
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import Optuna if available
try:
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class EnhancedCoffeeMLflowTracker:
    """
    Enhanced MLflow experiment tracker for comprehensive coffee text analytics research

    Features:
    - Model registry integration with versioning
    - Hyperparameter optimization tracking
    - Feature artifact management
    - Automated experiment comparison
    - Publication-quality results generation
    - Complete paper methodology validation
    """

    def __init__(self, experiment_name: str = "coffee-text-analytics-thesis"):
        """Initialize enhanced MLflow tracker with research capabilities"""
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        self.current_run_id = None
        self.model_registry_name = f"{experiment_name}-models"
        self.setup_experiment()

    def setup_experiment(self):
        """Setup MLflow experiment with enhanced tracking capabilities"""
        # Set tracking URI to local directory for storage efficiency
        mlflow.set_tracking_uri("file:./mlruns")

        # Enable autologging for supported libraries
        mlflow.sklearn.autolog(disable=True)  # We'll manually log for better control
        mlflow.xgboost.autolog(disable=True)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                self.logger.info(
                    f"Created new MLflow experiment: {self.experiment_name}"
                )
            else:
                self.experiment_id = experiment.experiment_id
                self.logger.info(
                    f"Using existing MLflow experiment: {self.experiment_name}"
                )
        except Exception as e:
            self.logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    def start_enhanced_run(
        self,
        run_name: str,
        experiment_config: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start enhanced MLflow run with comprehensive paper methodology tracking

        Args:
            run_name: Descriptive name for the run
            experiment_config: Complete experiment configuration
            tags: Additional tags for organization

        Returns:
            MLflow run ID
        """
        # Enhanced tags for paper methodology tracking
        default_tags = {
            "methodology_focus": "paper_reimplementation",
            "approach": "enhanced_experiment_tracking",
            "thesis_version": "corrected_methodology_implementation",
            "sample_fraction": str(experiment_config.get("sample_fraction", 1.0)),
            "feature_selection": experiment_config.get(
                "feature_selection_method", "corrected_lasso"
            ),
            "models": ",".join(experiment_config.get("models", [])),
            "paper_compliance": "enhanced_validation",
        }

        if tags:
            default_tags.update(tags)

        # Start MLflow run
        run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name, tags=default_tags
        )

        self.current_run_id = run.info.run_id

        # Log comprehensive experiment configuration
        mlflow.log_params(experiment_config)

        # Log enhanced paper methodology parameters
        methodology_params = {
            "paper_text_processing": "separate_desc_columns_specialized",
            "paper_feature_naming": "tfidf_desc_X_Y_format",
            "paper_lasso_method": "corrected_combined_text_features",
            "paper_preprocessing": "extractor_specialized_preprocessing",
            "paper_categorical": "one_hot_encoding_complete",
            "paper_hypertuning": "grid_randomized_search_combination",
            "paper_evaluation": "comprehensive_shap_analysis",
            "paper_model_registry": "automated_versioning",
        }
        mlflow.log_params(methodology_params)

        self.logger.info(
            f"Started enhanced MLflow run: {run_name} (ID: {run.info.run_id})"
        )
        return run.info.run_id

    def log_hyperparameter_optimization(
        self,
        model_name: str,
        param_grid: Dict[str, Any],
        cv_results: Union[GridSearchCV, RandomizedSearchCV, Dict[str, Any]],
        optimization_type: str = "grid_search",
    ):
        """
        Log hyperparameter optimization results with complete tracking

        Args:
            model_name: Name of the model being optimized
            param_grid: Parameter grid searched
            cv_results: Results from GridSearchCV or RandomizedSearchCV
            optimization_type: Type of optimization performed
        """
        self.logger.info(f"Logging hyperparameter optimization for {model_name}")

        # Log optimization metadata
        mlflow.log_params(
            {
                f"{model_name}_optimization_type": optimization_type,
                f"{model_name}_n_param_combinations": len(param_grid)
                if isinstance(param_grid, list)
                else "grid",
                f"{model_name}_cv_folds": getattr(cv_results, "cv", "unknown"),
            }
        )

        # Log parameter grid
        for param, values in param_grid.items():
            if isinstance(values, (list, tuple, np.ndarray)):
                mlflow.log_param(f"{model_name}_param_grid_{param}", str(values))
            else:
                mlflow.log_param(f"{model_name}_param_grid_{param}", values)

        # Log optimization results
        if hasattr(cv_results, "best_params_"):
            # GridSearchCV/RandomizedSearchCV object
            mlflow.log_params(
                {
                    f"{model_name}_best_{k}": v
                    for k, v in cv_results.best_params_.items()
                }
            )

            mlflow.log_metrics(
                {
                    f"{model_name}_best_cv_score": cv_results.best_score_,
                    f"{model_name}_optimization_time": getattr(
                        cv_results, "_total_time", 0
                    ),
                }
            )

            # Log detailed CV results
            self._log_cv_results_details(model_name, cv_results)

        elif isinstance(cv_results, dict):
            # Custom optimization results
            for key, value in cv_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{model_name}_{key}", value)
                else:
                    mlflow.log_param(f"{model_name}_{key}", str(value))

    def _log_cv_results_details(self, model_name: str, cv_results):
        """Log detailed cross-validation results"""
        try:
            # Create CV results visualization
            cv_scores = cv_results.cv_results_["mean_test_score"]
            cv_std = cv_results.cv_results_["std_test_score"]

            # Log CV statistics
            mlflow.log_metrics(
                {
                    f"{model_name}_cv_mean_score": np.mean(cv_scores),
                    f"{model_name}_cv_std_score": np.std(cv_scores),
                    f"{model_name}_cv_min_score": np.min(cv_scores),
                    f"{model_name}_cv_max_score": np.max(cv_scores),
                }
            )

            # Save CV results as artifact
            cv_results_df = pd.DataFrame(cv_results.cv_results_)
            cv_results_path = f"cv_results_{model_name}.csv"
            cv_results_df.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(cv_results_path, "hyperparameter_optimization")

            # Clean up temporary file
            Path(cv_results_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(
                f"Could not log detailed CV results for {model_name}: {e}"
            )

    def log_model_with_registry(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        performance_metrics: Dict[str, float],
        feature_names: Optional[List[str]] = None,
        register_model: bool = True,
    ) -> Optional[str]:
        """
        Log model with registry integration and comprehensive metadata

        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (xgboost, random_forest, etc.)
            performance_metrics: Performance metrics for model
            feature_names: List of feature names
            register_model: Whether to register model in registry

        Returns:
            Model version if registered, None otherwise
        """
        self.logger.info(f"Logging model {model_name} with registry integration")

        # Log model with MLflow
        model_info = None
        try:
            if model_type.lower() == "xgboost":
                model_info = mlflow.xgboost.log_model(
                    model,
                    f"models/{model_name}",
                    registered_model_name=self.model_registry_name
                    if register_model
                    else None,
                )
            else:
                model_info = mlflow.sklearn.log_model(
                    model,
                    f"models/{model_name}",
                    registered_model_name=self.model_registry_name
                    if register_model
                    else None,
                )
        except Exception as e:
            self.logger.warning(f"Could not log model {model_name}: {e}")
            return None

        # Log model metadata
        model_metadata = {
            f"{model_name}_type": model_type,
            f"{model_name}_n_features": len(feature_names)
            if feature_names
            else "unknown",
            f"{model_name}_registered": register_model,
        }
        mlflow.log_params(model_metadata)

        # Log performance metrics
        prefixed_metrics = {
            f"{model_name}_{metric}": value
            for metric, value in performance_metrics.items()
        }
        mlflow.log_metrics(prefixed_metrics)

        # Log feature names if available
        if feature_names:
            feature_names_path = f"feature_names_{model_name}.json"
            with open(feature_names_path, "w") as f:
                json.dump(feature_names, f)
            mlflow.log_artifact(feature_names_path, f"models/{model_name}")
            Path(feature_names_path).unlink(missing_ok=True)

        return (
            model_info.registered_model_version
            if model_info and register_model
            else None
        )

    def log_feature_artifacts(
        self,
        selected_features: Dict[str, List[str]],
        feature_importance: Optional[Dict[str, Any]] = None,
        feature_statistics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log feature-related artifacts for reproducibility and analysis

        Args:
            selected_features: Dictionary of selected features by category
            feature_importance: Feature importance scores and analysis
            feature_statistics: Feature statistics and distributions
        """
        self.logger.info("Logging feature artifacts for reproducibility")

        # Log selected features
        selected_features_path = "selected_features.json"
        with open(selected_features_path, "w") as f:
            json.dump(selected_features, f, indent=2)
        mlflow.log_artifact(selected_features_path, "features")

        # Log feature counts
        feature_counts = {
            f"n_features_{category}": len(features)
            for category, features in selected_features.items()
        }
        total_features = sum(feature_counts.values())
        feature_counts["total_features"] = total_features
        mlflow.log_metrics(feature_counts)

        # Log feature importance if available
        if feature_importance:
            importance_path = "feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(feature_importance, f, indent=2)
            mlflow.log_artifact(importance_path, "features")

            # Create feature importance plot
            if SHAP_AVAILABLE and "shap_values" in feature_importance:
                self._create_shap_plots(feature_importance)

        # Log feature statistics if available
        if feature_statistics:
            stats_path = "feature_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(feature_statistics, f, indent=2)
            mlflow.log_artifact(stats_path, "features")

        # Clean up temporary files
        for temp_file in [
            selected_features_path,
            "feature_importance.json",
            "feature_statistics.json",
        ]:
            Path(temp_file).unlink(missing_ok=True)

    def _create_shap_plots(self, feature_importance: Dict[str, Any]):
        """Create and log SHAP analysis plots"""
        try:
            if "shap_values" not in feature_importance:
                return

            shap_values = feature_importance["shap_values"]
            feature_names = feature_importance.get("feature_names", [])

            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
            mlflow.log_artifact("shap_summary.png", "features/shap")
            plt.close()

            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, feature_names=feature_names, plot_type="bar", show=False
            )
            plt.tight_layout()
            plt.savefig("shap_importance.png", dpi=300, bbox_inches="tight")
            mlflow.log_artifact("shap_importance.png", "features/shap")
            plt.close()

            # Clean up temporary files
            for temp_file in ["shap_summary.png", "shap_importance.png"]:
                Path(temp_file).unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(f"Could not create SHAP plots: {e}")

    def log_experiment_comparison(
        self, comparison_results: Dict[str, Any], create_visualization: bool = True
    ):
        """
        Log comprehensive experiment comparison results

        Args:
            comparison_results: Results from model comparison
            create_visualization: Whether to create comparison visualizations
        """
        self.logger.info("Logging experiment comparison results")

        # Log comparison summary
        if "summary_metrics" in comparison_results:
            summary = comparison_results["summary_metrics"]
            for metric_name, model_scores in summary.items():
                for model_name, score in model_scores.items():
                    mlflow.log_metric(f"comparison_{metric_name}_{model_name}", score)

        # Log best models
        if "best_models" in comparison_results:
            best_models = comparison_results["best_models"]
            mlflow.log_params(
                {f"best_model_{metric}": model for metric, model in best_models.items()}
            )

        # Create and log comparison visualizations
        if create_visualization:
            self._create_comparison_plots(comparison_results)

        # Log detailed comparison report
        if "comparison_report" in comparison_results:
            report_path = "model_comparison_report.txt"
            with open(report_path, "w") as f:
                f.write(comparison_results["comparison_report"])
            mlflow.log_artifact(report_path, "comparison")
            Path(report_path).unlink(missing_ok=True)

    def _create_comparison_plots(self, comparison_results: Dict[str, Any]):
        """Create and log model comparison visualizations"""
        try:
            if "summary_metrics" not in comparison_results:
                return

            summary = comparison_results["summary_metrics"]

            # Create comparison bar plot
            metrics = list(summary.keys())
            models = list(next(iter(summary.values())).keys())

            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
            if len(metrics) == 1:
                axes = [axes]

            for i, metric in enumerate(metrics):
                model_scores = [summary[metric][model] for model in models]
                axes[i].bar(models, model_scores)
                axes[i].set_title(f"{metric.upper()} Comparison")
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
            mlflow.log_artifact("model_comparison.png", "comparison")
            plt.close()

            # Clean up
            Path("model_comparison.png").unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(f"Could not create comparison plots: {e}")

    def end_run(self):
        """End current MLflow run with summary logging"""
        if self.current_run_id:
            # Log run completion
            mlflow.log_metric("run_completed", 1.0)
            mlflow.log_param("completion_timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        mlflow.end_run()
        self.current_run_id = None
        self.logger.info("Enhanced MLflow run completed")

    def get_experiment_comparison(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get comparison of top N experiments for analysis

        Args:
            top_n: Number of top experiments to include

        Returns:
            Comparison results across experiments
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["metrics.r2 DESC"],
                max_results=top_n,
            )

            if runs.empty:
                return {"message": "No runs found for comparison"}

            # Extract key metrics for comparison
            comparison = {
                "total_runs": len(runs),
                "best_r2": runs["metrics.r2"].max()
                if "metrics.r2" in runs.columns
                else None,
                "avg_r2": runs["metrics.r2"].mean()
                if "metrics.r2" in runs.columns
                else None,
                "best_run_id": runs.iloc[0]["run_id"] if not runs.empty else None,
                "runs_summary": runs[
                    ["run_id", "metrics.r2", "metrics.rmse", "metrics.mae"]
                ].to_dict("records")
                if all(
                    col in runs.columns
                    for col in ["metrics.r2", "metrics.rmse", "metrics.mae"]
                )
                else None,
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Error getting experiment comparison: {e}")
            return {"error": str(e)}


# Convenience functions for enhanced usage
def setup_enhanced_coffee_mlflow(
    experiment_name: str = "coffee-text-analytics-thesis",
) -> EnhancedCoffeeMLflowTracker:
    """Quick setup for enhanced coffee analytics MLflow tracking"""
    return EnhancedCoffeeMLflowTracker(experiment_name)


# Legacy compatibility - keep original class for backward compatibility
class CoffeeMLflowTracker(EnhancedCoffeeMLflowTracker):
    """Legacy compatibility class - redirects to enhanced tracker"""

    pass


def setup_coffee_mlflow() -> CoffeeMLflowTracker:
    """Legacy compatibility function"""
    return CoffeeMLflowTracker()


class OptimizedCoffeeMLflowTracker(EnhancedCoffeeMLflowTracker):
    """
    Optimized MLflow tracker with Optuna integration for intelligent hyperparameter optimization

    Features:
    - 5-10x faster hyperparameter optimization (TPE algorithm)
    - Early pruning of unpromising trials
    - Multi-objective optimization
    - Automated hyperparameter importance analysis
    - Complete MLflow integration with study persistence
    """

    def __init__(self, experiment_name: str = "coffee-text-analytics-thesis"):
        """Initialize optimized tracker with Optuna capabilities"""
        super().__init__(experiment_name)
        self.optuna_studies = {}  # Store active studies
        self.study_storage = f"sqlite:///optuna_studies_{experiment_name}.db"

    def create_optuna_study(
        self,
        study_name: str,
        direction: str = "maximize",
        directions: Optional[List[str]] = None,
        resume: bool = True,
    ) -> optuna.Study:
        """
        Create or resume an Optuna study with MLflow integration

        Args:
            study_name: Name for the study
            direction: "maximize" or "minimize" (single-objective)
            directions: List of directions for multi-objective (e.g., ["maximize", "minimize"])
            resume: Whether to resume existing study or create new one

        Returns:
            Optuna study object
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        try:
            if directions:
                # Multi-objective optimization
                if resume:
                    try:
                        study = optuna.load_study(
                            study_name=study_name,
                            storage=self.study_storage,
                            directions=directions,
                        )
                        self.logger.info(f"Resumed multi-objective study: {study_name}")
                    except:
                        study = optuna.create_study(
                            study_name=study_name,
                            storage=self.study_storage,
                            directions=directions,
                            load_if_exists=False,
                        )
                        self.logger.info(
                            f"Created new multi-objective study: {study_name}"
                        )
                else:
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=self.study_storage,
                        directions=directions,
                        load_if_exists=False,
                    )
            else:
                # Single-objective optimization
                if resume:
                    try:
                        study = optuna.load_study(
                            study_name=study_name, storage=self.study_storage
                        )
                        self.logger.info(f"Resumed study: {study_name}")
                    except:
                        study = optuna.create_study(
                            study_name=study_name,
                            storage=self.study_storage,
                            direction=direction,
                            load_if_exists=False,
                        )
                        self.logger.info(f"Created new study: {study_name}")
                else:
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=self.study_storage,
                        direction=direction,
                        load_if_exists=True,  # Allow overwriting for testing
                    )

            self.optuna_studies[study_name] = study
            return study

        except Exception as e:
            self.logger.error(f"Error creating Optuna study: {e}")
            raise

    def optimize_hyperparameters(
        self,
        objective_function: callable,
        study_name: str,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = "maximize",
        directions: Optional[List[str]] = None,
        pruner_type: str = "hyperband",
        sampler_type: str = "tpe",
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna with MLflow integration

        Args:
            objective_function: Function to optimize (should return metric to optimize)
            study_name: Name for the optimization study
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (optional)
            direction: "maximize" or "minimize" for single-objective
            directions: List of directions for multi-objective
            pruner_type: "hyperband", "median", or "none"
            sampler_type: "tpe", "random", or "cmaes"

        Returns:
            Optimization results with best parameters and metrics
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.logger.info(f"Starting hyperparameter optimization: {study_name}")

        # Create study
        study = self.create_optuna_study(study_name, direction, directions)

        # Configure pruner
        if pruner_type == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        elif pruner_type == "median":
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        # Configure sampler
        if sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.RandomSampler()

        # Update study with sampler and pruner
        study.sampler = sampler
        study.pruner = pruner

        # Create MLflow callback for automatic logging
        # Note: We handle MLflow logging in our objective function, so no callback needed
        mlflow_callback = None

        # Run optimization
        if mlflow_callback:
            study.optimize(
                objective_function,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[mlflow_callback],
            )
        else:
            study.optimize(objective_function, n_trials=n_trials, timeout=timeout)

        # Log optimization results
        self._log_optuna_results(study)

        # Return comprehensive results
        return self._extract_optuna_results(study)

    def _log_optuna_results(self, study: optuna.Study):
        """Log Optuna optimization results to MLflow"""
        try:
            # Log best parameters
            if hasattr(study, "best_params"):
                mlflow.log_params(
                    {f"optuna_best_{k}": v for k, v in study.best_params.items()}
                )

            # Log best values
            if hasattr(study, "best_values"):
                # Multi-objective
                for i, value in enumerate(study.best_values):
                    mlflow.log_metric(f"optuna_best_value_{i}", value)
            elif hasattr(study, "best_value"):
                # Single-objective
                mlflow.log_metric("optuna_best_value", study.best_value)

            # Log study statistics
            mlflow.log_metrics(
                {
                    "optuna_n_trials": len(study.trials),
                    "optuna_n_complete_trials": len(
                        [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                    ),
                    "optuna_n_pruned_trials": len(
                        [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.PRUNED
                        ]
                    ),
                    "optuna_n_failed_trials": len(
                        [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.FAIL
                        ]
                    ),
                }
            )

            # Create and log hyperparameter importance plot
            if len(study.trials) > 10:  # Need enough trials for importance analysis
                self._create_optuna_importance_plot(study)

            # Create and log optimization history plot
            self._create_optuna_history_plot(study)

        except Exception as e:
            self.logger.warning(f"Could not log Optuna results: {e}")

    def _create_optuna_importance_plot(self, study: optuna.Study):
        """Create and log hyperparameter importance plot"""
        try:
            import optuna.visualization as vis

            # Create importance plot
            fig = vis.plot_param_importances(study)

            # Save as image
            importance_path = "optuna_param_importance.png"
            fig.write_image(importance_path)
            mlflow.log_artifact(importance_path, "optuna")

            # Clean up
            Path(importance_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(f"Could not create Optuna importance plot: {e}")

    def _create_optuna_history_plot(self, study: optuna.Study):
        """Create and log optimization history plot"""
        try:
            import optuna.visualization as vis

            # Create history plot
            fig = vis.plot_optimization_history(study)

            # Save as image
            history_path = "optuna_optimization_history.png"
            fig.write_image(history_path)
            mlflow.log_artifact(history_path, "optuna")

            # Clean up
            Path(history_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(f"Could not create Optuna history plot: {e}")

    def _extract_optuna_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Extract comprehensive results from Optuna study"""
        results = {
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "optimization_complete": True,
        }

        # Best parameters and values
        if hasattr(study, "best_params"):
            results["best_params"] = study.best_params

        if hasattr(study, "best_values"):
            results["best_values"] = study.best_values
        elif hasattr(study, "best_value"):
            results["best_value"] = study.best_value

        # Trial statistics
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]

        results["trial_stats"] = {
            "complete": len(complete_trials),
            "pruned": len(pruned_trials),
            "failed": len(failed_trials),
            "pruning_efficiency": len(pruned_trials) / len(study.trials)
            if study.trials
            else 0,
        }

        # Hyperparameter importance (if enough trials)
        if len(complete_trials) > 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                results["param_importance"] = importance
            except:
                pass

        return results

    def create_coffee_objective(
        self,
        train_func: callable,
        X_train: Any,
        X_val: Any,
        y_train: Any,
        y_val: Any,
        param_space: Dict[str, Any],
        multi_objective: bool = False,
    ) -> callable:
        """
        Create objective function for coffee model hyperparameter optimization

        Args:
            train_func: Function that trains model and returns metrics
            X_train, X_val, y_train, y_val: Training and validation data
            param_space: Hyperparameter space definition
            multi_objective: Whether to return multiple objectives

        Returns:
            Objective function for Optuna optimization
        """

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )

            # Start nested MLflow run for this trial
            with mlflow.start_run(nested=True):
                # Log trial parameters
                mlflow.log_params(params)

                # Train model and get metrics
                try:
                    metrics = train_func(params, X_train, X_val, y_train, y_val)

                    # Log metrics
                    mlflow.log_metrics(metrics)

                    # Return objective value(s)
                    if multi_objective:
                        return [
                            metrics["r2"],
                            -metrics["training_time"],
                        ]  # Maximize R², minimize time
                    else:
                        return metrics.get("r2", 0.0)  # Maximize R²

                except Exception as e:
                    self.logger.warning(f"Trial failed: {e}")
                    # Log failure
                    mlflow.log_param("trial_status", "failed")
                    mlflow.log_param("error", str(e))

                    if multi_objective:
                        return [0.0, float("inf")]  # Poor performance for failed trials
                    else:
                        return 0.0

        return objective


# Updated convenience function
def setup_optimized_coffee_mlflow(
    experiment_name: str = "coffee-text-analytics-thesis",
) -> OptimizedCoffeeMLflowTracker:
    """Quick setup for optimized coffee analytics MLflow tracking with Optuna"""
    return OptimizedCoffeeMLflowTracker(experiment_name)


if __name__ == "__main__":
    # Test enhanced MLflow integration
    print("Testing Enhanced Coffee MLflow Integration...")
    tracker = setup_enhanced_coffee_mlflow()

    # Test configuration
    test_config = {
        "sample_fraction": 0.1,
        "models": ["xgboost", "random_forest", "linear"],
        "feature_selection_method": "corrected_lasso",
        "text_columns": ["desc_1", "desc_2", "desc_3"],
    }

    run_id = tracker.start_enhanced_run(
        "enhanced_test_run", test_config, tags={"test": "enhanced_functionality"}
    )

    # Test logging capabilities
    tracker.log_feature_artifacts(
        {
            "tfidf": ["tfidf_desc_1_0", "tfidf_desc_1_1"],
            "bert": ["bert_desc_1_0", "bert_desc_1_1"],
        }
    )

    tracker.end_run()
    print(f"✅ Enhanced MLflow test completed. Run ID: {run_id}")
    print("🔍 View results with: mlflow ui")

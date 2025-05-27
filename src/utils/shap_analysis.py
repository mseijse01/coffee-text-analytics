"""
Comprehensive SHAP Analysis Utilities

This module implements extensive SHAP analysis across all models following thesis methodology.
Provides feature importance analysis, summary plots, dependence plots, and model comparison.

Following thesis approach for comprehensive model interpretability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings

# SHAP imports with error handling
try:
    import shap

    SHAP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("SHAP available for comprehensive analysis")
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "SHAP not available. Install shap package for interpretability analysis."
    )

warnings.filterwarnings("ignore", category=FutureWarning)


class ComprehensiveSHAPAnalyzer:
    """
    Comprehensive SHAP analysis for all model types.

    Implements thesis methodology for feature importance analysis across:
    - Linear models (Linear, Ridge, Lasso)
    - Tree-based models (Random Forest, XGBoost, Decision Tree)
    - Kernel models (SVR)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize comprehensive SHAP analyzer.

        Args:
            config: Configuration dictionary with analysis parameters
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP package required for comprehensive analysis")

        self.config = config or {}
        self.explainers_ = {}
        self.shap_values_ = {}
        self.feature_names_ = None
        self.analysis_results_ = {}

        # Analysis configuration
        self.max_display = self.config.get("max_display", 20)
        self.sample_size = self.config.get("sample_size", 100)
        self.random_state = self.config.get("random_state", 57)

        logger.info("Comprehensive SHAP analyzer initialized")

    def analyze_all_models(
        self,
        models: Dict[str, Any],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive SHAP analysis across all models.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            feature_names: Feature names for interpretation

        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info("🔍 Starting Comprehensive SHAP Analysis (Thesis Methodology)")
        logger.info(f"Analyzing {len(models)} models with SHAP interpretability")

        # Prepare data
        if isinstance(X_test, pd.DataFrame):
            self.feature_names_ = list(X_test.columns)
            X_test_array = X_test.values
        else:
            self.feature_names_ = feature_names or [
                f"feature_{i}" for i in range(X_test.shape[1])
            ]
            X_test_array = X_test

        if isinstance(y_test, pd.Series):
            y_test_array = y_test.values
        else:
            y_test_array = y_test

        # Sample data for SHAP analysis (computational efficiency)
        if len(X_test_array) > self.sample_size:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(
                len(X_test_array), size=self.sample_size, replace=False
            )
            X_sample = X_test_array[sample_indices]
            y_sample = y_test_array[sample_indices]
            logger.info(f"Sampling {self.sample_size} instances for SHAP analysis")
        else:
            X_sample = X_test_array
            y_sample = y_test_array

        # Analyze each model
        for model_name, model in models.items():
            try:
                logger.info(f"📊 Analyzing {model_name} with SHAP...")
                self._analyze_single_model(model_name, model, X_sample, y_sample)
            except Exception as e:
                logger.error(f"SHAP analysis failed for {model_name}: {e}")
                continue

        # Generate comprehensive comparison
        self._generate_model_comparison()

        logger.info("✅ Comprehensive SHAP analysis completed")
        return self.analysis_results_

    def _analyze_single_model(
        self, model_name: str, model: Any, X_sample: np.ndarray, y_sample: np.ndarray
    ):
        """Analyze a single model with appropriate SHAP explainer."""

        # Get the underlying sklearn model
        if hasattr(model, "model_"):
            sklearn_model = model.model_
        else:
            sklearn_model = model

        # Choose appropriate explainer based on model type
        explainer_type = self._get_explainer_type(model_name, sklearn_model)

        try:
            if explainer_type == "linear":
                explainer = shap.LinearExplainer(sklearn_model, X_sample)
                shap_values = explainer.shap_values(X_sample)

            elif explainer_type == "tree":
                explainer = shap.TreeExplainer(sklearn_model)
                shap_values = explainer.shap_values(X_sample)

            elif explainer_type == "kernel":
                # Use a smaller background sample for kernel explainer
                background_size = min(50, len(X_sample))
                background = shap.sample(X_sample, background_size)
                explainer = shap.KernelExplainer(sklearn_model.predict, background)
                shap_values = explainer.shap_values(
                    X_sample[:50]
                )  # Limit for computational efficiency

            else:
                # Default to Permutation explainer
                explainer = shap.PermutationExplainer(sklearn_model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)

            # Store results
            self.explainers_[model_name] = explainer
            self.shap_values_[model_name] = shap_values

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values)

            # Store analysis results
            self.analysis_results_[model_name] = {
                "explainer_type": explainer_type,
                "shap_values": shap_values,
                "feature_importance": feature_importance,
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                "feature_names": self.feature_names_,
                "sample_size": len(X_sample),
            }

            logger.info(
                f"✅ {model_name} SHAP analysis completed ({explainer_type} explainer)"
            )

        except Exception as e:
            logger.error(f"Failed to create SHAP explainer for {model_name}: {e}")
            # Fallback to permutation explainer
            try:
                explainer = shap.PermutationExplainer(sklearn_model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)

                self.explainers_[model_name] = explainer
                self.shap_values_[model_name] = shap_values

                feature_importance = self._calculate_feature_importance(shap_values)

                self.analysis_results_[model_name] = {
                    "explainer_type": "permutation",
                    "shap_values": shap_values,
                    "feature_importance": feature_importance,
                    "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                    "feature_names": self.feature_names_,
                    "sample_size": len(X_sample),
                }

                logger.info(
                    f"✅ {model_name} SHAP analysis completed (fallback permutation explainer)"
                )

            except Exception as e2:
                logger.error(
                    f"Fallback SHAP analysis also failed for {model_name}: {e2}"
                )

    def _get_explainer_type(self, model_name: str, sklearn_model: Any) -> str:
        """Determine the appropriate SHAP explainer type for a model."""

        # Linear models
        if any(
            linear_type in str(type(sklearn_model)).lower()
            for linear_type in ["linear", "ridge", "lasso"]
        ):
            return "linear"

        # Tree-based models
        if any(
            tree_type in str(type(sklearn_model)).lower()
            for tree_type in ["forest", "tree", "xgb", "gradient"]
        ):
            return "tree"

        # SVR and other kernel methods
        if "svr" in str(type(sklearn_model)).lower():
            return "kernel"

        # Default
        return "permutation"

    def _calculate_feature_importance(
        self, shap_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance from SHAP values."""

        # Mean absolute SHAP values
        importance_scores = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance dictionary
        feature_importance = {}
        for i, score in enumerate(importance_scores):
            if i < len(self.feature_names_):
                feature_importance[self.feature_names_[i]] = float(score)

        return feature_importance

    def _generate_model_comparison(self):
        """Generate comprehensive model comparison analysis."""

        if not self.analysis_results_:
            logger.warning("No SHAP analysis results available for comparison")
            return

        logger.info("📈 Generating SHAP model comparison analysis...")

        # Compare feature importance across models
        comparison_data = []
        all_features = set()

        for model_name, results in self.analysis_results_.items():
            feature_importance = results["feature_importance"]
            all_features.update(feature_importance.keys())

            for feature, importance in feature_importance.items():
                comparison_data.append(
                    {"model": model_name, "feature": feature, "importance": importance}
                )

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Find top features across all models
        feature_importance_summary = (
            comparison_df.groupby("feature")["importance"]
            .agg(["mean", "std", "min", "max"])
            .sort_values("mean", ascending=False)
        )

        # Store comparison results
        self.analysis_results_["comparison"] = {
            "feature_importance_summary": feature_importance_summary,
            "comparison_data": comparison_df,
            "top_features": feature_importance_summary.head(
                self.max_display
            ).index.tolist(),
            "model_agreement": self._calculate_model_agreement(comparison_df),
        }

        logger.info("✅ SHAP model comparison completed")

    def _calculate_model_agreement(
        self, comparison_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate agreement between models on feature importance."""

        # Pivot to get model vs feature importance matrix
        importance_matrix = comparison_df.pivot(
            index="feature", columns="model", values="importance"
        ).fillna(0)

        # Calculate correlation between models
        model_correlations = importance_matrix.corr()

        # Calculate average agreement
        n_models = len(model_correlations)
        total_correlation = 0
        count = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                total_correlation += model_correlations.iloc[i, j]
                count += 1

        average_agreement = total_correlation / count if count > 0 else 0

        return {
            "average_agreement": average_agreement,
            "model_correlations": model_correlations.to_dict(),
            "agreement_interpretation": self._interpret_agreement(average_agreement),
        }

    def _interpret_agreement(self, agreement: float) -> str:
        """Interpret model agreement score."""
        if agreement >= 0.8:
            return "Very High Agreement"
        elif agreement >= 0.6:
            return "High Agreement"
        elif agreement >= 0.4:
            return "Moderate Agreement"
        elif agreement >= 0.2:
            return "Low Agreement"
        else:
            return "Very Low Agreement"

    def plot_summary_comparison(
        self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """Create comprehensive SHAP summary plot comparing all models."""

        if "comparison" not in self.analysis_results_:
            raise ValueError(
                "No comparison analysis available. Run analyze_all_models first."
            )

        comparison_data = self.analysis_results_["comparison"]["comparison_data"]
        top_features = self.analysis_results_["comparison"]["top_features"][
            : self.max_display
        ]

        # Filter to top features
        plot_data = comparison_data[comparison_data["feature"].isin(top_features)]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            "Comprehensive SHAP Analysis - Model Comparison",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Feature importance heatmap
        ax1 = axes[0, 0]
        pivot_data = plot_data.pivot(
            index="feature", columns="model", values="importance"
        )
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="viridis", ax=ax1)
        ax1.set_title("Feature Importance Heatmap")
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Features")

        # 2. Top features bar plot
        ax2 = axes[0, 1]
        feature_summary = self.analysis_results_["comparison"][
            "feature_importance_summary"
        ]
        top_10 = feature_summary.head(10)
        bars = ax2.barh(range(len(top_10)), top_10["mean"], xerr=top_10["std"])
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels(top_10.index, fontsize=8)
        ax2.set_xlabel("Mean SHAP Importance")
        ax2.set_title("Top 10 Features (Mean ± Std)")
        ax2.invert_yaxis()

        # 3. Model agreement
        ax3 = axes[1, 0]
        agreement_data = self.analysis_results_["comparison"]["model_agreement"]
        corr_matrix = pd.DataFrame(agreement_data["model_correlations"])
        sns.heatmap(
            corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax3
        )
        ax3.set_title(
            f"Model Agreement\n({agreement_data['agreement_interpretation']})"
        )

        # 4. Feature importance distribution
        ax4 = axes[1, 1]
        for model in plot_data["model"].unique():
            model_data = plot_data[plot_data["model"] == model]
            ax4.hist(model_data["importance"], alpha=0.6, label=model, bins=20)
        ax4.set_xlabel("SHAP Importance")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Feature Importance Distribution")
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP summary comparison saved to {save_path}")

        return fig

    def plot_individual_summary(
        self, model_name: str, save_path: Optional[str] = None, plot_type: str = "bar"
    ) -> plt.Figure:
        """Create SHAP summary plot for individual model."""

        if model_name not in self.analysis_results_:
            raise ValueError(f"No SHAP analysis available for {model_name}")

        results = self.analysis_results_[model_name]
        shap_values = results["shap_values"]

        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))

        if plot_type == "bar":
            shap.summary_plot(
                shap_values,
                feature_names=self.feature_names_,
                plot_type="bar",
                max_display=self.max_display,
                show=False,
            )
        else:
            shap.summary_plot(
                shap_values,
                feature_names=self.feature_names_,
                max_display=self.max_display,
                show=False,
            )

        plt.title(f"SHAP Summary - {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP summary for {model_name} saved to {save_path}")

        return plt.gcf()

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive SHAP analysis report."""

        if not self.analysis_results_:
            return "No SHAP analysis results available."

        report = []
        report.append("=" * 60)
        report.append("🔍 COMPREHENSIVE SHAP ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Following thesis methodology for feature importance analysis")
        report.append("")

        # Model overview
        report.append("📊 MODELS ANALYZED:")
        for model_name, results in self.analysis_results_.items():
            if model_name != "comparison":
                explainer_type = results.get("explainer_type", "unknown")
                sample_size = results.get("sample_size", "unknown")
                report.append(
                    f"  • {model_name}: {explainer_type} explainer ({sample_size} samples)"
                )
        report.append("")

        # Top features across all models
        if "comparison" in self.analysis_results_:
            comparison = self.analysis_results_["comparison"]
            report.append("🏆 TOP FEATURES (ACROSS ALL MODELS):")
            feature_summary = comparison["feature_importance_summary"]

            for i, (feature, stats) in enumerate(
                feature_summary.head(10).iterrows(), 1
            ):
                report.append(f"  {i:2d}. {feature}")
                report.append(f"      Mean importance: {stats['mean']:.4f}")
                report.append(f"      Std deviation:   {stats['std']:.4f}")
                report.append(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                report.append("")

            # Model agreement
            agreement = comparison["model_agreement"]
            report.append("🤝 MODEL AGREEMENT ANALYSIS:")
            report.append(f"  Average agreement: {agreement['average_agreement']:.3f}")
            report.append(f"  Interpretation: {agreement['agreement_interpretation']}")
            report.append("")

        # Individual model insights
        report.append("🔍 INDIVIDUAL MODEL INSIGHTS:")
        for model_name, results in self.analysis_results_.items():
            if model_name != "comparison":
                report.append(f"\n  📈 {model_name.upper()}:")

                # Top 5 features for this model
                feature_importance = results["feature_importance"]
                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:5]

                for i, (feature, importance) in enumerate(sorted_features, 1):
                    report.append(f"    {i}. {feature}: {importance:.4f}")

        report.append("")
        report.append("=" * 60)
        report.append("✅ COMPREHENSIVE SHAP ANALYSIS COMPLETE")
        report.append("=" * 60)

        return "\n".join(report)

    def save_analysis(self, filepath: Union[str, Path]):
        """Save comprehensive SHAP analysis results."""

        save_data = {
            "analysis_results": self.analysis_results_,
            "feature_names": self.feature_names_,
            "config": self.config,
            "comprehensive_report": self.generate_comprehensive_report(),
        }

        filepath = Path(filepath)

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Comprehensive SHAP analysis saved to {filepath}")


def run_comprehensive_shap_analysis(
    models: Dict[str, Any],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive SHAP analysis following thesis methodology.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save results
        config: Analysis configuration

    Returns:
        Comprehensive analysis results
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP package not available. Install with: pip install shap")
        return {}

    logger.info("🚀 Starting Comprehensive SHAP Analysis (Thesis Methodology)")

    # Initialize analyzer
    analyzer = ComprehensiveSHAPAnalyzer(config)

    # Run analysis
    results = analyzer.analyze_all_models(models, X_test, y_test)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    try:
        # Summary comparison plot
        fig = analyzer.plot_summary_comparison(
            save_path=str(output_dir / "shap_comprehensive_summary.png")
        )
        plt.close(fig)

        # Individual model plots
        for model_name in models.keys():
            if model_name in analyzer.analysis_results_:
                # Bar plot
                fig = analyzer.plot_individual_summary(
                    model_name,
                    save_path=str(output_dir / f"shap_summary_{model_name}_bar.png"),
                    plot_type="bar",
                )
                plt.close(fig)

                # Dot plot
                fig = analyzer.plot_individual_summary(
                    model_name,
                    save_path=str(output_dir / f"shap_summary_{model_name}_dot.png"),
                    plot_type="dot",
                )
                plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating SHAP plots: {e}")

    # Save analysis results
    analyzer.save_analysis(output_dir / "comprehensive_shap_analysis.pkl")

    # Generate and save report
    report = analyzer.generate_comprehensive_report()
    with open(output_dir / "shap_analysis_report.txt", "w") as f:
        f.write(report)

    logger.info("✅ Comprehensive SHAP Analysis completed successfully!")
    logger.info(f"Results saved to: {output_dir}")

    return results

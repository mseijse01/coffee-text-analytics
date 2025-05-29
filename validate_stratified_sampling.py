#!/usr/bin/env python3
"""
Stratified Sampling Validation Script

This script validates that our stratified sampling implementation properly
maintains the distribution of coffee ratings between train and test sets.

Usage:
    python validate_stratified_sampling.py [--sample_fraction 0.1]
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import config
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(sample_fraction=None):
    """Load the processed data."""
    # Try to load selected features first, then fall back to original features
    selected_features_path = config.paths.processed / "coffee_features_selected.csv"
    if selected_features_path.exists():
        data_path = selected_features_path
        logger.info(f"Loading selected features from {data_path}")
    else:
        data_path = config.paths.get_features_data_path()
        logger.info(f"Loading original features from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")

    # Apply sampling if requested
    if sample_fraction:
        n_samples = int(len(df) * sample_fraction)
        df = df.sample(n=n_samples, random_state=57).reset_index(drop=True)
        logger.info(f"Applied {sample_fraction:.1%} sampling: {df.shape}")

    return df


def prepare_features_and_target(df):
    """Prepare features and target variable."""
    target_column = config.models.target_column

    # Exclude non-feature columns (same logic as main.py)
    potential_exclude_columns = (
        config.models.text_columns
        + [target_column]
        + [
            "id",
            "slug",
            "all_text",
            "name",
            "location",
            "origin",
            "est_price",
            "review_date",
            "agtron",
            "aroma",
            "acid",
            "body",
            "flavor",
            "aftertaste",
            "with_milk",
            "price_value",
            "price_unit",
            "price_standardized",
            "processed_desc_1",
            "processed_desc_2",
            "processed_desc_3",
            "merged_text",
            "processed_text",
            "url",
            "loc",
        ]
    )

    exclude_columns = [col for col in potential_exclude_columns if col in df.columns]
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    X = df[feature_columns]
    y = df[target_column]

    return X, y


def validate_stratified_sampling(X, y, n_bins=5, test_size=0.3, random_state=57):
    """
    Validate stratified sampling implementation.

    Args:
        X: Feature matrix
        y: Target variable
        n_bins: Number of bins for stratification
        test_size: Test set size
        random_state: Random seed

    Returns:
        Dictionary with validation results
    """
    logger.info("🔍 Validating Stratified Sampling Implementation")
    logger.info(f"Target variable: {y.name}")
    logger.info(f"Data size: {len(y)} samples")
    logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Create stratified bins (same as main.py)
    y_binned = pd.cut(y, bins=n_bins, labels=False)

    # Check for any NaN values in binning
    if y_binned.isna().any():
        logger.warning(f"Found {y_binned.isna().sum()} NaN values in binning")
        # Remove NaN values
        valid_indices = ~y_binned.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        y_binned = y_binned[valid_indices]

    logger.info(f"Stratification bins: {n_bins}")
    logger.info("Bin distribution in full dataset:")
    bin_counts = pd.Series(y_binned).value_counts().sort_index()
    for bin_idx, count in bin_counts.items():
        percentage = (count / len(y_binned)) * 100
        logger.info(f"  Bin {bin_idx}: {count} samples ({percentage:.1f}%)")

    # Perform stratified split
    X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
        X,
        y,
        y_binned,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binned,
    )

    logger.info(f"Train set size: {len(y_train)} ({len(y_train) / len(y) * 100:.1f}%)")
    logger.info(f"Test set size: {len(y_test)} ({len(y_test) / len(y) * 100:.1f}%)")

    # Analyze distribution consistency
    results = {
        "original_stats": {
            "mean": y.mean(),
            "std": y.std(),
            "min": y.min(),
            "max": y.max(),
            "skewness": stats.skew(y),
            "kurtosis": stats.kurtosis(y),
        },
        "train_stats": {
            "mean": y_train.mean(),
            "std": y_train.std(),
            "min": y_train.min(),
            "max": y_train.max(),
            "skewness": stats.skew(y_train),
            "kurtosis": stats.kurtosis(y_train),
        },
        "test_stats": {
            "mean": y_test.mean(),
            "std": y_test.std(),
            "min": y_test.min(),
            "max": y_test.max(),
            "skewness": stats.skew(y_test),
            "kurtosis": stats.kurtosis(y_test),
        },
        "bin_distributions": {
            "original": pd.Series(y_binned).value_counts().sort_index(),
            "train": pd.Series(bins_train).value_counts().sort_index(),
            "test": pd.Series(bins_test).value_counts().sort_index(),
        },
    }

    # Calculate distribution differences
    logger.info("\n📊 DISTRIBUTION ANALYSIS:")
    logger.info("=" * 50)

    # Statistical comparison
    logger.info("Statistical Measures:")
    for measure in ["mean", "std", "skewness", "kurtosis"]:
        orig = results["original_stats"][measure]
        train = results["train_stats"][measure]
        test = results["test_stats"][measure]

        train_diff = abs(train - orig) / abs(orig) * 100 if orig != 0 else 0
        test_diff = abs(test - orig) / abs(orig) * 100 if orig != 0 else 0

        logger.info(
            f"  {measure.capitalize():10s}: Original={orig:.4f}, Train={train:.4f} ({train_diff:+.1f}%), Test={test:.4f} ({test_diff:+.1f}%)"
        )

    # Bin distribution comparison
    logger.info("\nBin Distribution Consistency:")
    original_dist = results["bin_distributions"]["original"]
    train_dist = results["bin_distributions"]["train"]
    test_dist = results["bin_distributions"]["test"]

    for bin_idx in sorted(original_dist.index):
        orig_pct = (original_dist[bin_idx] / len(y_binned)) * 100
        train_pct = (train_dist.get(bin_idx, 0) / len(bins_train)) * 100
        test_pct = (test_dist.get(bin_idx, 0) / len(bins_test)) * 100

        train_diff = train_pct - orig_pct
        test_diff = test_pct - orig_pct

        logger.info(
            f"  Bin {bin_idx}: Original={orig_pct:.1f}%, Train={train_pct:.1f}% ({train_diff:+.1f}%), Test={test_pct:.1f}% ({test_diff:+.1f}%)"
        )

    # Statistical tests
    logger.info("\n🧪 STATISTICAL TESTS:")
    logger.info("=" * 50)

    # Kolmogorov-Smirnov test for distribution similarity
    ks_train_stat, ks_train_p = stats.ks_2samp(y, y_train)
    ks_test_stat, ks_test_p = stats.ks_2samp(y, y_test)

    logger.info(
        f"Kolmogorov-Smirnov Test (Original vs Train): statistic={ks_train_stat:.4f}, p-value={ks_train_p:.4f}"
    )
    logger.info(
        f"Kolmogorov-Smirnov Test (Original vs Test):  statistic={ks_test_stat:.4f}, p-value={ks_test_p:.4f}"
    )

    # Chi-square test for bin distribution
    try:
        # Prepare contingency table
        train_expected = (
            train_dist.reindex(original_dist.index, fill_value=0) / len(bins_train)
        ) * len(bins_train)
        test_expected = (
            test_dist.reindex(original_dist.index, fill_value=0) / len(bins_test)
        ) * len(bins_test)

        chi2_train, chi2_train_p = stats.chisquare(
            train_dist.reindex(original_dist.index, fill_value=0), train_expected
        )
        chi2_test, chi2_test_p = stats.chisquare(
            test_dist.reindex(original_dist.index, fill_value=0), test_expected
        )

        logger.info(
            f"Chi-square Test (Train bins):  statistic={chi2_train:.4f}, p-value={chi2_train_p:.4f}"
        )
        logger.info(
            f"Chi-square Test (Test bins):   statistic={chi2_test:.4f}, p-value={chi2_test_p:.4f}"
        )

        results["statistical_tests"] = {
            "ks_train": {"statistic": ks_train_stat, "p_value": ks_train_p},
            "ks_test": {"statistic": ks_test_stat, "p_value": ks_test_p},
            "chi2_train": {"statistic": chi2_train, "p_value": chi2_train_p},
            "chi2_test": {"statistic": chi2_test, "p_value": chi2_test_p},
        }
    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")
        results["statistical_tests"] = {
            "ks_train": {"statistic": ks_train_stat, "p_value": ks_train_p},
            "ks_test": {"statistic": ks_test_stat, "p_value": ks_test_p},
        }

    # Overall assessment
    logger.info("\n✅ STRATIFICATION ASSESSMENT:")
    logger.info("=" * 50)

    # Check if distributions are similar (p-value > 0.05 means similar)
    train_similar = ks_train_p > 0.05
    test_similar = ks_test_p > 0.05

    logger.info(
        f"Train set distribution similarity: {'✅ PASS' if train_similar else '❌ FAIL'} (p={ks_train_p:.4f})"
    )
    logger.info(
        f"Test set distribution similarity:  {'✅ PASS' if test_similar else '❌ FAIL'} (p={ks_test_p:.4f})"
    )

    # Check mean differences (should be < 5%)
    mean_train_diff = (
        abs(results["train_stats"]["mean"] - results["original_stats"]["mean"])
        / abs(results["original_stats"]["mean"])
        * 100
    )
    mean_test_diff = (
        abs(results["test_stats"]["mean"] - results["original_stats"]["mean"])
        / abs(results["original_stats"]["mean"])
        * 100
    )

    mean_train_ok = mean_train_diff < 5.0
    mean_test_ok = mean_test_diff < 5.0

    logger.info(
        f"Train set mean difference: {'✅ PASS' if mean_train_ok else '❌ FAIL'} ({mean_train_diff:.1f}% difference)"
    )
    logger.info(
        f"Test set mean difference:  {'✅ PASS' if mean_test_ok else '❌ FAIL'} ({mean_test_diff:.1f}% difference)"
    )

    overall_pass = train_similar and test_similar and mean_train_ok and mean_test_ok
    logger.info(
        f"\n🎯 OVERALL STRATIFICATION: {'✅ EXCELLENT' if overall_pass else '⚠️ NEEDS REVIEW'}"
    )

    results["assessment"] = {
        "train_distribution_similar": train_similar,
        "test_distribution_similar": test_similar,
        "train_mean_ok": mean_train_ok,
        "test_mean_ok": mean_test_ok,
        "overall_pass": overall_pass,
        "mean_train_diff_pct": mean_train_diff,
        "mean_test_diff_pct": mean_test_diff,
    }

    return results


def create_visualizations(results, output_dir):
    """Create visualizations for stratification validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📊 Creating stratification visualizations...")

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Stratified Sampling Validation", fontsize=16, fontweight="bold")

    # 1. Distribution comparison (histograms)
    ax1 = axes[0, 0]

    # Get the actual target values for plotting
    # We'll need to reconstruct this from the results
    # For now, let's create a conceptual plot

    ax1.set_title("Target Variable Distribution Comparison")
    ax1.set_xlabel("Coffee Rating")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Bin distribution comparison
    ax2 = axes[0, 1]

    bin_data = results["bin_distributions"]
    bins = sorted(bin_data["original"].index)

    orig_pcts = [
        (bin_data["original"][b] / bin_data["original"].sum()) * 100 for b in bins
    ]
    train_pcts = [
        (bin_data["train"].get(b, 0) / bin_data["train"].sum()) * 100 for b in bins
    ]
    test_pcts = [
        (bin_data["test"].get(b, 0) / bin_data["test"].sum()) * 100 for b in bins
    ]

    x = np.arange(len(bins))
    width = 0.25

    ax2.bar(x - width, orig_pcts, width, label="Original", alpha=0.8)
    ax2.bar(x, train_pcts, width, label="Train", alpha=0.8)
    ax2.bar(x + width, test_pcts, width, label="Test", alpha=0.8)

    ax2.set_xlabel("Stratification Bins")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Bin Distribution Consistency")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Bin {b}" for b in bins])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Statistical measures comparison
    ax3 = axes[1, 0]

    measures = ["mean", "std", "skewness", "kurtosis"]
    orig_values = [results["original_stats"][m] for m in measures]
    train_values = [results["train_stats"][m] for m in measures]
    test_values = [results["test_stats"][m] for m in measures]

    x = np.arange(len(measures))

    ax3.bar(x - width, orig_values, width, label="Original", alpha=0.8)
    ax3.bar(x, train_values, width, label="Train", alpha=0.8)
    ax3.bar(x + width, test_values, width, label="Test", alpha=0.8)

    ax3.set_xlabel("Statistical Measures")
    ax3.set_ylabel("Values")
    ax3.set_title("Statistical Measures Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(measures)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Assessment summary
    ax4 = axes[1, 1]

    assessment = results["assessment"]
    tests = ["Train Distribution", "Test Distribution", "Train Mean", "Test Mean"]
    results_bool = [
        assessment["train_distribution_similar"],
        assessment["test_distribution_similar"],
        assessment["train_mean_ok"],
        assessment["test_mean_ok"],
    ]

    colors = ["green" if r else "red" for r in results_bool]

    ax4.bar(tests, [1 if r else 0 for r in results_bool], color=colors, alpha=0.7)
    ax4.set_ylabel("Pass/Fail")
    ax4.set_title("Stratification Assessment")
    ax4.set_ylim(0, 1.2)
    ax4.set_xticklabels(tests, rotation=45)

    # Add pass/fail labels
    for i, (test, result) in enumerate(zip(tests, results_bool)):
        ax4.text(
            i,
            0.5,
            "✅ PASS" if result else "❌ FAIL",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "stratification_validation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def generate_report(results, output_path):
    """Generate a comprehensive validation report."""
    logger.info("📝 Generating stratification validation report...")

    report = []
    report.append("# Stratified Sampling Validation Report")
    report.append("=" * 50)
    report.append("")

    # Executive Summary
    assessment = results["assessment"]
    report.append("## Executive Summary")
    report.append("")
    report.append(
        f"**Overall Assessment**: {'✅ EXCELLENT' if assessment['overall_pass'] else '⚠️ NEEDS REVIEW'}"
    )
    report.append("")

    # Statistical Summary
    report.append("## Statistical Summary")
    report.append("")

    orig_stats = results["original_stats"]
    train_stats = results["train_stats"]
    test_stats = results["test_stats"]

    report.append("| Measure | Original | Train | Test | Train Diff | Test Diff |")
    report.append("|---------|----------|-------|------|------------|-----------|")

    for measure in ["mean", "std", "skewness", "kurtosis"]:
        orig = orig_stats[measure]
        train = train_stats[measure]
        test = test_stats[measure]

        train_diff = ((train - orig) / abs(orig) * 100) if orig != 0 else 0
        test_diff = ((test - orig) / abs(orig) * 100) if orig != 0 else 0

        report.append(
            f"| {measure.capitalize()} | {orig:.4f} | {train:.4f} | {test:.4f} | {train_diff:+.1f}% | {test_diff:+.1f}% |"
        )

    report.append("")

    # Bin Distribution Analysis
    report.append("## Bin Distribution Analysis")
    report.append("")

    bin_data = results["bin_distributions"]
    bins = sorted(bin_data["original"].index)

    report.append("| Bin | Original | Train | Test | Train Diff | Test Diff |")
    report.append("|-----|----------|-------|------|------------|-----------|")

    for bin_idx in bins:
        orig_pct = (bin_data["original"][bin_idx] / bin_data["original"].sum()) * 100
        train_pct = (bin_data["train"].get(bin_idx, 0) / bin_data["train"].sum()) * 100
        test_pct = (bin_data["test"].get(bin_idx, 0) / bin_data["test"].sum()) * 100

        train_diff = train_pct - orig_pct
        test_diff = test_pct - orig_pct

        report.append(
            f"| {bin_idx} | {orig_pct:.1f}% | {train_pct:.1f}% | {test_pct:.1f}% | {train_diff:+.1f}% | {test_diff:+.1f}% |"
        )

    report.append("")

    # Statistical Tests
    if "statistical_tests" in results:
        report.append("## Statistical Tests")
        report.append("")

        tests = results["statistical_tests"]

        report.append("### Kolmogorov-Smirnov Test (Distribution Similarity)")
        report.append(
            f"- **Train vs Original**: statistic={tests['ks_train']['statistic']:.4f}, p-value={tests['ks_train']['p_value']:.4f}"
        )
        report.append(
            f"- **Test vs Original**: statistic={tests['ks_test']['statistic']:.4f}, p-value={tests['ks_test']['p_value']:.4f}"
        )
        report.append("")

        if "chi2_train" in tests:
            report.append("### Chi-square Test (Bin Distribution)")
            report.append(
                f"- **Train bins**: statistic={tests['chi2_train']['statistic']:.4f}, p-value={tests['chi2_train']['p_value']:.4f}"
            )
            report.append(
                f"- **Test bins**: statistic={tests['chi2_test']['statistic']:.4f}, p-value={tests['chi2_test']['p_value']:.4f}"
            )
            report.append("")

    # Assessment Details
    report.append("## Assessment Details")
    report.append("")

    report.append("### Criteria and Results")
    report.append("")
    report.append(
        f"1. **Train Distribution Similarity**: {'✅ PASS' if assessment['train_distribution_similar'] else '❌ FAIL'}"
    )
    report.append(
        f"2. **Test Distribution Similarity**: {'✅ PASS' if assessment['test_distribution_similar'] else '❌ FAIL'}"
    )
    report.append(
        f"3. **Train Mean Difference < 5%**: {'✅ PASS' if assessment['train_mean_ok'] else '❌ FAIL'} ({assessment['mean_train_diff_pct']:.1f}%)"
    )
    report.append(
        f"4. **Test Mean Difference < 5%**: {'✅ PASS' if assessment['test_mean_ok'] else '❌ FAIL'} ({assessment['mean_test_diff_pct']:.1f}%)"
    )
    report.append("")

    # Thesis Alignment
    report.append("## Thesis Methodology Alignment")
    report.append("")
    report.append(
        "✅ **Stratified Sampling**: Using pd.cut() to create 5 bins for stratification"
    )
    report.append("✅ **70/30 Split**: Maintaining thesis-specified train/test ratio")
    report.append("✅ **Random Seed**: Using seed 57 for reproducibility")
    report.append(
        "✅ **Distribution Preservation**: Ensuring rating distribution consistency"
    )
    report.append("")

    # Conclusions
    report.append("## Conclusions")
    report.append("")

    if assessment["overall_pass"]:
        report.append(
            "🎯 **Stratified sampling is working correctly**. The implementation successfully maintains"
        )
        report.append(
            "the distribution of coffee ratings between train and test sets, following thesis methodology."
        )
    else:
        report.append(
            "⚠️ **Stratified sampling needs review**. Some distribution differences detected."
        )
        report.append(
            "Consider adjusting the number of bins or stratification approach."
        )

    report.append("")
    report.append("---")
    report.append("*Report generated by validate_stratified_sampling.py*")

    # Save report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    logger.info(f"Report saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate stratified sampling implementation"
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=None,
        help="Fraction of data to use for validation (e.g., 0.1 for 10%)",
    )

    args = parser.parse_args()

    try:
        # Load data
        df = load_data(args.sample_fraction)

        # Prepare features and target
        X, y = prepare_features_and_target(df)

        # Validate stratified sampling
        results = validate_stratified_sampling(X, y)

        # Create output directory
        output_dir = Path("output/stratified_sampling_validation")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        create_visualizations(results, output_dir)

        # Generate report
        generate_report(results, output_dir / "stratification_validation_report.md")

        # Save detailed results
        import pickle

        with open(output_dir / "validation_results.pkl", "wb") as f:
            pickle.dump(results, f)

        logger.info(f"✅ Stratified sampling validation completed!")
        logger.info(f"📊 Results saved to: {output_dir}")

        # Print final assessment
        assessment = results["assessment"]
        print(
            f"\n🎯 FINAL ASSESSMENT: {'✅ EXCELLENT' if assessment['overall_pass'] else '⚠️ NEEDS REVIEW'}"
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()

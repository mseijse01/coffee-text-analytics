"""
Visualization utilities for coffee review data analysis.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    logger.warning("Seaborn not installed. Some visualizations will be limited.")
    SEABORN_AVAILABLE = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    logger.warning(
        "WordCloud not installed. Word cloud visualizations will not be available."
    )
    WORDCLOUD_AVAILABLE = False


def save_figure(fig, filename, output_dir, dpi=300):
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure
        filename: Name for the file
        output_dir: Output directory
        dpi: Resolution (dots per inch)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add file extension if not provided
    if not filename.endswith((".png", ".jpg", ".pdf")):
        filename = f"{filename}.png"

    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved figure to {filepath}")


def plot_rating_distribution(df, rating_column="rating", title=None):
    """
    Visualize the distribution of coffee ratings.

    Args:
        df: DataFrame containing rating data
        rating_column: Name of rating column
        title: Custom title (default: Rating Distribution)

    Returns:
        Matplotlib figure
    """
    if rating_column not in df.columns:
        logger.warning(f"Column '{rating_column}' not found in DataFrame")
        return None

    try:
        # Set up figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with KDE
        if SEABORN_AVAILABLE:
            sns.histplot(df[rating_column].dropna(), kde=True, ax=ax)
        else:
            ax.hist(df[rating_column].dropna(), bins=20, alpha=0.7, density=True)

        # Add mean and median lines
        mean_val = df[rating_column].mean()
        median_val = df[rating_column].median()

        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        ax.axvline(
            median_val, color="green", linestyle="--", label=f"Median: {median_val:.2f}"
        )

        # Add labels and title
        ax.set_xlabel(rating_column.capitalize())
        ax.set_ylabel("Frequency")
        ax.set_title(title or f"{rating_column.capitalize()} Distribution")
        ax.legend()

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating rating distribution plot: {e}")
        return None


def plot_correlation_matrix(df, columns=None, title="Feature Correlation Matrix"):
    """
    Create a correlation matrix heatmap.

    Args:
        df: DataFrame with features
        columns: Specific columns to include (if None, uses all numeric columns)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if not SEABORN_AVAILABLE:
        logger.warning("Seaborn not available. Correlation matrix will be basic.")

    try:
        # Select numeric columns if not specified
        if columns is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            # Limit to avoid overcrowded plots
            if len(numeric_cols) > 15:
                logger.warning(
                    f"Too many numeric columns ({len(numeric_cols)}), limiting to top 15"
                )
                # Prioritize certain columns if they exist
                priority_cols = [
                    "rating",
                    "price_standardized",
                    "sentiment_score",
                    "dominant_lda_topic",
                ]
                selected_cols = [col for col in priority_cols if col in numeric_cols]
                remaining_slots = 15 - len(selected_cols)
                if remaining_slots > 0:
                    other_cols = [
                        col for col in numeric_cols if col not in selected_cols
                    ][:remaining_slots]
                    columns = selected_cols + other_cols
                else:
                    columns = selected_cols[:15]
            else:
                columns = numeric_cols

        # Filter to only include columns that exist
        columns = [col for col in columns if col in df.columns]

        if len(columns) < 2:
            logger.warning("Not enough columns for correlation matrix")
            return None

        # Calculate correlation
        corr = df[columns].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        if SEABORN_AVAILABLE:
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Generate heatmap with better styling
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )
        else:
            # Basic matplotlib version
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)

            # Add text annotations
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                    )

            ax.set_xticks(range(len(corr)))
            ax.set_yticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

        ax.set_title(title)
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return None


def plot_word_clouds(model_file, vectorizer_file, output_dir, n_topics=10):
    """
    Generate word clouds for topic models.

    Args:
        model_file: Path to topic model file
        vectorizer_file: Path to vectorizer file
        output_dir: Directory to save outputs
        n_topics: Number of topics to visualize
    """
    if not WORDCLOUD_AVAILABLE:
        logger.warning("WordCloud not installed. Skipping word cloud generation.")
        return

    try:
        # Load model and vectorizer
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        with open(vectorizer_file, "rb") as f:
            vectorizer = pickle.load(f)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Create word clouds for each topic
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(model_file).split(".")[0]

        for topic_idx, topic in enumerate(model.components_[:n_topics]):
            # Get top words and their weights
            top_indices = topic.argsort()[:-21:-1]  # Top 20 words
            top_words = {feature_names[i]: topic[i] for i in top_indices}

            # Generate word cloud
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", random_state=42
            ).generate_from_frequencies(top_words)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_title(f"{model_name.upper()} - Topic {topic_idx + 1}")
            ax.axis("off")

            # Save figure
            save_figure(
                fig, f"{model_name}_topic_{topic_idx + 1}_wordcloud", output_dir
            )
            plt.close(fig)

        logger.info(f"Generated {n_topics} word clouds for {model_name}")
    except Exception as e:
        logger.error(f"Error generating word clouds: {e}")


def plot_model_comparison(results_file, output_dir):
    """
    Create visualizations comparing model performance.

    Args:
        results_file: Path to model results JSON file
        output_dir: Directory to save outputs
    """
    try:
        # Load results
        with open(results_file, "r") as f:
            results = json.load(f)

        if not results:
            logger.warning("No model results found")
            return

        # Extract metrics
        models = list(results.keys())
        metrics = {
            "rmse": [results[model].get("rmse", 0) for model in models],
            "mae": [results[model].get("mae", 0) for model in models],
            "r2": [results[model].get("r2", 0) for model in models],
        }

        # Create comparison plots
        for metric_name, metric_values in metrics.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create bar chart
            bars = ax.bar(models, metric_values, color="skyblue")

            # Add value labels
            for bar, val in zip(bars, metric_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                )

            # Set labels and title
            ax.set_xlabel("Model")
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f"Model Comparison - {metric_name.upper()}")

            # Adjust y-axis range for R²
            if metric_name == "r2":
                ax.set_ylim([0, max(1.0, max(metric_values) * 1.1)])

            # Style improvements
            if SEABORN_AVAILABLE:
                sns.despine()

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save figure
            save_figure(fig, f"model_comparison_{metric_name}", output_dir)
            plt.close(fig)

        logger.info("Created model comparison visualizations")
    except Exception as e:
        logger.error(f"Error creating model comparison: {e}")


def plot_feature_correlation_to_rating(
    df, target_column="rating", n_features=10, output_dir=None
):
    """
    Visualize features with strongest correlation to rating.

    Args:
        df: DataFrame with features
        target_column: Target column name
        n_features: Number of top features to show
        output_dir: Directory to save outputs

    Returns:
        Matplotlib figure
    """
    if target_column not in df.columns:
        logger.warning(f"Target column '{target_column}' not found in DataFrame")
        return None

    try:
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=["number"]).columns

        # Calculate correlation with target
        correlations = (
            df[numeric_cols].corrwith(df[target_column]).sort_values(ascending=False)
        )

        # Get top positive and negative correlations
        top_pos = correlations[correlations.index != target_column].head(n_features)
        top_neg = (
            correlations[correlations.index != target_column]
            .tail(n_features)
            .iloc[::-1]
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Combine and sort for visualization
        combined = pd.concat([top_pos, top_neg])
        colors = ["green" if c >= 0 else "red" for c in combined.values]

        # Create bar chart
        bars = ax.barh(combined.index, combined.values, color=colors)

        # Add value labels
        for bar, val in zip(bars, combined.values):
            ax.text(
                val + 0.01 if val >= 0 else val - 0.06,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                color="black" if val >= 0 else "white",
            )

        # Set labels and title
        ax.set_xlabel(f"Correlation with {target_column.capitalize()}")
        ax.set_title(f"Features Most Correlated with {target_column.capitalize()}")

        # Add a vertical line at x=0
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)

        # Style improvements
        if SEABORN_AVAILABLE:
            sns.despine()

        plt.tight_layout()

        # Save figure if output_dir provided
        if output_dir:
            save_figure(fig, f"feature_correlation_to_{target_column}", output_dir)

        return fig
    except Exception as e:
        logger.error(f"Error creating feature correlation plot: {e}")
        return None


def create_visualizations(
    features_file, models_dir="models", output_dir="output/figures"
):
    """
    Create visualizations from features and model results.

    Args:
        features_file: Path to features CSV file
        models_dir: Directory containing models
        output_dir: Directory to save visualizations
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load features data
        logger.info(f"Loading features from {features_file}")
        df = pd.read_csv(features_file)

        if df.empty:
            raise ValueError(f"No data found in {features_file}")

        # 1. Rating distribution
        if "rating" in df.columns:
            logger.info("Creating rating distribution plot")
            fig = plot_rating_distribution(df, "rating")
            if fig:
                save_figure(fig, "rating_distribution", output_dir)
                plt.close(fig)

        # 2. Feature correlation matrix
        logger.info("Creating feature correlation matrix")
        fig = plot_correlation_matrix(df)
        if fig:
            save_figure(fig, "feature_correlation_matrix", output_dir)
            plt.close(fig)

        # 3. Feature correlation to rating
        if "rating" in df.columns:
            logger.info("Creating feature correlation to rating plot")
            fig = plot_feature_correlation_to_rating(
                df, "rating", output_dir=output_dir
            )
            if fig:
                plt.close(fig)

        # 4. Word clouds for topic models if available
        vectorizer_file = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(vectorizer_file):
            # Check for LDA model
            lda_file = os.path.join(models_dir, "lda_model.pkl")
            if os.path.exists(lda_file):
                logger.info("Creating LDA topic word clouds")
                plot_word_clouds(lda_file, vectorizer_file, output_dir)

            # Check for NMF model
            nmf_file = os.path.join(models_dir, "nmf_model.pkl")
            if os.path.exists(nmf_file):
                logger.info("Creating NMF topic word clouds")
                plot_word_clouds(nmf_file, vectorizer_file, output_dir)

        # 5. Model comparison if results exist
        results_file = os.path.join("output", "model_results.json")
        if os.path.exists(results_file):
            logger.info("Creating model comparison visualizations")
            plot_model_comparison(results_file, output_dir)

        logger.info(f"All visualizations saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise

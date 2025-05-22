"""Visualization functions for coffee review analysis."""

from pathlib import Path
from typing import Union, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from src.config.settings import PATHS


def save_figure(
    fig: go.Figure,
    filename: str,
    path: Path = PATHS["figures"],
) -> None:
    """
    Save a plotly figure to the figures directory.

    Args:
        fig: Plotly figure object
        filename: Name for the saved figure
        path: Directory to save figure
    """
    fig.write_html(path / f"{filename}.html")
    fig.write_image(path / f"{filename}.png")


def plot_boxplots(data: pl.DataFrame, columns: List[str]) -> None:
    """Generate box plots for specified columns."""
    for col in columns:
        fig = px.box(
            data.to_pandas(),
            x=col,
            title=f"Distribution of {col}",
            height=300,
            width=600,
            orientation="h",
        )
        fig.show()


def plot_kde(data: pl.DataFrame, columns: List[str]) -> None:
    """Generate KDE plots for specified columns."""
    for col in columns:
        fig = px.histogram(
            data.to_pandas(),
            x=col,
            title=f"Distribution of {col}",
            height=300,
            width=600,
            marginal="kde",
        )
        fig.show()


def plot_categorical_distributions(data: pl.DataFrame, columns: List[str]) -> None:
    """Plot distributions for categorical columns."""
    for col in columns:
        value_counts = data.select(pl.col(col)).to_pandas()[col].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {col}",
            labels={"x": col, "y": "Count"},
        )
        fig.show()

"""
Coffee Text Analytics - Configuration Management

This module provides comprehensive configuration management for the coffee text analytics project.
It supports environment-specific settings, model parameters, data paths, and visualization settings.

Features:
- Environment-specific configurations (development, production, testing)
- Centralized path management
- Model hyperparameters
- Feature extraction parameters
- Visualization settings
- Logging configuration
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class PathConfig:
    """Configuration for project paths."""

    # Base directories
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data: Path = field(init=False)
    raw: Path = field(init=False)
    processed: Path = field(init=False)
    models: Path = field(init=False)
    output: Path = field(init=False)
    figures: Path = field(init=False)
    tests: Path = field(init=False)
    docs: Path = field(init=False)

    # Data files
    raw_data_file: str = "coffee_clean.csv"
    processed_data_file: str = "coffee_processed.csv"
    features_data_file: str = "coffee_features.csv"

    def __post_init__(self):
        """Initialize derived paths."""
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        self.models = self.root / "models"
        self.output = self.root / "output"
        self.figures = self.output / "figures"
        self.tests = self.root / "tests"
        self.docs = self.root / "docs"

    def create_directories(self) -> None:
        """Create all project directories if they don't exist."""
        directories = [
            self.data,
            self.raw,
            self.processed,
            self.models,
            self.output,
            self.figures,
            self.tests,
            self.docs,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_raw_data_path(self) -> Path:
        """Get the full path to the raw data file."""
        return self.raw / self.raw_data_file

    def get_processed_data_path(self) -> Path:
        """Get the full path to the processed data file."""
        return self.processed / self.processed_data_file

    def get_features_data_path(self) -> Path:
        """Get the full path to the features data file."""
        return self.processed / self.features_data_file


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""

    # Target variable
    target_column: str = "rating"

    # Text columns for analysis
    text_columns: List[str] = field(
        default_factory=lambda: ["desc_1", "desc_2", "desc_3"]
    )

    # Sensory attribute columns
    sensory_columns: List[str] = field(
        default_factory=lambda: ["aroma", "acid", "body", "flavor", "aftertaste"]
    )

    # Models to train (aligned with thesis methodology)
    models_to_train: List[str] = field(
        default_factory=lambda: [
            "linear",
            "ridge",
            "lasso",
            "random_forest",
            "xgboost",
            "svr",
            "decision_tree",
            "mnir",
        ]
    )

    # Model hyperparameters
    random_forest_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 57,
        }
    )

    xgboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 57,
        }
    )

    linear_params: Dict[str, Any] = field(default_factory=lambda: {"random_state": 57})

    # Ridge regression parameters
    ridge_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 1.0,
            "alpha_grid": [0.1, 1.0, 10.0, 100.0],
            "cv": 5,
            "random_state": 57,
        }
    )

    # Lasso regression parameters (thesis methodology)
    lasso_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 1.0,
            "alpha_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "cv": 5,
            "random_state": 57,
        }
    )

    # SVR specific parameters (following thesis methodology)
    svr_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "epsilon": 0.1,
            "random_state": 57,
        }
    )

    # Decision Tree specific parameters
    decision_tree_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 57,
        }
    )

    # MNIR specific parameters
    mnir_params: Dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.1, "max_iter": 1000, "random_state": 57}
    )

    # Cross-validation settings (thesis uses 5-fold CV)
    cv_folds: int = 5
    test_size: float = 0.3  # Thesis uses 70/30 split
    random_state: int = 57

    # Lasso-specific parameters for feature selection (thesis methodology)
    lasso_alpha_range: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    )
    lasso_cv_folds: int = 5

    # Feature selection settings
    feature_selection_enabled: bool = True
    feature_selection_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "cv_folds": 5,
            "max_features_per_group": 200,
            "min_features_per_group": 10,
            "selection_threshold": "mean",
            "random_state": 57,
            "scale_features": True,
        }
    )

    # Box-Cox transformation settings (thesis methodology)
    box_cox_enabled: bool = False  # Default: no transformation (thesis conclusion)
    box_cox_dual_pipeline: bool = (
        False  # Run both with and without Box-Cox for comparison
    )
    box_cox_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "lambda_range": (-2, 2),  # Range for lambda parameter search
            "method": "mle",  # Maximum likelihood estimation
            "alpha": 0.05,  # Significance level for normality tests
            "save_comparison": True,  # Save comparison results
            "random_state": 57,
        }
    )

    # Two-Step Hyperparameter Tuning (Signature Approach)
    # Phase 1: Randomized Search → Phase 2: Grid Search
    two_step_tuning_enabled: bool = True  # Enable signature two-step approach

    # Randomized Search parameters (Phase 1: Wide exploration)
    randomized_search_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_iter": 50,  # Number of parameter settings sampled
            "cv": 3,  # Use 3-fold CV for speed in randomized search
            "scoring": "r2",
            "n_jobs": -1,
            "random_state": 57,
            "verbose": 1,
        }
    )

    # Grid Search parameters (Phase 2: Fine-tuning)
    grid_search_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "cv": 5,  # Use 5-fold CV for final tuning
            "scoring": "r2",
            "n_jobs": -1,
            "verbose": 1,
        }
    )

    # Two-step parameter grids for each model
    random_forest_two_step: Dict[str, Any] = field(
        default_factory=lambda: {
            "randomized_params": {
                "n_estimators": [50, 100, 200, 300, 500],
                "max_depth": [None, 5, 10, 15, 20, 25],
                "min_samples_split": [2, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 4, 6, 8],
                "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
                "bootstrap": [True, False],
            },
            "grid_refinement_factor": 3,  # How many values around best to test
        }
    )

    xgboost_two_step: Dict[str, Any] = field(
        default_factory=lambda: {
            "randomized_params": {
                "n_estimators": [50, 100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6, 7, 8, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1, 10],
                "reg_lambda": [0, 0.01, 0.1, 1, 10],
            },
            "grid_refinement_factor": 3,
        }
    )

    svr_two_step: Dict[str, Any] = field(
        default_factory=lambda: {
            "randomized_params": {
                "kernel": ["rbf", "linear", "poly"],
                "C": [0.01, 0.1, 1, 10, 100, 1000],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "epsilon": [0.001, 0.01, 0.1, 0.2, 0.5],
                "degree": [2, 3, 4],  # For poly kernel
            },
            "grid_refinement_factor": 3,
        }
    )


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # TF-IDF parameters
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 3)  # unigrams, bigrams, trigrams
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95

    # Topic modeling parameters
    n_topics: int = 10
    lda_random_state: int = 57
    nmf_random_state: int = 57

    # BERT parameters
    bert_model_name: str = "distilbert-base-uncased"
    bert_max_length: int = 512
    bert_batch_size: int = 16

    # GloVe parameters
    glove_dimensions: int = 300
    glove_model_name: str = "glove-wiki-gigaword-300"

    # Sentiment analysis
    sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"

    # Text preprocessing
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_word_length: int = 2
    max_word_length: int = 20


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Data quality thresholds
    min_rating: float = 80.0
    max_missing_percentage: float = 50.0

    # Text processing
    min_text_length: int = 10
    max_text_length: int = 1000

    # Country standardization
    standardize_countries: bool = True

    # Price standardization
    standardize_prices: bool = True
    target_currency: str = "USD"
    target_unit: str = "kg"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations and plots."""

    # Plot settings
    figure_width: int = 800
    figure_height: int = 600
    template: str = "plotly_white"
    color_palette: List[str] = field(
        default_factory=lambda: [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    # Font settings
    font_family: str = "Arial"
    font_size: int = 12
    title_font_size: int = 16

    # Export settings
    export_format: str = "png"
    export_dpi: int = 300
    export_width: int = 1200
    export_height: int = 800


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "coffee_analytics.log"


class Config:
    """Main configuration class that combines all configuration components."""

    def __init__(self, environment: str = None):
        """
        Initialize configuration based on environment.

        Args:
            environment: Environment name (development, production, testing)
        """
        self.environment = environment or os.getenv("COFFEE_ENV", "development")

        # Initialize configuration components
        self.paths = PathConfig()
        self.models = ModelConfig()
        self.features = FeatureConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()

        # Apply environment-specific overrides
        self._apply_environment_config()

        # Create directories
        self.paths.create_directories()

        # Configure logging
        self._configure_logging()

        # Configure plotting
        self._configure_plotting()

    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == "testing":
            # Testing environment overrides
            self.data.min_rating = 70.0  # Lower threshold for testing
            self.features.n_topics = 5  # Fewer topics for faster testing
            self.models.cv_folds = 3  # Fewer folds for faster testing
            self.logging.level = "DEBUG"

        elif self.environment == "production":
            # Production environment overrides
            self.data.min_rating = 85.0  # Higher quality threshold
            self.features.n_topics = 15  # More topics for better analysis
            self.models.cv_folds = 10  # More robust validation
            self.logging.level = "WARNING"
            self.logging.file_handler = True

        elif self.environment == "development":
            # Development environment (default settings)
            self.logging.level = "INFO"
            self.logging.console_handler = True

    def _configure_logging(self):
        """Configure logging based on settings."""
        # Clear existing handlers
        logging.getLogger().handlers.clear()

        # Set logging level
        level = getattr(logging, self.logging.level.upper())
        logging.getLogger().setLevel(level)

        # Create formatter
        formatter = logging.Formatter(self.logging.format)

        # Add console handler if enabled
        if self.logging.console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)

        # Add file handler if enabled
        if self.logging.file_handler:
            log_path = self.paths.root / self.logging.log_file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)

    def _configure_plotting(self):
        """Configure default plotting settings."""
        # Create custom template
        template = go.layout.Template(
            layout=dict(
                font=dict(
                    family=self.visualization.font_family,
                    size=self.visualization.font_size,
                ),
                title=dict(
                    x=0.5,
                    xanchor="center",
                    font=dict(size=self.visualization.title_font_size),
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                colorway=self.visualization.color_palette,
                width=self.visualization.figure_width,
                height=self.visualization.figure_height,
                xaxis=dict(gridcolor="lightgray", showgrid=True, zeroline=False),
                yaxis=dict(gridcolor="lightgray", showgrid=True, zeroline=False),
            )
        )

        # Set as default template
        pio.templates["coffee_analytics"] = template
        pio.templates.default = "coffee_analytics"

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of hyperparameters
        """
        param_mapping = {
            "random_forest": self.models.random_forest_params,
            "xgboost": self.models.xgboost_params,
            "linear": self.models.linear_params,
            "svr": self.models.svr_params,
            "decision_tree": self.models.decision_tree_params,
            "mnir": self.models.mnir_params,
        }

        return param_mapping.get(model_name, {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "environment": self.environment,
            "paths": {
                "root": str(self.paths.root),
                "data": str(self.paths.data),
                "raw": str(self.paths.raw),
                "processed": str(self.paths.processed),
                "models": str(self.paths.models),
                "output": str(self.paths.output),
                "figures": str(self.paths.figures),
            },
            "models": {
                "target_column": self.models.target_column,
                "text_columns": self.models.text_columns,
                "models_to_train": self.models.models_to_train,
                "cv_folds": self.models.cv_folds,
                "test_size": self.models.test_size,
            },
            "features": {
                "tfidf_max_features": self.features.tfidf_max_features,
                "n_topics": self.features.n_topics,
                "bert_model_name": self.features.bert_model_name,
            },
            "data": {
                "min_rating": self.data.min_rating,
                "standardize_countries": self.data.standardize_countries,
                "standardize_prices": self.data.standardize_prices,
            },
        }


# Global configuration instance
config = Config()

# Backward compatibility exports
PATHS = {
    "root": config.paths.root,
    "data": config.paths.data,
    "raw": config.paths.raw,
    "processed": config.paths.processed,
    "models": config.paths.models,
    "output": config.paths.output,
    "figures": config.paths.figures,
}

# Export main configuration components for easy access
__all__ = [
    "Config",
    "PathConfig",
    "ModelConfig",
    "FeatureConfig",
    "DataConfig",
    "VisualizationConfig",
    "LoggingConfig",
    "config",
    "PATHS",
]

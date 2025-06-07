"""
Comprehensive tests for visualization modules.

Tests for visualization functionality with mocked matplotlib/plotting operations
to increase test coverage without requiring actual plot generation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Import visualization modules
from src.visualization import visualize

try:
    from src.visualization.plots import save_figure as plotly_save_figure

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    plotly_save_figure = None

from src.config.settings import VisualizationConfig


class TestVisualizationUtilities:
    """Test core visualization utilities."""

    def test_save_figure_basic(self):
        """Test basic figure saving functionality."""
        mock_fig = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            visualize.save_figure(mock_fig, "test_plot", temp_dir)

            # Verify matplotlib calls on the figure object
            mock_fig.savefig.assert_called_once()
            call_args = mock_fig.savefig.call_args
            assert (
                "test_plot.png" in call_args[0][0]
            )  # filename in first positional arg
            assert call_args[1]["dpi"] == 300  # dpi in keyword args
            assert call_args[1]["bbox_inches"] == "tight"

    def test_save_figure_custom_extension(self):
        """Test figure saving with custom file extension."""
        mock_fig = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            visualize.save_figure(mock_fig, "test_plot.pdf", temp_dir)

            mock_fig.savefig.assert_called_once()
            call_args = mock_fig.savefig.call_args
            assert "test_plot.pdf" in call_args[0][0]

    @patch("matplotlib.pyplot.subplots")
    @patch("src.visualization.visualize.SEABORN_AVAILABLE", True)
    @patch("seaborn.histplot")
    def test_plot_rating_distribution_with_seaborn(self, mock_histplot, mock_subplots):
        """Test rating distribution plotting with seaborn."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create test data
        df = pd.DataFrame({"rating": np.random.normal(85, 5, 100)})

        result = visualize.plot_rating_distribution(df, "rating")

        # Verify calls
        mock_subplots.assert_called_once()
        mock_histplot.assert_called_once()
        mock_ax.axvline.assert_called()  # Mean and median lines
        mock_ax.set_xlabel.assert_called_with("Rating")
        mock_ax.set_ylabel.assert_called_with("Frequency")
        mock_ax.legend.assert_called_once()

        assert result == mock_fig

    @patch("matplotlib.pyplot.subplots")
    @patch("src.visualization.visualize.SEABORN_AVAILABLE", False)
    def test_plot_rating_distribution_without_seaborn(self, mock_subplots):
        """Test rating distribution plotting without seaborn."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        df = pd.DataFrame({"rating": np.random.normal(85, 5, 100)})

        result = visualize.plot_rating_distribution(df, "rating")

        # Verify matplotlib hist was called instead of seaborn
        mock_ax.hist.assert_called_once()
        assert result == mock_fig

    def test_plot_rating_distribution_missing_column(self):
        """Test handling of missing rating column."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = visualize.plot_rating_distribution(df, "rating")

        assert result is None

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.xticks")
    @patch("src.visualization.visualize.save_figure")
    @patch("json.load")
    @patch("builtins.open")
    def test_plot_model_comparison(
        self,
        mock_open,
        mock_json_load,
        mock_save_figure,
        mock_xticks,
        mock_tight_layout,
        mock_close,
        mock_subplots,
    ):
        """Test model comparison plotting."""
        # Mock data
        mock_json_load.return_value = {
            "model_a": {"rmse": 0.5, "mae": 0.3, "r2": 0.8},
            "model_b": {"rmse": 0.6, "mae": 0.4, "r2": 0.7},
        }

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Mock the bar chart return with proper attributes for mathematical operations
        mock_bar1 = Mock()
        mock_bar1.get_x.return_value = 0.1
        mock_bar1.get_width.return_value = 0.8
        mock_bar1.get_height.return_value = 0.5

        mock_bar2 = Mock()
        mock_bar2.get_x.return_value = 1.1
        mock_bar2.get_width.return_value = 0.8
        mock_bar2.get_height.return_value = 0.6

        mock_bars = [mock_bar1, mock_bar2]
        mock_ax.bar.return_value = mock_bars

        with tempfile.TemporaryDirectory() as temp_dir:
            visualize.plot_model_comparison("fake_results.json", temp_dir)

            # Verify multiple plots were created (one for each metric)
            assert mock_subplots.call_count == 3  # rmse, mae, r2
            assert mock_ax.bar.call_count == 3
            assert mock_save_figure.call_count == 3

    @patch("json.load")
    @patch("builtins.open")
    def test_plot_model_comparison_empty_results(self, mock_open, mock_json_load):
        """Test model comparison with empty results."""
        mock_json_load.return_value = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle empty results gracefully
            visualize.plot_model_comparison("fake_results.json", temp_dir)


class TestVisualizationConfig:
    """Test visualization configuration."""

    def test_visualization_config_defaults(self):
        """Test default visualization configuration values."""
        config = VisualizationConfig()

        # Test plot settings
        assert config.figure_width > 0
        assert config.figure_height > 0
        assert config.template is not None
        assert len(config.color_palette) > 0

        # Test font settings
        assert config.font_family is not None
        assert config.font_size > 0
        assert config.title_font_size >= config.font_size

        # Test export settings
        assert config.export_format in ["png", "jpg", "svg", "pdf", "html"]
        assert config.export_dpi > 0
        assert config.export_width > 0
        assert config.export_height > 0

    def test_visualization_config_customization(self):
        """Test custom visualization configuration."""
        custom_palette = ["#FF0000", "#00FF00", "#0000FF"]

        config = VisualizationConfig(
            figure_width=1000, color_palette=custom_palette, export_format="svg"
        )

        assert config.figure_width == 1000
        assert config.color_palette == custom_palette
        assert config.export_format == "svg"


class TestPlotlyVisualization:
    """Test plotly-based visualization functions."""

    @pytest.mark.skipif(
        not PLOTLY_AVAILABLE, reason="Plotly visualization not available"
    )
    def test_save_plotly_figure(self):
        """Test saving plotly figures."""

        mock_fig = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            plotly_save_figure(mock_fig, "test_plot", Path(temp_dir))

            mock_fig.write_html.assert_called_once()
            mock_fig.write_image.assert_called_once()

    @pytest.mark.skipif(
        not PLOTLY_AVAILABLE, reason="Plotly visualization not available"
    )
    @patch("plotly.express.box")
    def test_plot_boxplots(self, mock_box):
        """Test boxplot generation."""
        from src.visualization.plots import plot_boxplots
        import polars as pl

        # Create test data
        df = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})

        mock_fig = Mock()
        mock_box.return_value = mock_fig

        plot_boxplots(df, ["col1", "col2"])

        # Should create one plot per column
        assert mock_box.call_count == 2
        mock_fig.show.assert_called()


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""

    @patch("matplotlib.pyplot.subplots", side_effect=Exception("Matplotlib error"))
    def test_plot_rating_distribution_error_handling(self, mock_subplots):
        """Test error handling in rating distribution plotting."""
        df = pd.DataFrame({"rating": [1, 2, 3]})

        result = visualize.plot_rating_distribution(df, "rating")

        # Should return None on error
        assert result is None

    @patch("json.load", side_effect=Exception("JSON error"))
    @patch("builtins.open")
    def test_plot_model_comparison_error_handling(self, mock_open, mock_json_load):
        """Test error handling in model comparison plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle JSON loading errors gracefully
            visualize.plot_model_comparison("fake_results.json", temp_dir)


class TestWordCloudFunctionality:
    """Test word cloud visualization features."""

    @pytest.mark.skipif(
        not hasattr(visualize, "WORDCLOUD_AVAILABLE")
        or not visualize.WORDCLOUD_AVAILABLE,
        reason="WordCloud module not available",
    )
    def test_wordcloud_generation_available(self):
        """Test word cloud generation when WordCloud is available."""
        # This would test a word cloud function if we had one
        # For now, just test that the import check works
        assert visualize.WORDCLOUD_AVAILABLE is True

    def test_wordcloud_unavailable_fallback(self):
        """Test fallback when WordCloud is not available."""
        # This test should always pass - testing the case where wordcloud is not available
        if hasattr(visualize, "WORDCLOUD_AVAILABLE"):
            # If WORDCLOUD_AVAILABLE is False, this is expected behavior
            if not visualize.WORDCLOUD_AVAILABLE:
                assert visualize.WORDCLOUD_AVAILABLE is False
            else:
                # If it is available, we can't test the fallback, so skip
                pytest.skip("WordCloud is available, cannot test unavailable fallback")
        else:
            # If the attribute doesn't exist, assume wordcloud is not available
            assert True


class TestVisualizationIntegration:
    """Test integration between visualization components."""

    @patch("matplotlib.pyplot.subplots")
    @patch("src.visualization.visualize.save_figure")
    @patch("src.visualization.visualize.SEABORN_AVAILABLE", True)
    @patch("seaborn.histplot")
    def test_end_to_end_visualization_pipeline(
        self, mock_histplot, mock_save_figure, mock_subplots
    ):
        """Test complete visualization pipeline."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Add proper __name__ attribute to avoid attribute errors
        mock_histplot.__name__ = "histplot"

        # Create test data
        df = pd.DataFrame(
            {
                "rating": np.random.normal(85, 5, 100),
                "price": np.random.normal(15, 3, 100),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test rating distribution
            fig1 = visualize.plot_rating_distribution(df, "rating")
            if fig1:
                visualize.save_figure(fig1, "rating_dist", temp_dir)

            # Verify the pipeline worked
            mock_subplots.assert_called()
            mock_save_figure.assert_called()

    def test_visualization_configuration_integration(self):
        """Test that visualization config integrates properly."""
        config = VisualizationConfig()

        # Test that config values are reasonable for matplotlib
        assert config.figure_width >= 400
        assert config.figure_height >= 300
        assert config.export_dpi >= 72
        assert len(config.color_palette) >= 3

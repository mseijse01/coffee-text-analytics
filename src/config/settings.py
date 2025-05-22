"""Project configuration settings."""

from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio


# Project paths
def setup_project_paths():
    """Configure project paths."""
    project_root = Path().absolute().parent

    paths = {
        "root": project_root,
        "data": project_root / "data",
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "results": project_root / "results",
        "figures": project_root / "results" / "figures",
    }

    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


PATHS = setup_project_paths()

# Plotting configuration
PLOT_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="Arial", size=12),
        title=dict(x=0.5, xanchor="center"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="lightgray", showgrid=True, zeroline=False),
    )
)


def configure_plotting():
    """Configure default plotting settings."""
    pio.templates.default = PLOT_TEMPLATE
    pio.templates["custom"] = PLOT_TEMPLATE

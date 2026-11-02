"""Shared styling for publication figures."""

import re
from typing import List, Union

import plotly.graph_objects as go

THEME_ACCENT = "#FF4569"
THEME_LIGHT = "#D9D9D9"
THEME_DARK = "#2E2E2E"

MODELS = ["gnn", "kpconv", "cosmo", "egnn", "se3_transformer"]
TOY_PERFORMANCE_MODELS = ["gnn", "kpconv", "egnn", "se3_transformer", "cosmo"]
MODEL_LABELS = {
    "gnn": "GNN",
    "kpconv": "KPConv",
    "cosmo": "Cosmo",
    "cosmo_ig": "Cosmo IG",
    "egnn": "EGNN",
    "se3_transformer": "SE3T",
}

# Shared BioLiP table rows and attribution-figure columns: (arch, method, label).
BIOLIP_MODEL_COLUMNS = [
    ("cosmo", "auto", "Cosmo"),
    ("cosmo", "ig", "Cosmo IG"),
    ("kpconv", "auto", "KPConv"),
    ("gnn", "auto", "GNN"),
    ("egnn", "auto", "EGNN"),
    ("se3_transformer", "auto", "SE3T"),
]

_MODEL_ALT = "|".join(re.escape(name) for name in MODELS)
TOY_RUN_PATTERN = re.compile(
    rf"^toy_(?P<model>{_MODEL_ALT})_(?P<ship>[^_]+)_(?P<seed>\d+)$"
)
TOY3D_RUN_PATTERN = re.compile(
    rf"^toy3d_(?P<model>{_MODEL_ALT})_(?P<ship>[^_]+)_(?P<seed>\d+)$"
)
ENZYME_RUN_PATTERN = re.compile(
    rf"^enzyme_(?P<model>{_MODEL_ALT})_(?P<seed>\d+)$"
)
BIOLIP_RUN_PATTERN = re.compile(
    rf"^biolip_(?P<model>{_MODEL_ALT})_(?P<seed>\d+)$"
)
QM9_RUN_PATTERN = re.compile(
    rf"^qm9_(?P<model>{_MODEL_ALT})_(?P<property>[A-Za-z0-9]+)_(?P<seed>\d+)$"
)

# Five model colors derived from the theme palette (light → accent → dark).
MODEL_COLORS = {
    "gnn": THEME_LIGHT,
    "kpconv": "#B8B8B8",
    "cosmo": THEME_ACCENT,
    "cosmo_ig": "#FF8AA0",
    "egnn": "#6B6B6B",
    "se3_transformer": THEME_DARK,
}

# BioLiP bars follow KPConv then GNN, so those two greys are swapped.
BIOLIP_MODEL_COLORS = {
    **MODEL_COLORS,
    "kpconv": MODEL_COLORS["gnn"],
    "gnn": MODEL_COLORS["kpconv"],
}

BINARY_COLORSCALE: List[List[Union[float, str]]] = [
    [0.0, THEME_LIGHT],
    [1.0, THEME_ACCENT],
]

# Zero at grey; ±max|s| at black / pink.
ATTRIBUTION_COLORSCALE: List[List[Union[float, str]]] = [
    [0.0, THEME_DARK],
    [0.5, THEME_LIGHT],
    [1.0, THEME_ACCENT],
]

GREYSCALE_COLORSCALE: List[List[Union[float, str]]] = [
    [0.0, THEME_LIGHT],
    [1.0, THEME_DARK],
]

LEGEND_LAYOUT = dict(
    orientation="h",
    yanchor="bottom",
    y=1.04,
    x=0.5,
    xanchor="center",
    bgcolor="rgba(255,255,255,0)",
)

FONT_FAMILY = "DejaVu Sans Mono, Menlo, Monaco, Consolas, monospace"
FONT_SIZE = 12
FONT = dict(family=FONT_FAMILY, size=FONT_SIZE)
ANNOTATION_FONT = dict(family=FONT_FAMILY, size=FONT_SIZE)

DEFAULT_MARGINS = dict(t=90, b=40, l=60, r=30)
BAR_FRAME_COLOR = "#333"


def apply_bar_frame(fig: go.Figure, *, row=None, col=None) -> None:
    """Draw a ``#333`` box around bar-chart axes."""
    kwargs = dict(showline=True, mirror=True, linewidth=1, linecolor=BAR_FRAME_COLOR)
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col
    fig.update_xaxes(**kwargs)
    fig.update_yaxes(**kwargs)


def apply_layout(fig: go.Figure, *, height: int, width: int, show_legend: bool = True, **extra) -> go.Figure:
    layout = dict(
        template="plotly_white",
        height=height,
        width=width,
        font=FONT.copy(),
        margin=DEFAULT_MARGINS.copy(),
        showlegend=show_legend,
    )
    if show_legend:
        layout["legend"] = LEGEND_LAYOUT.copy()
    layout.update(extra)
    fig.update_layout(**layout)
    fig.update_annotations(font=ANNOTATION_FONT.copy())
    return fig

#!/usr/bin/env python3
"""Figure: single positive toy ship example (scramble dataset).

Shows one present (positive) molecule from B_SHIP00 in square aspect,
with motif nodes/edges highlighted in the theme accent color.
"""

import os

import plotly.graph_objects as go

from plot_style import THEME_ACCENT, THEME_LIGHT, apply_layout
from plot_toy_performance import (
    ShipExample,
    _axis_range,
    _edge_segments,
    _load_ship_example,
)

ROOT = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(ROOT, "figures")

SCRAMBLE_SHIP = "ship00"
POSITIVE_LABEL = 1


def load_scramble_positive_example() -> ShipExample:
    seed = abs(hash(("example", SCRAMBLE_SHIP, POSITIVE_LABEL))) % (2**32)
    return _load_ship_example(SCRAMBLE_SHIP, POSITIVE_LABEL, seed)


def _add_ground_truth_graph(fig: go.Figure, example: ShipExample) -> None:
    pos = example.pos
    for is_motif, color, width in (
        (False, THEME_LIGHT, 1.6),
        (True, THEME_ACCENT, 2.4),
    ):
        xs, ys = _edge_segments(pos, example.edges, example.edge_is_motif == is_motif)
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for is_motif, color in ((False, THEME_LIGHT), (True, THEME_ACCENT)):
        mask = example.atom_is_motif == is_motif
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=pos[mask, 0],
                y=pos[mask, 1],
                mode="markers",
                marker=dict(size=14 if is_motif else 10, color=color, line=dict(width=0)),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def build_figure(example: ShipExample) -> go.Figure:
    size = 360
    margin = 12
    fig = go.Figure()
    _add_ground_truth_graph(fig, example)

    x_range, y_range = _axis_range(example.pos)
    fig.update_xaxes(visible=False, range=x_range, constrain="domain")
    fig.update_yaxes(
        visible=False,
        range=y_range,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    apply_layout(
        fig,
        height=size,
        width=size,
        margin=dict(t=margin, b=margin, l=margin, r=margin),
        showlegend=False,
    )
    return fig


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    example = load_scramble_positive_example()
    fig = build_figure(example)
    for ext in ("png", "svg", "pdf"):
        out = os.path.join(FIGURES_DIR, f"toy_example.{ext}")
        fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

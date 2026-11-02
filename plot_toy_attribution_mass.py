#!/usr/bin/env python3
"""Toy attribution mass (motif vs noise).

Computes the percentage of positive raw score mass assigned to motif
nodes vs noise nodes, then plots it in the same grid as ``toy_attribution``
(but without the ground-truth column). Colormap min-max is not used here.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_style import THEME_ACCENT, THEME_LIGHT, apply_layout
from plot_toy_attribution import (
    ATTRIBUTION_COLUMNS,
    ATTRIBUTION_SHIPS,
    load_attribution_examples,
    _run_prefix,
)
from plot_toy_performance import SHIP_LABELS
from models.io import toy_attribution

ROOT = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(ROOT, "figures")


def _add_row_labels(fig: go.Figure, ships: list[str]) -> None:
    for row, ship in enumerate(ships, start=1):
        fig.add_annotation(
            text=SHIP_LABELS[ship],
            xref="x domain",
            yref="y domain",
            x=-0.18,
            y=0.5,
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            textangle=90,
            row=row,
            col=1,
        )


def _mass_percent(scores, motif_mask) -> Tuple[float, float]:
    """Share of ``ReLU(raw)`` mass on motif vs noise. No colormap normalization."""
    scores = np.clip(np.asarray(scores, dtype=float).reshape(-1), 0.0, None)
    motif_mask = motif_mask.astype(bool).reshape(-1)
    motif_mass = float(scores[motif_mask].sum())
    total_mass = float(scores.sum())
    if total_mass <= 1e-12:
        return 0.0, 0.0
    motif_pct = 100.0 * motif_mass / total_mass
    noise_pct = 100.0 - motif_pct
    return motif_pct, noise_pct


def build_figure(examples) -> go.Figure:
    ships = ATTRIBUTION_SHIPS
    n_rows = len(ships)
    n_cols = len(ATTRIBUTION_COLUMNS)

    column_titles = [title for _, _, title in ATTRIBUTION_COLUMNS]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=column_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.03,
    )

    for row, ship in enumerate(ships, start=1):
        example = examples[ship]
        motif_mask = example.atom_is_motif.astype(bool)

        for col, (arch, method, _title) in enumerate(ATTRIBUTION_COLUMNS, start=1):
            scores = toy_attribution(example, ship, arch, method=method)
            if scores is None or len(scores) != len(motif_mask):
                raise FileNotFoundError(f"Missing attribution for {arch=}, {ship=}")

            motif_pct, noise_pct = _mass_percent(scores, motif_mask)
            # Use a single categorical y value so motif/noise segments render
            # in the same horizontal bar (one row per panel).
            y_val = [0]

            # Motif fraction (pink)
            fig.add_trace(
                go.Bar(
                    orientation="h",
                    x=[motif_pct],
                    y=y_val,
                    marker_color=THEME_ACCENT,
                    name="Motif Vertices",
                    showlegend=(row == 1 and col == 1),
                    text=[f"{motif_pct:.0f}%"],
                    textposition="inside",
                    insidetextanchor="middle",
                    cliponaxis=False,
                ),
                row=row,
                col=col,
            )

            # Noise fraction (grey)
            fig.add_trace(
                go.Bar(
                    orientation="h",
                    x=[noise_pct],
                    y=y_val,
                    marker_color=THEME_LIGHT,
                    name="Noise Vertices",
                    showlegend=(row == 1 and col == 1),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            # Clean axes; horizontal bars need numeric x-axis only.
            fig.update_xaxes(
                range=[0, 100],
                tickvals=[0, 50, 100],
                tickfont=dict(size=10),
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                showticklabels=False,
                showgrid=False,
                row=row,
                col=col,
            )

    _add_row_labels(fig, ships)
    return fig


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    examples = load_attribution_examples()
    fig = build_figure(examples)
    apply_layout(
        fig,
        barmode="stack",
        height=350,
        width=int(140 * len(ATTRIBUTION_COLUMNS)),
        margin=dict(t=60, b=78, l=60, r=20),
        show_legend=True,
        legend=dict(
            title=dict(text="Attribution mass (%) distributed to"),
            traceorder="reversed",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    out_base = os.path.join(FIGURES_DIR, f"{_run_prefix()}_attribution_mass")
    for ext in ("png", "svg", "pdf"):
        out = f"{out_base}.{ext}"
        fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Figure: toy ship performance table (examples + bar plots).

Layout:
  Row 1: scatter plot of one example per perturbation (columns = datasets)
  Row 2: grouped bar plots of model accuracy per perturbation

Reads ``results/{toy|toy3d}_{model}_{ship}_{seed}/test_results.yaml``. Missing runs
raise instead of substituting placeholder accuracies.
"""

import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import plotly.graph_objects as go
import yaml
from bioverse.factory import BenchmarkFactory
from plotly.subplots import make_subplots

from plot_style import (
    MODEL_COLORS,
    MODEL_LABELS,
    THEME_ACCENT,
    THEME_LIGHT,
    TOY3D_RUN_PATTERN,
    TOY_PERFORMANCE_MODELS,
    TOY_RUN_PATTERN,
    apply_bar_frame,
    apply_layout,
)

from models import io as models_io

ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

SHIPS = ["ship00", "shipr0", "shipch", "shiptl", "shippc"]
SHIP_LABELS = {
    "ship00": "Scramble",
    "shipr0": "Rotation",
    "shipch": "Chirality",
    "shiptl": "Tilt",
    "shippc": "Picasso",
}

SHIP_BENCHMARKS = {
    "ship00": "B_SHIP00",
    "shipr0": "B_SHIPR0",
    "shipch": "B_SHIPCH",
    "shiptl": "B_SHIPTL",
    "shippc": "B_SHIPPC",
}


def _run_prefix() -> str:
    return models_io.TOY_RUN_PREFIX


def _run_pattern():
    return TOY3D_RUN_PATTERN if _run_prefix() == "toy3d" else TOY_RUN_PATTERN


def _figure_stem(kind: str) -> str:
    return f"{_run_prefix()}_{kind}"


@dataclass(frozen=True)
class ShipExample:
    pos: np.ndarray
    atom_is_motif: np.ndarray
    edges: np.ndarray
    edge_is_motif: np.ndarray
    present: int


def _extract_ship_example(X, y) -> ShipExample:
    pos = ak.to_numpy(X.atom_pos)[:, :2]
    edges = ak.to_numpy(X.molecule_edges).reshape(-1, 2)
    return ShipExample(
        pos=pos,
        atom_is_motif=ak.to_numpy(X.atom_is_motif).astype(bool),
        edges=edges.astype(int),
        edge_is_motif=ak.to_numpy(X.molecule_edge_is_motif).reshape(-1).astype(bool),
        present=int(ak.to_numpy(y.target)[0]),
    )


def _load_ship_example(
    ship: str,
    present: int,
    seed: int,
    partition: str = "train",
    rotated: Optional[bool] = None,
) -> ShipExample:
    benchmark = SHIP_BENCHMARKS[ship]
    loader = BenchmarkFactory(benchmark).loader(
        partition,
        batch_size=1,
        shuffle=True,
        random_seed=seed,
        progress=False,
    )
    for (X, y), _ in loader:
        if int(ak.to_numpy(y.target)[0]) != present:
            continue
        if rotated is not None:
            if not hasattr(X, "molecule_rotated"):
                raise RuntimeError(
                    f"{benchmark} missing molecule_rotated field for rotated={rotated}"
                )
            is_rotated = bool(ak.to_numpy(ak.ravel(X.molecule_rotated))[0])
            if is_rotated != rotated:
                continue
        return _extract_ship_example(X, y)
    extra = f", rotated={rotated}" if rotated is not None else ""
    raise RuntimeError(
        f"No example with present={present}{extra} found for {benchmark} ({partition})"
    )


def load_ship_examples() -> Dict[str, ShipExample]:
    examples: Dict[str, ShipExample] = {}
    for ship in SHIPS:
        present = 1 if ship == "shipr0" else 0
        seed = abs(hash(("example", ship, present))) % (2**32)
        if ship == "shipr0":
            examples[ship] = _load_ship_example(
                ship, present, seed, partition="test", rotated=True
            )
        else:
            examples[ship] = _load_ship_example(ship, present, seed)
    return examples


def _axis_range(pos: np.ndarray, pad: float = 0.08) -> Tuple[List[float], List[float]]:
    xmin, ymin = pos.min(axis=0)
    xmax, ymax = pos.max(axis=0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    half *= 1 + pad
    return [cx - half, cx + half], [cy - half, cy + half]


def _edge_segments(pos: np.ndarray, edges: np.ndarray, mask: np.ndarray) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for i, j in edges[mask]:
        xs.extend([pos[i, 0], pos[j, 0], None])
        ys.extend([pos[i, 1], pos[j, 1], None])
    return xs, ys


def _add_ship_graph(fig: go.Figure, example: ShipExample, row: int, col: int) -> None:
    pos = example.pos
    for is_motif, color, width in (
        (False, THEME_LIGHT, 0.8),
        (True, THEME_ACCENT, 1.2),
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
            ),
            row=row,
            col=col,
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
                marker=dict(size=7 if is_motif else 5, color=color, line=dict(width=0)),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    x_range, y_range = _axis_range(pos)
    fig.update_xaxes(visible=False, range=x_range, constrain="domain", row=row, col=col)
    fig.update_yaxes(
        visible=False,
        range=y_range,
        scaleanchor=f"x{col if col > 1 else ''}",
        scaleratio=1,
        constrain="domain",
        row=row,
        col=col,
    )


def _read_metric(run_dir: str) -> Optional[float]:
    path = os.path.join(run_dir, "test_results.yaml")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        for key in ("Accuracy", "Binary Accuracy", "Balanced Binary Accuracy", "Bal.Acc."):
            if key in data:
                val = float(data[key])
                return val * 100 if val <= 1.0 else val
        if len(data) == 1:
            val = float(next(iter(data.values())))
            return val * 100 if val <= 1.0 else val
    elif isinstance(data, (int, float)):
        val = float(data)
        return val * 100 if val <= 1.0 else val
    return None


def collect_results() -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    missing: List[str] = []
    for model in TOY_PERFORMANCE_MODELS:
        for ship in SHIPS:
            metric = None
            for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{_run_prefix()}_{model}_{ship}_*"))):
                name = os.path.basename(run_dir)
                if _run_pattern().match(name) is None:
                    continue
                metric = _read_metric(run_dir)
                if metric is not None:
                    scores[(model, ship)] = metric
                    break
            if (model, ship) not in scores:
                missing.append(f"{_run_prefix()}_{model}_{ship}_*/test_results.yaml")
    if missing:
        raise FileNotFoundError(
            "Refusing to plot placeholder accuracies. Missing test results:\n  "
            + "\n  ".join(missing)
        )
    return scores


def _row_heights_for_square_scatter(
    n_cols: int,
    plot_width: float,
    plot_height: float,
    *,
    horizontal_spacing: float,
    vertical_spacing: float,
) -> Tuple[float, float]:
    """Row weights so the scatter row is square; bar row fills the remainder."""
    col_width_frac = (1 - (n_cols - 1) * horizontal_spacing) / n_cols
    scatter_row_frac = col_width_frac * plot_width / plot_height
    bar_row_frac = 1 - vertical_spacing - scatter_row_frac
    return scatter_row_frac, bar_row_frac


def _shrink_bar_row_domains(fig: go.Figure, n_cols: int, factor: float) -> None:
    """Shorten bar-row subplot height while keeping scatter row square."""
    yaxis = fig.layout.yaxis6
    y0, y1 = yaxis.domain
    new_top = y0 + (y1 - y0) * factor
    for col in range(1, n_cols + 1):
        fig.update_yaxes(domain=[y0, new_top], row=2, col=col)


def _center_scatter_row(
    fig: go.Figure,
    n_cols: int,
    *,
    bar_gap: float = 0.04,
    zone_top: float = 1.0,
) -> None:
    """Vertically center the scatter row between titles and the bar row."""
    row2_top = fig.layout.yaxis6.domain[1]
    row1_bottom, row1_top = fig.layout.yaxis.domain
    row1_height = row1_top - row1_bottom

    zone_bottom = row2_top + bar_gap
    zone_center = (zone_bottom + zone_top) / 2
    new_bottom = zone_center - row1_height / 2
    new_top = zone_center + row1_height / 2

    if new_top > zone_top:
        new_top = zone_top
        new_bottom = new_top - row1_height
    if new_bottom < zone_bottom:
        new_bottom = zone_bottom
        new_top = new_bottom + row1_height

    for col in range(1, n_cols + 1):
        fig.update_yaxes(domain=[new_bottom, new_top], row=1, col=col)


def build_figure(
    scores: Dict[Tuple[str, str], float],
    examples: Dict[str, ShipExample],
) -> go.Figure:
    n_cols = len(SHIPS)
    fig_width = 800
    fig_height = 400
    top_margin = 48
    bottom_margin = 58
    left_margin = 88
    right_margin = 48
    horizontal_spacing = 0.05
    vertical_spacing = 0.02
    bar_row_height_factor = 2 / 3

    plot_width = fig_width - left_margin - right_margin
    plot_height = fig_height - top_margin - bottom_margin
    scatter_row_weight, bar_row_weight = _row_heights_for_square_scatter(
        n_cols,
        plot_width,
        plot_height,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )
    fig = make_subplots(
        rows=2,
        cols=n_cols,
        subplot_titles=[SHIP_LABELS[s] for s in SHIPS],
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        row_heights=[scatter_row_weight, bar_row_weight],
        shared_yaxes="rows",
    )

    for col, ship in enumerate(SHIPS, start=1):
        _add_ship_graph(fig, examples[ship], row=1, col=col)

        for mi, model in enumerate(TOY_PERFORMANCE_MODELS):
            fig.add_trace(
                go.Bar(
                    x=[mi],
                    y=[scores[(model, ship)]],
                    marker_color=MODEL_COLORS[model],
                    name=MODEL_LABELS[model],
                    showlegend=(col == 1),
                    legendgroup=model,
                ),
                row=2,
                col=col,
            )
        fig.update_yaxes(
            title_text="Accuracy (%)" if col == 1 else "",
            range=[0, 100],
            tick0=0,
            dtick=25,
            showticklabels=col == 1,
            row=2,
            col=col,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=col)

    _shrink_bar_row_domains(fig, n_cols, bar_row_height_factor)

    apply_layout(
        fig,
        barmode="group",
        height=fig_height,
        width=fig_width,
        margin=dict(t=top_margin, b=bottom_margin, l=left_margin, r=right_margin),
        legend=dict(
            orientation="h",
            xref="paper",
            xanchor="center",
            x=0.5,
            yref="paper",
            yanchor="top",
            y=-0.14,
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    for annotation in fig.layout.annotations[:n_cols]:
        annotation.yshift = -8

    _center_scatter_row(fig, n_cols)
    apply_bar_frame(fig, row=2)
    return fig


def export_source_data(scores: Dict[Tuple[str, str], float]) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"source_data_{_figure_stem('performance')}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "ship", "accuracy"])
        for model in TOY_PERFORMANCE_MODELS:
            for ship in SHIPS:
                writer.writerow([model, ship, scores[(model, ship)]])


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    scores = collect_results()
    examples = load_ship_examples()
    fig = build_figure(scores, examples)
    export_source_data(scores)
    for ext in ("png", "svg", "pdf"):
        out = os.path.join(FIGURES_DIR, f"{_figure_stem('performance')}.{ext}")
        fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

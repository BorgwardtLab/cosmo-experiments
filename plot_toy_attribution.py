#!/usr/bin/env python3
"""Figure: toy ship attribution comparison.

Layout:
  Rows: one per dataset (perturbation)
  Columns: ground truth, then BioLiP order (Cosmo, Cosmo IG, KPConv, GNN, EGNN, SE3T)

Reads attribution from trained checkpoints. Missing scores raise instead of
substituting noise.
"""

import glob
import os
from typing import Dict, Optional

import awkward as ak
import numpy as np
import plotly.graph_objects as go
from bioverse.factory import BenchmarkFactory
from plotly.subplots import make_subplots

from plot_style import (
    ATTRIBUTION_COLORSCALE,
    BIOLIP_MODEL_COLUMNS,
    THEME_ACCENT,
    THEME_LIGHT,
    TOY3D_RUN_PATTERN,
    TOY_RUN_PATTERN,
    apply_layout,
)
from plot_toy_performance import (
    SHIP_BENCHMARKS,
    SHIP_LABELS,
    ShipExample,
    _extract_ship_example,
    _axis_range,
    _edge_segments,
    _read_metric,
)
from models import io as models_io

ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")


def _run_prefix() -> str:
    return models_io.TOY_RUN_PREFIX


def _model_dim() -> int:
    return models_io.TOY_MODEL_DIM


def _run_pattern():
    return TOY3D_RUN_PATTERN if _run_prefix() == "toy3d" else TOY_RUN_PATTERN


# Scramble and rotation are omitted — they are less informative for attribution.
ATTRIBUTION_SHIPS = ["shipch", "shiptl", "shippc"]
ATTRIBUTION_COLUMNS = list(BIOLIP_MODEL_COLUMNS)
_WELL_CLASSIFIED_MODELS = list(dict.fromkeys(arch for arch, _method, _label in ATTRIBUTION_COLUMNS))
# Skip models at chance: they cannot satisfy logit > 0 on positives.
_CHANCE_ACCURACY = 55.0


def _test_accuracy(arch: str, ship: str) -> Optional[float]:
    for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{_run_prefix()}_{arch}_{ship}_*"))):
        if _run_pattern().match(os.path.basename(run_dir)) is None:
            continue
        metric = _read_metric(run_dir)
        if metric is not None:
            return metric
    return None


def _required_models(ship: str) -> list:
    required = []
    for arch in _WELL_CLASSIFIED_MODELS:
        acc = _test_accuracy(arch, ship)
        if acc is not None and acc <= _CHANCE_ACCURACY:
            continue
        required.append(arch)
    return required or ["cosmo"]


def _load_attribution(run_dir: str, n_nodes: int) -> Optional[np.ndarray]:
    path = os.path.join(run_dir, "attributions.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    if "scores" in data and len(data["scores"]) == n_nodes:
        return data["scores"]
    return None


def _predict_logit(example: ShipExample, ship: str, arch: str) -> float:
    from models.io import get_toy_model, ship_example_to_data

    model = get_toy_model(arch, ship, dim=_model_dim(), run_prefix=_run_prefix())
    if model is None:
        raise FileNotFoundError(f"Missing checkpoint for {_run_prefix()}_{arch}_{ship}_*")
    data = ship_example_to_data(example, dim=_model_dim())
    return float(model._predict(data)[0].reshape(-1)[0])


def _is_well_classified(
    example: ShipExample, ship: str, min_logit: float, models: list
) -> bool:
    for arch in models:
        if _predict_logit(example, ship, arch) <= min_logit:
            return False
    return True


def _load_well_classified_example(ship: str, min_logit: float = 0.0) -> ShipExample:
    required = _required_models(ship)
    benchmark = SHIP_BENCHMARKS[ship]
    seed = abs(hash(("attribution", ship, 1, "well_classified"))) % (2**32)
    loader = BenchmarkFactory(benchmark).loader(
        "test",
        batch_size=1,
        shuffle=True,
        random_seed=seed,
        progress=False,
    )
    for (X, y), _ in loader:
        if int(ak.to_numpy(y.target)[0]) != 1:
            continue
        example = _extract_ship_example(X, y)
        if _is_well_classified(example, ship, min_logit=min_logit, models=required):
            return example
    raise RuntimeError(
        f"No positive test example for {ship} is well-classified by "
        f"{', '.join(required)} with logit > {min_logit:.2f}"
    )


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


def _set_square_axes(fig: go.Figure, pos: np.ndarray, row: int, col: int) -> None:
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


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    hex_color = color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(channel)) for channel in rgb))


def _sym_unit(scores: np.ndarray) -> np.ndarray:
    """Map raw scores onto ``[0, 1]`` with 0 at grey and ends at ±max|s|."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    peak = float(np.abs(scores).max()) if scores.size else 0.0
    if peak <= 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float32)
    return (0.5 + 0.5 * (scores / peak)).astype(np.float32)


def _color_from_scale(value: float) -> str:
    """Interpolate ``ATTRIBUTION_COLORSCALE`` at ``value`` in ``[0, 1]``."""
    stops = ATTRIBUTION_COLORSCALE
    value = float(np.clip(value, 0.0, 1.0))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if value <= t1 or i == len(stops) - 2:
            span = t1 - t0
            u = 0.0 if span == 0 else (value - t0) / span
            u = min(max(u, 0.0), 1.0)
            rgb0 = _hex_to_rgb(str(c0))
            rgb1 = _hex_to_rgb(str(c1))
            mixed = tuple(rgb0[j] + u * (rgb1[j] - rgb0[j]) for j in range(3))
            return _rgb_to_hex(mixed)
    return str(stops[-1][1])


def _add_attribution_edges(
    fig: go.Figure,
    example: ShipExample,
    scores: np.ndarray,
    row: int,
    col: int,
) -> None:
    if len(example.edges) == 0:
        return
    pos = example.pos
    scores = np.asarray(scores, dtype=float).reshape(-1)
    for is_motif, width in ((False, 0.8), (True, 1.2)):
        mask = example.edge_is_motif == is_motif
        for i, j in example.edges[mask]:
            color = _color_from_scale(0.5 * (scores[int(i)] + scores[int(j)]))
            fig.add_trace(
                go.Scatter(
                    x=[pos[i, 0], pos[j, 0]],
                    y=[pos[i, 1], pos[j, 1]],
                    mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )


def _add_ground_truth_panel(fig: go.Figure, example: ShipExample, row: int, col: int) -> None:
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
    _set_square_axes(fig, pos, row, col)


def _add_attribution_panel(
    fig: go.Figure,
    example: ShipExample,
    scores: np.ndarray,
    row: int,
    col: int,
) -> None:
    pos = example.pos
    scores = _sym_unit(scores)
    _add_attribution_edges(fig, example, scores, row, col)
    for is_motif in (False, True):
        mask = example.atom_is_motif == is_motif
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=pos[mask, 0],
                y=pos[mask, 1],
                mode="markers",
                marker=dict(
                    size=7 if is_motif else 5,
                    color=scores[mask],
                    colorscale=ATTRIBUTION_COLORSCALE,
                    cmin=0,
                    cmax=1,
                    showscale=False,
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    _set_square_axes(fig, pos, row, col)


def build_figure(examples: Dict[str, ShipExample]) -> go.Figure:
    ships = ATTRIBUTION_SHIPS
    n_rows = len(ships)
    n_cols = 1 + len(ATTRIBUTION_COLUMNS)
    fig_width = int(120 * n_cols)
    fig_height = int(600 * n_rows / 5)
    horizontal_spacing = 0.012
    vertical_spacing = 0.012
    top_margin = 38
    bottom_margin = 14
    left_margin = 54
    right_margin = 12

    column_titles = ["Ground Truth"] + [title for _, _, title in ATTRIBUTION_COLUMNS]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=column_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        row_heights=[1.0] * n_rows,
    )

    for row, ship in enumerate(ships, start=1):
        example = examples[ship]
        _add_ground_truth_panel(fig, example, row=row, col=1)

        from models.io import toy_attribution

        missing: list[str] = []
        for col_offset, (model, method, title) in enumerate(ATTRIBUTION_COLUMNS, start=2):
            motif = example.atom_is_motif.astype(float)
            scores = toy_attribution(example, ship, model, method=method)
            if (scores is None or len(scores) != len(motif)) and method != "ig":
                cached = None
                for run_dir in glob.glob(os.path.join(RESULTS_DIR, f"{_run_prefix()}_{model}_{ship}_*")):
                    cached = _load_attribution(run_dir, len(motif))
                    if cached is not None:
                        break
                scores = cached
            if scores is None or len(scores) != len(motif):
                missing.append(f"{title} / {SHIP_LABELS[ship]}")
                continue
            _add_attribution_panel(fig, example, scores, row=row, col=col_offset)
        if missing:
            raise FileNotFoundError(
                "Refusing to plot placeholder attribution. Missing scores:\n  "
                + "\n  ".join(missing)
            )

    apply_layout(
        fig,
        height=fig_height,
        width=fig_width,
        margin=dict(t=top_margin, b=bottom_margin, l=left_margin, r=right_margin),
        show_legend=False,
    )
    for annotation in fig.layout.annotations[:n_cols]:
        annotation.yshift = -4

    _add_row_labels(fig, ships)
    return fig


def load_attribution_examples() -> Dict[str, ShipExample]:
    """Use positive test ships correctly classified by models that beat chance."""
    examples: Dict[str, ShipExample] = {}
    for ship in ATTRIBUTION_SHIPS:
        examples[ship] = _load_well_classified_example(ship, min_logit=0.0)
    return examples


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    examples = load_attribution_examples()
    fig = build_figure(examples)
    for ext in ("png", "svg", "pdf"):
        out = os.path.join(FIGURES_DIR, f"{_run_prefix()}_attribution.{ext}")
        fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

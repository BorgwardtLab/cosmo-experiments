#!/usr/bin/env python3
"""BioLiP attribution on protein–ligand complexes as Mol* surfaces.

Layout:
  Rows: one example complex
  Columns: Binding site, Cosmo, Cosmo IG, KPConv, GNN, EGNN, SE3T

The protein is a surface colored by |attribution| (or BioLiP contact residues
for ground truth). The ligand is drawn as ball-and-stick in every panel.
"""

import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from biolip_molstar import (
    BIOLIP_ATTRIBUTION_COLUMNS,
    _attribution_archs,
    _column_id,
    load_biolip_examples,
    render_all_panels,
    require_biolip_checkpoints,
)
from plot_style import apply_layout

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"


def _axis_suffix(row: int, col: int, n_cols: int) -> str:
    index = (row - 1) * n_cols + col
    return "" if index == 1 else str(index)


def _subplot_domain(fig: go.Figure, row: int, col: int, n_cols: int) -> tuple[float, float, float, float]:
    suffix = _axis_suffix(row, col, n_cols)
    xaxis = fig.layout[f"xaxis{suffix}"]
    yaxis = fig.layout[f"yaxis{suffix}"]
    return xaxis.domain[0], xaxis.domain[1], yaxis.domain[0], yaxis.domain[1]


def _add_row_labels(fig: go.Figure, examples, n_cols: int) -> None:
    for row, example in enumerate(examples, start=1):
        x0, _, y0, y1 = _subplot_domain(fig, row, 1, n_cols)
        fig.add_annotation(
            text=f"{example.pdb_id.upper()} {example.ligand_id}",
            xref="paper",
            yref="paper",
            x=x0,
            y=(y0 + y1) / 2,
            xanchor="right",
            yanchor="middle",
            xshift=-2,
            showarrow=False,
            textangle=90,
        )


def _trimmed_panel_path(image_path: Path, pad_frac: float = 0.04) -> Path:
    """Crop near-white margins, pad back to a square, and write a sibling PNG."""
    out_path = image_path.with_name(image_path.stem + "_tight.png")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    ink = (rgb < 248).any(axis=2)
    if ink.any():
        rows, cols = np.where(ink)
        crop = rgb[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]
    else:
        crop = rgb
    height, width = crop.shape[:2]
    pad = max(2, int(round(pad_frac * max(height, width))))
    side = max(height, width) + 2 * pad
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = crop
    Image.fromarray(canvas).save(out_path)
    return out_path


def _add_panel_image(fig: go.Figure, image_path: Path, row: int, col: int, n_cols: int) -> None:
    x0, x1, y0, y1 = _subplot_domain(fig, row, col, n_cols)
    fig.add_layout_image(
        dict(
            source=str(_trimmed_panel_path(image_path)),
            xref="paper",
            yref="paper",
            x=x0,
            y=y1,
            sizex=x1 - x0,
            sizey=y1 - y0,
            xanchor="left",
            yanchor="top",
            layer="below",
            sizing="stretch",
        )
    )
    fig.update_xaxes(visible=False, showticklabels=False, row=row, col=col)
    fig.update_yaxes(visible=False, showticklabels=False, row=row, col=col)


def build_figure(examples, panel_paths: dict[str, dict[str, Path]]) -> go.Figure:
    n_rows = len(examples)
    attr_cols = BIOLIP_ATTRIBUTION_COLUMNS
    n_cols = 1 + len(attr_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=["Binding site"] + [label for _arch, _method, label in attr_cols],
        vertical_spacing=0.010,
        horizontal_spacing=0.012,
        row_heights=[1.0] * n_rows,
    )
    columns = ["ground_truth", *[_column_id(arch, method) for arch, method, _ in attr_cols]]
    for row, example in enumerate(examples, start=1):
        for col, column in enumerate(columns, start=1):
            _add_panel_image(
                fig,
                panel_paths[example.molecule_id][column],
                row=row,
                col=col,
                n_cols=n_cols,
            )
    apply_layout(
        fig,
        height=int(600 * n_rows / 5),
        width=int(120 * n_cols),
        margin=dict(t=40, b=8, l=22, r=10),
        show_legend=False,
    )
    fig.update_yaxes(automargin=False, ticks="", showticklabels=False, title_text="")
    fig.update_xaxes(automargin=False)
    for annotation in fig.layout.annotations[:n_cols]:
        annotation.yshift = 2
    _add_row_labels(fig, examples, n_cols)
    return fig


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    require_biolip_checkpoints(_attribution_archs())
    force = "--force" in sys.argv
    print(
        "Re-rendering all Mol* panels and recomputing attributions..."
        if force
        else "Rendering Mol* panels (reusing checkpoint-valid caches when available)..."
    )
    examples = load_biolip_examples()
    panel_paths = render_all_panels(force=force)
    fig = build_figure(examples, panel_paths)
    for ext in ("png", "svg", "pdf"):
        out = FIGURES_DIR / f"biolip_attribution.{ext}"
        fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

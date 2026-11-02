#!/usr/bin/env python3
"""LaTeX table and bar figure of BioLiP model comparison.

Table columns: MAE (pKd), Spearman ρ, AUROC, AUPRC, Recall.
Bar figure: one panel per percentage metric (no MAE), models as bars.
Rows match the attribution figure: Cosmo, Cosmo IG, KPConv, GNN, EGNN, SE3T.

Reads ``results/biolip_{model}_{seed}/``. Attribution columns are used
only when ``attr_results.yaml`` was written for the current
``checkpoint.pt`` (matching ``checkpoint_sha256``) and, for IG rows,
``ig_inputs: features``. This script never loads data or recomputes scores.
"""

import csv
import os
from typing import Dict, List, Optional

import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eval_common import attr_results_for_checkpoint, site_metrics_from_attr
from models.io import find_completed_run_dir
from plot_style import (
    BIOLIP_MODEL_COLUMNS,
    BIOLIP_MODEL_COLORS,
    MODEL_LABELS,
    apply_bar_frame,
    apply_layout,
)

# (run arch, attribution method, table label)
TABLE_ROWS = BIOLIP_MODEL_COLUMNS

ROOT = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(ROOT, "figures")
MISSING_LATEX = "---"
MISSING_CSV = "NA"


def _read_yaml_metric(path: str, keys: List[str]) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data:
            return float(data[key])
    return None


def collect_table_rows() -> List[Dict]:
    rows: List[Dict] = []
    for model, method, label in TABLE_ROWS:
        run_dir = find_completed_run_dir(f"biolip_{model}_*")
        mae = spearman = auroc = auprc = recall = None
        if run_dir is not None:
            mae = _read_yaml_metric(
                os.path.join(run_dir, "test_results.yaml"),
                ["MAE", "MAE (target)", "MeanAbsoluteErrorMetric"],
            )
            spearman = _read_yaml_metric(
                os.path.join(run_dir, "test_results.yaml"),
                ["Spearman", "Spearman (target)", "SpearmansRhoMetric"],
            )
            attr = attr_results_for_checkpoint(run_dir)
            auroc, auprc, recall = site_metrics_from_attr(attr, method=method)
        rows.append(
            {
                "model": "cosmo_ig" if method == "ig" else model,
                "label": label,
                "mae": mae,
                "spearman": spearman,
                "auroc": auroc,
                "auprc": auprc,
                "recall": recall,
            }
        )
    return rows


def format_mae(value: Optional[float]) -> str:
    if value is None:
        return MISSING_LATEX
    return f"{value:.2f}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return MISSING_LATEX
    return f"{100.0 * value:.1f}"


def csv_value(value) -> object:
    return MISSING_CSV if value is None else value


def render_latex(rows: List[Dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Protein--ligand binding affinity and site attribution on "
        r"D\_BIOLIP. "
        r"MAE is mean absolute error on pKd ($-\log_{10}$ Kd). "
        r"Spearman $\rho$ is the rank correlation between predicted and true "
        r"pKd. "
        r"AUROC and AUPRC rank $|$attribution$|$ on protein residues "
        r"(chain mask) against BioLiP contact residues. "
        r"Recall is the fraction of the $n_{\mathrm{site}}$ highest-scoring "
        r"residues that are contacts. "
        r"$\rho$, AUROC, AUPRC, and Recall are percentages. "
        r"Ligand atom identities are collapsed to one token (coordinates kept). "
        r"All models max-pool protein residues (ligand masked at readout). "
        r"Cosmo max-pools lifted frames onto nodes first. "
        r"Graphs are residue $k$NN with $k=6$ plus all protein--ligand pairs "
        r"within $8$\,\AA{}; batch size 32 complexes; 50 epochs. "
        r"Cosmo and KPConv use kernel shells $[1.5, 4.5, 8.0]$ and hidden 64; "
        r"EGNN uses hidden 256 and SE3T hidden 512. "
        r"KPConv uses nearest-kernel assignment (KPConvX). "
        r"Baselines and Cosmo IG use integrated gradients on node features "
        r"(coordinates held fixed). "
        r"All models trained with matched batch size, epochs, and depth.}",
        r"\label{tab:biolip_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & MAE (pKd) & Spearman $\rho$ (\%) & AUROC (\%) & AUPRC (\%) & Recall (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {format_mae(row['mae'])} & "
            f"{format_percent(row['spearman'])} & "
            f"{format_percent(row['auroc'])} & "
            f"{format_percent(row['auprc'])} & "
            f"{format_percent(row['recall'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


BAR_METRICS = [
    ("spearman", "Spearman ρ (%)"),
    ("auroc", "AUROC (%)"),
    ("auprc", "AUPRC (%)"),
    ("recall", "Recall (%)"),
]


def build_bar_figure(rows: List[Dict]) -> go.Figure:
    n_cols = len(BAR_METRICS)
    fig_width = 800
    fig_height = 280
    top_margin = 48
    bottom_margin = 58
    left_margin = 56
    right_margin = 24
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[title for _key, title in BAR_METRICS],
        horizontal_spacing=0.06,
        shared_yaxes=True,
    )
    for col, (key, _title) in enumerate(BAR_METRICS, start=1):
        for mi, row in enumerate(rows):
            value = row[key]
            if value is None:
                raise FileNotFoundError(
                    f"Missing {key} for {row['label']}; refusing to plot placeholders."
                )
            model_id = row["model"]
            fig.add_trace(
                go.Bar(
                    x=[mi],
                    y=[100.0 * float(value)],
                    marker_color=BIOLIP_MODEL_COLORS[model_id],
                    name=MODEL_LABELS[model_id],
                    showlegend=(col == 1),
                    legendgroup=model_id,
                ),
                row=1,
                col=col,
            )
        fig.update_yaxes(
            range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            dtick=25,
            ticks="outside" if col == 1 else "",
            showticklabels=col == 1,
            automargin=False,
            row=1,
            col=col,
        )
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=1,
            col=col,
        )
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
            y=-0.18,
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    for annotation in fig.layout.annotations[:n_cols]:
        annotation.yshift = 8
    apply_bar_frame(fig, row=1)
    return fig


def export_csv(rows: List[Dict]) -> None:
    path = os.path.join(FIGURES_DIR, "source_data_biolip_table.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mae", "spearman", "auroc", "auprc", "recall"])
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    csv_value(row["mae"]),
                    csv_value(row["spearman"]),
                    csv_value(row["auroc"]),
                    csv_value(row["auprc"]),
                    csv_value(row["recall"]),
                ]
            )


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = collect_table_rows()
    missing = [
        f"{row['label']}:{name}"
        for row in rows
        for name, value in (
            ("mae", row["mae"]),
            ("spearman", row["spearman"]),
            ("auroc", row["auroc"]),
            ("auprc", row["auprc"]),
            ("recall", row["recall"]),
        )
        if value is None
    ]
    if missing:
        print("Missing values: " + ", ".join(missing))
    tex = render_latex(rows)
    tex_path = os.path.join(FIGURES_DIR, "biolip_comparison.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    export_csv(rows)
    print(f"Wrote {tex_path}")
    fig = build_bar_figure(rows)
    for ext in ("png", "svg", "pdf"):
        out = os.path.join(FIGURES_DIR, f"biolip_performance.{ext}")
        fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

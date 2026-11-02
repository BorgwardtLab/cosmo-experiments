#!/usr/bin/env python3
"""Frameless vertical colorbars for assembling figure legends.

Black–grey–pink is the toy attribution scale (``ATTRIBUTION_COLORSCALE``).
White–pink is the BioLiP Mol* surface scale (white to ``THEME_ACCENT``).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_style import ATTRIBUTION_COLORSCALE, THEME_ACCENT

ROOT = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(ROOT, "figures")

WHITE_PINK_COLORSCALE = [
    [0.0, "#FFFFFF"],
    [1.0, THEME_ACCENT],
]

SCALES = (
    ("black_grey_pink", ATTRIBUTION_COLORSCALE),
    ("white_pink", WHITE_PINK_COLORSCALE),
)

# High value (pink) at the top.
_STRIP = np.linspace(0.0, 1.0, 512).reshape(-1, 1)


def _colormap(stops) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "scale",
        [(float(offset), color) for offset, color in stops],
    )


def build_strip(colorscale) -> plt.Figure:
    fig = plt.figure(figsize=(0.22, 2.6), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(
        _STRIP,
        aspect="auto",
        cmap=_colormap(colorscale),
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax.set_axis_off()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, _STRIP.shape[0] - 0.5)
    ax.margins(0)
    fig.patch.set_alpha(0.0)
    return fig


def _write(fig: plt.Figure, stem: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        out = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(
            out,
            dpi=200,
            pad_inches=0,
            facecolor="none",
            edgecolor="none",
            transparent=(ext != "png"),
        )
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    for name, colorscale in SCALES:
        _write(build_strip(colorscale), f"colorscale_{name}")


if __name__ == "__main__":
    main()

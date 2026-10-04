"""Shared matplotlib style for all paper figures.

Global rcParams (TrueType-embedded fonts, sizes), the dashed grey grid, the
Tableau10 palette, a white text halo, and a PDF+PNG saver.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
})

GRID_KW = dict(linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
HALO = [patheffects.Stroke(linewidth=2, foreground="white"), patheffects.Normal()]

# Tableau10
TAB10 = {
    "blue":   "#4e79a7",
    "orange": "#f28e2b",
    "red":    "#e15759",
    "teal":   "#76b7b2",
    "green":  "#59a14f",
    "purple": "#b07aa1",
    "brown":  "#9c755f",
    "pink":   "#ff9da7",
    "yellow": "#edc948",
    "gray":   "#bab0ac",
}

# Stable per-dataset identity colors
DATASET_COLORS = {
    "ppmatch":   TAB10["green"],
    "valentine": TAB10["orange"],
    "hdxsm":     TAB10["purple"],
    "oc3-fo":    TAB10["red"],
}

NICE = {"ppmatch": "PowerPlantBench", "valentine": "Valentine",
        "hdxsm": "HDXSM", "oc3-fo": "OC3-FO"}


def save(fig, out_dir: Path, name: str) -> None:
    """Vector PDF (tight) + 200-dpi PNG (tight)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out_dir / name}.pdf/.png")

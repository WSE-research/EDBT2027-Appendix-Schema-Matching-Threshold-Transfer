"""Figure 2 (fig:capacity) — MCC gain from the oracle threshold over tau=0
versus model size, per family. Lines + ±1σ band (std over the four target
datasets) for multi-size families; diamond + errorbar for single-size models.

Run:  python -m analysis.make_fig_capacity
Out:  figures/fig_capacity_trend.{pdf,png}   (file name as referenced by the tex)
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import paper_style as ps
from .paper_style import plt

FAM = {  # model -> (family, params in B)
    "gemma-3-4b": ("Gemma-3", 4), "gemma-3-12b": ("Gemma-3", 12), "gemma-3-27b": ("Gemma-3", 27),
    "gemma-4-31b": ("Gemma-4", 31),
    "qwen3-8b": ("Qwen3", 8), "qwen3-14b": ("Qwen3", 14), "qwen3-32b": ("Qwen3", 32),
    "llama-3.3-70b": ("Llama-3.3", 70),
}
FAMCOL = {"Gemma-3": ps.TAB10["blue"], "Qwen3": ps.TAB10["orange"],
          "Gemma-4": ps.TAB10["green"], "Llama-3.3": ps.TAB10["red"]}
MULTI = ["Gemma-3", "Qwen3"]          # families with >1 size (can draw a line)
SINGLE = ["Gemma-4", "Llama-3.3"]


def per_model_gain() -> dict:
    """model -> dict(params, fam, dorc = ΔMCC oracle−τ0 per dataset (array/4))."""
    out = {}
    for m in C.models():
        if m not in FAM:        # e.g. phi-4 when included via env — no family slot
            continue
        dorc = []
        for d in C.DATASETS:
            rows = C.load(m, d)
            t0 = C.mcc_at(rows, 0.0)
            orc = C.mcc_at(rows, C.best_tau(rows, C.mcc_at))
            dorc.append(orc - t0)
        out[m] = dict(params=FAM[m][1], fam=FAM[m][0], dorc=np.array(dorc))
    return out


def fam_series(data: dict, fam: str):
    ms = sorted([m for m in data if data[m]["fam"] == fam], key=lambda m: data[m]["params"])
    x = np.array([data[m]["params"] for m in ms])
    y = np.array([data[m]["dorc"].mean() for m in ms])
    e = np.array([data[m]["dorc"].std() for m in ms])
    return x, y, e


def main() -> None:
    data = per_model_gain()
    # slightly smaller fonts than the global style (column-width figure)
    with plt.rc_context({"axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5}):
        fig, ax = plt.subplots(figsize=(3.6, 2.15))
        for fam in MULTI:
            x, y, e = fam_series(data, fam)
            c = FAMCOL[fam]
            ax.plot(x, y, "-o", color=c, label=fam, lw=1.8, ms=5, zorder=3)
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.18, zorder=1)
        for fam in SINGLE:
            x, y, e = fam_series(data, fam)
            ax.errorbar(x, y, yerr=e, fmt="D", color=FAMCOL[fam], ms=6,
                        capsize=3, label=fam, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([4, 8, 12, 27, 70])
        ax.set_xticklabels(["4", "8", "12", "27", "70"])
        ax.minorticks_off()
        ax.grid(True, **ps.GRID_KW)
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_xlabel("Model size (B params, log)")
        ax.set_ylabel(r"$\Delta$MCC (oracle $-$ $\tau{=}0$)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        ps.save(fig, C.FIG_DIR, "fig_capacity_trend")

    # console digest of the in-text numbers (§5.4 capacity claims)
    for fam in MULTI + SINGLE:
        x, y, _ = fam_series(data, fam)
        pts = ", ".join(f"{int(b)}B {g:+.2f}" for b, g in zip(x, y))
        print(f"  ΔMCC(oracle−τ0) {fam}: {pts}")


if __name__ == "__main__":
    main()

"""Figure 2 (fig:capacity) — MCC gain from the oracle threshold over tau=0
versus model size, per family. Lines + ±1σ band (std over the four target
datasets) for multi-size families; diamond + errorbar for single-size models.

Run:  python -m src.analysis.make_fig_capacity
Out:  results/plots/fig_capacity_trend.{pdf,png}   (file name as referenced by the tex)
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
        if m not in FAM:        # unknown model from a custom re-run — no family slot
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
    # Printed size: the paper includes this figure at 0.72 columnwidth of the A4
    # EDBT template (column 232.75 pt), so it is built at exactly that width
    # and prints at scale 1.0 with 7-7.5 pt text.
    width_in = 0.72 * (489.50787 - 24) / 2 / 72.27
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig, ax = plt.subplots(figsize=(width_in, 1.36))
        for fam in MULTI:
            x, y, e = fam_series(data, fam)
            c = FAMCOL[fam]
            ax.plot(x, y, "-o", color=c, label=fam, lw=1.5, ms=3.5, zorder=3)
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.18, zorder=1)
        for fam in SINGLE:
            x, y, e = fam_series(data, fam)
            ax.errorbar(x, y, yerr=e, fmt="D", color=FAMCOL[fam], ms=4,
                        capsize=2, elinewidth=1, label=fam, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([4, 8, 12, 27, 70])
        ax.set_xticklabels(["4", "8", "12", "27", "70"])
        ax.minorticks_off()
        ax.grid(True, **ps.GRID_KW)
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_xlabel("model size (B params, log)")
        ax.set_ylabel(r"$\Delta$MCC (oracle $-$ $\tau{=}0$)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="upper right", handlelength=1.4,
                  borderaxespad=0.2, labelspacing=0.3)
        fig.tight_layout(pad=0.3)
        ps.save(fig, C.FIG_DIR, "fig_capacity_trend")

    # console digest of the in-text numbers (§5.4 capacity claims)
    for fam in MULTI + SINGLE:
        x, y, _ = fam_series(data, fam)
        pts = ", ".join(f"{int(b)}B {g:+.2f}" for b, g in zip(x, y))
        print(f"  ΔMCC(oracle−τ0) {fam}: {pts}")

    # per-model raw values (promised by the Figure-2 caption)
    md = ["# Capacity trend — per-model raw values (Figure 2)", "",
          "ΔMCC = MCC at the in-dataset MCC-optimal threshold minus MCC at "
          "τ=0, per dataset; mean±std over the four datasets is what Figure 2 "
          "plots.", "",
          "| Model | Family | Params (B) | " +
          " | ".join(f"ΔMCC {C.SHORT[d]}" for d in C.DATASETS) +
          " | mean | std |",
          "|---|---|---|" + "---|" * (len(C.DATASETS) + 2)]
    for m in sorted(data, key=lambda m: (data[m]["fam"], data[m]["params"])):
        d = data[m]
        md.append(f"| {m} | {d['fam']} | {d['params']} | " +
                  " | ".join(f"{v:+.3f}" for v in d["dorc"]) +
                  f" | {d['dorc'].mean():+.3f} | {d['dorc'].std():.3f} |")
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "capacity_per_model.md").write_text("\n".join(md) + "\n",
                                                     encoding="utf-8")
    print(f"  wrote {C.OUT_DIR / 'capacity_per_model.md'}")


if __name__ == "__main__":
    main()

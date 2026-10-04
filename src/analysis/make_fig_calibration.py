"""Figure 1 (fig:calibration) — error rate among accepted matches vs verbalized
confidence, per dataset, pooled over the eight paper models (grey lines:
individual models). 1x4 small multiples, dotted 10% error guide.

Run:  python -m src.analysis.make_fig_calibration
Out:  results/plots/fig2_calibration.{pdf,png}   (file name as referenced by the tex)
"""
from __future__ import annotations

from . import common as C
from .paper_style import plt, GRID_KW, DATASET_COLORS, NICE


def err_by_conf(rows) -> dict[int, float | None]:
    """Error rate among accepted matches, by integer confidence bin (0..10)."""
    out = {}
    for c in range(11):
        sub = [r for r in rows if r["predicted_target"] is not None
               and r["confidence"] is not None and int(round(r["confidence"])) == c]
        if not sub:
            out[c] = None
            continue
        wrong = sum(1 for r in sub if not (r["gt_target"] is not None
                    and r["predicted_target"] == r["gt_target"]))
        out[c] = wrong / len(sub)
    return out


def pooled_err_by_conf(dataset: str, models: list[str]) -> dict[int, float | None]:
    """Counts pooled over models per confidence bin (not a mean of rates)."""
    bins = {c: [0, 0] for c in range(11)}   # conf -> [wrong, total accepted]
    for m in models:
        for r in C.load(m, dataset):
            if r["predicted_target"] is None or r["confidence"] is None:
                continue
            c = max(0, min(10, int(round(r["confidence"]))))
            wrong = not (r["gt_target"] is not None
                         and r["predicted_target"] == r["gt_target"])
            bins[c][1] += 1
            bins[c][0] += int(wrong)
    return {c: (w / t if t else None) for c, (w, t) in bins.items()}


# Printed size: the paper includes this figure at 0.93 textwidth of the A4
# EDBT template (textwidth 489.5 pt), so it is built at exactly that width
# and prints at scale 1.0 with 7-8.5 pt text.
PRINT_WIDTH_IN = 0.93 * 489.50787 / 72.27


def main() -> None:
    M = C.models()
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, axes = plt.subplots(1, 4, figsize=(PRINT_WIDTH_IN, 1.30),
                                 sharex=True, sharey=True)
        confs = list(range(11))
        for ax, d in zip(axes, C.DATASETS):
            for m in M:                                   # faint per-model curves
                eb = err_by_conf(C.load(m, d))
                xs = [c for c in confs if eb[c] is not None]
                ys = [eb[c] for c in xs]
                ax.plot(xs, ys, "-", color="gray", lw=0.5, alpha=0.30, zorder=2)
            pooled = pooled_err_by_conf(d, M)
            xs = [c for c in confs if pooled[c] is not None]
            ys = [pooled[c] for c in xs]
            ax.plot(xs, ys, "o-", color=DATASET_COLORS[d], lw=1.5, ms=2.8, zorder=4)
            ax.axhline(0.10, ls=":", color="black", lw=0.8, alpha=0.6)
            ax.set_title(NICE[d], fontsize=8.5, color=DATASET_COLORS[d],
                         fontweight="bold", pad=3)
            ax.set_xlabel("verbalized confidence")
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.grid(True, **GRID_KW)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("error rate among\naccepted matches")
        axes[0].text(0.3, 0.13, "10% error", fontsize=7, color="black", alpha=0.7)
        fig.tight_layout(pad=0.3, w_pad=0.8)
        from .paper_style import save
        save(fig, C.FIG_DIR, "fig2_calibration")


if __name__ == "__main__":
    main()

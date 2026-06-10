"""Figure 1 (fig:calibration) — error rate among accepted matches vs verbalized
confidence, per dataset, pooled over the eight paper models (grey lines:
individual models). 1x4 small multiples, dotted 10% error guide.

Run:  python -m analysis.make_fig_calibration
Out:  figures/fig2_calibration.{pdf,png}   (file name as referenced by the tex)
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


def main() -> None:
    M = C.models()
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.88), sharex=True, sharey=True)
    confs = list(range(11))
    for ax, d in zip(axes, C.DATASETS):
        for m in M:                                   # faint per-model curves
            eb = err_by_conf(C.load(m, d))
            xs = [c for c in confs if eb[c] is not None]
            ys = [eb[c] for c in xs]
            ax.plot(xs, ys, "-", color="gray", lw=0.7, alpha=0.30, zorder=2)
        pooled = pooled_err_by_conf(d, M)
        xs = [c for c in confs if pooled[c] is not None]
        ys = [pooled[c] for c in xs]
        ax.plot(xs, ys, "o-", color=DATASET_COLORS[d], lw=2.2, ms=4, zorder=4)
        ax.axhline(0.10, ls=":", color="black", lw=0.9, alpha=0.6)
        ax.set_title(NICE[d], fontsize=10, color=DATASET_COLORS[d], fontweight="bold")
        ax.set_xlabel("verbalized confidence", fontsize=9)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("error rate among accepted")
    axes[0].text(0.3, 0.12, "10% error", fontsize=7, color="black", alpha=0.7)
    fig.tight_layout()
    from .paper_style import save
    save(fig, C.FIG_DIR, "fig2_calibration")


if __name__ == "__main__":
    main()

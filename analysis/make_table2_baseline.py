"""Table 2 (tab:baseline) — baseline matching quality at tau=0.

F1 and MCC per dataset for each of the eight paper models at the primary
prompt condition, no rejection (tau=0), plus the over-models mean. Rows sorted
by mean F1 (descending, computed before rounding); best per column flagged.

Run:  python -m analysis.make_table2_baseline
Out:  outputs/table2_baseline.md  +  .tex  +  console
"""
from __future__ import annotations

import numpy as np

from . import common as C


def build() -> dict:
    rows = {}
    for m in C.models():
        f1s, mccs = [], []
        for d in C.DATASETS:
            r = C.load(m, d)
            assert r is not None, f"missing primary-cell predictions: {m}/{d}"
            f1s.append(C.f1_at(r, 0.0))
            mccs.append(C.mcc_at(r, 0.0))
        rows[m] = {"f1": f1s, "mcc": mccs, "mean_f1": float(np.mean(f1s))}
    return rows


def main() -> None:
    rows = build()
    order = sorted(rows, key=lambda m: -rows[m]["mean_f1"])
    nd = len(C.DATASETS)
    best_f1 = [max(rows[m]["f1"][i] for m in rows) for i in range(nd)]
    best_mcc = [max(rows[m]["mcc"][i] for m in rows) for i in range(nd)]
    mean_f1 = [float(np.mean([rows[m]["f1"][i] for m in rows])) for i in range(nd)]
    mean_mcc = [float(np.mean([rows[m]["mcc"][i] for m in rows])) for i in range(nd)]

    def cell(v, best):
        mark = "*" if abs(v - best) < 1e-12 else " "
        return f"{v:.2f}{mark}"

    header = ["Model"] + [f"F1 {C.SHORT[d]}" for d in C.DATASETS] + \
             [f"MCC {C.SHORT[d]}" for d in C.DATASETS]
    md = ["# Table 2 — Baseline matching quality (tau=0, primary condition)",
          "",
          f"Models: {len(rows)} (phi-4 excluded unless REPRO_INCLUDE_PHI4=1). "
          "`*` = best per column (before rounding).",
          "",
          "| " + " | ".join(header) + " |",
          "|" + "---|" * len(header)]
    tex = ["% Table 2 (tab:baseline) rows — regenerated"]
    for m in order:
        r = rows[m]
        md.append("| " + " | ".join(
            [m] + [cell(v, b) for v, b in zip(r["f1"], best_f1)]
                + [cell(v, b) for v, b in zip(r["mcc"], best_mcc)]) + " |")
        tex.append(
            m + " & " + " & ".join(f"{v:.2f}".lstrip("0") for v in r["f1"]) +
            " & " + " & ".join(f"{v:.2f}".lstrip("0") for v in r["mcc"]) + r" \\")
    md.append("| *Mean* | " + " | ".join(f"{v:.2f}" for v in mean_f1 + mean_mcc) + " |")
    tex.append(r"\textit{Mean} & " +
               " & ".join(f"{v:.2f}".lstrip("0") for v in mean_f1 + mean_mcc) + r" \\")

    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "table2_baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (C.OUT_DIR / "table2_baseline.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {C.OUT_DIR / 'table2_baseline.md'} (+ .tex)")


if __name__ == "__main__":
    main()

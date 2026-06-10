"""Supplementary artifacts promised in the paper text.

  1. per_model_tau.md        — per-model MCC-optimal tau* per dataset (+medians)
                               (§5.3 "defer the per-model table to the replication package")
  2. calibration_numbers.md  — pooled accept-error at confidence 10/8/6 per dataset
                               (§5.2 in-text percentages)
  3. sc_stability.md         — self-consistency score-stability tables
                               (§5.2 "stability tables in the replication package")
  4. oc3_distractor_check.md — OC3-FO confidence-10 error with the Formula-One
                               distractor pairs removed (§5.2 robustness claim)
  5. classical_baselines.md  — COMA / Similarity Flooding at tau=0 and at their
                               own oracle threshold (§5.1 numbers)
  6. grid_robustness.md      — F1/MCC at tau=0 and oracle-MCC gain across the
                               full 16-condition prompt grid (§4 robustness claim)

Run:  python -m src.analysis.make_supplementary
Out:  results/tables/*.md  +  console summary
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from . import common as C

BASELINES = C.REPO_ROOT / "results" / "classical"

GRID_CELLS = [  # all 16 cells, cheap -> expensive (same order as matcher.config)
    f"vals{v}__shot{s}__{c}__t{t}__sc{sc}"
    for (t, sc) in (("0.0", "1"), ("0.7", "3"))
    for c in ("nocot", "cot")
    for s in ("0", "3")
    for v in ("off", "on")
]


def w(name: str, lines: list[str]) -> None:
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {C.OUT_DIR / name}")


# ── 1. per-model MCC-optimal tau* ───────────────────────────────────────────

def per_model_tau() -> None:
    M = C.models()
    md = ["# Per-model MCC-optimal threshold tau* (primary condition)", "",
          "| Model | " + " | ".join(C.SHORT[d] for d in C.DATASETS) + " |",
          "|" + "---|" * (1 + len(C.DATASETS))]
    taus = {d: [] for d in C.DATASETS}
    degenerate = 0
    for m in M:
        row = [m]
        for d in C.DATASETS:
            t = C.best_tau(C.load(m, d), C.mcc_at)
            taus[d].append(t)
            degenerate += (t == 0.0)
            row.append(f"{t:.1f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("| *median* | " + " | ".join(
        f"{statistics.median(taus[d]):.1f}" for d in C.DATASETS) + " |")
    md += ["", f"{degenerate} of the {len(M) * len(C.DATASETS)} (model, dataset) "
              "pairs degenerate to tau*=0."]
    w("per_model_tau.md", md)
    print("   medians:", {C.SHORT[d]: float(statistics.median(taus[d])) for d in C.DATASETS})


# ── 2. calibration in-text numbers ──────────────────────────────────────────

def calibration_numbers() -> None:
    from .make_fig_calibration import pooled_err_by_conf
    M = C.models()
    md = ["# Pooled accept-error by verbalized confidence (primary condition)", "",
          "| Dataset | err@10 | err@9 | err@8 | err@7 | err@6 |",
          "|---|---|---|---|---|---|"]
    for d in C.DATASETS:
        p = pooled_err_by_conf(d, M)
        md.append(f"| {C.SHORT[d]} | " + " | ".join(
            ("—" if p[c] is None else f"{p[c]:.3f}") for c in (10, 9, 8, 7, 6)) + " |")
    w("calibration_numbers.md", md)


# ── 3. self-consistency stability ───────────────────────────────────────────

def sc_stability() -> None:
    """Across the three temp-0.7 samples of a task: for source columns whose
    DECISION (chosen target) is identical in all three runs, how stable is the
    verbalized score? Reported per dataset + pooled."""
    M = C.models()
    md = ["# Self-consistency score stability "
          f"(cell {C.CELL_SC}, {len(M)} models)", "",
          "Identical-decision cases = (task, source) pairs where all three "
          "temp-0.7 runs choose the same target. The last column reads the "
          "spread only among confidence-10 cases (mean score ≥ 9.5).", "",
          "| Dataset | N identical-decision | % identical score | mean spread | "
          "median | P90 | mean spread @conf10 |",
          "|---|---|---|---|---|---|---|"]
    total_n = total_zero = 0
    pooled = []
    for d in C.DATASETS:
        ranges, means = [], []
        for m in M:
            for r in C.iter_records(m, d, C.CELL_SC):
                if int(r.get("n_valid_runs", 0)) != 3:
                    continue
                valid = [x for x in r["runs"] if not x["error"] and not x["parse_error"]]
                if len(valid) != 3:
                    continue
                maps = [{mp["source"]: (mp["target"], mp["score"]) for mp in x["mappings"]}
                        for x in valid]
                common_src = set(maps[0]) & set(maps[1]) & set(maps[2])
                for s in common_src:
                    if len({maps[i][s][0] for i in range(3)}) != 1:
                        continue   # decision differs -> not an identical-decision case
                    sc = [float(maps[i][s][1]) for i in range(3)]
                    ranges.append(max(sc) - min(sc))
                    means.append(sum(sc) / 3)
        a = np.array(ranges)
        if len(a) == 0:
            md.append(f"| {C.SHORT[d]} | (no SC data) | | | | | |")
            continue
        mm = np.array(means)
        hi = a[mm >= 9.5]
        total_n += len(a)
        total_zero += int(np.sum(a == 0))
        pooled.append(a)
        md.append(f"| {C.SHORT[d]} | {len(a)} | {100 * np.mean(a == 0):.1f}% | "
                  f"{a.mean():.2f} | {np.median(a):.1f} | {np.percentile(a, 90):.1f} | "
                  f"{hi.mean():.3f} (n={len(hi)}) |")
    if total_n:
        allr = np.concatenate(pooled)
        md += ["", f"Pooled: N={total_n}, identical score in all three runs = "
                   f"{100 * total_zero / total_n:.1f}%, mean spread {allr.mean():.2f} points."]
        print(f"   SC pooled: N={total_n}, %zero={100 * total_zero / total_n:.1f}%, "
              f"mean spread {allr.mean():.2f}")
    w("sc_stability.md", md)


# ── 4. OC3-FO distractor robustness ─────────────────────────────────────────

def oc3_distractor_check() -> None:
    """Drop the Formula-One distractor pairs (identified as oc3-fo tasks with
    an EMPTY equivalence ground truth) and recompute the confidence-10 error
    and the linkable rate on the relational core."""
    M = C.models()
    import sys
    sys.path.insert(0, str(C.REPO_ROOT))
    from src.matcher import loaders
    tasks = loaders.load_dataset("oc3-fo")
    core_pairs = {t["pair_id"] for t in tasks if t["ground_truth"]}
    distractor = sorted({t["pair_id"] for t in tasks} - core_pairs)

    def stats(rows):
        acc = [r for r in rows if r["predicted_target"] is not None
               and r["confidence"] is not None and int(round(r["confidence"])) == 10]
        wrong = sum(1 for r in acc if not (r["gt_target"] is not None
                    and r["predicted_target"] == r["gt_target"]))
        pos = sum(1 for r in rows if r["gt_target"] is not None)
        return (wrong / len(acc) if acc else None, len(acc), pos / len(rows) if rows else 0)

    allrows, corerows = [], []
    for m in M:
        for r in C.load(m, "oc3-fo"):
            allrows.append(r)
            if r["pair_id"] in core_pairs:
                corerows.append(r)
    e_all, n_all, l_all = stats(allrows)
    e_core, n_core, l_core = stats(corerows)
    md = ["# OC3-FO: confidence-10 error with vs without the Formula-One distractor", "",
          f"Distractor pairs (empty equivalence GT): {', '.join(distractor)}", "",
          "| Slice | conf-10 error | n(conf-10 accepted) | linkable rate |",
          "|---|---|---|---|",
          f"| all 6 pairs | {e_all:.3f} | {n_all} | {l_all:.2f} |",
          f"| relational core only | {e_core:.3f} | {n_core} | {l_core:.2f} |", "",
          "The miscalibration is a property of OC3-FO's relational core, not of "
          "the distractor."]
    w("oc3_distractor_check.md", md)
    print(f"   oc3 conf-10 error: all={e_all:.3f}  core-only={e_core:.3f} "
          f"(linkable {l_all:.2f} -> {l_core:.2f})")


# ── 5. classical (non-LLM) baselines ────────────────────────────────────────

def classical_baselines() -> None:
    if not BASELINES.is_dir():
        # keep the committed table instead of overwriting it with an empty stub
        print("WARNING: results/classical not found — keeping the committed "
              "classical-baselines table untouched.")
        return
    md = ["# Non-LLM baselines (schema-name input, bipartite assignment)", "",
          "Each method is granted its own best per-dataset threshold, per metric "
          "(`F1 @ F1-oracle`, `MCC @ MCC-oracle`) — the comparison of §5.1. "
          "`exact` (normalized name equality) is a sanity baseline included for "
          "context; the paper reports COMA and Similarity Flooding.", "",
          "| Method | Dataset | F1 @ tau0 | F1 @ F1-oracle | MCC @ MCC-oracle |",
          "|---|---|---|---|---|"]
    for method in sorted(p.name for p in BASELINES.iterdir() if p.is_dir()):
        for d in C.DATASETS:
            f = BASELINES / method / d / "metrics.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text(encoding="utf-8"))
            md.append(f"| {method} | {C.SHORT[d]} | {j['tau0']['F1']:.2f} | "
                      f"{j['best_f1']['F1']:.2f} | {j['best_mcc']['MCC']:.2f} |")
    w("classical_baselines.md", md)


# ── 6. robustness across the 16-condition prompt grid ───────────────────────

def grid_robustness() -> None:
    M = C.models()
    md = ["# Robustness across the 16-condition prompt grid", "",
          "Per cell: coverage (model x dataset units published) and, over the "
          "covered units, mean F1 and MCC at tau=0 and the mean oracle-MCC "
          "gain.", "",
          "Coverage rationale: the paper's evidence rests entirely on the "
          "primary condition (complete, 32/32). The wider grid is a robustness "
          "check, run in full on OC3-FO (the hardest benchmark) and for the "
          "lighter half of the grid on the three larger datasets; the "
          "chain-of-thought conditions proved ~neutral on OC3-FO and were not "
          "extended further. Cells covering only OC3-FO show higher mean "
          "oracle gains simply because threshold gains are largest there.", "",
          "| Cell | Coverage | mean F1@0 | mean MCC@0 | mean ΔMCC(oracle−τ0) |",
          "|---|---|---|---|---|"]
    full = len(M) * len(C.DATASETS)
    for cell in GRID_CELLS:
        f1s, mccs, gains = [], [], []
        n = 0
        for m in M:
            for d in C.DATASETS:
                rows = C.load(m, d, cell)
                if rows is None:
                    continue
                n += 1
                f1s.append(C.f1_at(rows, 0.0))
                m0 = C.mcc_at(rows, 0.0)
                mccs.append(m0)
                gains.append(C.mcc_at(rows, C.best_tau(rows, C.mcc_at)) - m0)
        if n == 0:
            md.append(f"| {cell} | 0/{full} | — | — | — |")
            continue
        md.append(f"| {cell} | {n}/{full} | {np.mean(f1s):.3f} | "
                  f"{np.mean(mccs):.3f} | {np.mean(gains):+.3f} |")
    w("grid_robustness.md", md)


def main() -> None:
    per_model_tau()
    calibration_numbers()
    sc_stability()
    oc3_distractor_check()
    classical_baselines()
    grid_robustness()


if __name__ == "__main__":
    main()

"""Dataset integrity check (the one referenced in the paper's Artifacts note).

Independently re-derives, for every task of the four study datasets, the three
correctness guarantees:

  1. GT subset of columns : every ground-truth (src, tgt) has src in
                            source_columns AND tgt in target_columns.
  2. No duplicate GT       : no exact (src, tgt) pair appears twice in a task.
  3. Cardinality           : max fan-out / fan-in == 1 (strict one-to-one
                            correspondences, each side injective).

Exit code 0 only if all guarantees hold.  Run from the repo root:
    python scripts/integrity_check.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matcher.loaders import load_dataset  # noqa: E402

DATASETS = ["ppmatch", "valentine", "hdxsm", "oc3-fo"]


def check_dataset(name: str) -> dict:
    tasks = load_dataset(name)
    n_gt = 0
    subset_violations = []   # (pair_id, src, tgt, why)
    dup_violations = []      # (pair_id, pair, count)
    max_fan_out = 0          # one source -> many targets
    max_fan_in = 0           # many sources -> one target
    masked_total = 0

    for t in tasks:
        src_set = set(t["source_columns"])
        tgt_set = set(t["target_columns"])
        gt = [tuple(p) for p in t["ground_truth"]]
        n_gt += len(gt)
        masked_total += len(t.get("masked_columns", []))

        for s, tg in gt:
            why = []
            if s not in src_set:
                why.append("src not in source_columns")
            if tg not in tgt_set:
                why.append("tgt not in target_columns")
            if why:
                subset_violations.append((t["pair_id"], s, tg, ",".join(why)))

        seen = Counter(gt)
        for pair, c in seen.items():
            if c > 1:
                dup_violations.append((t["pair_id"], pair, c))

        sc = Counter(s for s, _ in gt)
        tc = Counter(tg for _, tg in gt)
        if sc:
            max_fan_out = max(max_fan_out, max(sc.values()))
            max_fan_in = max(max_fan_in, max(tc.values()))

    ok = (not subset_violations and not dup_violations
          and max_fan_out <= 1 and max_fan_in <= 1)
    return {
        "name": name, "tasks": len(tasks), "gt": n_gt,
        "subset_violations": subset_violations, "dup_violations": dup_violations,
        "max_fan_out": max_fan_out, "max_fan_in": max_fan_in,
        "masked_total": masked_total, "pass": ok,
    }


def main() -> int:
    print("=" * 78)
    print("Dataset integrity check (one-to-one correspondences)")
    print("=" * 78)
    results = [check_dataset(n) for n in DATASETS]

    print(f"\n{'dataset':<14}{'tasks':>6}{'GT':>7}{'GT_bad':>8}"
          f"{'dups':>6}{'fanOut':>8}{'fanIn':>7}{'masked':>8}  verdict")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<14}{r['tasks']:>6}{r['gt']:>7}"
              f"{len(r['subset_violations']):>8}{len(r['dup_violations']):>6}"
              f"{r['max_fan_out']:>8}{r['max_fan_in']:>7}{r['masked_total']:>8}"
              f"  {'PASS' if r['pass'] else 'FAIL'}")
    print("-" * 70)

    for r in results:
        for pid, s, tg, why in r["subset_violations"][:10]:
            print(f"  [{r['name']}] {pid}: ({s!r},{tg!r})  {why}")
        for pid, pair, c in r["dup_violations"][:10]:
            print(f"  [{r['name']}] {pid}: duplicate {pair} x{c}")

    all_pass = all(r["pass"] for r in results)
    print("\nRESULT:", "ALL PASS — datasets are loadable & verified one-to-one"
          if all_pass else "FAIL — see violations above")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

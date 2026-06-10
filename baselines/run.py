"""Run a non-LLM baseline over the 4 study datasets and write comparable results.

Outputs (per method, per dataset):
  results/<method>/<dataset>/predictions.jsonl   # same schema as the LLM runs
  results/<method>/<dataset>/metrics.json        # tau0 / best-F1 / best-MCC

Pooled per-dataset metrics are computed over all source-column decisions,
exactly like the LLM evaluation (micro-averaged).

The results used in the paper (§5.1, "Against classical matchers") ship in
results/ — re-running is optional and needs the extra deps in requirements.txt.
"""
import argparse
import json
import os

import config
import data_adapter
import matcher
import metrics


def run_dataset(method, dataset, mode, model_name):
    tasks = data_adapter.load_tasks(dataset)
    all_rows = []
    for t in tasks:
        view = data_adapter.to_view(t)
        all_rows.extend(matcher.predict(view, method, model_name, mode))

    out_dir = os.path.join(config.RESULTS_DIR, method, dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = metrics.summarize(all_rows, config.THRESHOLDS)
    summary.update(method=method, dataset=dataset, assign=mode,
                   n_tasks=len(tasks),
                   model=model_name if method == "embedding" else None)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Non-LLM schema-matching baselines")
    ap.add_argument("--method",
                    choices=["exact", "embedding",
                             "coma", "similarity_flooding", "cupid"],
                    default="coma")
    ap.add_argument("--assign", choices=["greedy", "bipartite"], default="bipartite")
    ap.add_argument("--datasets", nargs="*", default=config.DATASETS)
    ap.add_argument("--cap", type=int, default=None,
                    help="override per-dataset task cap for a quick run")
    args = ap.parse_args()

    if args.cap is not None:
        for d in config.CAPS:
            config.CAPS[d] = args.cap

    print(f"method={args.method} assign={args.assign} datasets={args.datasets}")
    print(f"{'dataset':10} {'n':>5} | {'F1@0':>6} {'MCC@0':>6} {'err@0':>6} | "
          f"{'bestF1':>6} {'bestMCC':>7}")
    for d in args.datasets:
        s = run_dataset(args.method, d, args.assign, config.EMBED_MODEL)
        t0, bf, bm = s["tau0"], s["best_f1"], s["best_mcc"]
        print(f"{d:10} {s['n_sources']:>5} | {t0['F1']:>6.3f} {t0['MCC']:>6.3f} "
              f"{t0['accept_error']:>6.3f} | {bf['F1']:>6.3f} {bm['MCC']:>7.3f}")


if __name__ == "__main__":
    main()

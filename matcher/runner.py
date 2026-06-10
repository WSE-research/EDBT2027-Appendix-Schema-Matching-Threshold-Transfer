"""Orchestrator: datasets x grid cells x models x tasks (full-schema matcher).

For every (dataset, cell, model) it runs the full-schema forward matcher over
every task concurrently and writes, per run dir under results/:
  - records.jsonl    : one line per TASK — FULL detail (messages, every run's
                       raw response/usage/cost/latency, vote aggregation,
                       per-source decisions). Maximal logging.
  - predictions.jsonl: compact one-liner per scored SOURCE COLUMN
                       (gt, pred, confidence, agreement, correct).
  - run_meta.json    : model, cell, totals, metrics, stop info.
  - metrics.json     : micro P/R/F1.

The published logs of the paper live in runs/ (same layout, gzipped); fresh
re-runs land in results/ so they never mix with the published data.

Ordering: datasets small -> large (valentine LAST), cells cheap -> expensive,
then models. So the three small datasets complete fully before the big one.

Resume-safe: re-running skips TASKS already in records.jsonl.
Graceful stop: an out-of-credits (HTTP 402) error stops the whole experiment
cleanly (partial results flushed, run_meta marks the stop) — just re-run to
resume from where it left off.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config, data, matcher, metrics, openrouter
from .openrouter import InsufficientCreditsError


class StopExperiment(Exception):
    """Raised to stop ALL loops gracefully (out of credits). Run is resumable."""


class _JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        self._fh.close()


def _done_tasks(records_path: Path) -> set[str]:
    """pair_ids already fully recorded (task-level resume)."""
    done = set()
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["pair_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _few_shot(all_tasks: list[dict], current_pair: str, k: int) -> list[dict]:
    """Deterministically pick k OTHER tasks from the same dataset as examples."""
    pool = [t for t in all_tasks if t["pair_id"] != current_pair]
    if not pool or k <= 0:
        return []
    rng = random.Random(f"{config.SEED}:{current_pair}")
    rng.shuffle(pool)
    return pool[:k]


def run_unit(model: dict, dataset: str, ablation: dict, log, stop_event: threading.Event) -> dict:
    dirname = config.run_dirname(ablation)
    run_dir = config.RESULTS_DIR / model["name"] / dataset / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    preds_path = run_dir / "predictions.jsonl"

    cap = config.cap_for(dataset)
    tasks, total_available = data.load_tasks(dataset, cap)
    done = _done_tasks(records_path)
    units = [t for t in tasks if t["pair_id"] not in done]

    n_shot = ablation["n_shot"]
    label = f"{model['name']} | {dataset} | {dirname}"
    log(f"  [{label}] {len(tasks)}/{total_available} tasks, "
        f"{len(units)} to run ({len(done)} done){' [CAPPED]' if cap else ''}")

    rec_w = _JsonlWriter(records_path)
    pred_w = _JsonlWriter(preds_path)
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "reasoning_tokens": 0, "cost": 0.0, "latency_ms": 0.0,
              "hallucinated": 0, "parse_errors": 0, "errored_tasks": 0,
              "masked_excluded": 0}
    consec_fail = 0
    aborted = False
    stopped_reason = None

    def work(task):
        if stop_event.is_set():
            return None
        shots = _few_shot(tasks, task["pair_id"], n_shot) if n_shot else []
        res = matcher.match_task(
            task, use_values=ablation["use_values"], use_cot=ablation["use_cot"],
            shot_examples=shots, temp=ablation["temp"], sc_runs=ablation["sc_runs"],
            model_id=model["id"],
            price_in=model.get("price_in", 0.0), price_out=model.get("price_out", 0.0),
        )
        return task, res

    try:
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
            futures = {ex.submit(work, t): t for t in units}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    out = fut.result()
                except InsufficientCreditsError as e:
                    stop_event.set()
                    stopped_reason = "insufficient_credits"
                    log(f"    !! OUT OF CREDITS — graceful stop. "
                        f"Partial results saved; re-run to resume. ({e})")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                if out is None:        # skipped because stop_event was set
                    continue
                task, res = out
                gt_map = {s: t for s, t in task["ground_truth"]}
                masked = task.get("masked_columns") or []

                # accumulate usage across this task's runs — count EVERY billed
                # call: the final call (att['usage']) AND each retried/failed
                # billed call captured in the transport trace (truncation/empty).
                for r in res["runs"]:
                    for att in r.get("attempts", []):
                        billed = []
                        if att.get("usage"):
                            billed.append(att["usage"])
                        for tr in (att.get("transport_trace") or []):
                            if tr.get("usage"):
                                billed.append(tr["usage"])
                        for u in billed:
                            totals["calls"] += 1
                            totals["prompt_tokens"] += u.get("prompt_tokens", 0)
                            totals["completion_tokens"] += u.get("completion_tokens", 0)
                            totals["reasoning_tokens"] += u.get("reasoning_tokens", 0)
                            totals["cost"] += u.get("cost", 0) or 0
                    if r.get("latency_ms"):
                        totals["latency_ms"] += r["latency_ms"]
                if res["any_hallucinated"]:
                    totals["hallucinated"] += 1
                if any(r["parse_error"] for r in res["runs"]):
                    totals["parse_errors"] += 1

                # per-source predictions (exclude oc3-fo masked columns). An
                # all_failed task (0 valid runs) produced NO usable answer — do
                # NOT score its columns as abstentions (that would deflate
                # recall); log it for audit + count it errored, write no preds.
                scored = []
                if res["all_failed"]:
                    totals["errored_tasks"] += 1
                    consec_fail += 1
                else:
                    consec_fail = 0
                    for d in res["per_source"]:
                        src = d["source"]
                        if metrics.is_masked(src, masked):
                            totals["masked_excluded"] += 1
                            continue
                        gt = gt_map.get(src)
                        pred_tgt = d["predicted_target"]
                        row = {"dataset": dataset, "model": model["name"],
                               "pair_id": task["pair_id"], "source_column": src,
                               "gt_target": gt, "predicted_target": pred_tgt,
                               "confidence": d["confidence"],
                               "agreement": d["agreement"],
                               "votes": d["votes"],
                               "correct": (pred_tgt == gt)}
                        scored.append(row)
                        pred_w.write(row)

                rec_w.write({
                    "dataset": dataset, "model": model["name"],
                    "pair_id": task["pair_id"], "source_id": task["source_id"],
                    "target_id": task["target_id"],
                    "ground_truth": task["ground_truth"],
                    "masked_columns": masked,
                    "n_scored": len(scored),
                    **{k: v for k, v in res.items() if k != "messages"},
                    "messages": res["messages"],
                })

                if i % 25 == 0:
                    log(f"    {i}/{len(units)} tasks | cost so far ${totals['cost']:.3f}")
                if consec_fail >= 15:
                    aborted = True
                    log("    !! 15 consecutive fully-failed tasks — aborting this "
                        "unit (likely a model-specific format/availability issue).")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    finally:
        rec_w.close()
        pred_w.close()

    # metrics over predictions whose TASK is committed (pair_id present in
    # records.jsonl = the commit marker), deduped by (pair_id, source_column)
    # keeping the LAST. This ignores orphaned/superseded prediction rows from a
    # task that crashed before its record was written (then re-ran on resume).
    committed = _done_tasks(records_path)
    by_key: dict[tuple, dict] = {}
    if preds_path.exists():
        with open(preds_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("pair_id") not in committed:
                    continue
                by_key[(r.get("pair_id"), r.get("source_column"))] = r
    m = metrics.summarize(list(by_key.values()))

    meta = {
        "dataset": dataset, "model": model, "ablation": ablation,
        "run_dirname": dirname,
        "cap": {"cap_for_dataset": cap, "tasks_used": len(tasks),
                "tasks_available": total_available,
                "capped": cap is not None and total_available > cap},
        "totals": {**totals, "cost": round(totals["cost"], 4),
                   "latency_ms": round(totals["latency_ms"], 1)},
        "metrics": m,
        "aborted": aborted,
        "stopped_reason": stopped_reason,
    }
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"  -> {label}: F1={m['f1']} P={m['precision']} R={m['recall']} "
        f"(TP{m['tp']}/FP{m['fp']}/FN{m['fn']}/TN{m['tn']}) cost=${totals['cost']:.3f}")

    if stopped_reason:
        raise StopExperiment(stopped_reason)
    return meta


def run_all(datasets=None, ablations=None, models=None, ablation_outer=False):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(config.LOG_DIR / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    datasets = config.order_datasets(datasets or config.DATASETS)  # valentine LAST
    ablations = ablations or config.ABLATIONS
    models = models or config.models()
    stop_event = threading.Event()

    # fail fast: build the client once (raises immediately on a missing API key)
    # and avoid a lazy-init race across worker threads.
    openrouter.get_client()

    # Iteration order. ablation_outer ("cheap cells first") runs the cheapest
    # cell across ALL datasets (incl. valentine) before the expensive cells
    # -> a complete temp0 grid early. Otherwise each dataset finishes fully
    # before the next (valentine last). Both visit the SAME (model,dataset,cell)
    # run dirs, so resume (records.jsonl) is order-independent — nothing is redone.
    if ablation_outer:
        order = [(ds, ab, m) for ab in ablations for ds in datasets for m in models]
        order_desc = "cheap-cells-first (ablation-outer)"
    else:
        order = [(ds, ab, m) for ds in datasets for ab in ablations for m in models]
        order_desc = "dataset-complete (dataset-outer, valentine last)"

    log(f"=== full-schema matcher run | {len(models)} models x {len(ablations)} "
        f"configs x {len(datasets)} datasets | "
        f"order={order_desc} | workers={config.MAX_WORKERS} ===")
    log(f"    models: {[m['name'] for m in models]}")

    summary = []
    consec_unit_aborts = 0
    try:
        for dataset, ablation, model in order:
            if stop_event.is_set():
                raise StopExperiment("insufficient_credits")
            try:
                meta = run_unit(model, dataset, ablation, log, stop_event)
                summary.append(meta)
                if meta.get("aborted"):     # 15 consecutive fully-failed tasks
                    consec_unit_aborts += 1
                    log(f"  !! unit aborted (15 consecutive fails): "
                        f"{meta['run_dirname']}")
                    if consec_unit_aborts >= 2:
                        log("  !! 2 consecutive unit aborts — likely systemic "
                            "(auth/model/format). Stopping.")
                        raise StopExperiment("systemic_failure")
                else:
                    consec_unit_aborts = 0
            except StopExperiment:
                raise
            except Exception as e:
                log(f"  !! unit FAILED: {e}\n{traceback.format_exc()}")
                consec_unit_aborts += 1
                if consec_unit_aborts >= 2:
                    raise StopExperiment("systemic_failure")
    except StopExperiment as e:
        log(f"=== STOPPED gracefully ({e}). Re-run to resume. ===")
    finally:
        total_cost = sum(s["totals"]["cost"] for s in summary)
        log(f"=== {len(summary)} run dirs completed, total cost ${total_cost:.3f} ===")
        log_fh.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full-schema LLM matcher experiment")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of datasets")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--only-temp0", action="store_true",
                    help="run only the temp=0 configs (cheap baseline)")
    ap.add_argument("--primary-only", action="store_true",
                    help="run only the paper's primary cell "
                         "(zero-shot, no values, no CoT, greedy)")
    ap.add_argument("--cheap-first", action="store_true",
                    help="run cheapest cell across ALL datasets first "
                         "(ablation-outer) instead of finishing each dataset first")
    ap.add_argument("--cot", choices=["both", "on", "off"], default="both",
                    help="restrict the CoT axis for THIS run only (reversible: "
                         "existing results are untouched, re-running later with "
                         "a different value fills the rest)")
    ap.add_argument("--only-sc", action="store_true",
                    help="run only the temp0.7 self-consistency cells (complement of --only-temp0)")
    ap.add_argument("--shot", choices=["both", "0", "3"], default="both",
                    help="restrict the few-shot axis (reversible)")
    ap.add_argument("--values", choices=["both", "on", "off"], default="both",
                    help="restrict the sample-values axis (reversible)")
    args = ap.parse_args()

    abls = list(config.ABLATIONS)
    if args.primary_only:
        abls = [a for a in abls if a["temp"] == 0.0 and not a["use_cot"]
                and a["n_shot"] == 0 and not a["use_values"]]
    if args.only_temp0:
        abls = [a for a in abls if a["temp"] == 0.0]
    if args.only_sc:
        abls = [a for a in abls if a["sc_runs"] > 1]
    if args.cot == "off":
        abls = [a for a in abls if not a["use_cot"]]
    elif args.cot == "on":
        abls = [a for a in abls if a["use_cot"]]
    if args.shot == "0":
        abls = [a for a in abls if a["n_shot"] == 0]
    elif args.shot == "3":
        abls = [a for a in abls if a["n_shot"] == 3]
    if args.values == "on":
        abls = [a for a in abls if a["use_values"]]
    elif args.values == "off":
        abls = [a for a in abls if not a["use_values"]]
    sel_models = None
    if args.models:
        sel_models = [m for m in config.MODELS if m["name"] in set(args.models)]
    run_all(datasets=args.datasets, ablations=abls, models=sel_models,
            ablation_outer=args.cheap_first)

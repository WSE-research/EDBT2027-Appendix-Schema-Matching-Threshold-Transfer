"""Smoke test (lean — the only test). No API calls by default.

Checks end to end WITHOUT spending tokens:
  1. data bridge: all 4 datasets load from the bundled data/, task shape is
     sane, sample values are extractable.
  2. prompt: full-schema messages build for several grid cells (CoT on/off,
     values on/off, 3-shot) and contain the expected structure.
  3. parser: a canned reason-first response is parsed + grounded; hallucinated
     column names are dropped + counted.
  4. aggregation: self-consistency majority vote reduces to per-source best.
  5. metrics: TP/FP/FN/TN/WRONG classification + masking behave correctly.
  6. runs/: the published logs are present and readable (one cell spot-check).

  --live additionally fires ONE real LLM call (model-served check + 1 match_task).

Exit 0 on success, 1 on failure.  Run:  python -m matcher.smoke_test [--live]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys

from . import config, data, matcher, metrics, prompt


def _check_data() -> dict:
    first = {}
    for ds in config.DATASETS:
        tasks, total = data.load_tasks(ds, 2)
        assert tasks, f"{ds}: no tasks"
        t = tasks[0]
        for key in ("pair_id", "source_id", "target_id", "source_columns",
                    "target_columns", "ground_truth"):
            assert key in t, f"{ds}: task missing '{key}'"
        assert t["source_columns"] and t["target_columns"], f"{ds}: empty schema"
        sv = sum(1 for _ in (t.get("source_values") or {}))
        print(f"[ok] {ds}: {len(tasks)}/{total} tasks; task0 '{t['pair_id']}' "
              f"src={len(t['source_columns'])} tgt={len(t['target_columns'])} "
              f"gt={len(t['ground_truth'])} src_values_for={sv}")
        first[ds] = t
    return first


def _check_prompt(task: dict) -> None:
    shots = [task]  # reuse the same task as a trivial example
    for use_cot in (False, True):
        for use_values in (False, True):
            for n_shot in (0, 3):
                msgs = prompt.build_messages(
                    task, use_values=use_values, use_cot=use_cot,
                    shot_examples=(shots * 3) if n_shot else None,
                )
                assert len(msgs) == 2 and msgs[0]["role"] == "system"
                sys_p = msgs[0]["content"]
                assert ('"mappings"' in sys_p), "system prompt missing output format"
                assert ("step by step" in sys_p) == use_cot, "CoT instruction mismatch"
                user = msgs[1]["content"]
                assert task["source_id"] in user and task["target_id"] in user
                if n_shot:
                    assert "worked examples" in user, "few-shot block missing"
    print("[ok] prompt: full-schema messages build across CoT/values/shot cells")


def _check_parser_and_aggregation(task: dict) -> None:
    sc = task["source_columns"]
    tc = task["target_columns"]
    real = (sc[0], tc[0])
    # reason-first CoT style: prose, then a trailing JSON object; one hallucinated target.
    content = (
        "Let me think step by step. The first source column clearly matches.\n"
        '{"mappings": [{"source": "%s", "target": "%s", "score": 9},'
        ' {"source": "%s", "target": "__NOT_A_REAL_TARGET__", "score": 8}]}'
        % (real[0], real[1], sc[1] if len(sc) > 1 else sc[0])
    )
    p = prompt.parse_response(content, sc, tc)
    assert not p["parse_error"], "should parse reason-first JSON"
    assert p["hallucinated"] == 1, f"expected 1 hallucination, got {p['hallucinated']}"
    assert any(m["source"] == real[0] and m["target"] == real[1] for m in p["mappings"])
    print(f"[ok] parser: grounded {len(p['mappings'])} mapping(s), "
          f"dropped {p['hallucinated']} hallucinated")

    # aggregation: 3 runs, the real pair in all 3 (survives), a noise pair in 1 (dropped)
    good = {"mappings": [{"source": real[0], "target": real[1], "score": 9.0}],
            "hallucinated": 0, "parse_error": False, "error": None}
    noise = {"mappings": [{"source": real[0], "target": real[1], "score": 7.0},
                          {"source": sc[1] if len(sc) > 1 else sc[0],
                           "target": tc[1] if len(tc) > 1 else tc[0], "score": 6.0}],
             "hallucinated": 0, "parse_error": False, "error": None}
    runs = [dict(good, run_idx=0), dict(good, run_idx=1), dict(noise, run_idx=2)]
    agg = matcher._aggregate(runs, sc, sc_runs=3)
    chosen = next(d for d in agg["per_source"] if d["source"] == real[0])
    assert chosen["predicted_target"] == real[1], "majority pair should win"
    assert chosen["votes"] == 3, f"expected 3 votes, got {chosen['votes']}"
    dropped = next((d for d in agg["per_source"]
                    if d["source"] == (sc[1] if len(sc) > 1 else "___")), None)
    if dropped is not None:
        assert dropped["predicted_target"] is None, "1-vote pair should be dropped (<2)"
    print(f"[ok] aggregation: majority vote kept the 3/3 pair (conf={chosen['confidence']}), "
          f"dropped the 1/3 pair")

    # threshold clamp: 1 valid run + 2 errored runs at sc_runs=3 -> the single
    # valid run's pair must SURVIVE (not be dropped by a fixed threshold of 2).
    erro = {"mappings": [], "hallucinated": 0, "parse_error": True, "error": "boom"}
    runs2 = [dict(good, run_idx=0), dict(erro, run_idx=1), dict(erro, run_idx=2)]
    agg2 = matcher._aggregate(runs2, sc, sc_runs=3)
    assert agg2["n_valid_runs"] == 1 and not agg2["all_failed"]
    keep = next(d for d in agg2["per_source"] if d["source"] == real[0])
    assert keep["predicted_target"] == real[1], "1-of-3-valid pair must survive (clamped threshold)"
    # all-failed: every run errors -> all_failed True, every decision None
    agg3 = matcher._aggregate([dict(erro, run_idx=i) for i in range(3)], sc, sc_runs=3)
    assert agg3["all_failed"] and all(d["predicted_target"] is None for d in agg3["per_source"])
    print("[ok] degraded/failed: 1/3-valid survives (clamp); all-failed -> all None (runner skips)")


def _check_metrics() -> None:
    preds = [
        {"gt_target": "A", "predicted_target": "A"},     # TP
        {"gt_target": None, "predicted_target": "B"},    # FP
        {"gt_target": "C", "predicted_target": None},    # FN
        {"gt_target": None, "predicted_target": None},   # TN
        {"gt_target": "D", "predicted_target": "X"},     # WRONG -> FP+FN
    ]
    m = metrics.summarize(preds)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 2, 2, 1), m
    assert metrics.is_masked("MYSQL:orders.ship_addr", ["MYSQL:orders.ship_addr"])
    assert metrics.is_masked("orders.ship_addr", ["MYSQL:orders.ship_addr"])
    assert not metrics.is_masked("orders.id", ["MYSQL:orders.ship_addr"])
    print(f"[ok] metrics: P/R/F1 + masking correct (TP{m['tp']}/FP{m['fp']}/"
          f"FN{m['fn']}/TN{m['tn']}, F1={m['f1']})")


def _check_published_runs() -> None:
    """Spot-check the published logs: the primary cell of one model/dataset."""
    cell = "valsoff__shot0__nocot__t0.0__sc1__scopeoff"
    base = config.RUNS_DIR
    if not base.is_dir():
        print("[warn] runs/ not present — skipping published-log check")
        return
    models = sorted(p.name for p in base.iterdir() if p.is_dir())
    assert models, "runs/ exists but holds no model directories"
    probe = base / models[0] / "oc3-fo" / cell
    preds = probe / "predictions.jsonl.gz"
    recs = probe / "records.jsonl.gz"
    assert preds.exists(), f"missing {preds}"
    assert recs.exists(), f"missing {recs}"
    with gzip.open(preds, "rt", encoding="utf-8") as f:
        row = json.loads(next(f))
    for key in ("pair_id", "source_column", "gt_target", "predicted_target", "confidence"):
        assert key in row, f"predictions row missing '{key}'"
    with gzip.open(recs, "rt", encoding="utf-8") as f:
        rec = json.loads(next(f))
    assert rec.get("messages"), "records row missing the prompt messages"
    assert rec.get("runs") and rec["runs"][0].get("attempts"), "records row missing runs/attempts"
    assert "raw_response" in rec["runs"][0]["attempts"][0], "records row missing raw_response"
    print(f"[ok] runs/: {len(models)} models published; spot-checked "
          f"{models[0]}/oc3-fo/{cell} (prompt + raw response + scores present)")


def _check_live(task: dict) -> int:
    from . import openrouter
    model = config.MODELS[0]
    try:
        openrouter.chat([{"role": "user", "content": 'Reply with JSON only: {"ok":1}'}],
                        model["id"], 0.0)
        print(f"[ok] live: model served: {model['id']}")
    except Exception as e:
        print(f"[fail] live: model '{model['id']}' not served: {e!r}")
        return 1
    res = matcher.match_task(task, use_values=False, use_cot=False,
                             temp=0.0, sc_runs=1, model_id=model["id"])
    run0 = res["runs"][0]
    if run0.get("error"):
        print(f"[fail] live: match_task errored: {run0['error']}")
        return 1
    print(f"[ok] live: match_task('{task['pair_id']}') -> {len(res['mappings'])} "
          f"mappings, {res['n_valid_runs']} valid run(s), halluc={res['any_hallucinated']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also fire ONE real LLM call")
    args = ap.parse_args()
    try:
        first = _check_data()
        _check_prompt(first["ppmatch"])
        _check_parser_and_aggregation(first["ppmatch"])
        _check_metrics()
        _check_published_runs()
        if args.live:
            rc = _check_live(first["oc3-fo"])
            if rc:
                return rc
    except Exception as e:
        import traceback
        print(f"[fail] {e!r}")
        traceback.print_exc()
        return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

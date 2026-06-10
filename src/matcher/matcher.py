"""Full-schema forward matcher for ONE task (one schema pair), with optional
self-consistency.

One LLM call presents the whole source schema vs the whole target schema and
returns a list of scored mappings. For sc_runs>1 (temp>0) the task is matched
several times and the (source,target) pairs are combined by majority vote;
per-run scores and vote agreement are kept as calibration signals. The mapping
set is then reduced to a per-source single-best decision (highest-scoring
surviving target, else abstain) for the P/R/F1 metric.

Every individual LLM call (raw messages + raw response + usage + cost + latency)
is returned for logging. An out-of-credits error is NOT swallowed — it
propagates so the runner can stop the experiment gracefully and resumably.
"""
from __future__ import annotations

from collections import defaultdict

from . import config, openrouter, prompt
from .openrouter import InsufficientCreditsError


def match_task(
    task: dict,
    *,
    use_values: bool = False,
    use_cot: bool = False,
    shot_examples: list[dict] | None = None,
    temp: float = 0.0,
    sc_runs: int = 1,
    model_id: str | None = None,
    price_in: float = 0.0,
    price_out: float = 0.0,
) -> dict:
    model_id = model_id or config.MODELS[0]["id"]
    messages = prompt.build_messages(
        task, use_values=use_values, use_cot=use_cot,
        shot_examples=shot_examples, value_limit=config.MAX_VALUES_PER_COLUMN,
    )
    src_cols = task["source_columns"]
    tgt_cols = task["target_columns"]

    runs = [_one_run(messages, src_cols, tgt_cols, model_id, temp, i, price_in, price_out)
            for i in range(sc_runs)]

    agg = _aggregate(runs, src_cols, sc_runs)
    return {
        "n_source": len(src_cols),
        "n_target": len(tgt_cols),
        "use_values": use_values,
        "use_cot": use_cot,
        "n_shot": len(shot_examples or []),
        "temperature": temp,
        "sc_runs": sc_runs,
        "model_id": model_id,
        "messages": messages,          # identical across runs; logged once
        **agg,
        "runs": runs,
    }


def _one_run(messages, src_cols, tgt_cols, model_id, temp, run_idx,
             price_in=0.0, price_out=0.0) -> dict:
    """One self-consistency run = up to REPARSE_RETRIES+1 calls until parseable.

    Out-of-credits propagates (graceful stop); all other errors are logged with
    their full transport trace (incl. billed-but-failed retries).
    """
    attempts = []
    parsed = None
    for _ in range(config.REPARSE_RETRIES + 1):
        try:
            call = openrouter.chat(messages, model_id, temp, price_in, price_out)
        except InsufficientCreditsError:
            raise  # graceful-stop signal — do not swallow
        except Exception as e:  # exhausted retries / 4xx / transport
            attempts.append({
                "error": repr(e),
                "transport_trace": getattr(e, "transport_trace", None),
                "transport_retries": getattr(e, "transport_retries", None),
            })
            break
        parsed = prompt.parse_response(call["content"], src_cols, tgt_cols)
        attempts.append({
            "raw_response": call["content"],          # text content
            "response_dump": call.get("raw"),         # full provider response (routing/id)
            "served_model_id": call.get("model_id"),
            "finish_reason": call["finish_reason"],
            "usage": call["usage"],
            "latency_ms": call["latency_ms"],
            "transport_retries": call["retries"],
            "transport_trace": call.get("transport_trace"),
            "parsed": parsed,
        })
        if not parsed["parse_error"]:
            break

    last = attempts[-1] if attempts else {}
    return {
        "run_idx": run_idx,
        "mappings": (parsed or {}).get("mappings", []),
        "hallucinated": (parsed or {}).get("hallucinated", 0),
        "parse_error": (parsed or {}).get("parse_error", True) if parsed else True,
        "error": last.get("error"),
        "attempts": attempts,
        "usage": last.get("usage", {}),
        "latency_ms": last.get("latency_ms"),
    }


def _aggregate(runs: list[dict], src_cols: list[str], sc_runs: int) -> dict:
    """Majority-vote the (source,target) pairs across runs, then reduce to a
    per-source single-best decision.

    threshold = floor(sc_runs/2)+1 (majority); sc_runs=1 -> threshold 1.
    confidence = mean score over the runs backing the surviving pair (0-10).
    agreement  = votes / number of non-failed runs.
    """
    valid = [r for r in runs if not r["error"] and not r["parse_error"]]
    n_valid = len(valid)
    all_failed = n_valid == 0

    # votes + scores per (source, target) pair
    votes: dict[tuple, int] = defaultdict(int)
    scores: dict[tuple, list] = defaultdict(list)
    for r in valid:
        # within a run a source may appear once; dedupe by (s,t) to be safe
        seen = set()
        for m in r["mappings"]:
            key = (m["source"], m["target"])
            if key in seen:
                continue
            seen.add(key)
            votes[key] += 1
            scores[key].append(m["score"])

    # majority threshold over the runs that ACTUALLY succeeded (not the
    # configured sc_runs): if 2/3 runs failed transiently, the one good run must
    # not be silently dropped (which would score the task as all-abstain).
    threshold = min(sc_runs // 2 + 1, n_valid) if n_valid > 0 else 1
    survivors = {k: v for k, v in votes.items() if v >= threshold}

    # per-source single best among survivors (highest mean score)
    best_by_src: dict[str, dict] = {}
    for (s, t), v in survivors.items():
        mean_score = round(sum(scores[(s, t)]) / len(scores[(s, t)]), 3)
        cand = {"target": t, "confidence": mean_score,
                "agreement": round(v / n_valid, 3) if n_valid else 0.0, "votes": v}
        cur = best_by_src.get(s)
        if cur is None or mean_score > cur["confidence"]:
            best_by_src[s] = cand

    per_source = []
    for s in src_cols:
        b = best_by_src.get(s)
        per_source.append({
            "source": s,
            "predicted_target": b["target"] if b else None,
            "confidence": b["confidence"] if b else None,
            "agreement": b["agreement"] if b else (0.0 if not all_failed else None),
            "votes": b["votes"] if b else 0,
        })

    aggregated_mappings = [
        {"source": s, "target": t, "confidence": round(sum(scores[(s, t)]) / len(scores[(s, t)]), 3),
         "agreement": round(v / n_valid, 3) if n_valid else 0.0, "votes": v}
        for (s, t), v in survivors.items()
    ]

    return {
        "per_source": per_source,
        "mappings": aggregated_mappings,
        "n_valid_runs": n_valid,
        "any_hallucinated": any(r["hallucinated"] for r in runs),
        "all_failed": all_failed,
    }

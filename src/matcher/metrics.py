"""Per-source-column classification + micro P/R/F1 for a full-schema run.

The full-schema matcher is reduced to a per-source single-best decision (see
matcher._aggregate), so each scored source column has 0 or 1 predicted target
and 0 or 1 ground-truth target — mapping cleanly to TP / FP / FN / TN.
Confidence is NOT used here; it is logged for the threshold study (analysis/).

oc3-fo subsumption columns (task['masked_columns'], "schema:table.attribute")
are EXCLUDED from scoring — a matcher that correctly finds a subsumption link
must not be charged a false positive (equivalence-only scoring convention).
"""
from __future__ import annotations


def _norm(s: str) -> str:
    # strip an optional "schema:" prefix and lower-case for tolerant matching
    return str(s).split(":", 1)[-1].strip().casefold()


def is_masked(source_col: str, masked_columns) -> bool:
    if not masked_columns:
        return False
    if source_col in masked_columns:
        return True
    c = _norm(source_col)
    for m in masked_columns:
        mn = _norm(m)
        if c == mn or c.endswith("." + mn) or mn.endswith("." + c):
            return True
    return False


def classify(gt_target: str | None, pred_target: str | None) -> str:
    if gt_target is None and pred_target is None:
        return "TN"
    if gt_target is None and pred_target is not None:
        return "FP"
    if gt_target is not None and pred_target is None:
        return "FN"
    if pred_target == gt_target:
        return "TP"
    return "WRONG"   # predicted a target, but the wrong one: counts as FP and FN


def summarize(predictions: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for p in predictions:
        k = classify(p.get("gt_target"), p.get("predicted_target"))
        if k == "TP":
            tp += 1
        elif k == "FP":
            fp += 1
        elif k == "FN":
            fn += 1
        elif k == "TN":
            tn += 1
        else:  # WRONG
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_decisions": len(predictions),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

"""Metrics — identical confusion semantics to analysis/common.py so the
non-LLM baseline is directly comparable to the LLM matcher.

Decision per source column at threshold tau:
  kept = predicted_target is not None and confidence >= tau
  kept   & pred==gt (gt not None)      -> TP
  kept   & otherwise                   -> FP  (+FN if a gt target existed)
  not kept & gt exists                 -> FN
  not kept & gt is None                -> TN
"""
import math


def confusion(rows, tau):
    tp = fp = fn = tn = 0
    for r in rows:
        gt, pred, conf = r["gt_target"], r["predicted_target"], r["confidence"]
        kept = pred is not None and conf is not None and conf >= tau
        if kept:
            if gt is not None and pred == gt:
                tp += 1
            else:
                fp += 1
                if gt is not None:
                    fn += 1
        else:
            if gt is not None:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def metrics_at(rows, tau):
    tp, fp, fn, tn = confusion(rows, tau)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    err = fp / (tp + fp) if (tp + fp) else 0.0   # accept-error = 1 - P
    return dict(tau=tau, tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F1=F1,
                MCC=mcc, accept_error=err, n_accept=tp + fp)


def sweep(rows, thresholds):
    return [metrics_at(rows, t) for t in thresholds]


def best(sw, key):
    return max(sw, key=lambda m: m[key])


def summarize(rows, thresholds):
    """Report tau=0 and the best-F1 / best-MCC operating points (like the LLM)."""
    sw = sweep(rows, thresholds)
    return {
        "n_sources": len(rows),
        "tau0": metrics_at(rows, 0.0),
        "best_f1": best(sw, "F1"),
        "best_mcc": best(sw, "MCC"),
    }

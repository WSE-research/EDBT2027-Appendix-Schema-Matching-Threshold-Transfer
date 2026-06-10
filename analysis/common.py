"""Shared loading + metric core for all analysis scripts.

Single source of truth for the decision semantics used by every table and
figure (identical to the engine that produced the paper's numbers):

Decision semantics (predictions.jsonl, one row per source column):
  predicted_target is None -> abstain (no-match); confidence is None there.
  gt_target is None        -> no true match exists (negative column).
At threshold tau a predicted match is KEPT iff confidence >= tau, else demoted
to no-match. Confusion counts:
  TP = kept match & predicted==gt ;  FP = kept match & predicted!=gt
  FN = gt-match exists & not correctly kept ; TN = gt None & nothing kept
A wrong kept match on a column that HAS a gt match counts as BOTH fp and fn.

Data source: the published logs in runs/ by default (gzipped); set the env var
REPRO_RUNS_DIR (or pass --runs-dir where offered) to point at a fresh re-run
(e.g. the results/ directory produced by matcher.runner). The model set is
discovered from the run directories (the paper's M=8).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# Console output contains Unicode (Δ, τ, ±). On Windows a redirected stdout
# defaults to a legacy codepage and would crash — force UTF-8, lossily if needed.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("REPRO_RUNS_DIR", REPO_ROOT / "runs"))
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = REPO_ROOT / "figures"

DATASETS = ["ppmatch", "valentine", "hdxsm", "oc3-fo"]
SHORT = {"ppmatch": "ppm", "valentine": "val", "hdxsm": "hdx", "oc3-fo": "oc3"}

# The paper's primary prompt condition (zero-shot, no values, no CoT, greedy)
# and the self-consistency condition used for the stability analysis.
CELL_PRIMARY = "valsoff__shot0__nocot__t0.0__sc1__scopeoff"
CELL_SC = "valsoff__shot0__nocot__t0.7__sc3__scopeoff"

THRESHOLDS = [round(t, 1) for t in np.arange(0.0, 10.01, 0.5)]


def models() -> list[str]:
    """Model names discovered in the runs dir."""
    if not RUNS_DIR.is_dir():
        raise SystemExit(f"runs directory not found: {RUNS_DIR}")
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())


def _open_maybe_gz(base: Path):
    """Open <base>.gz if present, else <base>. Returns None if neither exists."""
    gz = base.with_suffix(base.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    if base.exists():
        return open(base, encoding="utf-8")
    return None


@lru_cache(maxsize=512)
def load(model: str, dataset: str, cell: str = CELL_PRIMARY) -> tuple | None:
    """Per-source-column prediction rows of one run, or None if absent."""
    fh = _open_maybe_gz(RUNS_DIR / model / dataset / cell / "predictions.jsonl")
    if fh is None:
        return None
    with fh:
        return tuple(json.loads(l) for l in fh)


def iter_records(model: str, dataset: str, cell: str):
    """Yield full task records (prompt + every raw response) of one run."""
    fh = _open_maybe_gz(RUNS_DIR / model / dataset / cell / "records.jsonl")
    if fh is None:
        return
    with fh:
        for line in fh:
            yield json.loads(line)


# --------------------------------------------------------------------------
# metric core
# --------------------------------------------------------------------------

def confusion(rows, tau: float) -> tuple[int, int, int, int]:
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


def prf(rows, tau: float) -> dict:
    tp, fp, fn, tn = confusion(rows, tau)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    ae = fp / (tp + fp) if (tp + fp) else 0.0   # accept-error = 1 - precision
    return dict(tau=tau, tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F1=F1, ae=ae,
                n_accept=tp + fp)


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def mcc_at(rows, tau: float) -> float:
    return mcc(*confusion(rows, tau))


def f1_at(rows, tau: float) -> float:
    return prf(rows, tau)["F1"]


def ae_at(rows, tau: float) -> float:
    return prf(rows, tau)["ae"]


def best_tau(rows, fn=mcc_at) -> float:
    """Lowest threshold that maximises fn (ties resolve to the smaller tau)."""
    bv, bt = -2.0, 0.0
    for t in THRESHOLDS:
        v = fn(rows, t)
        if v > bv + 1e-12:
            bv, bt = v, t
    return bt


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std())

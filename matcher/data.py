"""Task loading + per-column sample values.

Loads the four study datasets (oc3-fo, ppmatch, hdxsm, valentine) through the
bundled loaders (loaders.py, verified by scripts/integrity_check.py) and
enriches each task with up to N sample values per column for the sample-values
grid axis. Value extraction is per-dataset and fully defensive: any failure
degrades that column to name-only, never crashes a run.

Note on the bundled Valentine copy: instance CSVs ship truncated to the first
100 data rows (the originals total ~3 GB). Sample values are drawn from the
first VALUE_SCAN_ROWS=60 rows only, so prompts are identical to those built
from the full download (see data/DATASETS.md).
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from . import config, loaders


def load_tasks(dataset: str, max_tasks: int | None) -> tuple[list[dict], int]:
    """Return (tasks, total_available). Each task gets source_values /
    target_values dicts (possibly empty). Cap is applied deterministically."""
    tasks = loaders.load_dataset(dataset)
    total = len(tasks)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    for t in tasks:
        sv, tv = _sample_values(dataset, t)
        t["source_values"] = sv
        t["target_values"] = tv
    return tasks, total


# --------------------------------------------------------------------------
# sample-value extraction
# --------------------------------------------------------------------------

def _sample_values(dataset: str, task: dict) -> tuple[dict, dict]:
    try:
        if dataset == "ppmatch":
            return _ppmatch_values(task), {}
        if dataset == "valentine":
            return _valentine_values(task)
        if dataset == "hdxsm":
            return _hdxsm_values(task)
        # oc3-fo: schema-only, no instance data
        return {}, {}
    except Exception:
        return {}, {}


def _csv_samples(path: Path, wanted: set[str] | None = None) -> dict:
    """{column: [up to N distinct non-empty sample values]} from the first
    VALUE_SCAN_ROWS rows of a CSV."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    n = config.MAX_VALUES_PER_COLUMN
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if wanted is None or c in wanted]
        seen: dict[str, set] = {c: set() for c in cols}
        for c in cols:
            out[c] = []
        for i, row in enumerate(reader):
            if i >= config.VALUE_SCAN_ROWS:
                break
            for c in cols:
                if len(out[c]) >= n:
                    continue
                v = (row.get(c) or "").strip()
                if v and v != c and v not in seen[c]:
                    seen[c].add(v)
                    out[c].append(v)
    return out


@lru_cache(maxsize=1)
def _ppmatch_metadata() -> dict:
    p = config.DATA_DIR / "ppmatch" / "column_metadata.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ppmatch_values(task: dict) -> dict:
    meta = _ppmatch_metadata().get(task["source_id"], {})
    out = {}
    for col in task["source_columns"]:
        ex = meta.get(col, {}).get("examples") if isinstance(meta.get(col), dict) else None
        if ex:
            out[col] = list(ex)[: config.MAX_VALUES_PER_COLUMN]
    return out


def _valentine_dir(task: dict) -> Path:
    return config.DATA_DIR / "valentine" / "Valentine-datasets" / Path(task["pair_id"])


def _valentine_values(task: dict) -> tuple[dict, dict]:
    d = _valentine_dir(task)
    src = next(iter(d.glob("*_source.csv")), None)
    tgt = next(iter(d.glob("*_target.csv")), None)
    sv = _csv_samples(src, set(task["source_columns"])) if src else {}
    tv = _csv_samples(tgt, set(task["target_columns"])) if tgt else {}
    return sv, tv


def _hdxsm_values(task: dict) -> tuple[dict, dict]:
    d = config.DATA_DIR / "hdxsm" / "hxd_datasets_0.2" / task["pair_id"]
    sv = _csv_samples(d / "Table1.csv", set(task["source_columns"]))
    tv = _csv_samples(d / "Table2.csv", set(task["target_columns"]))
    return sv, tv

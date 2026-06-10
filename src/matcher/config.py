"""Central configuration for the full-schema LLM schema-matching experiment.

Everything tunable lives here. Each prompt-grid cell is encoded into the output
directory name (see run_dirname) so that every axis — sample values, few-shot,
CoT, temperature/self-consistency — is visible at a glance from the folder
layout (and is also written into run_meta.json).

The paper's primary condition is zero-shot / no values / no CoT / greedy
(temperature 0, single sample): cell "valsoff__shot0__nocot__t0.0__sc1".
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env at the repo root (optional convenience; plain env vars also work)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# --------------------------------------------------------------------------
# Paths (all repo-local; the repository is self-contained)
# --------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent              # .../src/matcher
REPO_ROOT = _PKG_DIR.parents[1]                         # repository root

DATA_DIR = REPO_ROOT / "data"                           # bundled datasets
RUNS_DIR = REPO_ROOT / "results" / "llm"                # published, pre-computed logs
RESULTS_DIR = REPO_ROOT / "results" / "rerun"           # fresh re-runs land here
LOG_DIR = REPO_ROOT / "logs"

# --------------------------------------------------------------------------
# Models under test (OpenRouter ids): the paper's eight open-weight, dense
# models (three families, 4B-70B). Cost is read from the API response when
# available; price_in/price_out are a fallback estimate ($/MTok).
# --------------------------------------------------------------------------
MODELS = [
    {"name": "gemma-3-4b",    "id": "google/gemma-3-4b-it",            "price_in": 0.04, "price_out": 0.08},
    {"name": "gemma-3-12b",   "id": "google/gemma-3-12b-it",           "price_in": 0.04, "price_out": 0.13},
    {"name": "gemma-3-27b",   "id": "google/gemma-3-27b-it",           "price_in": 0.08, "price_out": 0.16},
    {"name": "gemma-4-31b",   "id": "google/gemma-4-31b-it",           "price_in": 0.14, "price_out": 0.40},
    {"name": "qwen3-8b",      "id": "qwen/qwen3-8b",                   "price_in": 0.05, "price_out": 0.40},
    {"name": "qwen3-14b",     "id": "qwen/qwen3-14b",                  "price_in": 0.06, "price_out": 0.24},
    {"name": "qwen3-32b",     "id": "qwen/qwen3-32b",                  "price_in": 0.08, "price_out": 0.24},
    {"name": "llama-3.3-70b", "id": "meta-llama/llama-3.3-70b-instruct", "price_in": 0.10, "price_out": 0.32},
]


def models() -> list[dict]:
    """Active model list. MATCHER_MODELS=name1,name2 narrows it (e.g. for a
    quick pilot)."""
    sel = os.environ.get("MATCHER_MODELS", "").strip()
    if not sel:
        return MODELS
    wanted = {s.strip() for s in sel.split(",") if s.strip()}
    return [m for m in MODELS if m["name"] in wanted]


# --------------------------------------------------------------------------
# Datasets (ordered small -> large so the three small datasets complete fully
# before the big one — a crashed/credit-exhausted run still yields complete
# coverage of oc3-fo, ppmatch, hdxsm).
# --------------------------------------------------------------------------
DATASETS = ["oc3-fo", "ppmatch", "hdxsm", "valentine"]


def order_datasets(names) -> list[str]:
    """Force the small -> large order (valentine LAST) regardless of input
    order. Unknown names sort to the end."""
    rank = {d: i for i, d in enumerate(DATASETS)}
    return sorted(names, key=lambda d: rank.get(d, len(rank)))


# --------------------------------------------------------------------------
# Prompt grid (16 cells), ordered cheap -> expensive. Axes:
#   - temperature / self-consistency : 0.0 single run  vs  0.7 + 3-run majority
#   - chain-of-thought (reason-first): off / on
#   - few-shot                       : 0-shot / 3-shot
#   - sample values in the prompt    : off / on
# Self-consistency (temp 0.7 x3) is the dominant cost driver, so it is last.
# --------------------------------------------------------------------------
_TEMP_SC = [(0.0, 1), (0.7, 3)]
_BOOL = [False, True]

ABLATIONS = [
    {"use_values": v, "n_shot": s, "use_cot": c, "temp": t, "sc_runs": sc}
    for (t, sc) in _TEMP_SC          # temp0 (cheap) first
    for c in _BOOL                   # no-CoT before CoT
    for s in (0, 3)                  # 0-shot before 3-shot
    for v in _BOOL                   # values off before on
]
assert len(ABLATIONS) == 16, f"expected 16 grid cells, got {len(ABLATIONS)}"

# --------------------------------------------------------------------------
# Run-size controls. NO CAPPING by default (full task set per dataset).
# MATCHER_MAX_TASKS can cap for a quick debug pilot; if set, the cap is logged
# loudly in run_meta.json and on the console (no silent cap).
# --------------------------------------------------------------------------
_MAX_TASKS_ENV = os.environ.get("MATCHER_MAX_TASKS", "").strip()
MAX_TASKS_PER_DATASET = int(_MAX_TASKS_ENV) if _MAX_TASKS_ENV else None   # None = full


def cap_for(dataset: str) -> int | None:
    return MAX_TASKS_PER_DATASET


FEW_SHOT_K = 3                    # examples drawn when n_shot=3
SEED = 42                         # deterministic few-shot selection
MAX_VALUES_PER_COLUMN = 5         # sample values shown per column when use_values=True
VALUE_SCAN_ROWS = 60              # rows scanned in a raw CSV to collect sample values

# --------------------------------------------------------------------------
# API / retry
# --------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 2048                 # full-schema answer + reasoning
MAX_TOKENS_CEILING = 8192         # truncation -> double up to this
MAX_WORKERS = int(os.environ.get("MATCHER_WORKERS", "8"))
MAX_RETRIES = 5
REPARSE_RETRIES = 2               # extra calls if the response is unparseable JSON
REQUEST_TIMEOUT = 120             # seconds (full-schema + reasoning can be slow)


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("sk-or-..."):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.template to .env and fill "
            "in a real key (only needed to RE-RUN the LLM experiments; the "
            "bundled results/ reproduce everything without a key)."
        )
    return key


def run_dirname(ablation: dict) -> str:
    """Encode the full grid cell into the results directory name."""
    vals = "on" if ablation["use_values"] else "off"
    cot = "cot" if ablation["use_cot"] else "nocot"
    return (f"vals{vals}__shot{ablation['n_shot']}__{cot}"
            f"__t{ablation['temp']}__sc{ablation['sc_runs']}")

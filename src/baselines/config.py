"""Configuration for the non-LLM baseline experiment (COMA / Similarity
Flooding / zero-shot embedding / exact-name)."""
import os
import sys
from pathlib import Path

# The 4 study datasets, loaded via the SAME bundled loaders as the LLM matcher
# so task definitions match exactly.
DATASETS = ["ppmatch", "valentine", "hdxsm", "oc3-fo"]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(Path(HERE).parents[1])
sys.path.insert(0, REPO_ROOT)

# Optional per-dataset task cap (None = all). Valentine is ~18x the others;
# cap it for quick iterations, drop the cap for the final numbers.
CAPS = {
    "ppmatch": None,
    "valentine": None,
    "hdxsm": None,
    "oc3-fo": None,
}

# Zero-shot encoder for --method embedding.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Score-threshold sweep (cosine / similarity in [0,1]); mirrors the LLM tau sweep.
THRESHOLDS = [round(0.02 * i, 2) for i in range(0, 51)]   # 0.00 .. 1.00 step 0.02

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "classical")

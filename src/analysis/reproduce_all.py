"""Reproduce every table and figure of the paper from the published logs in
results/llm/ — no API key, no LLM calls.

Run:  python reproduce.py   (or: python -m src.analysis.reproduce_all)
Out:  results/tables/  +  results/plots/  (fig2_calibration,
      fig_capacity_trend)
"""
from __future__ import annotations

import time

from . import (make_characterization, make_fig_calibration, make_fig_capacity,
               make_supplementary, make_table2_baseline, make_table3_operating_point)
from . import common as C

STEPS = [
    ("Table 1  — benchmark characterization + perturbation stress test",
     make_characterization.main),
    ("Table 2  — baseline F1/MCC at tau=0", make_table2_baseline.main),
    ("Table 3  — operating-point transfer matrix", make_table3_operating_point.main),
    ("Figure 1 — confidence calibration", make_fig_calibration.main),
    ("Figure 2 — capacity trend", make_fig_capacity.main),
    ("Supplementary — tau*, calibration, SC stability, distractor, baselines, grid",
     make_supplementary.main),
]


def main() -> None:
    print(f"Reproducing all paper artifacts from {C.RUNS_DIR}\n"
          f"models: {C.models()}\n")
    t0 = time.time()
    for title, fn in STEPS:
        print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))
        fn()
    print(f"\nDone in {time.time() - t0:.0f}s. Tables/supplementary -> {C.OUT_DIR}, "
          f"figures -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()

"""Single entry point for the replication package.

Default (no flags): regenerate every table and figure of the paper from the
published logs in results/llm/ — no API key, no LLM calls, ~20 seconds.
Outputs: results/tables/ + results/plots/.

--run-experiments: re-run the LLM matching experiments via OpenRouter
(needs OPENROUTER_API_KEY in .env; see README). Fresh runs are written to
results/rerun/ in the same format as the published logs; analyse them with
REPRO_RUNS_DIR=results/rerun python reproduce.py.

Examples:
    python reproduce.py                                   # tables + figures
    python reproduce.py --run-experiments --primary-only  # paper's main condition
    python reproduce.py --run-experiments --models gemma-4-31b --datasets oc3-fo
    python reproduce.py --run-experiments                 # full 16-condition grid
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce the paper's results (default: from published logs)")
    ap.add_argument("--run-experiments", action="store_true",
                    help="re-run the LLM experiments via OpenRouter instead of "
                         "regenerating tables/figures (writes to results/rerun/)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names (with --run-experiments)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of datasets (with --run-experiments)")
    ap.add_argument("--primary-only", action="store_true",
                    help="run only the paper's primary condition "
                         "(zero-shot, no values, greedy)")
    ap.add_argument("--only-temp0", action="store_true",
                    help="run only the temperature-0 conditions (cheap half)")
    args = ap.parse_args()

    if not args.run_experiments:
        from src.analysis import reproduce_all
        reproduce_all.main()
        return

    from src.matcher import config, runner
    abls = list(config.ABLATIONS)
    if args.primary_only:
        abls = [a for a in abls if a["temp"] == 0.0 and not a["use_cot"]
                and a["n_shot"] == 0 and not a["use_values"]]
    if args.only_temp0:
        abls = [a for a in abls if a["temp"] == 0.0]
    models = None
    if args.models:
        models = [m for m in config.MODELS if m["name"] in set(args.models)]
    runner.run_all(datasets=args.datasets, ablations=abls, models=models)


if __name__ == "__main__":
    main()

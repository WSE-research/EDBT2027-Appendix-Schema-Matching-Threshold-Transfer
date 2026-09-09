# Replication Package: Calibrate Once, Match Everywhere? Cross-Dataset Transfer of False-Positive Rejection Thresholds in LLM-Based Schema Matching

This repository contains the complete replication package for the short paper
submitted to EDBT 2027. All experiment results are included — every table and
figure of the paper can be regenerated **without API access**: the published
logs hold the exact prompt messages, every raw model response, and the
per-column decisions with their verbalized confidence, for the paper's
8 models x 4 datasets.

The primary condition — the basis of every number in the paper — is published
complete for all 32 (model, dataset) pairs. The wider 16-condition prompt grid
is a robustness extra, published exactly as far as it was executed (287 of 512
cells — complete on OC3-FO, the hardest benchmark; per-cell coverage and
rationale in `results/tables/grid_robustness.md`).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproduce.py
```

Tables are written to `results/tables/`, figures to `results/plots/`
(~20 seconds, no API key needed). `notebooks/reproduce.ipynb` renders
everything inline (needs `pip install jupyter`).

Verify the bundled datasets:

```bash
python scripts/integrity_check.py    # GT⊆columns, no duplicates, strict 1:1
```

## Re-Running Experiments

To re-run the LLM matching experiments from scratch:

```bash
cp .env.template .env
# Edit .env and add your OpenRouter API key
python -m src.matcher.smoke_test          # end-to-end checks, zero API calls
python -m src.matcher.smoke_test --live   # + exactly ONE real call
python reproduce.py --run-experiments --primary-only
```

**Estimated cost:** a few USD for the 8-model primary condition; on the order
of tens of USD for the full 16-condition grid (self-consistency triples most
calls), depending on provider pricing. Real cost is read from the API and
stored per call in `records.jsonl.gz` (`usage.cost`).
**Estimated time:** a few hours per full grid pass, depending on rate limits.

To narrow a re-run:

```bash
python reproduce.py --run-experiments --models gemma-4-31b --datasets oc3-fo
python reproduce.py --run-experiments --only-temp0
MATCHER_MAX_TASKS=2 python reproduce.py --run-experiments   # 2-task pilot (bash)
# PowerShell: $env:MATCHER_MAX_TASKS="2"; python reproduce.py --run-experiments
```

Notes on re-running:

- Fresh runs land in `results/rerun/` (gitignored, same layout and file
  formats); the published logs in `results/llm/` are never touched. Analyse a
  re-run with `REPRO_RUNS_DIR=results/rerun python reproduce.py`
  (PowerShell: `$env:REPRO_RUNS_DIR="results/rerun"; python reproduce.py`).
- **Resume-safe:** tasks already in a run dir's `records.jsonl` are skipped;
  interrupting and re-running is always safe. Running out of OpenRouter
  credits stops the experiment gracefully; re-run to resume.
- LLM sampling is not bit-deterministic, so fresh responses will differ
  slightly from the published logs even at temperature 0; `results/llm/` is
  the exact data behind the paper's numbers.
- Windows: when **redirecting** console output to a file, set `PYTHONUTF8=1`
  (the reports contain Δ/τ characters; interactive consoles are fine).

## Directory Structure

```
├── reproduce.py              # Single entry point (tables/figures + optional --run-experiments)
├── requirements.txt          # Python dependencies
├── .env.template             # API key template (only for --run-experiments)
│
├── data/                     # The four benchmarks (see data/DATASETS.md for
│   ├── ppmatch/              #   provenance, licenses, and what was changed)
│   ├── valentine/            #   slim derivative (headers + first 100 records/CSV)
│   ├── hdxsm/
│   └── oc3-fo/
│
├── src/                      # Source code
│   ├── matcher/              # Full-schema LLM matcher: loaders, prompt builder +
│   │                         #   parser, OpenRouter client, self-consistency
│   │                         #   aggregation, resumable runner, smoke test
│   ├── analysis/             # One script per paper artifact + reproduce_all
│   └── baselines/            # Non-LLM baselines (COMA, Similarity Flooding)
│
├── scripts/
│   └── integrity_check.py    # Verifies the bundled datasets
│
├── notebooks/
│   └── reproduce.ipynb       # Runs the analysis and renders everything inline
│
└── results/                  # Pre-computed experiment results
    ├── llm/                  # The published logs: 8 models × 4 datasets × grid cells
    │   └── {model}/{dataset}/{cell}/
    │       ├── records.jsonl.gz      # every prompt + every raw response
    │       ├── predictions.jsonl.gz  # per-source-column decisions + confidence
    │       ├── metrics.json          # micro P/R/F1 of that run
    │       └── run_meta.json         # model, condition, totals (see caveats)
    ├── classical/            # COMA / Similarity Flooding / exact results
    ├── tables/               # Regenerated tables + supplementary (markdown/tex)
    ├── plots/                # Regenerated figures (PDF + PNG)
    └── rerun/                # Fresh --run-experiments output (gitignored)
```

What `python reproduce.py` regenerates:

| Artifact | Paper | Output |
|---|---|---|
| Table 1 — the four measured benchmark properties, plus Valentine's name-corruption stress test and the task-level rank correlations | §4, §5.3 | `results/tables/characterization.json` |
| Table 2 — baseline F1/MCC at τ=0, per model x dataset | §5.1 | `results/tables/table2_baseline.{md,tex}` |
| Table 3 — transfer matrix: MCC + accept-error at the source's MCC-optimal cutoff, mean±std, Δ vs τ=0 | §5.3 | `results/tables/table3_operating_point.{md,tex}` |
| Figure 1 — confidence calibration | §5.2 | `results/plots/fig2_calibration.{pdf,png}` (file name is historical — this is the paper's **Figure 1**) |
| Figure 2 — capacity trend (ΔMCC of the oracle threshold vs model size) | §5.4 | `results/plots/fig_capacity_trend.{pdf,png}` + per-model values in `results/tables/capacity_per_model.md` |
| per-model MCC-optimal τ* (+ medians) | §5.3 | `results/tables/per_model_tau.md` |
| pooled accept-error at confidence 10/9/8/7/6 | §5.2 | `results/tables/calibration_numbers.md` |
| self-consistency score-stability tables | §5.2 | `results/tables/sc_stability.md` |
| OC3-FO distractor-removal robustness check | §5.2 | `results/tables/oc3_distractor_check.md` |
| classical-matcher comparison (COMA, Sim. Flooding) | §5.1 | `results/tables/classical_baselines.md` |
| robustness across the 16-condition prompt grid (+ coverage) | §4 | `results/tables/grid_robustness.md` |

Each analysis script also runs standalone, e.g.
`python -m src.analysis.make_table3_operating_point`.

## Datasets

| Dataset | Domain | Tasks | GT correspondences |
|---|---|---|---|
| PowerPlantBench (`ppmatch`) | energy (multilingual EN/DE) | 24 | 209 |
| Valentine (`valentine`) | multi-domain | 544 | 8,644 |
| HDXSM (`hdxsm`) | humanitarian | 160 | 2,097 |
| OC3-FO (`oc3-fo`) | relational + distractor | 6 | 39 |

See `data/DATASETS.md` for provenance, licenses, the Valentine slim-copy note
(prompt-lossless, see there), and the OC3-FO subsumption-masking convention.

## Models Evaluated

**LLMs — open-weight (8):** Gemma-3-4B, Gemma-3-12B, Gemma-3-27B, Gemma-4-31B,
Qwen3-8B, Qwen3-14B, Qwen3-32B, Llama-3.3-70B — accessed through OpenRouter
(`src/matcher/config.py` holds the exact model IDs and per-MTok prices).
The analysis discovers the model set from the run directories, so a custom
re-run with a different model list is analysed the same way.

**Classical baselines (2 + sanity):** COMA, Similarity Flooding (schema-name
input, bipartite assignment, each granted its own oracle threshold), plus a
normalized-name-equality sanity baseline. See `src/baselines/README.md`.

## Results File Format

`results/llm/{model}/{dataset}/{cell}/` — the cell name encodes the prompt
condition: `vals{on|off}__shot{0|3}__{cot|nocot}__t{0.0|0.7}__sc{1|3}`
(sample values / few-shot / chain-of-thought / temperature / self-consistency
runs). The paper's primary condition is `valsoff__shot0__nocot__t0.0__sc1`.

**`records.jsonl.gz`** — one line per matching task:

```jsonc
{
  "pair_id": "...", "source_id": "...", "target_id": "...",
  "ground_truth": [["src_col", "tgt_col"], ...],
  "masked_columns": [...],            // oc3-fo subsumption masking
  "messages": [...],                  // the EXACT prompt (system + user)
  "runs": [                           // 1 (greedy) or 3 (self-consistency)
    {"attempts": [{"raw_response": "...",        // full model output text
                   "finish_reason": "...", "usage": {...tokens, cost...},
                   "latency_ms": ..., "parsed": {...}}],
     "mappings": [{"source": ..., "target": ..., "score": ...}], ...}
  ],
  "per_source": [                     // aggregated decision per source column
    {"source": ..., "predicted_target": ..., "confidence": ...,
     "agreement": ..., "votes": ...}
  ]
}
```

The sketch shows the load-bearing fields; records carry further
self-describing metadata (`model`, `temperature`, `n_shot`, `n_valid_runs`,
per-attempt `usage`/`transport_trace`, …). Relative to the raw experiment
logs, only the per-attempt `response_dump` — the provider's full JSON
envelope, whose text content is already in `raw_response` — was dropped for
size.

**`predictions.jsonl.gz`** — one line per scored source column; every number
in the paper derives from these rows (sole exception: the self-consistency
stability table reads the per-run scores from `records.jsonl.gz`):

```jsonc
{"dataset": ..., "model": ..., "pair_id": ..., "source_column": ...,
 "gt_target": ..., "predicted_target": ..., "confidence": 0..10,
 "agreement": ..., "votes": ..., "correct": true|false}
```

**`results/classical/{method}/{dataset}/metrics.json`** — `tau0`, `best_f1`,
and `best_mcc` operating points over the similarity-threshold sweep.

## Evaluation Conventions

- Decisions are per source column: predicted target or abstain; confidence on
  the native 0–10 scale. At threshold τ, a prediction is kept iff
  `confidence ≥ τ`. A wrong kept match on a column that has a true match
  counts as both FP and FN. Accept-error = FP / (TP+FP) = 1 − precision.
- Model output names are grounded to schema columns by strict
  exact/case-insensitive matching (`src/matcher/prompt.py: parse_response`);
  ungroundable mappings are dropped and counted as hallucinations.

## Notes

- **Seed:** few-shot example selection uses seed 42; greedy decoding
  (temperature 0) for the primary condition.
- **Self-consistency:** at temperature 0.7, each task is queried 3 times with
  majority voting over (source, target) pairs.
- **All-failed tasks:** tasks where every model call failed write no
  per-column decisions by design (≈0.3% of tasks, logged in
  `records.jsonl.gz`).
- **`run_meta.json` totals** reflect only the last runner invocation of a
  cell — for runs completed across several resumes, they read ≈0; per-call
  usage in `records.jsonl.gz` is complete and authoritative. Two
  self-consistency cells were interrupted before their final metrics pass and
  ship without `metrics.json`/`run_meta.json` (records + predictions are
  complete there too).

## License

Code: MIT (see `LICENSE`). Bundled datasets retain their original licenses and
attributions (see `data/DATASETS.md`).

# Calibrate Once, Match Everywhere? — Replication Package

Artifact for the EDBT 2027 short-paper submission
*"[Short Paper] Calibrate Once, Match Everywhere? Cross-Dataset Transfer of
False-Positive Rejection Thresholds in LLM-Based Schema Matching"*.

It contains **all code, all prompts, and all pre-computed scores** of the
study: the exact prompt messages, every raw model response, and the
per-column decisions with their verbalized confidence, for the paper's
8 models x 4 datasets. The primary condition — the basis of every number in the paper —
is published complete for all 32 (model, dataset) pairs. The wider
16-condition prompt grid is a robustness extra and is published as run
(287 of 512 cells): it was executed in full on OC3-FO (the hardest, decisive
benchmark) and for the cheaper no-CoT conditions elsewhere; the expensive
CoT conditions were not extended to the three large datasets after proving
~neutral on OC3-FO (exact per-cell coverage in `outputs/grid_robustness.md`).
Tasks where all model calls failed write no per-column decisions by design
(≈0.3% of tasks, logged in `records.jsonl.gz`).

There are two independent ways to use it:

| | What you get | Needs |
|---|---|---|
| **Path A — reproduce the statistics** | every table & figure of the paper, regenerated from the published logs in `runs/` | Python only — **no API key, no cost, ~1 minute** |
| **Path B — reproduce the LLM calls** | fresh model responses via OpenRouter, written to `results/` in the same format | an OpenRouter API key (`.env`) |

## Layout

```
matcher/        the experiment code: full-schema LLM matcher
                (config, loaders, prompt builder + parser, OpenRouter client,
                 self-consistency aggregation, runner, smoke test)
analysis/       one script per paper artifact (tables, figures, supplementary)
notebooks/      reproduce.ipynb — runs the analysis and renders everything inline
runs/           the PUBLISHED logs (read-only input for Path A):
                runs/<model>/<dataset>/<cell>/
                  records.jsonl.gz      every prompt + every raw response
                  predictions.jsonl.gz  per-source-column decisions + confidence
                  metrics.json          micro P/R/F1 of that run
                  run_meta.json         model, condition (`ablation`), totals
                                        (see caveat under "Published log formats")
data/           the four benchmarks + loaders provenance (see data/DATASETS.md)
baselines/      non-LLM baselines (COMA, Similarity Flooding) + their results
scripts/        integrity_check.py — verifies the bundled datasets
outputs/        regenerated tables + supplementary (written by Path A)
figures/        regenerated figures (written by Path A)
results/        fresh Path-B re-runs land here (gitignored, never mixes with runs/)
```

## Setup

Python ≥ 3.10 (tested on 3.11):

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows    (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Path A — reproduce every table and figure (no API key)

From the **repository root** (all `python -m …` commands assume this cwd):

```bash
python -m analysis.reproduce_all
```

(~20 s; or open `notebooks/reproduce.ipynb` — needs `pip install jupyter`,
which is not in requirements.txt). This reads only `runs/` and writes:

| Artifact | Paper | Output |
|---|---|---|
| Table 2 — baseline F1/MCC at τ=0, per model x dataset | §5.1 | `outputs/table2_baseline.{md,tex}` |
| Table 3 — transfer matrix: MCC + accept-error at the source's MCC-optimal cutoff, mean±std, Δ vs τ=0 | §5.3 | `outputs/table3_operating_point.{md,tex}` |
| Figure 1 — confidence calibration (error vs reported confidence) | §5.2 | `figures/fig2_calibration.{pdf,png}` (file name is historical — this is the paper's **Figure 1**) |
| Figure 2 — capacity trend (ΔMCC of the oracle threshold vs model size) | §5.4 | `figures/fig_capacity_trend.{pdf,png}` + per-model raw values in `outputs/capacity_per_model.md` |
| per-model MCC-optimal τ* (+ medians) | §5.3 | `outputs/per_model_tau.md` |
| pooled accept-error at confidence 10/9/8/7/6 | §5.2 | `outputs/calibration_numbers.md` |
| self-consistency score-stability tables | §5.2 | `outputs/sc_stability.md` |
| OC3-FO distractor-removal robustness check | §5.2 | `outputs/oc3_distractor_check.md` |
| classical-matcher comparison (COMA, Sim. Flooding) | §5.1 | `outputs/classical_baselines.md` |
| robustness across the full 16-condition prompt grid | §4 | `outputs/grid_robustness.md` |

Each script also runs standalone, e.g. `python -m analysis.make_table3_operating_point`.

To analyse a fresh Path-B re-run instead of the published logs, point the
analysis at it:

```bash
REPRO_RUNS_DIR=results python -m analysis.reproduce_all        # bash
$env:REPRO_RUNS_DIR="results"; python -m analysis.reproduce_all  # PowerShell
```

Windows note: when **redirecting** console output to a file, set
`PYTHONUTF8=1` (the reports contain Δ/τ characters; interactive consoles are
fine).

## Path B — re-run the LLM experiments (OpenRouter)

```bash
cp .env.example .env              # fill in OPENROUTER_API_KEY
python -m matcher.smoke_test      # end-to-end checks, zero API calls
python -m matcher.smoke_test --live   # + exactly ONE real call
```

Then:

```bash
# the paper's primary condition (zero-shot, no values, no CoT, greedy),
# all 8 models x 4 datasets — 734 tasks per model:
python -m matcher.runner --primary-only

# everything (full 16-condition grid incl. temp-0.7 self-consistency x3):
python -m matcher.runner

# useful narrowing flags (all reversible; see --help):
python -m matcher.runner --models gemma-4-31b --datasets oc3-fo ppmatch
python -m matcher.runner --only-temp0 --cheap-first
MATCHER_MAX_TASKS=2 python -m matcher.runner     # 2-task pilot, loudly logged (bash)
# PowerShell: $env:MATCHER_MAX_TASKS="2"; python -m matcher.runner
```

Notes:

- Output goes to `results/` (same layout and file formats as `runs/`); the
  published logs are never touched.
- **Resume-safe**: tasks already in a run dir's `records.jsonl` are skipped, so
  interrupting and re-running is always safe. Running out of OpenRouter
  credits stops the experiment gracefully; re-run to resume.
- **Cost**: the bundled models are small open-weight models (see
  `matcher/config.py` for per-MTok prices). The 8-model primary condition is
  on the order of a few dollars; the full grid (16 conditions, self-consistency
  triples most calls) is on the order of tens of dollars, depending on
  provider pricing. Real cost is read from the API and stored per call in
  `records.jsonl.gz` (`usage.cost`, `cost_source`) — the authoritative cost
  record; `run_meta.json` totals cover only the final runner invocation (≈0
  for runs completed across resumes, see below).
- LLM sampling is not bit-deterministic, so fresh responses will differ
  slightly from `runs/` even at temperature 0; the published logs are the
  exact data behind the paper's numbers.

## Published log formats

`runs/<model>/<dataset>/<cell>/` — the cell name encodes the prompt condition:
`vals{on|off}__shot{0|3}__{cot|nocot}__t{0.0|0.7}__sc{1|3}__scopeoff`
(sample values / few-shot / chain-of-thought / temperature / self-consistency
runs; the trailing `scopeoff` is a fixed legacy suffix). The paper's primary
condition is `valsoff__shot0__nocot__t0.0__sc1__scopeoff`.

`records.jsonl.gz` — one line per matching task:

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

(The schema sketch above shows the load-bearing fields; records carry further
self-describing metadata — `model`, `model_id`, `temperature`, `n_shot`,
`n_valid_runs`, per-attempt `usage`/`transport_trace`, etc. Relative to the
raw experiment logs, only the per-attempt `response_dump` — the provider's
full JSON envelope, whose text content is already in `raw_response` — was
dropped for size.)

Caveats: `run_meta.json` `totals` reflect only the **last** runner invocation
of a cell — for runs completed across several resumes they read ≈0 (per-call
usage in `records.jsonl.gz` is complete and authoritative); two
self-consistency cells were interrupted before their final metrics pass and
ship without `metrics.json`/`run_meta.json` (records + predictions are
complete there too). The `scoping`/`scope_v` keys in `run_meta.json` are the
same historical naming as the `scopeoff` suffix — always off/null in this
study.

`predictions.jsonl.gz` — one line per scored source column; every number in
the paper derives from these rows (sole exception: the self-consistency
stability table reads the per-run scores from `records.jsonl.gz`):

```jsonc
{"dataset": ..., "model": ..., "pair_id": ..., "source_column": ...,
 "gt_target": ..., "predicted_target": ..., "confidence": 0..10,
 "agreement": ..., "votes": ..., "correct": true|false}
```

## Models

The paper's eight open-weight models are bundled in the logs — Gemma-3
4/12/27B, Gemma-4-31B, Qwen3 8/14/32B, and Llama-3.3-70B — accessed through
OpenRouter (`matcher/config.py` holds the exact model ids and per-MTok
prices). The analysis discovers the model set from the run directories, so a
custom re-run with a different model list is analysed the same way.

## Datasets

See `data/DATASETS.md` for provenance, licenses, the Valentine slim-copy note,
and the OC3-FO masking convention. Verify the bundled data with:

```bash
python scripts/integrity_check.py     # GT⊆columns, no duplicates, strict 1:1
```

## Evaluation conventions

- Decisions are per source column: predicted target or abstain; confidence on
  the native 0–10 scale. At threshold τ a prediction is kept iff
  `confidence ≥ τ`. A wrong kept match on a column that has a true match
  counts as both FP and FN. Accept-error = FP / (TP+FP) = 1 − precision.
- Model output names are grounded to schema columns by strict
  exact/case-insensitive matching (`matcher/prompt.py: parse_response`);
  ungroundable mappings are dropped and counted as hallucinations.

## License

Code: MIT (see `LICENSE`). Bundled datasets retain their original licenses and
attributions (see `data/DATASETS.md`).

# Non-LLM baselines (COMA, Similarity Flooding)

Classical schema matchers compared against the LLM matcher in §5.1 of the
paper ("Against classical matchers"). All methods run on **column names only**
(header-only DataFrames, no instance values), so they apply uniformly to all
four datasets including the schema-only OC3-FO; assignment is global 1:1
(Hungarian, `--assign bipartite`).

The results used in the paper ship in `results/<method>/<dataset>/`
(`predictions.jsonl` + `metrics.json` with `tau0` / `best_f1` / `best_mcc`
operating points). The paper grants each classical matcher its own
*optimal* per-dataset threshold — i.e. the `best_f1` entry.

## Re-running (optional)

```bash
pip install scipy valentine          # COMA / Similarity Flooding / Cupid + Hungarian
cd baselines
python run.py --method coma
python run.py --method similarity_flooding
```

`--method exact` (normalized name equality) is dependency-free and a useful
smoke test. `--method embedding` additionally needs `sentence-transformers`.

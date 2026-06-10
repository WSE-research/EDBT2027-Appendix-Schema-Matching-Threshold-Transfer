# Non-LLM baselines (schema-name input, bipartite assignment)

Each method is granted its own best per-dataset threshold, per metric (`F1 @ F1-oracle`, `MCC @ MCC-oracle`) — the comparison of §5.1.

| Method | Dataset | F1 @ tau0 | F1 @ F1-oracle | MCC @ MCC-oracle |
|---|---|---|---|---|
| coma | ppm | 0.39 | 0.60 | 0.54 |
| coma | val | 0.69 | 0.78 | 0.48 |
| coma | hdx | 0.75 | 0.87 | 0.72 |
| coma | oc3 | 0.10 | 0.40 | 0.36 |
| exact | ppm | 0.23 | 0.38 | 0.40 |
| exact | val | 0.15 | 0.15 | 0.00 |
| exact | hdx | 0.49 | 0.65 | 0.36 |
| exact | oc3 | 0.01 | 0.07 | 0.18 |
| similarity_flooding | ppm | 0.33 | 0.49 | 0.46 |
| similarity_flooding | val | 0.56 | 0.58 | 0.16 |
| similarity_flooding | hdx | 0.77 | 0.85 | 0.65 |
| similarity_flooding | oc3 | 0.09 | 0.19 | 0.22 |

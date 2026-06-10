"""Non-LLM matchers: a score matrix (source x target) + 1:1 assignment.

Methods:
  exact     - normalized name equality (dependency-free smoke baseline)
  embedding - zero-shot sentence-embedding cosine
  coma / similarity_flooding - classical matchers via the `valentine` package,
              run on column NAMES only (header-only DataFrames), so they apply
              uniformly to all four datasets incl. schema-only OC3-FO.

Assignment:
  greedy    - per-source argmax (a target may be reused)
  bipartite - global 1:1 (Hungarian). Falls back to greedy if scipy is missing.
"""
import re
import numpy as np

_norm_re = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _norm_re.sub("", s.lower())


# ── score matrices ───────────────────────────────────────────────────────────
def _exact_scores(src, tgt):
    sn = [_norm(c) for c in src]
    tn = [_norm(c) for c in tgt]
    return np.array([[1.0 if a == b else 0.0 for b in tn] for a in sn])


_MODEL = None


def _embedding_scores(src, tgt, model_name):
    global _MODEL
    from sentence_transformers import SentenceTransformer
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    es = _MODEL.encode(src, normalize_embeddings=True)
    et = _MODEL.encode(tgt, normalize_embeddings=True)
    sim = es @ et.T                      # cosine in [-1, 1]
    return (sim + 1.0) / 2.0             # rescale to [0, 1] like a confidence


VALENTINE_METHODS = {"coma", "similarity_flooding"}


def _valentine_matcher(method):
    import valentine.algorithms as VA
    if method == "coma":
        return VA.Coma(use_instances=False, use_schema=True)   # schema-only mode
    if method == "similarity_flooding":
        return VA.SimilarityFlooding()
    raise ValueError(method)


def _valentine_scores(src, tgt, method):
    import pandas as pd
    from valentine import valentine_match
    df_s = pd.DataFrame(columns=list(src))   # header-only: names, no rows
    df_t = pd.DataFrame(columns=list(tgt))
    res = valentine_match([df_s, df_t], _valentine_matcher(method), df_names=["src", "tgt"])
    sidx = {c: i for i, c in enumerate(src)}
    tidx = {c: i for i, c in enumerate(tgt)}
    M = np.zeros((len(src), len(tgt)))
    for cp, score in res.items():
        # cp = ColumnPair(source_table, source_column, target_table, target_column)
        i, j = sidx.get(cp.source_column), tidx.get(cp.target_column)
        if i is not None and j is not None:
            M[i, j] = float(score)
    return M


def score_matrix(src, tgt, method, model_name=None):
    if not src or not tgt:
        return np.zeros((len(src), len(tgt)))
    if method == "exact":
        return _exact_scores(src, tgt)
    if method == "embedding":
        return _embedding_scores(src, tgt, model_name)
    if method in VALENTINE_METHODS:
        return _valentine_scores(src, tgt, method)
    raise ValueError(f"unknown method: {method}")


# ── assignment ───────────────────────────────────────────────────────────────
def assign(scores, src, tgt, mode="bipartite"):
    """Return per-source (predicted_target, score). Unassigned -> (None, 0.0)."""
    S, T = scores.shape
    out = {s: (None, 0.0) for s in src}
    if S == 0 or T == 0:
        return out
    if mode == "bipartite":
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-scores)   # maximize total score
            for r, c in zip(rows, cols):
                out[src[r]] = (tgt[c], float(scores[r, c]))
            return out
        except ImportError:
            mode = "greedy"
    # greedy: per-source argmax
    for r in range(S):
        c = int(np.argmax(scores[r]))
        out[src[r]] = (tgt[c], float(scores[r, c]))
    return out


def predict(view, method, model_name=None, mode="bipartite") -> list[dict]:
    """Produce predictions.jsonl rows for one task (schema mirrors the LLM runs)."""
    sm = score_matrix(view["source_cols"], view["target_cols"], method, model_name)
    a = assign(sm, view["source_cols"], view["target_cols"], mode)
    rows = []
    for s in view["source_cols"]:
        pred, score = a[s]
        rows.append({
            "source_col": s,
            "gt_target": view["gt_target"].get(s),
            "predicted_target": pred,
            "confidence": score,         # similarity in [0,1]; swept like tau
        })
    return rows

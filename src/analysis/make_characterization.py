"""Benchmark characterization: the four measured properties in Table 1, the
Valentine perturbation stress test discussed in Section 5.3, and the task-level
rank correlations quoted in the response letter.

Everything here is computed from the bundled schemas (data/, via
src/matcher/loaders.py) and the published prediction logs (results/llm/, via
src/analysis/common.py). No API key, no new inference.

The similarity measures are deliberately lexical and simple. The reviewer asked
for "lexical overlap" without naming a measure, and a name-similarity number a
reader cannot recompute from the column names alone is a worse answer than a
crude one they can.

    python -m src.analysis.make_characterization
    python -m src.analysis.make_characterization --selftest
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import mean, median

import numpy as np

from src.analysis import common
from src.matcher.loaders import load_dataset

DATASETS = ["valentine", "hdxsm", "ppmatch", "oc3-fo"]
NICE = {"ppmatch": "PowerPlantBench", "valentine": "Valentine", "hdxsm": "HDXSM", "oc3-fo": "OC3-FO"}
NOISE_ORDER = ["ec", "ac1", "ac2", "ac3", "ac4", "ac5"]
OUT_PATH = common.OUT_DIR / "characterization.json"


# --------------------------------------------------------------------------
# name normalization and the four similarity measures
# --------------------------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NONWORD = re.compile(r"[^a-z0-9]+")


def relation_of(col):
    """OC3-FO carries 'relation.column'; the others are single-table pairs."""
    return col.rsplit(".", 1)[0] if "." in col else None


def norm(col):
    """'Ord.grossCapacity_MW' -> 'gross capacity mw'."""
    base = col.rsplit(".", 1)[-1]
    base = _CAMEL.sub(" ", base)
    return _NONWORD.sub(" ", base.lower()).strip()


def tokens(name):
    return set(name.split())


def trigrams(name):
    s = "  " + name + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def sim_exact(a, b):
    return 1.0 if a and a == b else 0.0


def sim_jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sim_trigram(a, b):
    ga, gb = trigrams(a), trigrams(b)
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def sim_seq(a, b):
    return SequenceMatcher(None, a, b).ratio()


MEASURES = {"exact": sim_exact, "jaccard": sim_jaccard, "trigram": sim_trigram, "seqratio": sim_seq}

# Valentine encodes its fabricated pairs in the pair_id:
#   <domain>/<relation type>/<table>_<side>_<p1>_<p2>_<noise>_ev
# noise is 'ec' (verbatim column names) or 'ac1'..'ac5' (increasing abbreviation).
# p1 and p2 are the two generation percentages; which is rows and which is
# columns is not documented in the data, so they stay named p1/p2 and what they
# do is measured below rather than asserted.
_VAL_TAIL = re.compile(r"_(both|source|target)_(\d+)_(\d+)_(ec|ac\d)_ev$")


def valentine_facets(pair_id):
    parts = pair_id.split("/")
    if len(parts) != 3:
        return {}
    m = _VAL_TAIL.search(parts[2])
    if not m:
        return {"val_domain": parts[0], "val_reltype": parts[1]}
    return {
        "val_domain": parts[0],
        "val_reltype": parts[1],
        "val_side": m.group(1),
        "val_p1": int(m.group(2)),
        "val_p2": int(m.group(3)),
        "val_noise": m.group(4),
    }


# --------------------------------------------------------------------------
# per-dataset characterization: the four Table 1 properties and their siblings
# --------------------------------------------------------------------------

def characterize(tasks):
    src_sizes, tgt_sizes, rel_counts, linkable_shares, gt_counts = [], [], [], [], []
    gt_sim = {m: [] for m in MEASURES}          # lexical overlap of true correspondences
    margins, top1_correct = [], []              # name ambiguity, on linkable columns

    for t in tasks:
        src, tgt = t["source_columns"], t["target_columns"]
        gt = dict(t["ground_truth"])
        if not src or not tgt:
            continue
        src_sizes.append(len(src))
        tgt_sizes.append(len(tgt))
        rels = {r for c in src + tgt if (r := relation_of(c))}
        # no relation prefix means a pairwise single-table task: one relation per
        # side, by construction rather than measured
        rel_counts.append(len(rels) if rels else 2)
        linkable_shares.append(len(gt) / len(src))
        gt_counts.append(len(gt))

        nsrc = {c: norm(c) for c in src}
        ntgt = {c: norm(c) for c in tgt}

        for s, g in gt.items():
            if s in nsrc and g in ntgt:
                for name, fn in MEASURES.items():
                    gt_sim[name].append(fn(nsrc[s], ntgt[g]))

        for s in src:
            if s not in gt:
                continue
            scored = sorted(((sim_trigram(nsrc[s], ntgt[c]), c) for c in tgt), reverse=True)
            best, best_col = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            margins.append(best - second)
            top1_correct.append(1.0 if best_col == gt[s] else 0.0)

    def agg(v):
        if not v:
            return {"mean": 0.0, "median": 0.0, "n": 0}
        return {"mean": mean(v), "median": median(v), "n": len(v)}

    return {
        "tasks": len(tasks),
        "src_cols": agg(src_sizes),
        "tgt_cols": agg(tgt_sizes),
        "relations_per_task": agg(rel_counts),
        # pooled, not the per-task mean: a 6-column task should not weigh the
        # same as a 111-column one
        "linkable_share_pooled": sum(gt_counts) / sum(src_sizes) if src_sizes else 0.0,
        "gt_overlap": {m: agg(v) for m, v in gt_sim.items()},
        "ambiguity_margin": agg(margins),
        "lexically_decidable": agg(top1_correct),
    }


# --------------------------------------------------------------------------
# per-column properties, joined to predictions on (pair_id, source_column)
# --------------------------------------------------------------------------

def column_properties(dataset, tasks):
    out = {}
    facet_cache = {}
    for t in tasks:
        src, tgt = t["source_columns"], t["target_columns"]
        gt = dict(t["ground_truth"])
        if not src or not tgt:
            continue
        facets = (facet_cache.setdefault(t["pair_id"], valentine_facets(t["pair_id"]))
                  if dataset == "valentine" else {})
        nsrc = {c: norm(c) for c in src}
        ntgt = {c: norm(c) for c in tgt}

        for s in src:
            scored = sorted(((sim_trigram(nsrc[s], ntgt[c]), c) for c in tgt), reverse=True)
            best, best_col = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            g = gt.get(s)
            row = {
                "n_src": len(src),
                "n_tgt": len(tgt),
                "linkable_share_task": len(gt) / len(src),
                "is_linkable": g is not None,
                "margin": best - second,
                "top1_is_gt": (best_col == g) if g is not None else None,
                "gt_jaccard": sim_jaccard(nsrc[s], ntgt[g]) if (g is not None and g in ntgt) else None,
            }
            row.update(facets)
            out[(t["pair_id"], s)] = row
    return out


def enriched_rows(model, dataset, props):
    """Prediction rows of one run, each carrying its column properties."""
    preds = common.load(model, dataset)
    if preds is None:
        return []
    out = []
    for r in preds:
        p = props.get((r["pair_id"], r["source_column"]))
        if p is not None:
            out.append(dict(r, **p))
    return out


def metrics_at(rows, tau):
    tp, fp, fn, tn = common.confusion(rows, tau)
    return {"mcc": common.mcc(tp, fp, fn, tn),
            "accept_error": fp / (tp + fp) if (tp + fp) else 0.0}


def best_tau(rows):
    return common.best_tau(rows, fn=common.mcc_at)


# --------------------------------------------------------------------------
# Valentine's built-in perturbation design (Section 5.3, reviewer point O2)
# --------------------------------------------------------------------------

def valentine_perturbation(props, models):
    by_model = {m: enriched_rows(m, "valentine", props) for m in models}
    any_rows = next(iter(by_model.values()))

    per_noise = {}
    for lvl in NOISE_ORDER:
        taus, mcc_own, mcc_glob = [], [], []
        for m in models:
            rows = [r for r in by_model[m] if r.get("val_noise") == lvl]
            if not rows:
                continue
            t_glob, t_local = best_tau(by_model[m]), best_tau(rows)
            taus.append(t_local)
            mcc_own.append(metrics_at(rows, t_local)["mcc"])
            mcc_glob.append(metrics_at(rows, t_glob)["mcc"])
        if taus:
            per_noise[lvl] = {
                "tau_median": float(np.median(taus)),
                "mcc_oracle": float(np.mean(mcc_own)),
                "mcc_global": float(np.mean(mcc_glob)),
                "regret_pp": float(np.mean(np.array(mcc_own) - np.array(mcc_glob)) * 100),
                # the share of linkable columns whose lexically closest target is
                # the correct one, the property the corruption drives down
                "decidable": float(np.mean([r["top1_is_gt"] for r in any_rows
                                            if r.get("val_noise") == lvl and r["top1_is_gt"] is not None])),
            }

    def transfer_matrix(key, levels):
        out = {}
        for src in levels:
            out[str(src)] = {}
            for dst in levels:
                vals = []
                for m in models:
                    rs = [r for r in by_model[m] if r.get(key) == src]
                    rd = [r for r in by_model[m] if r.get(key) == dst]
                    if rs and rd:
                        vals.append(metrics_at(rd, best_tau(rd))["mcc"]
                                    - metrics_at(rd, best_tau(rs))["mcc"])
                if vals:
                    out[str(src)][str(dst)] = float(np.mean(vals) * 100)
        return out

    p2_levels = sorted({r["val_p2"] for r in any_rows if r.get("val_p2") is not None})
    return {
        "per_noise": per_noise,
        "noise_transfer_pp": transfer_matrix("val_noise", NOISE_ORDER),
        "p2_transfer_pp": transfer_matrix("val_p2", p2_levels),
        "p2_levels": p2_levels,
    }


# --------------------------------------------------------------------------
# task level: do the properties predict the transfer outcome? (response letter)
# --------------------------------------------------------------------------

def spearman(xs, ys):
    """Rank correlation. Pearson on average ranks, ties averaged.

    Implemented here so the analysis path keeps its numpy-only dependency.
    """
    def ranks(v):
        order = np.argsort(np.asarray(v, dtype=float), kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average the ranks of tied values
        sv = np.asarray(v, dtype=float)[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def task_correlations(props_by_ds, models):
    rows = []
    for d in DATASETS:
        for m in models:
            own_rows = enriched_rows(m, d, props_by_ds[d])
            if not own_rows:
                continue
            tau_own = best_tau(own_rows)
            # the threshold a practitioner would import: calibrated on the other three
            others = [best_tau(enriched_rows(m, o, props_by_ds[o]))
                      for o in DATASETS if o != d]
            tau_imported = float(np.median(others))

            by_task = defaultdict(list)
            for r in own_rows:
                by_task[r["pair_id"]].append(r)
            for pid, g in by_task.items():
                own, imp = metrics_at(g, tau_own), metrics_at(g, tau_imported)
                jac = [r["gt_jaccard"] for r in g if r["gt_jaccard"] is not None]
                t1 = [r["top1_is_gt"] for r in g if r["top1_is_gt"] is not None]
                rows.append({
                    "dataset": d, "model": m, "pair_id": pid,
                    "n_src": g[0]["n_src"],
                    "linkable_share": g[0]["linkable_share_task"],
                    "mean_margin": float(np.mean([r["margin"] for r in g])),
                    "mean_gt_jaccard": float(np.mean(jac)) if jac else None,
                    "top1_ok": float(np.mean(t1)) if t1 else None,
                    "mcc_imported": imp["mcc"],
                    "mcc_regret": own["mcc"] - imp["mcc"],
                    "ae_imported": imp["accept_error"],
                    "ae_regret": imp["accept_error"] - own["accept_error"],
                })

    properties = ["n_src", "linkable_share", "mean_margin", "mean_gt_jaccard", "top1_ok"]
    outcomes = ["mcc_imported", "mcc_regret", "ae_imported", "ae_regret"]
    corr = {}
    for p in properties:
        corr[p] = {}
        for o in outcomes:
            pairs = [(r[p], r[o]) for r in rows if r[p] is not None and r[o] is not None]
            if len(pairs) >= 20:
                corr[p][o] = {"rho": spearman([a for a, _ in pairs], [b for _, b in pairs]),
                              "n": len(pairs)}
    return {"n_rows": len(rows),
            "n_tasks": len({(r["dataset"], r["pair_id"]) for r in rows}),
            "spearman": corr}


# --------------------------------------------------------------------------

def selftest():
    assert norm("Ord.grossCapacity_MW") == "gross capacity mw", norm("Ord.grossCapacity_MW")
    assert relation_of("Ord.capacity") == "Ord" and relation_of("capacity") is None
    assert sim_exact("capacity", "capacity") == 1.0 and sim_exact("", "") == 0.0
    assert sim_jaccard("gross capacity", "net capacity") == 1 / 3
    assert sim_trigram("capacity", "capacity") == 1.0 and sim_trigram("capacity", "zzzz") < 0.1
    assert 0.0 < sim_seq("capacity", "capacitiy") < 1.0

    # a schema whose lexically closest target is the wrong one
    task = {"pair_id": "t", "source_columns": ["capacity"],
            "target_columns": ["capacity_net", "power"],
            "ground_truth": [("capacity", "power")]}
    r = characterize([task])
    assert r["lexically_decidable"]["mean"] == 0.0, r["lexically_decidable"]
    assert r["linkable_share_pooled"] == 1.0
    assert r["gt_overlap"]["exact"]["mean"] == 0.0
    assert r["relations_per_task"]["median"] == 2   # no prefix, so assumed pairwise

    cp = column_properties("test", [task])
    assert len(cp) == 1 and cp[("t", "capacity")]["top1_is_gt"] is False

    f = valentine_facets("ChEMBL/Joinable/assays_both_50_1_ac1_ev")
    assert f["val_noise"] == "ac1" and f["val_p1"] == 50 and f["val_p2"] == 1, f
    assert valentine_facets("Wikidata/Musicians/Musicians").get("val_noise") is None
    # underscores in the table name must not break the tail match
    assert valentine_facets("X/Y/order_items_both_50_70_ec_ev")["val_noise"] == "ec"

    # rank correlation against hand-checkable cases. The third: the squared rank
    # differences sum to 4, so rho = 1 - 6*4/(5*24) = 0.8. The fourth has ties on
    # both sides and matches scipy.stats.spearmanr to twelve decimals.
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4, 5], [1, 3, 2, 5, 4]) - 0.8) < 1e-9
    assert abs(spearman([1, 1, 1, 2, 3], [3, 1, 2, 2, 5]) - 0.573539334676) < 1e-9

    print("selftest ok")


def main():
    models = common.models()
    print(f"models: {len(models)}  ({', '.join(models)})\n")

    tasks_by_ds, props_by_ds, per_ds = {}, {}, {}
    for d in DATASETS:
        tasks_by_ds[d] = load_dataset(d)
        props_by_ds[d] = column_properties(d, tasks_by_ds[d])
        per_ds[d] = characterize(tasks_by_ds[d])

    w = 18
    order = ["valentine", "hdxsm", "ppmatch", "oc3-fo"]

    def line(label, fn):
        print(f"{label:<34s}" + "".join(f"{fn(per_ds[d]):>{w}}" for d in order))

    print("=" * (34 + w * len(order)))
    print(f"{'Table 1 properties':<34s}" + "".join(f"{NICE[d]:>{w}}" for d in order))
    print("=" * (34 + w * len(order)))
    line("Src   source attrs/task, median", lambda r: f"{r['src_cols']['median']:.1f}")
    line("Link  share with a match", lambda r: f"{r['linkable_share_pooled']:.2f}")
    line("Ident. identical names", lambda r: f"{r['gt_overlap']['exact']['mean']:.2f}")
    line("Dec.  decidable by name", lambda r: f"{r['lexically_decidable']['mean']:.2f}")
    print("-" * (34 + w * len(order)))
    line("relations/task, median", lambda r: f"{r['relations_per_task']['median']:.1f}")
    line("tasks", lambda r: r["tasks"])
    print("=" * (34 + w * len(order)))

    print("\nValentine perturbation (Section 5.3): six name-corruption steps")
    pert = valentine_perturbation(props_by_ds["valentine"], models)
    print(f"  {'step':<8s}{'tau*':>8s}{'MCC*':>9s}{'MCC@shared':>12s}{'regret':>10s}{'decidable':>11s}")
    for lvl in NOISE_ORDER:
        p = pert["per_noise"][lvl]
        print(f"  {lvl:<8s}{p['tau_median']:>8.1f}{p['mcc_oracle']:>9.3f}"
              f"{p['mcc_global']:>12.3f}{p['regret_pp']:>9.2f}pp{p['decidable']:>11.2f}")
    worst = max(p["regret_pp"] for p in pert["per_noise"].values())
    print(f"  -> one shared cutoff stays within {worst:.1f} pp of each step's own optimum")

    lv = pert["p2_levels"]
    hi, lo = str(lv[-1]), str(lv[0])
    print(f"\n  share of matchable attributes, transfer from the highest to the lowest setting:"
          f" {pert['p2_transfer_pp'][hi][lo]:.1f} pp")

    print("\nTask level (response letter): rank correlation over every transfer")
    tc = task_correlations(props_by_ds, models)
    print(f"  {tc['n_rows']} task-by-model rows over {tc['n_tasks']} distinct tasks")
    for o in ["ae_imported", "mcc_imported", "ae_regret", "mcc_regret"]:
        best = max((abs(v[o]["rho"]) for v in tc["spearman"].values() if o in v), default=0.0)
        print(f"  largest |rho| against {o:<14s} {best:.2f}")
    print("  The eight models repeat every task, so the rows are not independent;"
          " read the rho, not a p value.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"per_dataset": per_ds, "valentine_perturbation": pert,
                   "task_level": tc}, f, indent=2)
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()

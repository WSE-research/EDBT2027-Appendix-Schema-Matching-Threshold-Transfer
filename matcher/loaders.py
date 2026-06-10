"""Dataset loaders for the four study benchmarks (extracted verbatim from the
benchmark-collection unified loader used for the paper; only the data root is
repo-local here).

Each loader returns a list of MatchingTask dicts in a common format:
    {
        "dataset": str,           # dataset name
        "pair_id": str,           # unique pair identifier
        "source_id": str,         # source schema name
        "target_id": str,         # target schema name
        "source_columns": list[str],
        "target_columns": list[str],
        "ground_truth": list[tuple[str, str]],   # (source_col, target_col)
        "masked_columns": list[str],             # oc3-fo only (see load_oc3fo)
        "category": "schema-matching",
    }

Provenance and licenses of the bundled data: see data/DATASETS.md.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def load_dataset(name: str) -> list[dict]:
    dispatch = {
        "valentine": load_valentine,
        "ppmatch": load_ppmatch,
        "oc3-fo": load_oc3fo,
        "hdxsm": load_hdxsm,
    }
    if name not in dispatch:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dispatch.keys())}")
    return dispatch[name]()


# ---------------------------------------------------------------------------
# Valentine (Koutras et al., ICDE 2021)
# ---------------------------------------------------------------------------

# Known upstream GT typos in Valentine: the mapping JSON names a source column
# that does not exist in the source CSV header, but the intended column is
# unambiguous (verified against the header — only the "Label" suffix is
# missing). Keyed by pair_id -> {wrong_source_column: correct_source_column}.
_VALENTINE_GT_FIXES = {
    "Wikidata/Musicians/Musicians_unionable": {"givenName": "givenNameLabel"},
    "Wikidata/Musicians/Musicians_viewunion": {"givenName": "givenNameLabel"},
}


def load_valentine() -> list[dict]:
    base = DATA_ROOT / "valentine" / "Valentine-datasets"
    tasks = []

    for domain_dir in sorted(base.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
            continue
        for category_dir in sorted(domain_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for pair_dir in sorted(category_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                mapping_files = list(pair_dir.glob("*_mapping.json"))
                if not mapping_files:
                    continue
                mapping = json.loads(mapping_files[0].read_text(encoding="utf-8"))

                source_csvs = list(pair_dir.glob("*_source.csv"))
                target_csvs = list(pair_dir.glob("*_target.csv"))
                if not source_csvs or not target_csvs:
                    continue

                source_cols = _csv_columns(source_csvs[0])
                target_cols = _csv_columns(target_csvs[0])

                pair_id = f"{domain_dir.name}/{category_dir.name}/{pair_dir.name}"
                fix = _VALENTINE_GT_FIXES.get(pair_id, {})
                gt = [
                    (fix.get(m["source_column"], m["source_column"]), m["target_column"])
                    for m in mapping.get("matches", [])
                ]

                tasks.append({
                    "dataset": "valentine",
                    "pair_id": pair_id,
                    "source_id": source_csvs[0].stem,
                    "target_id": target_csvs[0].stem,
                    "source_columns": source_cols,
                    "target_columns": target_cols,
                    "ground_truth": gt,
                    "category": "schema-matching",
                })
    return tasks


# ---------------------------------------------------------------------------
# PPMatch / PowerPlantBench (energy domain, 24 sources x 18 target columns)
# ---------------------------------------------------------------------------

def load_ppmatch() -> list[dict]:
    base = DATA_ROOT / "ppmatch"
    gt = json.loads((base / "ground_truth.json").read_text(encoding="utf-8"))
    meta_path = base / "column_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    target_cols = list(gt["target_schema"])
    tasks = []
    for source_name, info in gt["sources"].items():
        mappings = info.get("mappings", {})
        no_match = info.get("no_match", [])
        source_cols = list(mappings.keys()) + list(no_match)
        ground_truth = [(src, tgt) for src, tgt in mappings.items()]
        src_meta = metadata.get(source_name)
        tasks.append({
            "dataset": "ppmatch",
            "pair_id": source_name,
            "source_id": source_name,
            "target_id": "powerplantmatching",
            "source_columns": source_cols,
            "target_columns": target_cols,
            "ground_truth": ground_truth,
            "category": "schema-matching",
            "source_instances": src_meta if src_meta else None,
            "target_instances": None,
        })
    return tasks


# ---------------------------------------------------------------------------
# OC3-FO (Traeger et al.)
# ---------------------------------------------------------------------------

def load_oc3fo() -> list[dict]:
    """OC3-FO as a strict-1:1 EQUIVALENCE matching task.

    The linkage file distinguishes two attribute relation types:
      - ``inter_identical``  (39 pairs): true semantic equivalence -> POSITIVES.
      - ``inter_sub_typed``  (16 pairs): subsumption / part-of, NOT equality.

    Following the field convention that schema matching scores equivalence only,
    the subsumption correspondences are MASKED rather than relabelled as
    negatives: for each schema pair, any column that participates in a
    subsumption link but NOT in any equivalence link is removed from that pair's
    candidate set. This prevents a model that correctly recognises a subsumption
    relation from being charged a false positive (which would bias the FP metric).
    Columns kept in the candidate set with no positive are genuine no-matches.
    """
    base = DATA_ROOT / "oc3-fo"
    schema_file = base / "schemas" / "OC3FO_schema_elements_dataset.csv"
    gt_file = base / "ground-truth" / "OC3_linkages.csv"

    # entity id -> (schema, "table.attribute"); attributes per schema
    elem: dict[str, tuple[str, str]] = {}
    attrs_by_schema: dict[str, list[str]] = {}
    with open(schema_file, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["type"] != "attribute":
                continue
            col = f"{row['parent_name']}.{row['name']}"
            elem[row["id"]] = (row["schema"], col)
            attrs_by_schema.setdefault(row["schema"], []).append(col)

    # correspondences split by relation_type, keyed by unordered schema pair
    # (source side is always the alphabetically smaller schema)
    identical: dict[tuple[str, str], list[tuple[str, str]]] = {}
    subtyped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    with open(gt_file, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["type"] != "attribute":
                continue
            a = elem.get(row["entity_a_id"])
            b = elem.get(row["entity_b_id"])
            if not a or not b:
                continue
            (sa, ca), (sb, cb) = a, b
            if sa == sb:
                continue  # intra-schema (guard; inter_* should not be intra)
            if sa < sb:
                key, corr = (sa, sb), (ca, cb)
            else:
                key, corr = (sb, sa), (cb, ca)
            bucket = identical if row["relation_type"] == "inter_identical" else subtyped
            bucket.setdefault(key, []).append(corr)

    schemas = sorted(attrs_by_schema.keys())
    tasks = []
    for i, s1 in enumerate(schemas):
        for s2 in schemas[i + 1:]:
            key = (s1, s2)
            pos = identical.get(key, [])
            sub = subtyped.get(key, [])
            pos_src = {c for c, _ in pos}
            pos_tgt = {c for _, c in pos}
            mask_src = {c for c, _ in sub} - pos_src
            mask_tgt = {c for _, c in sub} - pos_tgt
            src_cols = [c for c in sorted(attrs_by_schema[s1]) if c not in mask_src]
            tgt_cols = [c for c in sorted(attrs_by_schema[s2]) if c not in mask_tgt]
            masked = [f"{s1}:{c}" for c in sorted(mask_src)] + \
                     [f"{s2}:{c}" for c in sorted(mask_tgt)]
            tasks.append({
                "dataset": "oc3-fo",
                "pair_id": f"{s1}__{s2}",
                "source_id": s1,
                "target_id": s2,
                "source_columns": src_cols,
                "target_columns": tgt_cols,
                "ground_truth": pos,            # equivalence-only positives
                "masked_columns": masked,        # subsumption-only cols excluded from scoring
                "category": "schema-matching",
            })
    return tasks


# ---------------------------------------------------------------------------
# HDXSM (Humanitarian Data Exchange schema matching; introduced by SMUTF,
# Zhang et al., Information Systems 2024). Flat real-world humanitarian tables.
# ---------------------------------------------------------------------------

_HDXSM_MAPPING_RE = re.compile(r'^<\s*"(.*)"\s*,\s*"(.*)"\s*>$')


def _hdxsm_header(path: Path) -> list[str]:
    # All HDXSM files are valid UTF-8 (verified: 0 of 612 files fail strict
    # decode). Some headers contain a literal U+FFFD that is byte-identical in
    # both the table header and the mapping file, so exact matching still holds.
    with open(path, encoding="utf-8") as f:
        return next(csv.reader(f), [])


def _hdxsm_mappings(path: Path) -> list[tuple[str, str]]:
    """Parse mappings.txt lines of the form <"source col", "target col">."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _HDXSM_MAPPING_RE.match(line)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def _hdxsm_titles(meta_path: Path, fallback: str) -> tuple[str, str]:
    """meta.txt: line1 '<uuid>,<Table1 title>', line2 '<uuid>,<Table2 title>'."""
    if not meta_path.exists():
        return f"{fallback}_T1", f"{fallback}_T2"
    lines = meta_path.read_text(encoding="utf-8").splitlines()

    def title(idx: int, fb: str) -> str:
        if idx < len(lines) and "," in lines[idx]:
            return lines[idx].split(",", 1)[1].strip() or fb
        return fb

    return title(0, f"{fallback}_T1"), title(1, f"{fallback}_T2")


def load_hdxsm(include_quarantined: bool = False) -> list[dict]:
    """Load the strict-1:1, GT-consistent subset of HDXSM.

    A pair is QUARANTINED (excluded) unless it is both:
      (a) strict 1:1   — no source col maps to >1 target and vice versa, and
      (b) GT-consistent — every ground-truth column is present in its header.
    The quarantine manifest ships at data/hdxsm/hdxsm_quarantine.json.
    Pass include_quarantined=True to bypass the filter (diagnostics only).
    """
    base = DATA_ROOT / "hdxsm" / "hxd_datasets_0.2"
    if not base.is_dir():
        return []

    pair_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("pair_")],
        key=lambda p: int(p.name.split("_")[1]),
    )

    tasks = []
    quarantine = []
    for p in pair_dirs:
        t1, t2, mp = p / "Table1.csv", p / "Table2.csv", p / "mappings.txt"
        if not (t1.exists() and t2.exists() and mp.exists()):
            quarantine.append({"pair": p.name, "reasons": ["missing file"]})
            continue

        h1 = _hdxsm_header(t1)
        h2 = _hdxsm_header(t2)
        gt = _hdxsm_mappings(mp)

        reasons = []
        bad = [(s, t) for s, t in gt if s not in h1 or t not in h2]
        if bad:
            reasons.append(f"{len(bad)} GT col(s) absent from header")
        src_count = Counter(s for s, _ in gt)
        tgt_count = Counter(t for _, t in gt)
        fan_out = {s: c for s, c in src_count.items() if c > 1}
        fan_in = {t: c for t, c in tgt_count.items() if c > 1}
        if fan_out or fan_in:
            reasons.append(f"non-1:1 (fan_out={fan_out}, fan_in={fan_in})")

        if reasons and not include_quarantined:
            quarantine.append({
                "pair": p.name,
                "reasons": reasons,
                "bad_gt_examples": bad[:5],
            })
            continue

        s_title, t_title = _hdxsm_titles(p / "meta.txt", p.name)
        tasks.append({
            "dataset": "hdxsm",
            "pair_id": p.name,
            "source_id": s_title,
            "target_id": t_title,
            "source_columns": h1,
            "target_columns": h2,
            "ground_truth": gt,
            "category": "schema-matching",
            "source_instances": None,  # Table1/Table2 CSVs hold the rows if needed
            "target_instances": None,
        })

    return tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_columns(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        return list(header) if header else []


if __name__ == "__main__":
    for name in ("oc3-fo", "ppmatch", "hdxsm", "valentine"):
        tasks = load_dataset(name)
        n_gt = sum(len(t["ground_truth"]) for t in tasks)
        print(f"{name:<12} {len(tasks):>4} tasks   {n_gt:>6} GT correspondences")

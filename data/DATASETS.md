# Bundled datasets — provenance, licenses, and what was changed

All four benchmarks are publicly available. This package bundles them (one in a
slimmed form, see below) together with the exact loaders used for the paper
(`src/matcher/loaders.py`) and an integrity check (`scripts/integrity_check.py`)
that verifies, for every task: ground truth ⊆ declared columns, no duplicate
correspondences, and strict one-to-one cardinality.

| Dataset | Domain | Tasks | GT | Bundled as | Upstream license |
|---|---|---|---|---|---|
| PowerPlantBench (`ppmatch`) | energy | 24 | 209 | GT + column metadata (JSON, verbatim) | released with this package (MIT) |
| Valentine (`valentine`) | multi-domain | 544 | 8,644 | **slim derivative** (see below) | CC-BY-4.0 (Zenodo 5084605) |
| HDXSM (`hdxsm`) | humanitarian | 160 | 2,097 | verbatim | Apache-2.0 (SMUTF repository) |
| OC3-FO (`oc3-fo`) | relational | 6 | 39 | schemas + linkages (verbatim) | CC0-1.0 (CollaborativeScoping repository) |

## PowerPlantBench (`data/ppmatch/`)

24 real-world power-plant registries (e.g. national TSO and registry feeds,
mixed German/English attribute names) aligned to the 18-attribute target schema
of the open `powerplantmatching` library; introduced in the preliminary
analysis referenced by the paper. Bundled: `ground_truth.json` (per-source
mappings + no-match lists, which defines the task set) and
`column_metadata.json` (per-column example values used for the sample-values
prompt condition). The benchmark files are released under this package's MIT
license; the underlying registry data are public sources.

## Valentine (`data/valentine/`)

Koutras et al., "Valentine: Evaluating Matching Techniques for Dataset
Discovery" (ICDE 2021). Official distribution: Zenodo record
[5084605](https://zenodo.org/record/5084605), licensed **CC-BY-4.0**
(redistribution with attribution — given here).

Note: the per-pair `*.ini` files are upstream generator configs shipped
verbatim from the official archive; the absolute paths inside them belong to
the upstream dataset authors' environment, not to this package.

**Bundled as a slim derivative to keep the package small** (the originals total
~3 GB): every `*_mapping.json` ground-truth file is verbatim; every CSV keeps
its full header plus the first 100 data records (re-quoted with the Python csv
module). This is lossless for the experiments in the paper: schema columns come
from the header, and sample values are drawn from the first 60 records only
(`src/matcher/config.py: VALUE_SCAN_ROWS`), so prompts built from the slim copy are
identical to prompts built from the full download. To work with the full data,
download the Zenodo archive and replace `data/valentine/Valentine-datasets/`.

Two upstream ground-truth typos (a missing `Label` suffix in the
`Wikidata/Musicians` unionable/viewunion pairs) are corrected at load time and
documented in `src/matcher/loaders.py` (`_VALENTINE_GT_FIXES`).

## HDXSM (`data/hdxsm/`)

Humanitarian Data Exchange schema-matching pairs introduced by SMUTF
(Zhang et al., Information Systems 2024; arXiv:2402.01685). Ground truth
derives from HXL hashtags. Bundled verbatim (`hxd_datasets_0.2/`); the SMUTF
repository is licensed **Apache-2.0**, and the underlying HDX tables are
published open data.

The paper uses the strict subset of 160 of 204 pairs that are (a) strict 1:1
and (b) GT-consistent (every GT column present in its table header); the
excluded pairs and reasons are listed in `hdxsm_quarantine.json` and the filter
is implemented in `src/matcher/loaders.py: load_hdxsm`.

## OC3-FO (`data/oc3-fo/`)

Three partially overlapping relational customer-order schemas plus an unrelated
Formula-One schema as a controlled distractor (Traeger et al.; upstream
repository: `leotraeg/CollaborativeScoping` on GitHub, licensed **CC0-1.0**).
Bundled: `schemas/` and `ground-truth/` plus the upstream README. Two reading
notes: the upstream README also describes an `evaluation/` directory (the
upstream authors' own experiment results), which this package deliberately
omits; and its "annotated correspondences" count includes subsumption and
entity-level links — the equivalence-only attribute ground truth used here is
the 39 `inter_identical` pairs (16 `inter_sub_typed` column entries masked on
both sides yield the integrity check's "masked 21" column count).

Scoring convention (implemented in `src/matcher/loaders.py: load_oc3fo`):
equivalence-only. The 39 `inter_identical` linkages are positives; the 16
`inter_sub_typed` (subsumption) linkages are **masked, not negatives** — any
column participating only in a subsumption link is removed from that pair's
candidate set, so a matcher that correctly recognises a subsumption relation is
never charged a false positive.

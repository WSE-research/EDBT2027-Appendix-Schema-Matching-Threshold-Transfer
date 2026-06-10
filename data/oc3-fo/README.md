# OC3-FO — Multi-Source Schema Matching Dataset

Source: <https://github.com/leotraeg/CollaborativeScoping>
Paper: Traeger, Behrend, Karabatis. "Collaborative Scoping: Self-Supervised Linkability Assessment for Schema Matching." EDBT 2026.

## Overview

OC3-FO is a multi-source schema matching scenario containing **4 heterogeneous schemas** from different database vendors and domains:

- **OC-ORACLE** — Orders/Customers schema from Oracle sample DB (7 tables, 50 attributes)
- **OC-MYSQL** — Orders/Customers schema from MySQL tutorial DB (8 tables, 67 attributes)
- **OC-SAP** — Orders/Customers schema from SAP HANA sample project (3 tables, 40 attributes)
- **FORMULA** — Formula One schema from jolpica-f1 project (14 tables, 128 attributes), domain-unrelated to the other three

OC3 is the 3-schema subset (Oracle + MySQL + SAP only). OC3-FO extends it with the Formula One schema to test cross-domain robustness.

Total: 289 schema elements (32 tables + 257 attributes) across 4 schemas.

## Directory Structure

```
schemas/
  OC3FO_schema_elements_dataset.csv   — All tables and attributes with metadata and linkability labels
  OC3_collaborative_scoping.csv       — OC3 (3-schema) scoping results with per-schema anomaly scores
  OC3FO_collaborative_scoping.csv     — OC3-FO (4-schema) scoping results with per-schema anomaly scores

ground-truth/
  OC3_linkages.csv                    — 76 annotated correspondences between OC3 schemas
  OC3FO_linkages_cossimilarity.csv    — Pairwise cosine similarity between all table pairs across schemas

evaluation/
  evaluation_analysis_results_*.csv   — Aggregated AUC scores for Scoping vs Collaborative Scoping
  evaluation_raw_results_*.csv        — Raw evaluation metrics (full precision)
```

## Schema Elements File Format

`OC3FO_schema_elements_dataset.csv` columns:
- `id` — Unique element identifier (e.g., entity_44)
- `type` — "table" or "attribute"
- `parent_id` — Parent element (schema for tables, table for attributes)
- `schema` — Source schema: OC-ORACLE, OC-MYSQL, OC-SAP, FORMULA
- `name` — Element name
- `parent_name` — Parent element name
- `datatype` — Data type (NUMBER, VARCHAR2, DATE, FLOAT, BLOB, etc.)
- `constraints` — PRIMARY KEY, FOREIGN KEY, or empty
- `text_sequence` — Serialized text representation for embedding
- `label_linkability` — Ground truth: True = linkable across schemas, False = unlinkable

## Linkages File Format

`OC3_linkages.csv` contains annotated correspondences:
- `entity_a_id`, `entity_b_id` — Matched element pair
- `entity_a_source`, `entity_b_source` — Source schemas
- `canonical_name` — Semantic label for the correspondence
- `type` — "table" or "attribute"
- `relation_type` — `inter_identical` (exact match) or `inter_sub_typed` (subsumption)

## Key Statistics

| Metric | OC3 | OC3-FO |
|--------|-----|--------|
| Schemas | 3 | 4 |
| Tables | 18 | 32 |
| Attributes | 133 | 257 |
| Linkable elements | ~50% | ~26% |
| Ground truth correspondences | 76 | 76 (same OC3 pairs; FORMULA has 0 cross-domain matches) |

## Usage for Schema Matching

The dataset supports two tasks:
1. **Linkability assessment** — Binary classification of whether a schema element has correspondences in other schemas (using `label_linkability`).
2. **Schema matching** — Finding correspondences between elements across schemas (using `OC3_linkages.csv`).

The FORMULA schema acts as a "distractor" with no true matches to the OC schemas, testing a matcher's ability to handle domain-unrelated sources.

"""Full-schema forward schema-matching prompt + response parser.

Design:
  * Scope = FULL-SCHEMA FORWARD (N-to-M): the whole source schema vs the whole
    target schema in ONE call. The LLM reasons over the joint context and
    returns a mapping list.
  * Task framing = score-each-mapping 0.0-10.0, abstain by OMISSION.
  * CoT = REASON-FIRST: free-text step-by-step reasoning, THEN a trailing JSON
    object. NOT forced JSON-first (which would suppress the CoT effect).
  * The prompt is DOMAIN-AGNOSTIC (the study spans 4 domains) and states an
    explicit EQUIVALENCE definition to suppress broader/narrower (subsumption)
    false positives — important for the FP study and the oc3-fo handling.
  * no-CoT control is via a strong "JSON only" instruction (not API json-mode),
    to stay uniform across heterogeneous open models, several of which lack
    reliable response_format=json_object support on OpenRouter.
"""
from __future__ import annotations

import json
import re

_EQUIVALENCE = (
    "Definition of a MATCH: two columns match if and only if they represent the "
    "SAME real-world attribute and would hold interchangeable values. Columns "
    "that are merely related, or where one is broader or narrower than the other "
    "(e.g. 'city' vs 'full address', 'first name' vs 'full name'), DO NOT match."
)

_SCORING = (
    "For each source column that has an equivalent target column, output one "
    "mapping with a confidence score from 0.0 to 10.0:\n"
    "- 10.0 = you are absolutely certain this mapping is correct\n"
    "- 5.0 = you are unsure\n"
    "- 0.0 = you are absolutely certain this is NOT a match\n"
    "Only output a mapping when you see a plausible semantic correspondence "
    "(score >= 5.0). A source column with no equivalent target should simply be "
    "OMITTED — never force a match; abstaining is correct and expected."
)

_FORMAT = (
    'Output format — a single JSON object:\n'
    '{"mappings": [{"source": "<exact source column name>", '
    '"target": "<exact target column name>", "score": <number 0.0-10.0>}]}'
)

_COT_INSTR = (
    "Think step by step: first reason briefly about each source column's "
    "meaning, then compare it against the candidate target columns, then decide. "
    "After your reasoning, output your final answer as a single JSON object on "
    "its own line."
)
_NOCOT_INSTR = "Respond with ONLY the JSON object and nothing else."


def _system_prompt(use_cot: bool) -> str:
    return (
        "You are an expert in database and data-integration schema matching. You "
        "are given the columns of a SOURCE schema and a TARGET schema, and must "
        "determine which source columns are semantically equivalent to which "
        "target columns.\n\n"
        f"{_EQUIVALENCE}\n\n"
        f"{_SCORING}\n\n"
        f"{_COT_INSTR if use_cot else _NOCOT_INSTR}\n\n"
        f"{_FORMAT}"
    )


def _fmt_values(values, limit: int) -> str:
    if not values:
        return ""
    shown = [str(v) for v in list(values)[:limit] if str(v).strip()]
    return f"  (e.g. {', '.join(shown)})" if shown else ""


def _format_columns(columns, values: dict | None, use_values: bool, limit: int) -> str:
    lines = []
    for c in columns:
        line = f"  - {c}"
        if use_values and values:
            line += _fmt_values(values.get(c), limit)
        lines.append(line)
    return "\n".join(lines)


def _gold_mappings_json(task: dict) -> str:
    """The ground-truth mapping list of one example task, score=10.0."""
    maps = [{"source": s, "target": t, "score": 10.0} for s, t in task["ground_truth"]]
    return json.dumps({"mappings": maps}, ensure_ascii=False)


def _shot_block(shot_examples: list[dict], use_values: bool, limit: int) -> str:
    if not shot_examples:
        return ""
    blocks = [f"Here are {len(shot_examples)} worked examples from OTHER schema pairs:"]
    for i, ex in enumerate(shot_examples, 1):
        blocks.append(
            f"\nExample {i}:\n"
            f"SOURCE columns:\n{_format_columns(ex['source_columns'], ex.get('source_values'), use_values, limit)}\n"
            f"TARGET columns:\n{_format_columns(ex['target_columns'], ex.get('target_values'), use_values, limit)}\n"
            f"Correct answer: {_gold_mappings_json(ex)}"
        )
    return "\n".join(blocks) + "\n"


def build_messages(
    task: dict,
    *,
    use_values: bool = False,
    use_cot: bool = False,
    shot_examples: list[dict] | None = None,
    value_limit: int = 5,
) -> list[dict]:
    src = _format_columns(task["source_columns"], task.get("source_values"), use_values, value_limit)
    tgt = _format_columns(task["target_columns"], task.get("target_values"), use_values, value_limit)
    shots = _shot_block(shot_examples or [], use_values, value_limit)

    user = (
        f"SOURCE schema '{task['source_id']}' columns:\n{src}\n\n"
        f"TARGET schema '{task['target_id']}' columns:\n{tgt}\n\n"
        f"{shots}"
        "Which source columns are semantically equivalent to which target "
        "columns? Score each mapping 0.0-10.0 and omit source columns with no "
        "true match. Return the JSON object."
    )
    return [
        {"role": "system", "content": _system_prompt(use_cot)},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    # prefer the LAST {"mappings" ...} object (reason-first CoT puts it last)
    starts = [m.start() for m in re.finditer(r'\{\s*"mappings"', cleaned)]
    if not starts:
        b = cleaned.find("{")
        starts = [b] if b >= 0 else []
    for start in reversed(starts):
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _ground(name: str, columns: list[str], lut: dict) -> str | None:
    """Map a model-returned column name to an exact schema column."""
    if name in lut:
        return lut[name]
    s = str(name).strip()
    if s in lut:
        return lut[s]
    return lut.get(s.casefold())


def parse_response(content: str, source_columns: list[str], target_columns: list[str]) -> dict:
    """Ground the model's mappings against the real source/target columns.

    Returns {mappings: [{source, target, score}], hallucinated: int,
    parse_error: bool}. A mapping is dropped (and counted as a hallucination)
    if either side cannot be grounded to a real column.
    """
    data = _extract_json(content or "")
    if not isinstance(data, dict) or "mappings" not in data:
        return {"mappings": [], "hallucinated": 0, "parse_error": True}

    # case-insensitive lookups, exact names win
    src_lut = {c.casefold(): c for c in source_columns}
    src_lut.update({c: c for c in source_columns})
    tgt_lut = {c.casefold(): c for c in target_columns}
    tgt_lut.update({c: c for c in target_columns})

    out = []
    hallucinated = 0
    raw_maps = data.get("mappings") or []
    if not isinstance(raw_maps, list):
        return {"mappings": [], "hallucinated": 0, "parse_error": True}

    for m in raw_maps:
        if not isinstance(m, dict):
            hallucinated += 1
            continue
        gs = _ground(m.get("source", ""), source_columns, src_lut)
        gt = _ground(m.get("target", ""), target_columns, tgt_lut)
        if gs is None or gt is None:
            hallucinated += 1
            continue
        try:
            score = float(m.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(10.0, score))
        out.append({"source": gs, "target": gt, "score": round(score, 2)})

    return {"mappings": out, "hallucinated": hallucinated, "parse_error": False}

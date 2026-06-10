"""Bridge the bundled dataset loaders into the baseline's per-source view.

Each task becomes: source columns, target columns, and the ground-truth target
for every source column (None for unlinkable sources). This mirrors how the LLM
matcher's predictions.jsonl is structured, so metrics are directly comparable.
"""
import config  # noqa: F401  (inserts the repo root into sys.path)

from src.matcher import loaders


def load_tasks(dataset: str) -> list[dict]:
    """Return the dataset's tasks, capped per config.CAPS."""
    tasks = loaders.load_dataset(dataset)
    cap = config.CAPS.get(dataset)
    return tasks[:cap] if cap else tasks


def to_view(task: dict) -> dict:
    """Normalize a MatchingTask into the baseline view.

    Returns {source_cols, target_cols, gt_target}, where gt_target maps each
    source column to its single ground-truth target (or None if unlinkable).
    The study sets are 1:1, so each source has at most one GT target.
    """
    gt = {}
    for s, t in task.get("ground_truth", []):
        gt.setdefault(s, t)   # first wins; sets are 1:1 so this is unambiguous
    source_cols = list(task["source_columns"])
    target_cols = list(task["target_columns"])
    return {
        "pair_id": task.get("pair_id", task.get("source_id", "")),
        "source_cols": source_cols,
        "target_cols": target_cols,
        "gt_target": {s: gt.get(s) for s in source_cols},
    }

"""PopQA loading and preprocessing.

Implements the raw -> interim -> processed pipeline described in
docs/methodology.md. The Hugging Face `datasets` import is deferred to the
functions that actually need it, so importing this module (e.g. from
tests exercising normalize/foils/sampling) never requires `datasets` to be
installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conflict_eval.data.normalize import normalize_answer
from conflict_eval.data.sampling import sample_candidates

POPQA_FIELDS = [
    "id",
    "subj",
    "prop",
    "obj",
    "subj_id",
    "prop_id",
    "obj_id",
    "s_aliases",
    "o_aliases",
    "question",
    "possible_answers",
]


def download_raw(hf_dataset_id: str, split: str, raw_dir: str | Path) -> Path:
    """Download PopQA via `datasets.load_dataset` and dump it to
    `raw_dir` as JSONL, alongside a manifest recording the exact
    identifier/split/revision resolved, for reproducibility
    (docs/data/README.md).
    """
    import datasets  # deferred: heavy, network-using dependency

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset(hf_dataset_id, split=split)
    out_path = raw_dir / "popqa_raw.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in ds)

    manifest = {
        "hf_dataset_id": hf_dataset_id,
        "split": split,
        "num_rows": len(ds),
        "fields": ds.column_names,
    }
    with open(raw_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path


def load_raw_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _parse_alias_field(value: Any) -> list[str]:
    """PopQA's alias columns are stored as string-encoded Python lists
    (e.g. "['foo', 'bar']"). Fall back to treating the raw value as a
    single alias if it does not parse, rather than dropping it silently.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        if stripped:
            return [stripped]
    return []


def build_interim(raw_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize raw PopQA rows into interim records. Returns
    (interim_items, exclusions). An item is excluded here only for
    structural problems (missing question/gold answer) — KC/KW
    classification happens later, per-model, during baseline screening.
    """
    interim_items = []
    exclusions = []

    for row in raw_items:
        item_id = str(row.get("id"))
        question = row.get("question")
        gold = row.get("obj")

        if not question or not str(question).strip():
            exclusions.append({"id": item_id, "reason": "empty_question"})
            continue
        if not gold or not str(gold).strip():
            exclusions.append({"id": item_id, "reason": "empty_gold_answer"})
            continue

        aliases = _parse_alias_field(row.get("o_aliases"))
        possible_answers = _parse_alias_field(row.get("possible_answers"))
        all_aliases = sorted(set(aliases) | set(possible_answers))

        interim_items.append(
            {
                "id": item_id,
                "subj": row.get("subj"),
                "prop": row.get("prop"),
                "obj": gold,
                "question": str(question).strip(),
                "aliases": all_aliases,
                "gold_normalized": normalize_answer(gold),
            }
        )

    return interim_items, exclusions


def screen_candidates(interim_items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically subsample the interim pool for baseline
    screening (docs/methodology.md, section 1, step 3).
    """
    return sample_candidates(interim_items, n, seed)


def write_jsonl(items: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in items)

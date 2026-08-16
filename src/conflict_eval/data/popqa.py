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


class DatasetRevisionResolutionError(RuntimeError):
    """Raised when a real PopQA download cannot be pinned to a verified,
    exact Hugging Face commit SHA before loading. A real data-preparation
    run must fail clearly here rather than silently claiming an exact
    revision that was never actually verified (docs/decisions.md,
    "Resolve, pin, load, record").
    """


def resolve_dataset_revision(hf_dataset_id: str, requested_revision: str | None = "main") -> str | None:
    """Resolve `requested_revision` (default "main") to the exact,
    immutable Hugging Face commit SHA for the dataset repo
    `hf_dataset_id`, via a single `huggingface_hub` metadata request — no
    dataset file download.

    This is the SELECTION mechanism: the returned SHA is meant to be
    passed to `datasets.load_dataset(..., revision=...)` so the revision
    determines what gets loaded, rather than being inferred afterward
    from local cache state (docs/decisions.md, "Resolve, pin, load,
    record" — this deliberately replaces an earlier implementation that
    scanned the local cache post-download and could, in principle, pick
    up an unrelated cached revision rather than the one the current call
    actually used).

    `huggingface_hub` is not a direct project dependency here: it is
    already installed transitively via `transformers`/`datasets`, so it
    is imported lazily rather than re-declared in pyproject.toml.

    Returns `None` — never a guessed value — if Hub access is unavailable
    (offline, etc.) or the lookup otherwise fails.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        info = HfApi().dataset_info(hf_dataset_id, revision=requested_revision or "main")
    except Exception:  # noqa: BLE001 — any Hub/network failure must degrade to "unavailable", not crash resolution
        return None
    return info.sha


def build_manifest(
    hf_dataset_id: str,
    split: str,
    num_rows: int,
    fields: list[str],
    resolved_revision: str | None,
) -> dict[str, Any]:
    """Pure manifest-dict construction, factored out of download_raw so
    the manifest schema (in particular, that `resolved_revision` is
    always present, even as `None`) is unit-testable without a real
    dataset download.
    """
    return {
        "hf_dataset_id": hf_dataset_id,
        "split": split,
        "num_rows": num_rows,
        "fields": fields,
        "resolved_revision": resolved_revision,
    }


def download_raw(
    hf_dataset_id: str, split: str, raw_dir: str | Path, revision: str | None = "main"
) -> Path:
    """Download PopQA via `datasets.load_dataset` and dump it to
    `raw_dir` as JSONL, alongside a manifest recording the exact
    identifier/split/resolved revision, for reproducibility
    (data/README.md).

    Follows resolve -> pin -> load -> record: the exact commit SHA is
    resolved first, then passed explicitly to `load_dataset(revision=...)`,
    so the manifest records the revision that was actually used to load
    the data rather than one inferred afterward. Raises
    DatasetRevisionResolutionError rather than proceeding if that SHA
    cannot be resolved — a real data-preparation run must not silently
    claim an exact revision that was never verified.
    """
    import datasets  # deferred: heavy, network-using dependency

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    resolved_revision = resolve_dataset_revision(hf_dataset_id, revision)
    if resolved_revision is None:
        raise DatasetRevisionResolutionError(
            f"Could not resolve an exact Hugging Face commit SHA for dataset "
            f"{hf_dataset_id!r} (requested revision: {revision!r}). A real "
            "data-preparation run must not proceed with an unpinned/unverified "
            "dataset snapshot. Check Hub access (network connectivity) and retry."
        )

    ds = datasets.load_dataset(hf_dataset_id, split=split, revision=resolved_revision)
    out_path = raw_dir / "popqa_raw.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in ds)

    manifest = build_manifest(
        hf_dataset_id=hf_dataset_id,
        split=split,
        num_rows=len(ds),
        fields=ds.column_names,
        resolved_revision=resolved_revision,
    )
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

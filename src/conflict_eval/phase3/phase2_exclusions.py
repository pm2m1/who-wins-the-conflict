"""Deterministic extraction of the Phase 2 Qwen KW exclusion list (§15.1).

§15.1 requires Cohort A to be built from **fresh** Qwen KW items: the
items already used in the Phase 2 Qwen pilot are excluded, so the Phase 3
replication is not measured partly on the same observations it is
replicating.

The exclusion list is therefore derived from the **actual frozen Phase 2
artifact**, never from remembered or re-derived values. This module reads
that artifact, extracts the selected KW item ids, and writes an exclusion
file whose provenance points back at the source by digest.

`PHASE2_QWEN_KW_ITEM_COUNT` (30) is a check, not a source: if the artifact
yields a different count the extraction fails loudly rather than padding,
truncating, or "correcting" the empirical record.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from conflict_eval.phase3.constants import ITEM_ID_FIELD, PHASE2_QWEN_KW_ITEM_COUNT
from conflict_eval.phase3.runtime_capture import sha256_file, sha256_text


class Phase2ExclusionError(ValueError):
    """Raised when the Phase 2 exclusion list cannot be derived faithfully."""


@dataclasses.dataclass(frozen=True)
class Phase2Exclusions:
    item_ids: tuple[str, ...]
    source_paths: tuple[str, ...]
    source_sha256: dict[str, str]
    knowledge_group: str = "KW"
    model_key: str = "qwen"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "knowledge_group": self.knowledge_group,
            "excluded_item_ids": list(self.item_ids),
            "excluded_count": len(self.item_ids),
            "source_artifacts": list(self.source_paths),
            "source_artifact_sha256": dict(self.source_sha256),
            "rule": (
                "docs/phase3_scaled_study_design.md §15.1 -- Cohort A uses FRESH "
                "Qwen KW items; every item selected for the Phase 2 Qwen pilot is "
                "excluded."
            ),
        }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_phase2_qwen_kw_exclusions(
    trial_artifacts: list[str | Path],
    *,
    expected_count: int | None = PHASE2_QWEN_KW_ITEM_COUNT,
) -> Phase2Exclusions:
    """Extract the selected Phase 2 Qwen KW item ids from real artifacts.

    `trial_artifacts` are the committed Phase 2 pilot record files for Qwen
    (the built pilot sample and/or its trial results). Any record whose
    `knowledge_group` is `KW` contributes its `item_id`.

    Ordering is deterministic (sorted), and duplicates across artifacts
    collapse to one id -- the same item appearing in several conditions is
    still one excluded item.
    """
    if not trial_artifacts:
        raise Phase2ExclusionError(
            "No Phase 2 Qwen artifact supplied. The exclusion list must be derived "
            "from the real frozen artifact and is never reconstructed from memory "
            "(§15.1)."
        )

    ids: set[str] = set()
    sources: list[str] = []
    digests: dict[str, str] = {}
    for raw in trial_artifacts:
        path = Path(raw)
        if not path.exists():
            raise Phase2ExclusionError(
                f"Phase 2 artifact not found: {path}. Supply the actual frozen "
                "Phase 2 Qwen pilot artifact."
            )
        records = _iter_jsonl(path)
        if not records:
            raise Phase2ExclusionError(f"Phase 2 artifact {path} is empty.")
        found = 0
        for record in records:
            if record.get("knowledge_group") != "KW":
                continue
            if ITEM_ID_FIELD not in record:
                raise Phase2ExclusionError(
                    f"{path}: a KW record is missing {ITEM_ID_FIELD!r}; item "
                    "identity is required to exclude it (§15.1)."
                )
            ids.add(str(record[ITEM_ID_FIELD]))
            found += 1
        if found == 0:
            raise Phase2ExclusionError(
                f"{path} contains no KW records. This does not look like the Phase 2 "
                "Qwen pilot artifact; refusing to derive an empty exclusion list."
            )
        sources.append(str(path))
        digests[str(path)] = sha256_file(path)

    item_ids = tuple(sorted(ids))
    if expected_count is not None and len(item_ids) != expected_count:
        raise Phase2ExclusionError(
            f"Extracted {len(item_ids)} Phase 2 Qwen KW item ids but the frozen "
            f"pilot selected {expected_count} (§15.1). The artifact and the frozen "
            "design disagree -- resolve the discrepancy rather than adjusting "
            "either. No id is invented, padded or dropped here."
        )
    return Phase2Exclusions(
        item_ids=item_ids, source_paths=tuple(sources), source_sha256=digests
    )


def write_exclusion_file(
    exclusions: Phase2Exclusions, path: str | Path
) -> tuple[Path, str]:
    """Write the exclusion artifact and return its path and SHA256."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(exclusions.as_dict(), indent=2, sort_keys=True) + "\n"
    out.write_text(payload, encoding="utf-8", newline="\n")
    return out, sha256_text(payload)


def load_exclusion_file(path: str | Path) -> tuple[frozenset[str], str]:
    """Load a written exclusion artifact, returning its ids and digest."""
    out = Path(path)
    if not out.exists():
        raise Phase2ExclusionError(f"Exclusion artifact not found: {out}")
    data = json.loads(out.read_text(encoding="utf-8"))
    ids = data.get("excluded_item_ids")
    if not isinstance(ids, list) or not ids:
        raise Phase2ExclusionError(f"{out}: excluded_item_ids is missing or empty.")
    return frozenset(str(i) for i in ids), sha256_file(out)

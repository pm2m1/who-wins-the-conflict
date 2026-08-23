"""Tests for the §15.1 Phase 2 Qwen KW exclusion extraction.

Cohort A is the direct replication, measured on **fresh** Qwen KW items.
If the Phase 2 items leaked back in, the replication would be measured
partly on the observations it is replicating.

The property under test is that the list is *derived from the real
artifact* and never invented: a wrong count, an artifact with no KW
records, or a missing file all fail loudly rather than producing a
plausible-looking list.
"""

from __future__ import annotations

import json

import pytest

from conflict_eval.phase3.constants import PHASE2_QWEN_KW_ITEM_COUNT
from conflict_eval.phase3.phase2_exclusions import (
    Phase2ExclusionError,
    extract_phase2_qwen_kw_exclusions,
    load_exclusion_file,
    write_exclusion_file,
)
from conflict_eval.phase3.runtime_capture import sha256_file


def _artifact(tmp_path, name="qwen_pilot.jsonl", kw=30, kc=12, start=0):
    """A stand-in Phase 2 Qwen pilot artifact.

    Synthetic on purpose: the real artifact lives on the researcher's GPU
    host and is gitignored, so these tests pin the extraction RULE, not any
    particular id.
    """
    rows = [
        {"item_id": f"kw-{i:04d}", "knowledge_group": "KW", "model_id": "Qwen/Qwen2.5-7B-Instruct"}
        for i in range(start, start + kw)
    ] + [
        {"item_id": f"kc-{i:04d}", "knowledge_group": "KC"} for i in range(kc)
    ]
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_extraction_returns_exactly_the_kw_ids(tmp_path):
    path = _artifact(tmp_path)
    result = extract_phase2_qwen_kw_exclusions([path])
    assert len(result.item_ids) == PHASE2_QWEN_KW_ITEM_COUNT == 30
    assert all(i.startswith("kw-") for i in result.item_ids)
    assert not any(i.startswith("kc-") for i in result.item_ids)


def test_extraction_is_deterministically_ordered(tmp_path):
    path = _artifact(tmp_path)
    a = extract_phase2_qwen_kw_exclusions([path]).item_ids
    b = extract_phase2_qwen_kw_exclusions([path]).item_ids
    assert a == b == tuple(sorted(a))


def test_extraction_records_source_provenance_and_digest(tmp_path):
    path = _artifact(tmp_path)
    result = extract_phase2_qwen_kw_exclusions([path])
    assert result.source_paths == (str(path),)
    assert result.source_sha256[str(path)] == sha256_file(path)


def test_duplicate_ids_across_artifacts_collapse_to_one_item(tmp_path):
    """The same item appearing in several conditions is still one item."""
    a = _artifact(tmp_path, "a.jsonl")
    b = _artifact(tmp_path, "b.jsonl")
    result = extract_phase2_qwen_kw_exclusions([a, b])
    assert len(result.item_ids) == 30


def test_a_wrong_count_is_refused_rather_than_padded(tmp_path):
    path = _artifact(tmp_path, kw=29)
    with pytest.raises(Phase2ExclusionError, match="frozen pilot selected 30"):
        extract_phase2_qwen_kw_exclusions([path])


def test_an_oversized_list_is_also_refused(tmp_path):
    path = _artifact(tmp_path, kw=31)
    with pytest.raises(Phase2ExclusionError, match="frozen pilot selected 30"):
        extract_phase2_qwen_kw_exclusions([path])


def test_an_artifact_with_no_kw_records_is_refused(tmp_path):
    path = _artifact(tmp_path, kw=0, kc=5)
    with pytest.raises(Phase2ExclusionError, match="no KW records"):
        extract_phase2_qwen_kw_exclusions([path])


def test_a_missing_artifact_is_refused(tmp_path):
    with pytest.raises(Phase2ExclusionError, match="not found"):
        extract_phase2_qwen_kw_exclusions([tmp_path / "absent.jsonl"])


def test_no_artifact_at_all_is_refused():
    with pytest.raises(Phase2ExclusionError, match="never reconstructed from memory"):
        extract_phase2_qwen_kw_exclusions([])


def test_a_kw_record_without_item_id_is_refused(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"knowledge_group": "KW"}) + "\n", encoding="utf-8")
    with pytest.raises(Phase2ExclusionError, match="missing 'item_id'"):
        extract_phase2_qwen_kw_exclusions([path])


def test_written_exclusion_file_round_trips(tmp_path):
    result = extract_phase2_qwen_kw_exclusions([_artifact(tmp_path)])
    out, digest = write_exclusion_file(result, tmp_path / "exclusions.json")
    assert len(digest) == 64
    ids, file_digest = load_exclusion_file(out)
    assert ids == frozenset(result.item_ids)
    assert file_digest == sha256_file(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["excluded_count"] == 30
    assert data["source_artifact_sha256"]
    assert "§15.1" in data["rule"]


def test_an_excluded_id_can_never_enter_cohort_a_supply(tmp_path):
    """The end-to-end property §15.1 actually asks for."""
    from conflict_eval.phase3.screening import ScreeningState

    result = extract_phase2_qwen_kw_exclusions([_artifact(tmp_path)])
    excluded = frozenset(result.item_ids)
    records = [
        {
            "item_id": item_id,
            "knowledge_group": "KW",
            "primary_conflict_eligible": True,
            "relation": "country",
            "parametric_margin": 0.5,
        }
        for item_id in list(excluded)[:10]
    ] + [
        {
            "item_id": f"fresh-{i}",
            "knowledge_group": "KW",
            "primary_conflict_eligible": True,
            "relation": "country",
            "parametric_margin": float(i) / 100.0,
        }
        for i in range(20)
    ]
    state = ScreeningState("qwen", phase2_excluded_ids=excluded)
    report = state.add_block(records)
    assert sum(report.cohort_a_per_stratum.values()) == 20, (
        "excluded Phase 2 items must contribute zero Cohort A supply"
    )

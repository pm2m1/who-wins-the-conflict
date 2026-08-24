"""Tests for the Phase 3C freeze construction.

These pin the properties that make the freeze *trustworthy* rather than
merely produced: that a returned block carrying an evidence outcome is
refused rather than turned into a cohort, that rendering a planned prompt
never runs one, that deduplication respects model and item identity, and
that the derived artifacts are a deterministic function of the raw ones.

Records here are synthetic. The real screening artifacts live outside the
repository (`configs/phase3/freeze/README.md`), so what these tests pin is
the RULE, never a particular measured value.
"""

from __future__ import annotations

import json

import pytest

from conflict_eval.models.base import GenerationConfig
from conflict_eval.phase3 import freeze_build as fb
from conflict_eval.phase3.constants import MARGIN_STRATA
from conflict_eval.phase3.screening import ScreeningError

EVIDENCE_TEMPLATE = 'Source: {source}\n\nThe answer to "{question}" is "{asserted_answer}".'
BASELINE_TEMPLATE = "Question:\n{question}\n\nEvidence:\n{evidence_or_none}\n"


def _record(item_id, *, group="KW", relation="country", margin=1.0, correct=False):
    """One synthetic baseline record in the real Phase 2/3 schema."""
    record = {
        "item_id": str(item_id),
        "model_id": "Fake/Model",
        "model_revision": "a" * 40,
        "requested_revision": "a" * 40,
        "subject": f"subject-{item_id}",
        "relation": relation,
        "question": f"question about {item_id}?",
        "gold_answer": f"gold-{item_id}",
        "gold_aliases": [],
        "raw_generation": "Answer: x",
        "parsed_answer": f"mem-{item_id}",
        "parsed_decision": "answer",
        "parsed_confidence": 100,
        "normalized_answer": f"mem-{item_id}",
        "baseline_correct": correct,
        "prompt_version": "v1",
        "prompt": "irrelevant",
        "generation_config": {"do_sample": False, "max_new_tokens": 32, "num_beams": 1},
        "manual_review": False,
        "knowledge_group": group,
        "primary_conflict_eligible": True,
        "conflict_eligibility_reason": None,
        "memory_answer": f"gold-{item_id}" if group == "KC" else f"mem-{item_id}",
        "conflicting_context_answer": "other",
        "memory_logprob_normalized": -0.1,
        "conflicting_answer_logprob_normalized": -1.1,
        "parametric_margin": margin,
    }
    if group == "KC":
        record["foil_answer"] = f"foil-{item_id}"
        record["foil_source_item_id"] = "999"
        record["foil_generation_method"] = "same_relation_sample"
    return record


def _write_blocks(root, model_key, records, block_size=250):
    block_dir = root / model_key / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    for index in range(0, max(len(records), 1), block_size):
        chunk = records[index : index + block_size]
        path = block_dir / f"block_{index // block_size:04d}.jsonl"
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in chunk),
            encoding="utf-8",
        )
    return block_dir


# ---------------------------------------------------------------------------
# Outcome-blindness of the replay path
# ---------------------------------------------------------------------------


def test_a_returned_block_carrying_an_outcome_field_is_refused(tmp_path):
    """The single property that keeps selection honest.

    If a block came back with a Phase 3 outcome on it, the cohorts derived
    from it would be outcome-dependent. Replay must fail loudly rather than
    silently building a cohort from contaminated input (§11, §16).
    """
    poisoned = _record(1)
    poisoned["context_adopted"] = True
    _write_blocks(tmp_path, "qwen", [poisoned])
    with pytest.raises(ScreeningError, match="prohibited outcome field"):
        fb.replay_screening(tmp_path, "qwen")


def test_replay_refuses_a_model_with_no_returned_blocks(tmp_path):
    with pytest.raises(fb.FreezeBuildError, match="EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE"):
        fb.replay_screening(tmp_path, "qwen")


def test_replay_preserves_block_order_and_record_count(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(300)]
    _write_blocks(tmp_path, "qwen", records)
    finalized = fb.replay_screening(tmp_path, "qwen")
    assert finalized.blocks_screened == 2
    assert finalized.screened_total == 300
    assert [r["item_id"] for r in finalized.records] == [r["item_id"] for r in records]


# ---------------------------------------------------------------------------
# Derived artifacts
# ---------------------------------------------------------------------------


def test_derived_artifacts_are_deterministic_and_hashed(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(30)]
    _write_blocks(tmp_path, "qwen", records)
    finalized = fb.replay_screening(tmp_path, "qwen")

    first = fb.derive_model_artifacts(finalized, tmp_path / "a", raw_records=records)
    second = fb.derive_model_artifacts(finalized, tmp_path / "b", raw_records=records)
    digests_a = {k: v["sha256"] for k, v in first["artifacts"].items()}
    digests_b = {k: v["sha256"] for k, v in second["artifacts"].items()}
    assert digests_a == digests_b, "re-derivation must reproduce identical bytes"
    assert all(len(d) == 64 for d in digests_a.values())


def test_derived_artifacts_never_modify_the_raw_returned_blocks(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(20)]
    block_dir = _write_blocks(tmp_path, "qwen", records)
    before = {p.name: p.read_bytes() for p in block_dir.glob("*.jsonl")}
    finalized = fb.replay_screening(tmp_path, "qwen")
    fb.derive_model_artifacts(finalized, tmp_path / "derived", raw_records=records)
    after = {p.name: p.read_bytes() for p in block_dir.glob("*.jsonl")}
    assert before == after


def test_the_exclusion_file_accounts_for_every_ineligible_record(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(10)]
    dropped = _record(2000)
    dropped["knowledge_group"] = "excluded"
    dropped["exclusion_reason"] = "baseline_uncertain"
    dropped.pop("primary_conflict_eligible")
    records.append(dropped)
    _write_blocks(tmp_path, "qwen", records)
    finalized = fb.replay_screening(tmp_path, "qwen")
    derived = fb.derive_model_artifacts(finalized, tmp_path / "d", raw_records=records)
    payload = json.loads(
        (tmp_path / "d" / "qwen" / "screening_exclusions.json").read_text(encoding="utf-8")
    )
    assert payload["screened_total"] == 11
    assert payload["excluded_total"] == 1
    assert payload["eligible_total"] == 10
    assert payload["excluded"][0]["reason"] == "baseline_uncertain"
    assert derived["summary"]["eligible_total"] == 10


# ---------------------------------------------------------------------------
# Frozen margin boundaries
# ---------------------------------------------------------------------------


def test_frozen_strata_record_the_finalized_edges_not_a_recomputation(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(60)]
    _write_blocks(tmp_path, "qwen", records)
    finalized = fb.replay_screening(tmp_path, "qwen")
    strata = fb.frozen_margin_strata({"qwen": finalized})
    entry = strata["qwen|KW"]
    assert entry["edges"] == list(finalized.stratum_edges[("qwen", "KW")])
    assert entry["strata"] == list(MARGIN_STRATA)
    assert sum(entry["per_stratum_counts"].values()) == entry["assigned_items"] == 60


# ---------------------------------------------------------------------------
# Trial specification: rendering is not running
# ---------------------------------------------------------------------------


class _Entry:
    hf_model_id = "Fake/Model"
    revision = "a" * 40
    preferred_source = "a government website"
    dispreferred_source = "a social media post"
    runs_model_specific_arm = True
    condition_set = ("C0", "K1", "K2", "K3", "K4", "M1", "M2")


class _Config:
    seed = 42

    def __init__(self, entry=None):
        self._entry = entry or _Entry()

    def model(self, key):  # one stand-in model entry for every key
        return self._entry


def _trial_setup(tmp_path, records, model_key="qwen"):
    _write_blocks(tmp_path, model_key, records)
    finalized = fb.replay_screening(tmp_path, model_key)
    membership = {
        f"{model_key}|{r['item_id']}": ["A"] for r in records
    }
    return finalized, membership


def test_trial_specification_constructs_no_adapter_and_requests_no_generation(tmp_path):
    """The structural reason a trial specification cannot become a trial.

    `build_trial_specification` is handed no adapter factory and no model;
    there is no object in scope that could generate. This test pins that by
    checking the module never references a generation entry point.
    """
    import ast
    import inspect

    source = inspect.getsource(fb)
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "generate" not in called
    assert "score_candidate" not in called
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "HFCausalAdapter" not in imported


def test_trial_specification_covers_every_condition_for_an_enabled_arm(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(9)]
    finalized, membership = _trial_setup(tmp_path, records)
    rows, requests = fb.build_trial_specification(
        {"qwen": finalized},
        membership,
        _Config(),
        baseline_template=BASELINE_TEMPLATE,
        evidence_template=EVIDENCE_TEMPLATE,
        prompt_version="v1",
    )
    assert len(rows) == 9 * 7
    assert {r["condition"] for r in rows} == {"C0", "K1", "K2", "K3", "K4", "M1", "M2"}
    assert len(requests["qwen"]) == 63
    # C0 carries no evidence; every other condition does.
    c0 = [r for r in rows if r["condition"] == "C0"]
    assert all(r["source_label"] is None and r["asserted_answer"] is None for r in c0)


def test_a_disabled_arm_yields_no_m_conditions_at_all(tmp_path):
    class _Disabled(_Entry):
        preferred_source = None
        dispreferred_source = None
        runs_model_specific_arm = False
        condition_set = ("C0", "K1", "K2", "K3", "K4")

    records = [_record(1000 + i, margin=float(i)) for i in range(5)]
    finalized, membership = _trial_setup(tmp_path, records)
    rows, _ = fb.build_trial_specification(
        {"qwen": finalized},
        membership,
        _Config(_Disabled()),
        baseline_template=BASELINE_TEMPLATE,
        evidence_template=EVIDENCE_TEMPLATE,
        prompt_version="v1",
    )
    assert {r["condition"] for r in rows} == {"C0", "K1", "K2", "K3", "K4"}
    assert not [r for r in rows if r["condition"] in ("M1", "M2")]


def test_a_cohort_item_missing_from_the_screened_pool_is_refused(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(3)]
    finalized, membership = _trial_setup(tmp_path, records)
    membership["qwen|does-not-exist"] = ["A"]
    with pytest.raises(fb.FreezeBuildError, match="EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE"):
        fb.build_trial_specification(
            {"qwen": finalized},
            membership,
            _Config(),
            baseline_template=BASELINE_TEMPLATE,
            evidence_template=EVIDENCE_TEMPLATE,
            prompt_version="v1",
        )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedup(requests_by_model, config=None):
    return fb.build_dedup_map(
        requests_by_model,
        config or _Config(),
        prompt_version="v1",
        gen_config=GenerationConfig(),
    )


def test_a_coincident_source_pair_collapses_and_the_alias_is_recorded(tmp_path):
    """Qwen's frozen pair IS the common pair, so M1/M2 are prompt-identical
    to two K conditions and must be stored once (§19, §22)."""

    class _Coincident(_Entry):
        preferred_source = "a government website"
        dispreferred_source = "an anonymous online forum post"

    records = [_record(1000 + i, margin=float(i)) for i in range(4)]
    finalized, membership = _trial_setup(tmp_path, records)
    _, requests = fb.build_trial_specification(
        {"qwen": finalized},
        membership,
        _Config(_Coincident()),
        baseline_template=BASELINE_TEMPLATE,
        evidence_template=EVIDENCE_TEMPLATE,
        prompt_version="v1",
    )
    result = _dedup({"qwen": requests["qwen"]}, _Config(_Coincident()))
    per_model = result["per_model"]["qwen"]
    assert per_model["nominal_condition_slots"] == 4 * 7
    # KW items: M1/M2 assert the gold answer from the two common sources,
    # exactly as K1/K2 do -- two collapses per item.
    assert per_model["collapsed_by_deduplication"] == 8
    assert per_model["unique_planned_generations"] == 20
    aliased = [o for o in result["observations"] if o["is_aliased"]]
    conditions = {tuple(sorted(a["condition"] for a in o["aliased_conditions"]))
                  for o in aliased}
    assert conditions == {("K1", "M1"), ("K2", "M2")}


def test_two_models_never_deduplicate_against_each_other(tmp_path):
    """Even byte-identical prompts are different generations on different
    model artifacts (§22)."""
    records = [_record(1000 + i, margin=float(i)) for i in range(3)]
    _write_blocks(tmp_path, "qwen", records)
    _write_blocks(tmp_path, "llama", records)
    finalized = {
        key: fb.replay_screening(tmp_path, key) for key in ("qwen", "llama")
    }
    membership = {
        f"{key}|{r['item_id']}": ["C"] for key in finalized for r in records
    }
    _, requests = fb.build_trial_specification(
        finalized,
        membership,
        _Config(),
        baseline_template=BASELINE_TEMPLATE,
        evidence_template=EVIDENCE_TEMPLATE,
        prompt_version="v1",
    )
    result = _dedup(requests)
    ids_by_model: dict[str, set[str]] = {}
    for obs in result["observations"]:
        ids_by_model.setdefault(obs["model_key"], set()).add(obs["observation_id"])
    assert ids_by_model["qwen"].isdisjoint(ids_by_model["llama"])
    assert result["totals"]["unique_planned_generations"] == sum(
        len(v) for v in ids_by_model.values()
    )


def test_two_items_with_identical_text_stay_separate_observations(tmp_path):
    """Item identity is the scientific unit; identical rendering never
    merges two selected items (§22)."""
    a = _record(1000, margin=1.0)
    b = _record(2000, margin=2.0)
    b["question"] = a["question"]
    b["gold_answer"] = a["gold_answer"]
    b["memory_answer"] = a["memory_answer"]
    records = [a, b]
    finalized, membership = _trial_setup(tmp_path, records)
    _, requests = fb.build_trial_specification(
        {"qwen": finalized},
        membership,
        _Config(),
        baseline_template=BASELINE_TEMPLATE,
        evidence_template=EVIDENCE_TEMPLATE,
        prompt_version="v1",
    )
    result = _dedup({"qwen": requests["qwen"]})
    item_ids = {o["item_id"] for o in result["observations"]}
    assert item_ids == {"1000", "2000"}
    assert result["per_model"]["qwen"]["items"] == 2


def test_unique_generations_never_exceed_nominal_slots(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(6)]
    finalized, membership = _trial_setup(tmp_path, records)
    _, requests = fb.build_trial_specification(
        {"qwen": finalized},
        membership,
        _Config(),
        baseline_template=BASELINE_TEMPLATE,
        evidence_template=EVIDENCE_TEMPLATE,
        prompt_version="v1",
    )
    totals = _dedup({"qwen": requests["qwen"]})["totals"]
    assert totals["unique_planned_generations"] <= totals["nominal_condition_slots"]
    assert totals["collapsed_by_deduplication"] >= 0


# ---------------------------------------------------------------------------
# Cross-cohort membership
# ---------------------------------------------------------------------------


def test_membership_map_makes_cross_cohort_reuse_explicit(tmp_path):
    records = [_record(1000 + i, margin=float(i)) for i in range(40)]
    _write_blocks(tmp_path, "qwen", records)
    finalized = fb.replay_screening(tmp_path, "qwen")
    bundle = fb.build_cohorts(
        {"qwen": finalized},
        seed=42,
        phase2_excluded_ids=frozenset(),
        cohort_c_target=8,
    )
    membership = fb.cross_cohort_membership(bundle)
    assert membership, "every selected item must appear in the map"
    assert all(key.startswith("qwen|") for key in membership)
    reused = [labels for labels in membership.values() if len(labels) > 1]
    assert reused, "an item selected into several cohorts must show all of them"

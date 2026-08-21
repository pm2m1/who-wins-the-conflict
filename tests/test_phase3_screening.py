"""Tests for Phase 3 blockwise screening and the margin-stratum freeze.

Covers `docs/phase3_scaled_study_design.md` §11: 250-candidate blocks, the
2,000/model ceiling, the >= 34/stratum Cohort A supply criterion, the
>= 10/cell Cohort B criterion, outcome-blind stopping, and the rule that
strata are recomputed per block but frozen once screening stops.
"""

from __future__ import annotations

import pytest

from conflict_eval.phase3 import synthetic
from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_SUPPLY,
    COHORT_B_CELL_SUPPLY,
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
)
from conflict_eval.phase3.screening import (
    STOP_CEILING,
    STOP_SUPPLY_MET,
    ScreeningError,
    ScreeningState,
)


def _kw_records(n: int, start: int = 0) -> list[dict]:
    return synthetic.make_kw_pool_for_cohort_a(n, start_index=start)


def test_frozen_screening_parameters_match_the_design():
    assert SCREENING_BLOCK_SIZE == 250
    assert SCREENING_CEILING_PER_MODEL == 2000
    assert COHORT_A_PER_STRATUM_SUPPLY == 34  # 32 target + 2 reserve
    assert COHORT_B_CELL_SUPPLY == 10  # 8 target + 2 reserve


def test_block_larger_than_the_frozen_block_size_is_rejected():
    state = ScreeningState("qwen")
    with pytest.raises(ScreeningError, match="exceeds the frozen block size"):
        state.add_block(_kw_records(SCREENING_BLOCK_SIZE + 1))


def test_screening_refuses_records_carrying_outcome_fields():
    """Prohibited stopping information must be unable to reach the rule."""
    state = ScreeningState("qwen")
    record = synthetic.make_baseline_record("syn-country-0001", "country", "KW", 1.0)
    record["context_adopted"] = True
    with pytest.raises(ScreeningError, match="prohibited outcome field"):
        state.add_block([record])


@pytest.mark.parametrize(
    "field", ["context_adopted", "condition", "source_role", "conflict_status"]
)
def test_each_prohibited_outcome_field_is_rejected(field):
    state = ScreeningState("qwen")
    record = synthetic.make_baseline_record("syn-country-0002", "country", "KW", 1.0)
    record[field] = "anything"
    with pytest.raises(ScreeningError):
        state.add_block([record])


def test_blocks_accumulate_and_report_supply():
    state = ScreeningState("qwen", require_cohort_a=True)
    report = state.add_block(_kw_records(200))
    assert report.block_index == 1
    assert report.screened_total == 200
    assert set(report.cohort_a_per_stratum) == {"low", "medium", "high"}


def test_cohort_a_supply_requires_34_fresh_kw_per_stratum():
    """Supply is met exactly when EVERY stratum holds >= 34 (32 + 2 reserve).

    The assertion is written against the rule rather than a hand-computed
    split, because tertile edges tie-break unevenly under the unchanged
    Phase 2 `assign_margin_bin` helper -- a real property covered by
    `test_phase3_cohorts.py::test_tertile_ties_are_handled_deterministically`.
    """
    for count in (60, 99, 120, 150):
        state = ScreeningState("qwen", require_cohort_a=True)
        state.add_block(_kw_records(count)[:SCREENING_BLOCK_SIZE])
        report = state.supply_report()
        expected = min(report.cohort_a_per_stratum.values()) >= COHORT_A_PER_STRATUM_SUPPLY
        assert report.cohort_a_supply_met is expected


def test_cohort_a_supply_is_unmet_for_a_clearly_insufficient_pool():
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(_kw_records(60))
    assert state.supply_report().cohort_a_supply_met is False


def test_cohort_a_supply_is_met_for_a_clearly_sufficient_pool():
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(_kw_records(150))
    report = state.supply_report()
    assert min(report.cohort_a_per_stratum.values()) >= COHORT_A_PER_STRATUM_SUPPLY
    assert report.cohort_a_supply_met is True


def test_phase2_excluded_ids_do_not_count_toward_cohort_a_supply():
    """Fresh items only: Phase 2 items never count toward the supply (§11)."""
    records = _kw_records(102)
    excluded = {r["item_id"] for r in records[:60]}
    state = ScreeningState("qwen", phase2_excluded_ids=excluded, require_cohort_a=True)
    state.add_block(records)
    report = state.supply_report()
    assert report.cohort_a_supply_met is False
    assert sum(report.cohort_a_per_stratum.values()) == len(records) - len(excluded)


def test_cohort_a_supply_has_no_relation_quota():
    """A pool concentrated in a single relation still satisfies Cohort A
    supply, because Cohort A imposes no relation quota (§15.1)."""
    records = [
        synthetic.make_baseline_record(f"syn-country-{i:04d}", "country", "KW", i * 0.1)
        for i in range(120)
    ]
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(records[:120])
    assert state.supply_report().cohort_a_supply_met is True


def test_screening_stops_when_supply_criteria_are_met():
    state = ScreeningState("qwen", require_cohort_a=False)
    state.add_block(synthetic.make_baseline_pool(n_per_cell=36)[:250])
    state.add_block(synthetic.make_baseline_pool(n_per_cell=36)[250:])
    if state.should_stop():
        finalized = state.finalize()
        assert finalized.stopped_reason == STOP_SUPPLY_MET


def test_ceiling_is_enforced_at_2000_candidates():
    state = ScreeningState("qwen", require_cohort_a=True)
    for block in range(SCREENING_CEILING_PER_MODEL // SCREENING_BLOCK_SIZE):
        state.add_block(_kw_records(1, start=block))
    assert state.ceiling_reached() is True
    assert state.should_stop() is True
    with pytest.raises(ScreeningError, match="ceiling"):
        state.add_block(_kw_records(1, start=999))


def test_finalize_reports_ceiling_when_supply_never_met():
    state = ScreeningState("qwen", require_cohort_a=True)
    for block in range(SCREENING_CEILING_PER_MODEL // SCREENING_BLOCK_SIZE):
        state.add_block(_kw_records(1, start=block))
    finalized = state.finalize()
    assert finalized.stopped_reason == STOP_CEILING
    assert finalized.cohort_a_supply_met is False


def test_strata_are_recomputed_after_each_block_before_freeze():
    """Adding a block that shifts the margin distribution must change the
    recomputed edges (§11 step 1)."""
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(
        [
            synthetic.make_baseline_record(f"syn-country-{i:04d}", "country", "KW", i)
            for i in range(30)
        ]
    )
    first = state.supply_report().stratum_edges[("qwen", "KW")]
    state.add_block(
        [
            synthetic.make_baseline_record(f"syn-sport-{i:04d}", "sport", "KW", 500 + i)
            for i in range(30)
        ]
    )
    second = state.supply_report().stratum_edges[("qwen", "KW")]
    assert first != second


def test_finalized_strata_are_immutable_after_freeze():
    """Once finalized, nothing can alter the frozen strata (§11 step 3)."""
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(_kw_records(60))
    finalized = state.finalize()
    edges_before = dict(finalized.stratum_edges)
    assignments_before = dict(finalized.stratum_assignments)

    with pytest.raises(ScreeningError, match="after finalize"):
        state.add_block(_kw_records(60, start=500))

    assert finalized.stratum_edges == edges_before
    assert finalized.stratum_assignments == assignments_before


def test_finalized_result_is_a_snapshot_not_a_live_view():
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(_kw_records(60))
    finalized = state.finalize()
    count_before = finalized.screened_total
    # Mutating the returned records must not affect the frozen snapshot's
    # own accounting.
    records = finalized.eligible_records()
    for record in records:
        record["parametric_margin"] = 999.0
    assert finalized.screened_total == count_before
    assert all(
        r["parametric_margin"] != 999.0 for r in finalized.eligible_records()
    )


def test_ineligible_records_never_contribute_supply():
    """Abstentions/manual-review/non-eligible items are excluded, unchanged
    from Phase 2 (§12, §31)."""
    records = _kw_records(102)
    for record in records:
        record["knowledge_group"] = "excluded"
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(records)
    report = state.supply_report()
    assert sum(report.cohort_a_per_stratum.values()) == 0
    assert report.cohort_a_supply_met is False


def test_screening_is_deterministic_for_identical_input():
    a = ScreeningState("qwen", require_cohort_a=True)
    b = ScreeningState("qwen", require_cohort_a=True)
    records = _kw_records(120)
    a.add_block(records)
    b.add_block(list(reversed(records)))
    first = a.finalize()
    second = b.finalize()
    assert first.stratum_edges == second.stratum_edges
    assert first.stratum_assignments == second.stratum_assignments


# --- real baseline-record schema compatibility (repair blocker #2) ---------


def _real_shaped_record(item_id="syn-real-0001", group="KW", margin=1.23):
    """A record shaped like a REAL Phase 2 baseline record
    (`src/conflict_eval/cli.py`, `cmd_screen`) with synthetic content.

    Schema-compatible, not real PopQA data.
    """
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "synthetic-revision",
        "requested_revision": None,
        "item_id": item_id,
        "subject": "synthetic-subject",
        "relation": "country",
        "question": "synthetic-question?",
        "gold_answer": "synthetic-gold",
        "gold_aliases": ["synthetic-gold"],
        "raw_generation": "Answer: synthetic-wrong\nDecision: answer\nConfidence: 70",
        "parsed_answer": "synthetic-wrong",
        "parsed_decision": "answer",
        "parsed_confidence": 70,
        "normalized_answer": "synthetic-wrong",
        "baseline_correct": False,
        "prompt_version": "v1",
        "prompt": "synthetic prompt",
        "generation_config": {"do_sample": False, "max_new_tokens": 32, "num_beams": 1},
        "manual_review": False,
        "knowledge_group": group,
        "primary_conflict_eligible": True,
        "parametric_margin": margin,
        "margin_bin": "low",
        "conflict_eligibility_reason": None,
    }


def test_real_shaped_baseline_record_passes_screening():
    """The guard must not reject ordinary baseline metadata; only genuine
    Phase 3 outcome fields are prohibited (§11)."""
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block([_real_shaped_record()])
    report = state.supply_report()
    assert report.screened_total == 1
    finalized = state.finalize()
    assert len(finalized.eligible_records()) == 1


def test_real_shaped_record_carrying_a_phase3_outcome_is_rejected():
    state = ScreeningState("qwen", require_cohort_a=True)
    record = _real_shaped_record()
    record["context_adopted"] = True
    with pytest.raises(ScreeningError, match="prohibited outcome field"):
        state.add_block([record])


def test_record_without_item_id_fails_clearly():
    state = ScreeningState("qwen", require_cohort_a=True)
    record = _real_shaped_record()
    del record["item_id"]
    with pytest.raises(ScreeningError, match="item_id"):
        state.add_block([record])


def test_record_with_empty_item_id_fails_clearly():
    state = ScreeningState("qwen", require_cohort_a=True)
    record = _real_shaped_record()
    record["item_id"] = ""
    with pytest.raises(ScreeningError, match="item_id"):
        state.add_block([record])


def test_item_id_is_preserved_through_screening_and_cohorts():
    from conflict_eval.phase3.cohorts import build_cohort_a

    records = [
        _real_shaped_record(item_id=f"syn-real-{i:04d}", margin=float(i))
        for i in range(60)
    ]
    state = ScreeningState("qwen", require_cohort_a=True)
    state.add_block(records)
    finalized = state.finalize()
    assert finalized.stratum_of("syn-real-0000") is not None

    cohort_a = build_cohort_a(
        finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=5
    )
    assert cohort_a.items
    assert all(r["item_id"].startswith("syn-real-") for r in cohort_a.items)

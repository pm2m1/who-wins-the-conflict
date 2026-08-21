"""Tests for the three Phase 3 cohorts.

Covers `docs/phase3_scaled_study_design.md` §15 (Cohort A / Cohort B),
§16 (Cohort C) and §32 (the missing-cell ladder). The central structural
property under test is the frozen independence guarantee: **a Cohort B
eligibility failure never invalidates Cohort A** (§15, §34).
"""

from __future__ import annotations

import pytest

from conflict_eval.phase3 import synthetic
from conflict_eval.phase3.cohorts import (
    STATE_COMPLETE,
    STATE_DOWNSAMPLED,
    STATE_ELIGIBILITY_LIMITED,
    STATE_ELIGIBILITY_LIMITED_EXPLORATORY,
    build_cohort_a,
    build_cohort_b_group,
    build_cohort_c,
)
from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_TARGET,
    COHORT_A_TOTAL_TARGET,
    COHORT_B_CELL_MINIMUM,
    COHORT_B_CELL_TARGET,
    MARGIN_STRATA,
    PHASE3_RELATIONS,
)
from conflict_eval.phase3.screening import ScreeningState


def _finalize(records: list[dict], model_key: str = "qwen", excluded=None):
    state = ScreeningState(
        model_key, phase2_excluded_ids=excluded or set(), require_cohort_a=False
    )
    for start in range(0, len(records), 250):
        state.add_block(records[start : start + 250])
    return state.finalize()


def test_frozen_cohort_targets_match_the_design():
    assert COHORT_A_PER_STRATUM_TARGET == 32
    assert COHORT_A_TOTAL_TARGET == 96
    assert COHORT_B_CELL_TARGET == 8
    assert COHORT_B_CELL_MINIMUM == 6
    assert PHASE3_RELATIONS == ("country", "sport", "place of birth", "mother")


# --- Cohort A -------------------------------------------------------------


def test_cohort_a_excludes_supplied_phase2_ids():
    records = synthetic.make_kw_pool_for_cohort_a(120)
    excluded = {r["item_id"] for r in records[:40]}
    finalized = _finalize(records, excluded=excluded)
    result = build_cohort_a(
        finalized, seed=1, phase2_excluded_ids=excluded, per_stratum_target=4
    )
    selected = {r["item_id"] for r in result.items}
    assert selected.isdisjoint(excluded)
    assert result.excluded_phase2_count > 0


def test_cohort_a_selects_equal_counts_per_margin_stratum():
    records = synthetic.make_kw_pool_for_cohort_a(150)
    finalized = _finalize(records)
    result = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=10)
    assert result.state == STATE_COMPLETE
    assert result.per_stratum_selected == {s: 10 for s in MARGIN_STRATA}
    assert len(result.items) == 30


def test_cohort_a_imposes_no_relation_quota():
    """A pool dominated by one relation still yields a COMPLETE Cohort A --
    the whole point of dropping the relation quota (§15.1)."""
    records = [
        synthetic.make_baseline_record(f"syn-country-{i:04d}", "country", "KW", i * 0.1)
        for i in range(90)
    ]
    finalized = _finalize(records)
    result = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=10)
    assert result.state == STATE_COMPLETE
    assert set(result.relation_distribution) == {"country"}
    assert result.relation_dominance_share == pytest.approx(1.0)


def test_cohort_a_relation_dominance_flag_is_diagnostic_and_non_gating():
    records = [
        synthetic.make_baseline_record(f"syn-country-{i:04d}", "country", "KW", i * 0.1)
        for i in range(90)
    ]
    finalized = _finalize(records)
    result = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=10)
    # Flag is raised, yet the cohort is still COMPLETE and fully selected.
    assert result.relation_dominance_flag is True
    assert result.state == STATE_COMPLETE
    assert not result.is_eligibility_limited
    assert len(result.items) == 30


def test_cohort_a_is_eligibility_limited_only_when_strata_cannot_be_filled():
    records = synthetic.make_kw_pool_for_cohort_a(15)
    finalized = _finalize(records)
    result = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=32)
    assert result.state == STATE_ELIGIBILITY_LIMITED
    assert any(v > 0 for v in result.shortfall.values())


def test_cohort_a_ignores_non_primary_relations():
    records = [
        synthetic.make_baseline_record(f"syn-genre-{i:04d}", "genre", "KW", i * 0.1)
        for i in range(60)
    ]
    finalized = _finalize(records)
    result = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=4)
    assert result.items == ()


def test_cohort_a_only_uses_kw_items():
    records = synthetic.make_baseline_pool(n_per_cell=30)
    finalized = _finalize(records)
    result = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=6)
    assert all(r["knowledge_group"] == "KW" for r in result.items)


def test_cohort_a_selection_is_deterministic_for_a_given_seed():
    records = synthetic.make_kw_pool_for_cohort_a(150)
    finalized = _finalize(records)
    first = build_cohort_a(finalized, seed=7, phase2_excluded_ids=set(), per_stratum_target=10)
    second = build_cohort_a(finalized, seed=7, phase2_excluded_ids=set(), per_stratum_target=10)
    assert [r["item_id"] for r in first.items] == [r["item_id"] for r in second.items]


def test_cohort_a_selection_changes_with_the_seed():
    records = synthetic.make_kw_pool_for_cohort_a(150)
    finalized = _finalize(records)
    first = build_cohort_a(finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=10)
    second = build_cohort_a(finalized, seed=2, phase2_excluded_ids=set(), per_stratum_target=10)
    assert [r["item_id"] for r in first.items] != [r["item_id"] for r in second.items]


# --- Cohort B -------------------------------------------------------------


def _cell_counts(result, finalized):
    counts: dict[tuple[str, str], int] = {}
    for record in result.items:
        key = (record["relation"], finalized.stratum_of(record["item_id"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _pool(per_relation: dict[str, int], group: str = "KW", spread: float = 30.0):
    """Synthetic pool with an exact per-relation item count.

    Margins span the same range for every relation so the shared tertile
    edges split each relation comparably.
    """
    records = []
    for relation, count in per_relation.items():
        for i in range(count):
            records.append(
                synthetic.make_baseline_record(
                    synthetic.synthetic_item_id(f"{relation}-{group}", i),
                    relation,
                    group,
                    round(((i + 0.5) / max(count, 1)) * spread, 6),
                )
            )
    return records


def test_cohort_b_all_twelve_cells_at_target():
    """§32 rule 1: every cell >= 8 -> exactly 8 per cell, all 4 relations."""
    finalized = _finalize(_pool({r: 27 for r in PHASE3_RELATIONS}))
    result = build_cohort_b_group(finalized, "KW", seed=1)
    assert result.state == STATE_COMPLETE
    assert result.realized_cell_count == COHORT_B_CELL_TARGET
    assert result.qualifying_relations == PHASE3_RELATIONS
    assert result.confirmatory_eligible is True
    assert set(_cell_counts(result, finalized).values()) == {8}


def test_cohort_b_cells_above_target_are_capped_at_eight():
    """§32 rule 2: surplus cells are downsampled to the common target."""
    finalized = _finalize(_pool({r: 60 for r in PHASE3_RELATIONS}))
    result = build_cohort_b_group(finalized, "KW", seed=1)
    assert result.state == STATE_COMPLETE
    assert result.realized_cell_count == 8
    assert set(_cell_counts(result, finalized).values()) == {8}


@pytest.mark.parametrize("per_relation_total,expected_cell", [(24, 7), (21, 6)])
def test_cohort_b_downsamples_every_cell_to_the_common_realized_count(
    per_relation_total, expected_cell
):
    """§32 rule 3: all relations qualify but some cell < 8 -> every cell is
    reduced to the same realized count, preserving exact balance."""
    finalized = _finalize(_pool({r: per_relation_total for r in PHASE3_RELATIONS}))
    result = build_cohort_b_group(finalized, "KW", seed=1)
    assert result.state == STATE_DOWNSAMPLED
    assert result.realized_cell_count == expected_cell
    assert COHORT_B_CELL_MINIMUM <= result.realized_cell_count < COHORT_B_CELL_TARGET
    assert result.qualifying_relations == PHASE3_RELATIONS
    assert result.confirmatory_eligible is True
    assert set(_cell_counts(result, finalized).values()) == {expected_cell}


def test_cohort_b_one_short_relation_retains_the_other_three():
    """§32 rule 4, second bullet: the balanced estimate is computed over the
    remaining relations -- items are NOT discarded."""
    pool = _pool({r: 30 for r in PHASE3_RELATIONS if r != "mother"})
    pool += _pool({"mother": 9})  # ~3 per cell -> below the minimum
    finalized = _finalize(pool)
    result = build_cohort_b_group(finalized, "KW", seed=1)

    assert result.state == STATE_ELIGIBILITY_LIMITED
    assert result.items != ()  # the crucial repair
    assert "mother" in result.excluded_short_relations
    assert set(result.qualifying_relations) == set(PHASE3_RELATIONS) - {"mother"}
    assert len(result.qualifying_relations) == 3
    assert result.confirmatory_eligible is True  # 3 of 4 still confirmatory
    counts = _cell_counts(result, finalized)
    assert set(counts.values()) == {result.realized_cell_count}
    assert all(relation != "mother" for relation, _ in counts)
    assert result.reduction_reason


def test_cohort_b_completely_absent_relation_behaves_the_same_way():
    finalized = _finalize(_pool({r: 30 for r in PHASE3_RELATIONS if r != "sport"}))
    result = build_cohort_b_group(finalized, "KW", seed=1)
    assert result.state == STATE_ELIGIBILITY_LIMITED
    assert result.items != ()
    assert "sport" in result.excluded_short_relations
    assert len(result.qualifying_relations) == 3
    assert result.confirmatory_eligible is True


def test_cohort_b_two_short_relations_become_exploratory():
    """§32 rule 4, third bullet: fewer than three qualifying relations ->
    removed from the confirmatory families, still reported descriptively."""
    pool = _pool({"country": 30, "sport": 30})
    pool += _pool({"place of birth": 9, "mother": 9})
    finalized = _finalize(pool)
    result = build_cohort_b_group(finalized, "KW", seed=1)

    assert result.state == STATE_ELIGIBILITY_LIMITED_EXPLORATORY
    assert result.confirmatory_eligible is False
    assert len(result.qualifying_relations) == 2
    assert result.items != ()  # still described
    assert result.reduction_reason


def test_cohort_b_all_relations_short_is_eligibility_limited_with_no_items():
    finalized = _finalize(_pool({r: 9 for r in PHASE3_RELATIONS}))
    result = build_cohort_b_group(finalized, "KW", seed=1)
    assert result.state == STATE_ELIGIBILITY_LIMITED_EXPLORATORY
    assert result.confirmatory_eligible is False
    assert result.qualifying_relations == ()
    assert result.items == ()
    assert result.deficient_cells


def test_cohort_b_never_backfills_a_short_relation_from_a_full_one():
    """The short relation contributes nothing, and the retained relations
    are not inflated to compensate."""
    pool = _pool({r: 40 for r in PHASE3_RELATIONS if r != "mother"})
    pool += _pool({"mother": 9})
    finalized = _finalize(pool)
    result = build_cohort_b_group(finalized, "KW", seed=1)
    counts = _cell_counts(result, finalized)
    assert all(relation != "mother" for relation, _ in counts)
    assert set(counts.values()) == {result.realized_cell_count}
    assert result.realized_cell_count <= COHORT_B_CELL_TARGET


def test_cohort_b_is_deterministic_under_input_permutation():
    pool = _pool({r: 30 for r in PHASE3_RELATIONS})
    first = build_cohort_b_group(_finalize(pool), "KW", seed=5)
    second = build_cohort_b_group(_finalize(list(reversed(pool))), "KW", seed=5)
    assert [r["item_id"] for r in first.items] == [r["item_id"] for r in second.items]
    assert first.realized_cell_count == second.realized_cell_count


def test_cohort_b_failure_does_not_invalidate_cohort_a():
    """The frozen independence property (§15, §34): B can be reduced to
    EXPLORATORY while A is still COMPLETE from the same screening."""
    records = synthetic.make_baseline_pool(
        n_per_cell=40, relations=("country", "sport")
    )
    records += synthetic.make_baseline_pool(
        n_per_cell=2, relations=("place of birth", "mother"), start_index=900
    )
    finalized = _finalize(records)

    cohort_b = build_cohort_b_group(finalized, "KW", seed=1)
    cohort_a = build_cohort_a(
        finalized, seed=1, phase2_excluded_ids=set(), per_stratum_target=10
    )

    assert cohort_b.is_eligibility_limited
    assert cohort_b.confirmatory_eligible is False
    assert cohort_a.state == STATE_COMPLETE
    assert not cohort_a.is_eligibility_limited
    assert len(cohort_a.items) == 30


def test_cohort_b_kc_and_kw_are_built_independently():
    records = synthetic.make_baseline_pool(n_per_cell=36, groups=("KW",))
    records += synthetic.make_baseline_pool(
        n_per_cell=3, groups=("KC",), start_index=900
    )
    finalized = _finalize(records)
    assert build_cohort_b_group(finalized, "KW", seed=1).state == STATE_COMPLETE
    assert build_cohort_b_group(finalized, "KC", seed=1).is_eligibility_limited


def test_tertile_ties_are_handled_deterministically():
    """Repeated/tied margins must not crash or produce nondeterminism; the
    unchanged Phase 2 helper may split unevenly, which is acceptable
    because strata are sampling devices, not latent categories (§14)."""
    records = [
        synthetic.make_baseline_record(f"syn-country-{i:04d}", "country", "KW", 5.0)
        for i in range(30)
    ]
    first = _finalize(records)
    second = _finalize(list(reversed(records)))
    assert first.stratum_assignments == second.stratum_assignments


# --- Cohort C -------------------------------------------------------------


def test_cohort_c_preserves_per_model_knowledge_state():
    """A shared item may be KC for one model and KW for another; labels are
    never forced to agree (§16)."""
    shared_ids = [f"syn-country-{i:04d}" for i in range(30)]
    qwen_records = [
        synthetic.make_baseline_record(i, "country", "KW", n * 0.5)
        for n, i in enumerate(shared_ids)
    ]
    llama_records = [
        synthetic.make_baseline_record(i, "country", "KC", n * 0.5)
        for n, i in enumerate(shared_ids)
    ]
    result = build_cohort_c(
        {"qwen": _finalize(qwen_records), "llama": _finalize(llama_records, "llama")},
        seed=1,
        target_size=4,
    )
    assert result.items
    for item in result.items:
        assert item.knowledge_group_for("qwen") == "KW"
        assert item.knowledge_group_for("llama") == "KC"
    assert result.label_disagreement_count == len(result.items)
    assert result.label_agreement_count == 0


def test_cohort_c_only_retains_items_present_for_every_model():
    qwen_records = [
        synthetic.make_baseline_record(f"syn-country-{i:04d}", "country", "KW", i * 0.5)
        for i in range(30)
    ]
    llama_records = qwen_records[:10]
    result = build_cohort_c(
        {"qwen": _finalize(qwen_records), "llama": _finalize(llama_records, "llama")},
        seed=1,
        target_size=40,
    )
    selected = {i.item_id for i in result.items}
    assert selected <= {r["item_id"] for r in llama_records}
    assert result.candidates_considered == 10


def test_cohort_c_selection_is_deterministic():
    records = synthetic.make_baseline_pool(n_per_cell=12)
    finalized = {"qwen": _finalize(records), "llama": _finalize(records, "llama")}
    first = build_cohort_c(finalized, seed=3, target_size=8)
    second = build_cohort_c(finalized, seed=3, target_size=8)
    assert [i.item_id for i in first.items] == [i.item_id for i in second.items]


def test_cohort_c_requires_at_least_one_model():
    with pytest.raises(ValueError):
        build_cohort_c({}, seed=1, target_size=4)


# --- Cohort A exclusion-set API hardening (repair #4) ---------------------


def test_cohort_a_requires_an_explicit_exclusion_set():
    """Omitting the exclusion set must fail loudly: item freshness defines
    the direct replication (§15.1), so it cannot be skipped by accident."""
    finalized = _finalize(synthetic.make_kw_pool_for_cohort_a(60))
    with pytest.raises(TypeError):
        build_cohort_a(finalized, seed=1, per_stratum_target=4)


def test_cohort_a_exclusion_set_must_be_keyword_only():
    finalized = _finalize(synthetic.make_kw_pool_for_cohort_a(60))
    with pytest.raises(TypeError):
        build_cohort_a(finalized, 1, set(), 4)


def test_cohort_a_accepts_an_explicit_synthetic_exclusion_set():
    records = synthetic.make_kw_pool_for_cohort_a(90)
    excluded = {r["item_id"] for r in records[:10]}
    finalized = _finalize(records, excluded=excluded)
    result = build_cohort_a(
        finalized, seed=1, phase2_excluded_ids=excluded, per_stratum_target=5
    )
    assert result.items
    assert {r["item_id"] for r in result.items}.isdisjoint(excluded)


def test_cohort_a_exclusion_metadata_reaches_the_manifest():
    from conflict_eval.phase3.manifest import cohort_a_provenance

    records = synthetic.make_kw_pool_for_cohort_a(90)
    excluded = {r["item_id"] for r in records[:10]}
    finalized = _finalize(records, excluded=excluded)
    result = build_cohort_a(
        finalized, seed=1, phase2_excluded_ids=excluded, per_stratum_target=5
    )
    provenance = cohort_a_provenance(result, excluded)
    assert set(provenance["excluded_phase2_item_ids"]) == excluded
    assert provenance["realized_relation_distribution"]
    assert provenance["relation_dominance_flag"] in (True, False)
    assert provenance["relation_dominance_share"] is not None


def test_cohort_b_reduction_metadata_reaches_the_manifest():
    from conflict_eval.phase3.manifest import cohort_b_provenance

    pool = _pool({r: 30 for r in PHASE3_RELATIONS if r != "mother"})
    pool += _pool({"mother": 9})
    result = build_cohort_b_group(_finalize(pool), "KW", seed=1)
    provenance = cohort_b_provenance(result)
    assert provenance["excluded_short_relations"] == ["mother"]
    assert len(provenance["qualifying_relations"]) == 3
    assert provenance["original_cell_counts"]
    assert provenance["reduction_reason"]
    assert provenance["confirmatory_eligible"] is True
    assert provenance["realized_cell_count"] == result.realized_cell_count


def test_cohort_c_per_model_knowledge_reaches_the_manifest():
    from conflict_eval.phase3.manifest import cohort_c_provenance

    shared = [f"syn-country-{i:04d}" for i in range(30)]
    qwen = [
        synthetic.make_baseline_record(i, "country", "KW", n * 0.5)
        for n, i in enumerate(shared)
    ]
    llama = [
        synthetic.make_baseline_record(i, "country", "KC", n * 0.5)
        for n, i in enumerate(shared)
    ]
    result = build_cohort_c(
        {"qwen": _finalize(qwen), "llama": _finalize(llama, "llama")},
        seed=1,
        target_size=4,
    )
    provenance = cohort_c_provenance(result)
    assert provenance["items"]
    entry = provenance["items"][0]
    assert entry["per_model"]["qwen"]["knowledge_group"] == "KW"
    assert entry["per_model"]["llama"]["knowledge_group"] == "KC"
    assert entry["per_model"]["qwen"]["margin_stratum"] is not None

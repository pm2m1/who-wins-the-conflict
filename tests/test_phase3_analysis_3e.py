"""Tests for the Phase 3E preregistered analysis.

The statistical procedures themselves are tested in
`test_phase3_paired_stats.py`. What is pinned here is the part that decides
*what a number means*: the fixed §37 evaluation order, the §30 saturation
guard, the §28 family boundaries, and the §22 count-once rule.

Every case is synthetic and constructed to isolate one rule, because the
real observations must never be the thing a test is calibrated against.
"""

from __future__ import annotations

import pytest

from conflict_eval.phase3 import analysis_3e as a3e
from conflict_eval.phase3.paired_stats import paired_source_result


def _contrast(a_only, b_only, both, neither, **extra):
    """A contrast dict shaped exactly like the analysis produces."""
    outcomes = (
        [(True, True)] * both
        + [(True, False)] * a_only
        + [(False, True)] * b_only
        + [(False, False)] * neither
    )
    payload = a3e.result_to_dict(paired_source_result(outcomes))
    payload["estimable"] = True
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# §30 / §37: saturation is checked FIRST
# ---------------------------------------------------------------------------


def test_a_saturated_contrast_is_inconclusive_even_though_delta_is_positive():
    """The single most important guard in the whole classification.

    A ceiling regime produces a positive-but-tiny delta with almost no
    discordance. Read naively it looks like a weak effect; §37 requires it
    to be INCONCLUSIVE, and the other categories are not even considered.
    """
    contrast = _contrast(a_only=2, b_only=0, both=97, neither=1)
    assert contrast["diagnostics"]["saturated_uninformative"] is True
    result = a3e.classify_replication(contrast)
    assert result["category"] == a3e.INCONCLUSIVE
    assert any("SATURATED" in r for r in result["reasons"])


def test_a_floor_regime_is_also_inconclusive_not_a_null():
    contrast = _contrast(a_only=1, b_only=0, both=0, neither=99)
    assert contrast["diagnostics"]["near_boundary"] is True
    assert a3e.classify_replication(contrast)["category"] == a3e.INCONCLUSIVE


def test_too_few_discordant_pairs_is_inconclusive_regardless_of_direction():
    """Below the discordance floor the manipulation had no opportunity to
    express an effect, so neither a null nor an effect may be claimed."""
    contrast = _contrast(a_only=4, b_only=0, both=40, neither=40)
    assert contrast["diagnostics"]["saturated_uninformative"] is False
    result = a3e.classify_replication(contrast)
    assert result["category"] == a3e.INCONCLUSIVE
    assert any("discordant pairs" in r for r in result["reasons"])


def test_an_interval_covering_both_zero_and_the_phase2_point_is_inconclusive():
    """Compatible with no effect and with the full Phase 2 effect at once."""
    contrast = _contrast(a_only=7, b_only=4, both=5, neither=5)
    result = a3e.classify_replication(contrast)
    assert result["inputs"]["ci_excludes_zero"] is False
    assert result["inputs"]["ci_contains_phase2_point"] is True
    assert result["category"] == a3e.INCONCLUSIVE


# ---------------------------------------------------------------------------
# §37: the four primary categories
# ---------------------------------------------------------------------------


def test_full_replication_requires_the_interval_to_contain_the_phase2_point():
    contrast = _contrast(a_only=24, b_only=0, both=56, neither=16)
    result = a3e.classify_replication(contrast)
    assert result["category"] == a3e.FULL_REPLICATION
    assert result["inputs"]["ci_contains_phase2_point"] is True


def test_a_genuine_but_smaller_effect_is_attenuated_not_full():
    """A Phase 3 estimate below the pilot's is evidence about the effect's
    size, not evidence that Phase 3 is wrong (§37)."""
    contrast = _contrast(a_only=14, b_only=0, both=30, neither=156)
    result = a3e.classify_replication(contrast)
    assert result["inputs"]["delta"] < a3e.PHASE2_QWEN_CORRECTIVE_DELTA
    assert result["inputs"]["ci_excludes_zero"] is True
    assert result["inputs"]["ci_contains_phase2_point"] is False
    assert result["category"] == a3e.ATTENUATED_REPLICATION


def test_an_informative_negative_result_is_a_non_replication():
    contrast = _contrast(a_only=3, b_only=20, both=30, neither=43)
    result = a3e.classify_replication(contrast)
    assert result["inputs"]["delta"] < 0
    assert result["category"] == a3e.NON_REPLICATION


def test_an_eligibility_limited_cohort_a_forces_inconclusive():
    contrast = _contrast(a_only=24, b_only=0, both=56, neither=16)
    result = a3e.classify_replication(contrast, cohort_eligibility_limited=True)
    assert result["category"] == a3e.INCONCLUSIVE


# ---------------------------------------------------------------------------
# §37 / §28: secondary contrasts
# ---------------------------------------------------------------------------


def test_a_secondary_contrast_must_survive_holm_to_be_confirmed():
    contrast = _contrast(a_only=24, b_only=0, both=56, neither=16)
    assert (
        a3e.classify_secondary(contrast, holm_adjusted_p=0.001)["category"]
        == a3e.DIRECTIONAL_EFFECT_CONFIRMED
    )


def test_failing_holm_does_not_manufacture_a_non_replication():
    """§37 makes Holm survival a condition for calling a secondary contrast
    confirmed. It does not say a Holm-failing positive result is a
    NON-REPLICATION, whose definition is `Delta <= 0` or an interval
    containing 0."""
    contrast = _contrast(a_only=24, b_only=0, both=56, neither=16)
    result = a3e.classify_secondary(contrast, holm_adjusted_p=0.20)
    assert result["category"] == a3e.DIRECTIONAL_NOT_MULTIPLICITY_SURVIVING
    assert result["category"] != a3e.NON_REPLICATION


def test_a_counted_once_row_is_not_treated_as_a_failed_test():
    """Qwen's common-arm contrast shares observations with the primary; it
    is reported once and has no pass/fail of its own (§19, §22, §28)."""
    contrast = _contrast(
        a_only=15, b_only=0, both=32, neither=55,
        counted_once_with="cohort_a_qwen_corrective_frozen_pair",
    )
    result = a3e.classify_secondary(contrast, holm_adjusted_p=None)
    assert result["category"] == a3e.COUNTED_ONCE


def test_secondary_saturation_is_checked_before_holm():
    contrast = _contrast(a_only=0, b_only=0, both=53, neither=1)
    result = a3e.classify_secondary(contrast, holm_adjusted_p=0.001)
    assert result["category"] == a3e.INCONCLUSIVE


# ---------------------------------------------------------------------------
# §22: an aliased observation is counted once
# ---------------------------------------------------------------------------


def test_one_generation_serves_every_condition_that_aliases_it():
    manifest = {
        "deduplication_alias_map": {
            "qwen|1|K1": "obs1",
            "qwen|1|M1": "obs1",
            "qwen|1|K2": "obs2",
        }
    }
    records = [
        {"observation_id": "obs1", "context_adopted": True},
        {"observation_id": "obs2", "context_adopted": False},
    ]
    lookup = a3e.build_outcome_lookup(manifest, records)
    assert lookup[("qwen", "1", "K1")] is True
    assert lookup[("qwen", "1", "M1")] is True
    assert lookup[("qwen", "1", "K2")] is False
    # One item, one pair -- the shared observation does not create a second.
    assert a3e.paired_outcomes(lookup, "qwen", ["1"], "K1", "K2") == [(True, False)]


def test_a_missing_outcome_is_refused_rather_than_read_as_false():
    manifest = {"deduplication_alias_map": {"qwen|1|K1": "obs1"}}
    with pytest.raises(a3e.Phase3EError, match="non-boolean"):
        a3e.build_outcome_lookup(
            manifest, [{"observation_id": "obs1", "context_adopted": None}]
        )


def test_a_planned_trial_with_no_returned_observation_is_refused():
    manifest = {"deduplication_alias_map": {"qwen|1|K1": "missing"}}
    with pytest.raises(a3e.Phase3EError, match="was not returned"):
        a3e.build_outcome_lookup(manifest, [])


def test_an_incomplete_pair_cannot_contribute():
    lookup = {("qwen", "1", "K1"): True}
    assert a3e.paired_outcomes(lookup, "qwen", ["1"], "K1", "K2") == []


# ---------------------------------------------------------------------------
# §28 / §32: family membership
# ---------------------------------------------------------------------------


def test_an_ineligible_cohort_b_row_is_removed_from_the_holm_family():
    declared = ["a", "b"]
    results = {
        "a": {"holm_p_value": 0.01},
        "b": {
            "holm_p_value": 0.01,
            "eligibility_gate": {"eligible": False, "reason": "fewer than three"},
        },
    }
    members, adjusted = a3e.secondary_family(declared, results, {})
    assert sorted(adjusted) == ["a"]
    assert [m.included for m in members] == [True, False]
    assert "fewer than three" in next(m.reason for m in members if not m.included)


def test_a_row_with_no_single_test_statistic_is_recorded_not_silently_dropped():
    declared = ["a", "b"]
    results = {
        "a": {"holm_p_value": 0.01},
        "b": {"holm_p_value": None, "holm_exclusion_reason": "interval only"},
    }
    members, adjusted = a3e.secondary_family(declared, results, {})
    excluded = [m for m in members if not m.included]
    assert len(excluded) == 1
    assert excluded[0].reason == "interval only"
    assert sorted(adjusted) == ["a"]


def test_holm_uses_the_realized_family_size():
    declared = ["a", "b", "c"]
    results = {name: {"holm_p_value": 0.01} for name in declared}
    _, adjusted = a3e.secondary_family(declared, results, {})
    assert adjusted["a"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# §26.2: the H1 ladder
# ---------------------------------------------------------------------------


def _h1_inputs(separating: bool):
    lookup, attributes, items = {}, {}, []
    for i in range(60):
        item_id = str(i)
        items.append(item_id)
        margin = i / 10.0
        attributes[("m", item_id)] = {
            "knowledge_group": "KW",
            "relation": "country",
            "parametric_margin": margin,
            "margin_stratum": "low",
        }
        outcome = (margin > 3.0) if separating else (i % 3 == 0)
        lookup[("m", item_id, "K1")] = outcome
        lookup[("m", item_id, "K2")] = outcome
    return lookup, attributes, items


def test_h1_uses_the_primary_specification_when_it_is_estimable():
    lookup, attributes, items = _h1_inputs(separating=False)
    result = a3e.h1_margin_effect(
        lookup, attributes, model_key="m", item_ids=items,
        conflict_conditions=a3e.COMMON_CONFLICT_PAIR,
    )
    assert result["estimable"] is True
    assert result["method"].startswith("ordinary logistic regression")


def test_h1_falls_back_to_firth_under_separation_and_says_so():
    """The exact Phase 2 Llama failure. Firth's estimate is defined where
    the ordinary MLE diverges, and it must be labelled as penalized."""
    lookup, attributes, items = _h1_inputs(separating=True)
    result = a3e.h1_margin_effect(
        lookup, attributes, model_key="m", item_ids=items,
        conflict_conditions=a3e.COMMON_CONFLICT_PAIR,
    )
    assert result["estimable"] is True
    assert "Firth" in result["method"]
    assert "Firth" in result["status"]


def test_h1_reports_not_estimable_rather_than_inventing_a_null():
    lookup = {("m", "1", "K1"): True, ("m", "1", "K2"): True}
    attributes = {
        ("m", "1"): {
            "knowledge_group": "KW", "relation": "country",
            "parametric_margin": 1.0, "margin_stratum": "low",
        }
    }
    result = a3e.h1_margin_effect(
        lookup, attributes, model_key="m", item_ids=["1"],
        conflict_conditions=a3e.COMMON_CONFLICT_PAIR,
    )
    assert result["estimable"] is False
    assert result["status"] == "NOT ESTIMABLE"


# ---------------------------------------------------------------------------
# Common-arm pairing
# ---------------------------------------------------------------------------


def test_the_conflict_pair_depends_on_the_knowledge_group():
    """KW items conflict under correct evidence (K1/K2); KC items conflict
    under false evidence (K3/K4). Using one pair for both would compare
    agreement trials against conflict trials (§22)."""
    lookup = {
        ("m", "kw", "K1"): True, ("m", "kw", "K2"): False,
        ("m", "kc", "K3"): False, ("m", "kc", "K4"): True,
    }
    attributes = {
        ("m", "kw"): {"knowledge_group": "KW"},
        ("m", "kc"): {"knowledge_group": "KC"},
    }
    pairs = a3e.pooled_conflict_pairs(
        lookup, attributes, model_key="m", item_ids=["kw", "kc"],
        pair_by_group=a3e.COMMON_CONFLICT_PAIR,
    )
    assert sorted(pairs) == [(False, True), (True, False)]

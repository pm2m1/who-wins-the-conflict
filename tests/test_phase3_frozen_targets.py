"""Frozen-design target enforcement for the Phase 3 freeze/readiness layer.

Regression tests for the second-audit blocker: a non-synthetic manifest was
able to redefine the frozen Phase 3 design constants (Cohort A target 18 /
6-6-6, Cohort B target 3 / minimum 1) and still freeze.

The distinction these tests pin down is the one the frozen protocol makes:

- the **planned** design (96 items, 32/32/32 strata, Cohort B target 8 and
  minimum 6, 250-candidate blocks, a 2,000 ceiling, seven conditions, the
  frozen common source pair) is fixed by
  `docs/phase3_scaled_study_design.md` and may never be rewritten;
- the **realized** cohort may legitimately fall short -- a Cohort B cell at
  6 or 7, a three-relation fallback, or an eligibility-limited Cohort A --
  provided the manifest says so and records why (§32, §15.1, §34).

Rejecting the first while still accepting the second is the whole point.
"""

from __future__ import annotations

import pytest
from test_phase3_config_and_gate import (  # shared §36 fixture provenance
    ARTIFACT_HASHES_36,
    DATASET_36,
    ENVIRONMENT_36,
    HARDWARE_36,
    model_provenance_36,
)

from conflict_eval.phase3.config import load_phase3_config
from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_TARGET,
    COHORT_A_TOTAL_TARGET,
    COHORT_B_CELL_MINIMUM,
    COHORT_B_CELL_TARGET,
    FROZEN_MODEL_REVISIONS,
    MARGIN_STRATA,
)
from conflict_eval.phase3.manifest import (
    ManifestError,
    build_manifest,
    freeze_manifest,
    planned_cohort_a_design,
    planned_screening_design,
    validate_manifest,
)
from conflict_eval.phase3.real_run_gate import check_readiness

COMMITTED_CONFIG = "configs/phase3/phase3_study.yaml"
REL_ALL = ["country", "sport", "place of birth", "mother"]


def _design_manifest(
    *,
    a_planned_total=COHORT_A_TOTAL_TARGET,
    a_planned_strata=None,
    a_realized=COHORT_A_TOTAL_TARGET,
    a_strata=None,
    a_status="COMPLETE",
    a_shortfall=None,
    stopped="supply_criteria_met",
    b_target=COHORT_B_CELL_TARGET,
    b_minimum=COHORT_B_CELL_MINIMUM,
    b_realized_cell=COHORT_B_CELL_TARGET,
    b_qualifying=None,
    b_short=(),
    b_confirmatory=True,
):
    """A non-synthetic, otherwise-complete manifest whose design and
    realized values the caller can vary one at a time."""
    qualifying = list(REL_ALL if b_qualifying is None else b_qualifying)
    cohort_a = {
        **planned_cohort_a_design(),
        "selected_item_ids": [f"i{i}" for i in range(a_realized)],
        "realized_total": a_realized,
        "excluded_phase2_item_ids": [f"old-{i}" for i in range(30)],
        "realized_relation_distribution": {"country": a_realized},
        "relation_dominance_share": 1.0,
        "relation_dominance_flag": True,
        "per_stratum_selected": a_strata
        or {s: COHORT_A_PER_STRATUM_TARGET for s in MARGIN_STRATA},
        "status": a_status,
        "eligibility_limited": a_status != "COMPLETE",
        "shortfall": a_shortfall or dict.fromkeys(MARGIN_STRATA, 0),
    }
    cohort_a["planned_total_target"] = a_planned_total
    if a_planned_strata is not None:
        cohort_a["planned_per_stratum_target"] = a_planned_strata

    return build_manifest(
        seed=42,
        repository_commit="d684f39",
        dataset=dict(DATASET_36),
        artifact_hashes=dict(ARTIFACT_HASHES_36),
        environment=dict(ENVIRONMENT_36),
        hardware=dict(HARDWARE_36),
        models={
            "qwen": {
                "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": FROZEN_MODEL_REVISIONS["qwen"]["revision"],
                "model_specific_arm_enabled": True,
                "preferred_source": "a government website",
                "dispreferred_source": "an anonymous online forum post",
                "condition_set": ["C0", "K1", "K2", "K3", "K4", "M1", "M2"],
                **model_provenance_36(),
            }
        },
        prompt_version="v1",
        cohorts={
            "A": cohort_a,
            "B": {
                "qwen|KW": {
                    "target_cell_count": b_target,
                    "minimum_cell_count": b_minimum,
                    "original_cell_counts": {"country|low": 9},
                    "qualifying_relations": qualifying,
                    "excluded_short_relations": list(b_short),
                    "realized_relations": qualifying,
                    "realized_cell_count": b_realized_cell,
                    "status": "ELIGIBILITY_LIMITED" if b_short else "COMPLETE",
                    "confirmatory_eligible": b_confirmatory,
                    "selected_item_ids": ["i0"],
                    "reduction_reason": "short relation" if b_short else None,
                }
            },
            "C": {
                "items": [
                    {
                        "item_id": "i0",
                        "relation": "country",
                        "per_model": {
                            "qwen": {
                                "knowledge_group": "KW",
                                "parametric_margin": 1.0,
                                "margin_stratum": "low",
                            }
                        },
                    }
                ]
            },
        },
        cohort_membership_map={"o": ["A"]},
        deduplication_alias_map={"qwen|i0|M1": "o"},
        final_margin_strata={"qwen|KW": [1.0, 2.0]},
        screening={**planned_screening_design(), "stopped_reason": stopped},
        nominal_condition_slots=7,
        unique_observations=5,
        synthetic=False,
    )


def _rejects_at_all_layers(manifest, needle):
    """A frozen-design violation must be caught by `validate_manifest`,
    `freeze_manifest` AND the real-run gate -- not by one layer only."""
    problems = validate_manifest(manifest)
    assert any(needle in p for p in problems), problems
    with pytest.raises(ManifestError):
        freeze_manifest(manifest)
    config = load_phase3_config(COMMITTED_CONFIG)
    blockers = check_readiness(config, manifest.data).blockers
    assert any(needle in b for b in blockers), blockers


def test_design_baseline_manifest_is_accepted():
    manifest = _design_manifest()
    assert validate_manifest(manifest) == []
    assert freeze_manifest(manifest).is_frozen is True


# --- the exact second-audit exploit ---------------------------------------


def test_reduced_cohort_a_planned_target_is_rejected_at_all_layers():
    manifest = _design_manifest(
        a_planned_total=18,
        a_planned_strata=dict.fromkeys(MARGIN_STRATA, 6),
        a_realized=18,
        a_strata=dict.fromkeys(MARGIN_STRATA, 6),
    )
    _rejects_at_all_layers(manifest, "Cohort A planned target must be 96")


def test_reduced_cohort_a_planned_per_stratum_target_is_rejected():
    manifest = _design_manifest(a_planned_strata=dict.fromkeys(MARGIN_STRATA, 6))
    _rejects_at_all_layers(manifest, "planned per-stratum target must be")


def test_full_second_audit_exploit_is_rejected():
    """Cohort A 18 / 6-6-6 AND Cohort B target 3 / minimum 1 together."""
    manifest = _design_manifest(
        a_planned_total=18,
        a_planned_strata=dict.fromkeys(MARGIN_STRATA, 6),
        a_realized=18,
        a_strata=dict.fromkeys(MARGIN_STRATA, 6),
        b_target=3,
        b_minimum=1,
        b_realized_cell=3,
    )
    problems = validate_manifest(manifest)
    assert any("Cohort A planned target must be 96" in p for p in problems)
    assert any("target_cell_count must be 8" in p for p in problems)
    assert any("minimum_cell_count must be 6" in p for p in problems)
    with pytest.raises(ManifestError):
        freeze_manifest(manifest)


# --- Cohort B frozen design mutations (rejected) --------------------------


@pytest.mark.parametrize(
    "target,minimum,realized,needle",
    [
        (3, 1, 3, "target_cell_count must be 8"),
        (7, 6, 7, "target_cell_count must be 8"),
        (8, 5, 6, "minimum_cell_count must be 6"),
        (9, 6, 8, "target_cell_count must be 8"),
    ],
)
def test_cohort_b_design_mutations_are_rejected(target, minimum, realized, needle):
    manifest = _design_manifest(
        b_target=target, b_minimum=minimum, b_realized_cell=realized
    )
    _rejects_at_all_layers(manifest, needle)


# --- Cohort B legitimate realized reductions (accepted) -------------------


@pytest.mark.parametrize("realized", [8, 7, 6])
def test_cohort_b_legitimate_realized_cell_counts_are_accepted(realized):
    """§32 rules 1-3: a realized 6, 7 or 8 across all four relations is a
    valid reduction, not a design change."""
    manifest = _design_manifest(b_realized_cell=realized)
    assert validate_manifest(manifest) == []
    assert freeze_manifest(manifest).is_frozen is True


def test_cohort_b_three_relation_fallback_is_accepted():
    """§32 rule 4: three qualifying relations stays confirmatory."""
    manifest = _design_manifest(
        b_qualifying=REL_ALL[:3], b_short=["mother"], b_confirmatory=True
    )
    assert validate_manifest(manifest) == []
    assert freeze_manifest(manifest).is_frozen is True


def test_cohort_b_two_relation_exploratory_manifest_is_accepted():
    """Fewer than three relations is still a valid eligibility-limited
    manifest, provided it is marked non-confirmatory with provenance."""
    manifest = _design_manifest(
        b_qualifying=REL_ALL[:2], b_short=REL_ALL[2:], b_confirmatory=False
    )
    assert validate_manifest(manifest) == []


def test_cohort_b_confirmatory_flag_must_match_the_three_relation_rule():
    too_generous = _design_manifest(
        b_qualifying=REL_ALL[:2], b_short=REL_ALL[2:], b_confirmatory=True
    )
    assert any(
        "confirmatory family only with at least three" in p
        for p in validate_manifest(too_generous)
    )
    too_strict = _design_manifest(
        b_qualifying=REL_ALL[:3], b_short=["mother"], b_confirmatory=False
    )
    assert any(
        "confirmatory family only with at least three" in p
        for p in validate_manifest(too_strict)
    )


def test_cohort_b_realized_cell_above_target_is_rejected():
    manifest = _design_manifest(b_realized_cell=9)
    assert any("exceeds the frozen target" in p for p in validate_manifest(manifest))


# --- Cohort A planned-vs-realized distinction -----------------------------


def test_cohort_a_eligibility_limited_with_full_provenance_is_accepted():
    """VALID: planned target intact, realized short, status says so, and the
    shortfall/ceiling provenance explains why (§15.1, §34)."""
    manifest = _design_manifest(
        a_realized=40,
        a_strata={"low": 14, "medium": 13, "high": 13},
        a_status="ELIGIBILITY_LIMITED",
        a_shortfall={"low": 18, "medium": 19, "high": 19},
        stopped="ceiling_reached",
    )
    assert validate_manifest(manifest) == []
    assert manifest.data["cohorts"]["A"]["planned_total_target"] == COHORT_A_TOTAL_TARGET


def test_short_cohort_a_cannot_be_reported_as_complete():
    manifest = _design_manifest(
        a_realized=18, a_strata=dict.fromkeys(MARGIN_STRATA, 6)
    )
    assert any("marked COMPLETE but realized" in p for p in validate_manifest(manifest))


def test_eligibility_limited_cohort_a_requires_shortfall_provenance():
    manifest = _design_manifest(
        a_realized=40,
        a_strata={"low": 14, "medium": 13, "high": 13},
        a_status="ELIGIBILITY_LIMITED",
        stopped="ceiling_reached",
    )
    assert any("reports no shortfall" in p for p in validate_manifest(manifest))


def test_eligibility_limited_cohort_a_requires_an_acquisition_failure():
    """A short Cohort A is legitimate only as a ceiling/supply failure."""
    manifest = _design_manifest(
        a_realized=40,
        a_strata={"low": 14, "medium": 13, "high": 13},
        a_status="ELIGIBILITY_LIMITED",
        a_shortfall={"low": 18, "medium": 19, "high": 19},
        stopped="supply_criteria_met",
    )
    assert any(
        "acquisition failure at the screening ceiling" in p
        for p in validate_manifest(manifest)
    )


# --- frozen acquisition / condition / source constants --------------------


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("block_size", 500),
        ("ceiling_per_model", 100),
        ("cohort_a_supply_per_stratum", 10),
    ],
)
def test_screening_design_constants_are_enforced(field, bad_value):
    manifest = _design_manifest()
    manifest.data["screening"][field] = bad_value
    _rejects_at_all_layers(manifest, f"screening {field} must be")


def test_missing_screening_constant_is_rejected():
    manifest = _design_manifest()
    del manifest.data["screening"]["block_size"]
    assert any(
        "missing frozen constant 'block_size'" in p for p in validate_manifest(manifest)
    )


def test_condition_specification_cannot_be_redefined():
    manifest = _design_manifest()
    manifest.data["condition_specification"] = ["C0", "C1", "C2", "C3", "C4"]
    _rejects_at_all_layers(manifest, "condition specification must be")


def test_common_source_identities_cannot_be_redefined():
    manifest = _design_manifest()
    manifest.data["sources"]["common_source_b"] = "a personal blog"
    _rejects_at_all_layers(manifest, "common_source_b must be")


def test_frozen_design_violation_survives_boolean_spoofing():
    """Setting the flags by hand must not bypass a design mutation."""
    manifest = _design_manifest(b_target=3, b_minimum=1, b_realized_cell=3)
    manifest.data["ready_for_real_run"] = True
    manifest.data["frozen"] = True
    manifest.data["synthetic"] = False
    assert validate_manifest(manifest)
    with pytest.raises(ManifestError):
        freeze_manifest(manifest)


def test_synthetic_dryrun_manifest_keeps_the_frozen_planned_targets():
    """The dry run may realize a small, cheap cohort, but the PLANNED design
    it records must still be the frozen 96 / 32-32-32."""
    from conflict_eval.phase3.dryrun import run_synthetic_dryrun

    cohort_a = run_synthetic_dryrun(seed=42).manifest["cohorts"]["A"]
    assert cohort_a["planned_total_target"] == COHORT_A_TOTAL_TARGET
    assert cohort_a["planned_per_stratum_target"] == {
        s: COHORT_A_PER_STRATUM_TARGET for s in MARGIN_STRATA
    }
    assert cohort_a["realized_total"] < COHORT_A_TOTAL_TARGET

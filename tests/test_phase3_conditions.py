"""Tests for the Phase 3 seven-condition builder.

Covers `docs/phase3_scaled_study_design.md` §22 (the condition table and
the KC/KW conflict mapping) and §20.2 (a model whose source roles are still
unresolved cannot have M1/M2 built).
"""

from __future__ import annotations

import pytest

from conflict_eval.phase3.conditions import (
    UnresolvedSourceRolesError,
    build_phase3_conditions,
)
from conflict_eval.phase3.constants import (
    ARM_BASELINE,
    ARM_COMMON,
    ARM_MODEL_SPECIFIC,
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
    FROZEN_MODEL_SOURCE_PAIRS,
    NOMINAL_CONDITIONS,
)

QWEN = FROZEN_MODEL_SOURCE_PAIRS["qwen"]
LLAMA = FROZEN_MODEL_SOURCE_PAIRS["llama"]


def _build(group: str, pair=QWEN):
    return build_phase3_conditions(
        knowledge_group=group,
        gold_answer="gold",
        baseline_answer="memory",
        foil_answer="foil",
        model_preferred_source=pair["preferred_source"],
        model_dispreferred_source=pair["dispreferred_source"],
    )


def _by_condition(specs):
    return {s.condition: s for s in specs}


def test_frozen_common_pair_matches_the_design():
    assert COMMON_SOURCE_A == "a government website"
    assert COMMON_SOURCE_B == "an anonymous online forum post"


def test_frozen_model_pairs_match_the_design():
    assert QWEN == {
        "preferred_source": "a government website",
        "dispreferred_source": "an anonymous online forum post",
    }
    assert LLAMA == {
        "preferred_source": "a government website",
        "dispreferred_source": "a social media post",
    }


def test_exactly_seven_nominal_conditions_in_the_frozen_order():
    assert NOMINAL_CONDITIONS == ("C0", "K1", "K2", "K3", "K4", "M1", "M2")
    for group in ("KC", "KW"):
        specs = _build(group)
        assert [s.condition for s in specs] == list(NOMINAL_CONDITIONS)


def test_c0_has_no_evidence_and_no_source():
    spec = _by_condition(_build("KW"))["C0"]
    assert spec.arm == ARM_BASELINE
    assert spec.evidence_truth == "none"
    assert spec.source_label is None
    assert spec.asserted_answer is None
    assert spec.conflict_status == "none"


def test_common_arm_truth_and_source_mapping():
    specs = _by_condition(_build("KW"))
    assert specs["K1"].evidence_truth == "true"
    assert specs["K1"].source_label == COMMON_SOURCE_A
    assert specs["K2"].evidence_truth == "true"
    assert specs["K2"].source_label == COMMON_SOURCE_B
    assert specs["K3"].evidence_truth == "false"
    assert specs["K3"].source_label == COMMON_SOURCE_A
    assert specs["K4"].evidence_truth == "false"
    assert specs["K4"].source_label == COMMON_SOURCE_B
    assert all(specs[c].arm == ARM_COMMON for c in ("K1", "K2", "K3", "K4"))


def test_common_arm_uses_identity_roles_not_preference_roles():
    """The common labels are fixed source IDENTITIES, never automatically
    preferred/dispreferred for any model (§19)."""
    specs = _by_condition(_build("KW"))
    assert specs["K1"].source_role == "identity_a"
    assert specs["K2"].source_role == "identity_b"
    assert specs["K1"].source_role not in ("preferred", "dispreferred")


def test_kw_model_specific_arm_uses_correct_conflicting_evidence():
    """KW -> correct (gold) evidence conflicts with the wrong memory answer
    (corrective override) (§22)."""
    specs = _by_condition(_build("KW"))
    for condition in ("M1", "M2"):
        assert specs[condition].arm == ARM_MODEL_SPECIFIC
        assert specs[condition].evidence_truth == "true"
        assert specs[condition].asserted_answer == "gold"
        assert specs[condition].conflict_status == "conflict"


def test_kc_model_specific_arm_uses_false_conflicting_evidence():
    """KC -> false (foil) evidence conflicts with the correct memory answer
    (harmful override) (§22)."""
    specs = _by_condition(_build("KC"))
    for condition in ("M1", "M2"):
        assert specs[condition].evidence_truth == "false"
        assert specs[condition].asserted_answer == "foil"
        assert specs[condition].conflict_status == "conflict"


def test_model_specific_arm_carries_preference_roles():
    specs = _by_condition(_build("KW"))
    assert specs["M1"].source_role == "preferred"
    assert specs["M2"].source_role == "dispreferred"
    assert specs["M1"].source_label == QWEN["preferred_source"]
    assert specs["M2"].source_label == QWEN["dispreferred_source"]


def test_model_specific_arm_is_always_conflict_only():
    """It deliberately omits agreement conditions (§22)."""
    for group in ("KC", "KW"):
        specs = _by_condition(_build(group))
        assert specs["M1"].conflict_status == "conflict"
        assert specs["M2"].conflict_status == "conflict"


def test_kc_common_arm_agreement_and_conflict_statuses():
    specs = _by_condition(_build("KC"))
    assert specs["K1"].conflict_status == "agreement"  # correct evidence agrees
    assert specs["K3"].conflict_status == "conflict"  # foil conflicts


def test_kw_common_arm_agreement_and_conflict_statuses():
    specs = _by_condition(_build("KW"))
    assert specs["K1"].conflict_status == "conflict"  # gold conflicts
    assert specs["K3"].conflict_status == "agreement"  # restated wrong answer


def test_llama_model_specific_pair_differs_from_the_common_pair():
    specs = _by_condition(_build("KW", LLAMA))
    assert specs["M1"].source_label == COMMON_SOURCE_A  # coincides
    assert specs["M2"].source_label != COMMON_SOURCE_B  # differs


def test_unresolved_source_roles_are_refused():
    """A new model awaiting Phase 3C calibration cannot have M1/M2 built."""
    with pytest.raises(UnresolvedSourceRolesError):
        build_phase3_conditions(
            knowledge_group="KW",
            gold_answer="gold",
            baseline_answer="memory",
            foil_answer="foil",
            model_preferred_source=None,
            model_dispreferred_source=None,
        )


def test_partially_unresolved_source_roles_are_refused():
    with pytest.raises(UnresolvedSourceRolesError):
        build_phase3_conditions(
            knowledge_group="KW",
            gold_answer="gold",
            baseline_answer="memory",
            foil_answer="foil",
            model_preferred_source="a government website",
            model_dispreferred_source=None,
        )


def test_kc_without_a_foil_is_refused():
    with pytest.raises(ValueError, match="foil_answer"):
        build_phase3_conditions(
            knowledge_group="KC",
            gold_answer="gold",
            baseline_answer="gold",
            foil_answer=None,
            model_preferred_source=QWEN["preferred_source"],
            model_dispreferred_source=QWEN["dispreferred_source"],
        )


def test_invalid_knowledge_group_is_refused():
    with pytest.raises(ValueError, match="knowledge_group"):
        build_phase3_conditions(
            knowledge_group="excluded",
            gold_answer="gold",
            baseline_answer="memory",
            foil_answer="foil",
            model_preferred_source=QWEN["preferred_source"],
            model_dispreferred_source=QWEN["dispreferred_source"],
        )


def test_phase2_condition_builder_is_untouched():
    """Phase 2 must continue to behave exactly as before (backward
    compatibility): its builder still returns C0-C4."""
    from conflict_eval.experiment.conditions import build_conditions

    specs = build_conditions("KW", "gold", "memory", "foil", "pref", "dispref")
    assert [s.condition for s in specs] == ["C0", "C1", "C2", "C3", "C4"]

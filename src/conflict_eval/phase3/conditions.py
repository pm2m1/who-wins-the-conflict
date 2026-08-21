"""Phase 3 seven-condition construction.

Implements the frozen condition table in
`docs/phase3_scaled_study_design.md` §22:

    C0                      no evidence
    K1  correct  + common A     common fixed-source arm
    K2  correct  + common B
    K3  false    + common A
    K4  false    + common B
    M1  conflicting + model-specific source A   model-specific arm
    M2  conflicting + model-specific source B

"Conflicting" in the model-specific arm is resolved by knowledge group,
exactly as in Phase 2: **KC -> false (foil) evidence** (harmful override),
**KW -> correct (gold) evidence** (corrective override) (§22). The
model-specific arm therefore contributes only conflict trials.

This module deliberately mirrors, and does not modify, the Phase 2
`experiment/conditions.py` semantics: the Phase 2 C0-C4 builder is
untouched and still used by the frozen Phase 2 pipeline.
"""

from __future__ import annotations

import dataclasses

from conflict_eval.phase3.constants import (
    ARM_BASELINE,
    ARM_COMMON,
    ARM_MODEL_SPECIFIC,
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
)

_VALID_GROUPS = ("KC", "KW")


class UnresolvedSourceRolesError(ValueError):
    """Raised when a model's `M1`/`M2` sources are still unresolved.

    New Phase 3 models are calibrated in Phase 3C (§20.2); until then their
    source roles are `None` and the model-specific arm cannot be built. The
    frozen design forbids inventing them.
    """


@dataclasses.dataclass(frozen=True)
class Phase3TrialSpec:
    """One planned condition for one item.

    `source_role` is `"preferred"`/`"dispreferred"` only in the
    model-specific arm. In the common arm the labels are fixed source
    IDENTITIES, not roles (§19), so the role field is `"identity_a"` /
    `"identity_b"` there -- the distinction is deliberate and must not be
    collapsed.
    """

    condition: str
    arm: str
    evidence_truth: str  # "true" | "false" | "none"
    source_role: str
    source_label: str | None
    asserted_answer: str | None
    conflict_status: str  # "conflict" | "agreement" | "none"


def build_phase3_conditions(
    knowledge_group: str,
    gold_answer: str,
    baseline_answer: str,
    foil_answer: str | None,
    model_preferred_source: str | None,
    model_dispreferred_source: str | None,
    common_source_a: str = COMMON_SOURCE_A,
    common_source_b: str = COMMON_SOURCE_B,
) -> list[Phase3TrialSpec]:
    """Build the seven nominal conditions for one item.

    Raises `UnresolvedSourceRolesError` if the model-specific pair is not
    resolved -- the builder refuses to construct `M1`/`M2` for a model whose
    roles await Phase 3C calibration (§20.2).
    """
    if knowledge_group not in _VALID_GROUPS:
        raise ValueError(
            f"build_phase3_conditions requires knowledge_group in {_VALID_GROUPS}, "
            f"got {knowledge_group!r}"
        )
    if model_preferred_source is None or model_dispreferred_source is None:
        raise UnresolvedSourceRolesError(
            "Model-specific source roles are unresolved; M1/M2 cannot be built. "
            "Qwen and Llama use their frozen Phase 2 pairs; a new model's pair "
            "is set only after its Phase 3C calibration "
            "(docs/phase3_scaled_study_design.md, §20)."
        )

    if knowledge_group == "KC":
        if foil_answer is None:
            raise ValueError(
                "KC items require a foil_answer to build the false-evidence conditions"
            )
        false_evidence_answer = foil_answer
        # KC: correct evidence agrees with the (correct) memory answer;
        # false evidence conflicts with it.
        correct_status = "agreement"
        false_status = "conflict"
        conflicting_answer = false_evidence_answer
        conflicting_truth = "false"
    else:  # KW
        # KW: correct evidence conflicts with the (wrong) memory answer;
        # false evidence restating the baseline answer agrees with it.
        false_evidence_answer = baseline_answer
        correct_status = "conflict"
        false_status = "agreement"
        conflicting_answer = gold_answer
        conflicting_truth = "true"

    return [
        Phase3TrialSpec(
            "C0", ARM_BASELINE, "none", "none", None, None, "none"
        ),
        Phase3TrialSpec(
            "K1", ARM_COMMON, "true", "identity_a", common_source_a,
            gold_answer, correct_status,
        ),
        Phase3TrialSpec(
            "K2", ARM_COMMON, "true", "identity_b", common_source_b,
            gold_answer, correct_status,
        ),
        Phase3TrialSpec(
            "K3", ARM_COMMON, "false", "identity_a", common_source_a,
            false_evidence_answer, false_status,
        ),
        Phase3TrialSpec(
            "K4", ARM_COMMON, "false", "identity_b", common_source_b,
            false_evidence_answer, false_status,
        ),
        Phase3TrialSpec(
            "M1", ARM_MODEL_SPECIFIC, conflicting_truth, "preferred",
            model_preferred_source, conflicting_answer, "conflict",
        ),
        Phase3TrialSpec(
            "M2", ARM_MODEL_SPECIFIC, conflicting_truth, "dispreferred",
            model_dispreferred_source, conflicting_answer, "conflict",
        ),
    ]

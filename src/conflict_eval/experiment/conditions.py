"""Build the five experimental conditions (C0-C4) for one item.

Interpretation depends on the model-specific knowledge group, per
docs/phase2_research_design.md:

    KC: C0 baseline; C1/C2 (correct evidence) = agreement;
        C3/C4 (false evidence, using a foil) = primary conflict (harmful override)
    KW: C0 baseline; C1/C2 (correct evidence, gold) = primary conflict (corrective override);
        C3/C4 (false evidence, using the baseline wrong answer) = agreement
"""

from __future__ import annotations

import dataclasses

_VALID_GROUPS = ("KC", "KW")


@dataclasses.dataclass(frozen=True)
class TrialSpec:
    condition: str  # C0 | C1 | C2 | C3 | C4
    evidence_truth: str  # "true" | "false" | "none"
    source_role: str  # "preferred" | "dispreferred" | "none"
    conflict_status: str  # "conflict" | "agreement" | "none"
    source_label: str | None
    asserted_answer: str | None


def build_conditions(
    knowledge_group: str,
    gold_answer: str,
    baseline_answer: str,
    foil_answer: str | None,
    preferred_source: str,
    dispreferred_source: str,
) -> list[TrialSpec]:
    if knowledge_group not in _VALID_GROUPS:
        raise ValueError(
            f"build_conditions requires knowledge_group in {_VALID_GROUPS}, got {knowledge_group!r}"
        )

    if knowledge_group == "KC":
        if foil_answer is None:
            raise ValueError("KC items require a foil_answer to build the C3/C4 conflict trials")
        correct_evidence_answer = gold_answer
        false_evidence_answer = foil_answer
        c1_c2_status = "agreement"
        c3_c4_status = "conflict"
    else:  # KW
        correct_evidence_answer = gold_answer
        false_evidence_answer = baseline_answer
        c1_c2_status = "conflict"
        c3_c4_status = "agreement"

    return [
        TrialSpec("C0", "none", "none", "none", None, None),
        TrialSpec("C1", "true", "preferred", c1_c2_status, preferred_source, correct_evidence_answer),
        TrialSpec("C2", "true", "dispreferred", c1_c2_status, dispreferred_source, correct_evidence_answer),
        TrialSpec("C3", "false", "preferred", c3_c4_status, preferred_source, false_evidence_answer),
        TrialSpec("C4", "false", "dispreferred", c3_c4_status, dispreferred_source, false_evidence_answer),
    ]

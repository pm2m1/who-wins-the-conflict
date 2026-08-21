"""Analysis-status registry and multiplicity families.

Implements `docs/phase3_scaled_study_design.md` §28 (multiplicity) and §44
(analysis status table). Two frozen properties are enforced structurally:

- **The primary family contains exactly one test** -- the Cohort A Qwen
  corrective frozen-pair replication -- and therefore requires no
  multiplicity correction (§28). Registering a second primary analysis is
  an error, not a silent addition.
- **No analysis may be promoted** from EXPLORATORY/DIAGNOSTIC to
  confirmatory after results exist (§28, §44). Entries are frozen
  dataclasses held in an immutable registry.
"""

from __future__ import annotations

import dataclasses

from conflict_eval.phase3.constants import (
    ANALYSIS_STATUSES,
    STATUS_DIAGNOSTIC,
    STATUS_EXPLORATORY,
    STATUS_PRIMARY,
    STATUS_SECONDARY,
)

FAMILY_PRIMARY = "primary"
FAMILY_SECONDARY = "secondary"
FAMILY_NONE = "none"


class AnalysisStatusError(ValueError):
    """Raised on an illegal status/family declaration."""


@dataclasses.dataclass(frozen=True)
class AnalysisEntry:
    name: str
    cohort: str
    outcome: str
    contrast: str
    status: str
    family: str

    def __post_init__(self) -> None:
        if self.status not in ANALYSIS_STATUSES:
            raise AnalysisStatusError(
                f"Unknown analysis status {self.status!r}; expected one of "
                f"{list(ANALYSIS_STATUSES)}"
            )
        expected_family = {
            STATUS_PRIMARY: FAMILY_PRIMARY,
            STATUS_SECONDARY: FAMILY_SECONDARY,
            STATUS_EXPLORATORY: FAMILY_NONE,
            STATUS_DIAGNOSTIC: FAMILY_NONE,
        }[self.status]
        if self.family != expected_family:
            raise AnalysisStatusError(
                f"Analysis {self.name!r} has status {self.status!r} but family "
                f"{self.family!r}; the frozen design pairs it with "
                f"{expected_family!r} (§28)."
            )


class AnalysisRegistry:
    """An immutable set of declared analyses.

    Built once, from the frozen §44 table, before any Phase 3 result
    exists. There is deliberately no method to change an entry's status.
    """

    def __init__(self, entries: list[AnalysisEntry]) -> None:
        primary = [e for e in entries if e.status == STATUS_PRIMARY]
        if len(primary) > 1:
            raise AnalysisStatusError(
                "The primary confirmatory family contains exactly ONE test "
                f"(§28); got {len(primary)}: {[e.name for e in primary]}"
            )
        names = [e.name for e in entries]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise AnalysisStatusError(f"Duplicate analysis names: {sorted(duplicates)}")
        self._entries = tuple(entries)

    @property
    def entries(self) -> tuple[AnalysisEntry, ...]:
        return self._entries

    def by_status(self, status: str) -> tuple[AnalysisEntry, ...]:
        return tuple(e for e in self._entries if e.status == status)

    def primary(self) -> AnalysisEntry | None:
        found = self.by_status(STATUS_PRIMARY)
        return found[0] if found else None

    def secondary_family(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.by_status(STATUS_SECONDARY))

    def requires_multiplicity_correction(self, family: str) -> bool:
        """The primary family holds one test and needs no correction (§28);
        the secondary family is Holm-corrected within itself."""
        if family == FAMILY_PRIMARY:
            return False
        if family == FAMILY_SECONDARY:
            return len(self.by_status(STATUS_SECONDARY)) > 1
        return False


def default_registry() -> AnalysisRegistry:
    """The frozen §44 analysis-status table, as far as Phase 3B needs it.

    Entries mirror the frozen document; new-model rows are named
    generically because the exact models are unresolved until Phase 3C.
    """
    return AnalysisRegistry(
        [
            AnalysisEntry(
                name="cohort_a_qwen_corrective_frozen_pair",
                cohort="A",
                outcome="context_adopted",
                contrast="M1 vs M2 (frozen Phase 2 Qwen pair), KW",
                status=STATUS_PRIMARY,
                family=FAMILY_PRIMARY,
            ),
            AnalysisEntry(
                name="cohort_b_qwen_corrective",
                cohort="B",
                outcome="context_adopted",
                contrast="M1 vs M2, KW (generalization, not a direct replication)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="cohort_b_llama_corrective",
                cohort="B",
                outcome="context_adopted",
                contrast="M1 vs M2 (frozen Phase 2 Llama pair), KW",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="cohort_b_harmful_model_specific",
                cohort="B",
                outcome="context_adopted",
                contrast="M1 vs M2, KC",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="common_fixed_source_contrast",
                cohort="B",
                outcome="context_adopted",
                contrast="K1/K3 vs K2/K4 (common identities A vs B)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="cohort_c_cross_model",
                cohort="C",
                outcome="context_adopted",
                contrast="model x source",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="source_by_parametric_strength",
                cohort="B",
                outcome="context_adopted",
                contrast="margin x source, nonlinear",
                status=STATUS_EXPLORATORY,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="calibration_stability_qwen_llama",
                cohort="n/a",
                outcome="elicited pair",
                contrast="Phase 3 calibrated pair vs frozen Phase 2 pair",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="cohort_a_relation_distribution",
                cohort="A",
                outcome="composition",
                contrast="most-frequent-relation share",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="ceiling_floor_diagnostics",
                cohort="all",
                outcome="discordance/boundary/CI width",
                contrast="per contrast",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
        ]
    )

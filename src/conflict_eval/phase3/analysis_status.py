"""Analysis-status registry and multiplicity families.

Implements `docs/phase3_scaled_study_design.md` §28 (multiplicity), §29
(sensitivity analyses) and §44 (analysis status table). Four frozen
properties are enforced structurally:

- **The primary family contains exactly one test** -- the Cohort A Qwen
  corrective frozen-pair replication -- and therefore requires no
  multiplicity correction (§28). Registering a second primary analysis is
  an error, not a silent addition.
- **Cohort A never enters the secondary family** (§28).
- **Qwen's common-arm contrast is counted once** and never re-enters as an
  additional independent test, because it rests on the same observations as
  its frozen-pair contrast (§19, §22, §28, §44).
- **No analysis may be promoted** from EXPLORATORY/DIAGNOSTIC to
  confirmatory after results exist (§28, §44). Entries are frozen
  dataclasses held in an immutable registry.

WHAT THIS REGISTRY IS, AND IS NOT
---------------------------------
`default_registry()` is the **pre-outcome nominal §44 declaration**: every
analysis the frozen design names, with the status the frozen design assigns
it, fixed before any Phase 3 result exists.

It is **not** the realized Phase 3E Holm family, and nothing here computes
one. Realized secondary membership additionally depends on:

- each model's model-specific arm availability (§20.2, §34) -- resolved
  here only to the extent of marking impossible contrasts NOT APPLICABLE;
- `CohortBGroupResult.confirmatory_eligible`, which is decided by realized
  cell counts and the §32 relation ladder;
- the remaining frozen eligibility rules (§30 saturation, §32, §37).

No final family size or composition is computed, asserted, or hard-coded
anywhere in this module, and none may be. Phase 3E multiplicity selection
is NOT complete.
"""

from __future__ import annotations

import dataclasses

from conflict_eval.phase3.constants import (
    ANALYSIS_STATUSES,
    STATUS_DIAGNOSTIC,
    STATUS_EXPLORATORY,
    STATUS_NOT_APPLICABLE,
    STATUS_PRIMARY,
    STATUS_SECONDARY,
    STATUS_SECONDARY_MECHANISTIC,
)

FAMILY_PRIMARY = "primary"
FAMILY_SECONDARY = "secondary"
FAMILY_NONE = "none"

#: `contrast_kind` values. MODEL_SPECIFIC contrasts require an M1/M2 arm and
#: are therefore impossible for a model whose arm was disabled under §34.
#: COMMON contrasts rest on the K conditions and stay available to every
#: model, which is exactly what §34's common-arm fallback preserves.
KIND_MODEL_SPECIFIC = "model_specific"
KIND_COMMON = "common"


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
    #: `KIND_MODEL_SPECIFIC`, `KIND_COMMON`, or "" for analyses that depend
    #: on neither arm (H1 margin effects, composition diagnostics).
    contrast_kind: str = ""
    #: The model this analysis belongs to, or None for cross-model analyses.
    #: An EXPLICIT key -- `mark_not_applicable` matches on this and never on
    #: a substring of `name`, so a short or overlapping model key can never
    #: silently disable another model's analysis.
    model_key: str | None = None
    #: Set when this analysis rests on the same observations as another
    #: family member and must therefore be counted ONCE (§28). Such an entry
    #: is declared but excluded from `secondary_family()`.
    counted_once_with: str | None = None
    #: Populated only when status is NOT APPLICABLE (§34).
    not_applicable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ANALYSIS_STATUSES:
            raise AnalysisStatusError(
                f"Unknown analysis status {self.status!r}; expected one of "
                f"{list(ANALYSIS_STATUSES)}"
            )
        expected_family = {
            STATUS_PRIMARY: FAMILY_PRIMARY,
            STATUS_SECONDARY: FAMILY_SECONDARY,
            # §44 lists this status with multiplicity family "none".
            STATUS_SECONDARY_MECHANISTIC: FAMILY_NONE,
            STATUS_EXPLORATORY: FAMILY_NONE,
            STATUS_DIAGNOSTIC: FAMILY_NONE,
            # A never-measured contrast belongs to no multiplicity family.
            STATUS_NOT_APPLICABLE: FAMILY_NONE,
        }[self.status]
        if self.family != expected_family:
            raise AnalysisStatusError(
                f"Analysis {self.name!r} has status {self.status!r} but family "
                f"{self.family!r}; the frozen design pairs it with "
                f"{expected_family!r} (§28)."
            )
        if self.status == STATUS_NOT_APPLICABLE and not self.not_applicable_reason:
            raise AnalysisStatusError(
                f"Analysis {self.name!r} is NOT APPLICABLE but records no "
                "reason; a never-measured contrast must say why, so it is "
                "never mistaken for a null result (§34)."
            )
        if self.status == STATUS_PRIMARY and self.cohort != "A":
            raise AnalysisStatusError(
                f"Analysis {self.name!r} is PRIMARY but sits in cohort "
                f"{self.cohort!r}; the sole primary test is the Cohort A Qwen "
                "frozen-pair replication (§28, §44)."
            )
        if self.status == STATUS_SECONDARY and self.cohort == "A":
            raise AnalysisStatusError(
                f"Analysis {self.name!r} is SECONDARY but sits in cohort 'A'; "
                "Cohort A never enters the secondary family (§28)."
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
        known = set(names)
        for entry in entries:
            if entry.counted_once_with and entry.counted_once_with not in known:
                raise AnalysisStatusError(
                    f"Analysis {entry.name!r} is counted once with "
                    f"{entry.counted_once_with!r}, which is not registered."
                )
        self._entries = tuple(entries)

    @property
    def entries(self) -> tuple[AnalysisEntry, ...]:
        return self._entries

    def by_status(self, status: str) -> tuple[AnalysisEntry, ...]:
        return tuple(e for e in self._entries if e.status == status)

    def by_model(self, model_key: str) -> tuple[AnalysisEntry, ...]:
        return tuple(e for e in self._entries if e.model_key == model_key)

    def primary(self) -> AnalysisEntry | None:
        found = self.by_status(STATUS_PRIMARY)
        return found[0] if found else None

    def secondary_family(self) -> tuple[str, ...]:
        """Nominally declared secondary members, before realized eligibility.

        Entries marked `counted_once_with` are excluded: they rest on the
        same observations as another member and §28 counts them once.

        This is the NOMINAL declaration. It is not the realized Phase 3E
        Holm family -- see the module docstring.
        """
        return tuple(
            e.name
            for e in self.by_status(STATUS_SECONDARY)
            if e.counted_once_with is None
        )

    def requires_multiplicity_correction(self, family: str) -> bool:
        """The primary family holds one test and needs no correction (§28);
        the secondary family is Holm-corrected within itself."""
        if family == FAMILY_PRIMARY:
            return False
        if family == FAMILY_SECONDARY:
            return len(self.secondary_family()) > 1
        return False


def mark_not_applicable(
    registry: AnalysisRegistry, disabled_models: list[str] | tuple[str, ...]
) -> AnalysisRegistry:
    """Mark model-specific analyses NOT APPLICABLE for models whose arm was
    disabled under the frozen §34 rule.

    Matching is on the EXPLICIT `model_key` and `contrast_kind` fields --
    never on a substring of the analysis name -- so a short or overlapping
    model key cannot reach another model's analysis.

    Both the corrective (KW) and harmful (KC) model-specific contrasts are
    covered, since both require an M1/M2 arm. COMMON-arm analyses are left
    untouched: contributing to H2a is precisely what §34 preserves for these
    models.

    Crucially this is **not** a null result and must never be aggregated
    with one: the contrast was never measured, so it has no estimate, no
    interval and no p-value. It is also excluded from the Holm-controlled
    secondary family, because a family size inflated by never-run contrasts
    would penalise the tests that *were* run.

    Applied pre-outcome, from the recorded calibration state, so it can
    never be an after-the-fact reaction to results.
    """
    disabled = set(disabled_models)
    if not disabled:
        return registry
    updated: list[AnalysisEntry] = []
    for entry in registry.entries:
        impossible = (
            entry.status == STATUS_SECONDARY
            and entry.contrast_kind == KIND_MODEL_SPECIFIC
            and entry.model_key in disabled
        )
        if impossible:
            updated.append(
                dataclasses.replace(
                    entry,
                    status=STATUS_NOT_APPLICABLE,
                    family=FAMILY_NONE,
                    not_applicable_reason=(
                        f"{entry.model_key}'s model-specific arm was disabled by "
                        "the frozen §34 calibration rule, so M1/M2 were never "
                        "generated; this contrast was never measured and is NOT "
                        "a null result"
                    ),
                )
            )
        else:
            updated.append(entry)
    return AnalysisRegistry(updated)


#: Every model declared in the Phase 3 study, in a stable order. Used only
#: to enumerate the per-model rows §44 specifies "per model"; it says
#: nothing about which of them will prove eligible.
_MODELS: tuple[tuple[str, str], ...] = (
    ("qwen", "Qwen"),
    ("llama", "Llama"),
    ("mistral", "Mistral"),
    ("gemma", "Gemma"),
)


def default_registry() -> AnalysisRegistry:
    """The frozen §44 analysis-status table, declared pre-outcome.

    Every row below corresponds to an analysis named in the frozen design
    (§28, §29, §44). No analysis is invented here, and no realized Holm
    family is constructed -- see the module docstring for what still has to
    resolve before Phase 3E membership is known.
    """
    entries: list[AnalysisEntry] = [
        # --- sole primary test (§28, §44) ------------------------------
        AnalysisEntry(
            name="cohort_a_qwen_corrective_frozen_pair",
            cohort="A",
            outcome="context_adopted",
            contrast="M1 vs M2 (frozen Phase 2 Qwen pair), KW",
            status=STATUS_PRIMARY,
            family=FAMILY_PRIMARY,
            contrast_kind=KIND_MODEL_SPECIFIC,
            model_key="qwen",
        ),
    ]

    # --- Cohort B model-specific contrasts, per model (§44) -------------
    # Corrective (KW) and harmful (KC) both require an M1/M2 arm, so both
    # become NOT APPLICABLE for a model disabled under §34.
    for key, label in _MODELS:
        entries.append(
            AnalysisEntry(
                name=f"cohort_b_{key}_corrective",
                cohort="B",
                outcome="context_adopted",
                contrast=f"{label} M1 vs M2, KW (generalization, not a direct "
                "replication)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
                contrast_kind=KIND_MODEL_SPECIFIC,
                model_key=key,
            )
        )
    for key, label in _MODELS:
        entries.append(
            AnalysisEntry(
                name=f"cohort_b_{key}_harmful",
                cohort="B",
                outcome="context_adopted",
                contrast=f"{label} M1 vs M2, KC (H4)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
                contrast_kind=KIND_MODEL_SPECIFIC,
                model_key=key,
            )
        )

    # --- Common fixed-source contrasts, per model (H2a, §44) ------------
    # This is what a §34 common-arm-only model still contributes.
    for key, label in _MODELS:
        # §28/§44: Qwen is counted ONCE. Its common-arm conflict estimate
        # rests on the same observations as its frozen-pair contrast,
        # because the frozen common pair coincides with Qwen's Phase 2 pair
        # (§19, §22). Declared, but never an independent family member and
        # never corroboration of Cohort A.
        counted_once = (
            "cohort_b_qwen_corrective" if key == "qwen" else None
        )
        entries.append(
            AnalysisEntry(
                name=f"common_fixed_source_{key}",
                cohort="B",
                outcome="context_adopted",
                contrast=f"{label} common identity A vs B, conflict cells",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
                contrast_kind=KIND_COMMON,
                model_key=key,
                counted_once_with=counted_once,
            )
        )

    # --- H1 continuous margin, per model (§44) --------------------------
    # A margin effect on context adoption, measurable from common-arm
    # observations, so it survives a disabled model-specific arm.
    for key, label in _MODELS:
        entries.append(
            AnalysisEntry(
                name=f"parametric_strength_h1_{key}",
                cohort="B",
                outcome="context_adopted",
                contrast=f"{label} continuous margin",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
                model_key=key,
            )
        )

    # --- Cohort C shared-cohort contrasts (§44) -------------------------
    for key, label in _MODELS:
        entries.append(
            AnalysisEntry(
                name=f"cohort_c_model_specific_{key}",
                cohort="C",
                outcome="context_adopted",
                contrast=f"{label} model-specific contrast on shared items",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
                contrast_kind=KIND_MODEL_SPECIFIC,
                model_key=key,
            )
        )
    entries.append(
        AnalysisEntry(
            name="cohort_c_common_cross_model",
            cohort="C",
            outcome="context_adopted",
            contrast="common contrast on shared items, across models",
            status=STATUS_SECONDARY,
            family=FAMILY_SECONDARY,
            contrast_kind=KIND_COMMON,
        )
    )

    entries.extend(
        [
            # --- cross-model interaction (RQ-B, §44) --------------------
            AnalysisEntry(
                name="cross_model_model_by_source_interaction",
                cohort="B+C",
                outcome="context_adopted",
                contrast="model x source",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            # --- §29 sensitivities that §28 places in the family --------
            AnalysisEntry(
                name="leave_one_relation_out",
                cohort="B",
                outcome="context_adopted",
                contrast="contrast, per relation dropped (§29.1)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="country_only_sensitivity",
                cohort="B",
                outcome="context_adopted",
                contrast="contrast, country only (§29.2)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="shared_cohort_restriction",
                cohort="C",
                outcome="context_adopted",
                contrast="contrast recomputed on shared items only (§29.3)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="model_specific_cohort_restriction",
                cohort="B",
                outcome="context_adopted",
                contrast="complement of the shared-cohort restriction (§29.4)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="margin_standardization_robustness",
                cohort="B",
                outcome="context_adopted",
                contrast="rank vs z-score margin standardization (§14, §29.5)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            AnalysisEntry(
                name="bootstrap_interval_robustness",
                cohort="B",
                outcome="context_adopted",
                contrast="bootstrap over items vs paired analytic interval (§29.7)",
                status=STATUS_SECONDARY,
                family=FAMILY_SECONDARY,
            ),
            # --- SECONDARY (mechanistic), family none (§44) -------------
            AnalysisEntry(
                name="tentative_answer_content_vs_commitment",
                cohort="all",
                outcome="tentative content",
                contrast="by source and knowledge group",
                status=STATUS_SECONDARY_MECHANISTIC,
                family=FAMILY_NONE,
            ),
            # --- EXPLORATORY (§44, §29.6) -------------------------------
            AnalysisEntry(
                name="source_by_parametric_strength_h3",
                cohort="B",
                outcome="context_adopted",
                contrast="margin x source, nonlinear/intermediate strength",
                status=STATUS_EXPLORATORY,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="kc_kw_agreement_subset",
                cohort="C",
                outcome="context_adopted",
                contrast="shared items where all models share the knowledge label",
                status=STATUS_EXPLORATORY,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="self_reported_confidence",
                cohort="all",
                outcome="confidence",
                contrast="per condition",
                status=STATUS_EXPLORATORY,
                family=FAMILY_NONE,
            ),
            # --- DIAGNOSTIC (§44) ---------------------------------------
            AnalysisEntry(
                name="common_arm_agreement_control",
                cohort="B",
                outcome="context_adopted",
                contrast="common A vs B, agreement cells",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
                contrast_kind=KIND_COMMON,
            ),
            AnalysisEntry(
                name="calibration_stability_qwen_llama",
                cohort="n/a",
                outcome="elicited pair",
                contrast="Phase 3 calibrated pair vs frozen Phase 2 pair; may "
                "never redefine the confirmatory contrast (§20.1)",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="cohort_a_relation_distribution",
                cohort="A",
                outcome="composition",
                contrast="most-frequent-relation share + dominance flag (§15.1)",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="relation_specific_descriptives",
                cohort="all",
                outcome="context_adopted",
                contrast="per relation",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="margin_bin_displays",
                cohort="all",
                outcome="context_adopted",
                contrast="per stratum",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="parsing_failure_rate_check",
                cohort="all",
                outcome="malformed rate",
                contrast="per model (gating, §34)",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="abstention_rates",
                cohort="all",
                outcome="Decision == uncertain",
                contrast="per condition",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="parsed_answer_accuracy",
                cohort="all",
                outcome="final_correct",
                contrast="per condition",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="c0_reproducibility_check",
                cohort="all",
                outcome="exact match",
                contrast="C0 vs baseline (gating, §34)",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
            AnalysisEntry(
                name="ceiling_floor_diagnostics",
                cohort="all",
                outcome="discordance/boundary/CI width",
                contrast="per contrast (§30)",
                status=STATUS_DIAGNOSTIC,
                family=FAMILY_NONE,
            ),
        ]
    )
    return AnalysisRegistry(entries)

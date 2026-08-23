"""Hard safety gate against a premature real Phase 3 run.

The frozen design states, in capitals and without qualification:

> **NO REAL PHASE 3 MODEL MAY RUN BEFORE 3C IS FROZEN.**
> (`docs/phase3_scaled_study_design.md`, §41)

This module is the programmatic expression of that rule. It is deliberately
*deny-by-default*: `assert_ready_for_real_run` raises unless every listed
Phase 3C prerequisite is demonstrably present, and the Phase 3B codebase
ships no command that loads a real model, so there is nothing for a caller
to accidentally bypass the gate into.

The gate checks state, not intentions. A config that merely asserts
`ready_for_real_run: true` while fields remain unresolved is rejected --
the flag is not self-certifying.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from conflict_eval.phase3.config import Phase3Config
from conflict_eval.phase3.constants import PHASE2_QWEN_KW_ITEM_COUNT
from conflict_eval.phase3.manifest import Phase3Manifest, validate_manifest


class Phase3NotReadyError(RuntimeError):
    """Raised when a real Phase 3 run is attempted before 3C is frozen."""


def _as_manifest(manifest: Any) -> Phase3Manifest:
    """Accept either a raw manifest mapping or a `Phase3Manifest`.

    The gate's public signature takes a plain dict, but `validate_manifest`
    owns the rules and works on the wrapper, so normalize here rather than
    duplicating any rule.
    """
    if isinstance(manifest, Phase3Manifest):
        return manifest
    return Phase3Manifest(data=manifest)


@dataclasses.dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    blockers: tuple[str, ...]

    def describe(self) -> str:
        if self.ready:
            return "All Phase 3C prerequisites satisfied."
        lines = "\n".join(f"  - {b}" for b in self.blockers)
        return "Phase 3C prerequisites unmet:\n" + lines


def check_readiness(
    config: Phase3Config,
    manifest: dict | None = None,
) -> ReadinessReport:
    """Evaluate every Phase 3C prerequisite and report all blockers.

    Returns the full list rather than the first failure, so a future
    Phase 3C operator sees everything outstanding at once.
    """
    blockers: list[str] = []

    unresolved_models = config.unresolved_models()
    if unresolved_models:
        blockers.append(
            "exact model id/revision unresolved for: "
            f"{unresolved_models} (Phase 3C must resolve and freeze these; §7, §42.1)"
        )

    # UNRESOLVED arm state still blocks. A model whose arm was explicitly
    # DISABLED under the frozen §34 calibration rule does NOT block: running
    # the common arm only is a legitimate pre-specified state, provided the
    # reason and calibration provenance are recorded (enforced by the config
    # loader, so a not-yet-calibrated model can never reach this point
    # disguised as a deliberate common-arm-only model).
    unresolved_roles = config.unresolved_source_roles()
    if unresolved_roles:
        blockers.append(
            "model-specific arm state unresolved for: "
            f"{unresolved_roles} (each model must either enable the arm with "
            "both source roles, or explicitly disable it under the frozen §34 "
            "calibration rule with a recorded reason; §20.2, §34)"
        )

    # §36 requires each NEW model's calibration output SHA256, preference
    # matrix and stated reason INSIDE the pre-run freeze manifest. An
    # incomplete record blocks; it is never silently accepted, and a missing
    # artifact hash is named field-by-field so nobody is tempted to invent
    # one to clear the gate.
    for model_key, missing in sorted(config.calibration_provenance_gaps().items()):
        blockers.append(
            f"calibration provenance for new model {model_key!r} is missing "
            f"{missing}; §36 requires the calibration output SHA256, preference "
            "matrix and the researcher's stated reason before the freeze is "
            "sealed (§20.2, §36)"
        )

    dataset_revision = (config.dataset or {}).get("revision")
    if not dataset_revision:
        blockers.append("dataset.revision is not pinned (§8)")

    if not config.ready_for_real_run:
        blockers.append(
            "config does not declare ready_for_real_run (must be set only at the "
            "Phase 3C freeze, §36)"
        )

    if manifest is None:
        blockers.append("no pre-run freeze manifest supplied (§36)")
    else:
        # ONE authoritative manifest-validation path. The gate delegates to
        # the full `validate_manifest`, so "the gate passed" can never mean
        # something weaker than "the manifest is valid". Previously the gate
        # re-derived only `_frozen_design_problems`, which meant two
        # different definitions of a valid manifest could disagree.
        #
        # The config's model keys are always supplied here, so the §36
        # every-model invariant is enforced on the path that actually
        # authorizes a run.
        blockers.extend(
            validate_manifest(
                _as_manifest(manifest), expected_model_keys=sorted(config.models)
            )
        )
        if not manifest.get("frozen"):
            blockers.append("freeze manifest is not marked frozen (§36)")
        if manifest.get("synthetic"):
            blockers.append(
                "freeze manifest is SYNTHETIC; synthetic Phase 3B artifacts can "
                "never authorize a real run"
            )
        for field in (
            "repository_commit",
            "cohort_membership_map",
            "final_margin_strata",
            "condition_specification",
            "prompt_version",
        ):
            if not manifest.get(field):
                blockers.append(f"freeze manifest missing required field {field!r} (§36)")
        cohorts = manifest.get("cohorts") or {}
        cohort_a = cohorts.get("A")
        if not cohort_a:
            blockers.append("freeze manifest has no finalized Cohort A membership (§36)")
        elif isinstance(cohort_a, dict):
            # Cohort A freshness provenance must be supplied by 3C: the
            # frozen design excludes the Phase 2 Qwen KW items (§15.1), so a
            # real run with no exclusion list is not a direct replication.
            excluded = cohort_a.get("excluded_phase2_item_ids")
            if not excluded:
                blockers.append(
                    "Cohort A freshness exclusion list is missing/empty; Phase 3C "
                    "must supply the frozen Phase 2 Qwen KW item ids (§15.1)"
                )
            elif len(excluded) != PHASE2_QWEN_KW_ITEM_COUNT:
                # Size check only -- the ids themselves are never invented here.
                blockers.append(
                    f"Cohort A exclusion list has {len(excluded)} ids; the frozen "
                    f"Phase 2 pilot selected {PHASE2_QWEN_KW_ITEM_COUNT} Qwen KW "
                    "items (§15.1). Verify the supplied provenance."
                )
            if cohort_a.get("realized_relation_distribution") is None:
                blockers.append(
                    "Cohort A realized relation distribution missing; it is a "
                    "mandatory report alongside the primary estimate (§15.1, §26.1)"
                )
            if cohort_a.get("relation_dominance_flag") is None:
                blockers.append(
                    "Cohort A relation-dominance diagnostic missing (§15.1)"
                )

    return ReadinessReport(ready=not blockers, blockers=tuple(blockers))


def assert_ready_for_real_run(
    config: Phase3Config, manifest: dict | None = None
) -> None:
    """Raise `Phase3NotReadyError` unless every 3C prerequisite is met.

    Any future real-run entry point must call this first. Phase 3B provides
    no such entry point by design.
    """
    report = check_readiness(config, manifest)
    if not report.ready:
        raise Phase3NotReadyError(
            "Refusing to start a real Phase 3 run. "
            "NO REAL PHASE 3 MODEL MAY RUN BEFORE 3C IS FROZEN "
            "(docs/phase3_scaled_study_design.md, §41).\n"
            + report.describe()
        )

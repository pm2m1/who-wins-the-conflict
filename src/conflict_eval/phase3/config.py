"""Phase 3 study configuration schema and loader.

Separate from `conflict_eval.config` (the Phase 2 pilot schema), which is
left completely unchanged: Phase 2 must continue to behave exactly as
before (Phase 3B brief §19). This module adds a Phase 3 namespace rather
than retrofitting the Phase 2 loader.

The loader deliberately **accepts** unresolved fields -- the two approved
new model families have no exact release, repository id, or commit SHA
until Phase 3C (`docs/phase3_scaled_study_design.md`, §7, §42.1), and
representing that state honestly is a requirement, not a defect. What it
refuses is *executing* while those fields are unresolved; see
`real_run_gate.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from conflict_eval.phase3.calibration_provenance import (
    CalibrationProvenanceError,
    missing_required_fields,
    validate_calibration_provenance,
)
from conflict_eval.phase3.constants import (
    ARM_DISABLED_BY_CALIBRATION,
    ARM_ENABLED,
    ARM_UNRESOLVED,
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
    CONDITIONS_COMMON_ARM_ONLY,
    CONDITIONS_WITH_MODEL_SPECIFIC_ARM,
    FROZEN_MODEL_REVISIONS,
    FROZEN_MODEL_SOURCE_PAIRS,
    SYMBOLIC_REVISIONS,
)


class Phase3ConfigError(ValueError):
    """Raised when a Phase 3 configuration is structurally invalid."""


@dataclasses.dataclass(frozen=True)
class Phase3ModelEntry:
    """One model in the Phase 3 study.

    `resolved` is False for the approved-but-unresolved families. Their
    `hf_model_id` and `revision` stay `None` until Phase 3C; the loader
    never fills them in and never guesses.
    """

    key: str
    family: str
    hf_model_id: str | None
    revision: str | None
    role: str  # "replication" | "new"
    preferred_source: str | None
    dispreferred_source: str | None
    # Tri-state model-specific arm (§20.2, §34). `None` in the config means
    # UNRESOLVED -- not yet calibrated -- which still blocks a real run.
    model_specific_arm_enabled: bool | None = None
    model_specific_arm_reason: str | None = None
    calibration_provenance: dict[str, Any] | None = None

    @property
    def arm_state(self) -> str:
        """ENABLED / DISABLED_BY_CALIBRATION / UNRESOLVED (§20.2, §34)."""
        if self.model_specific_arm_enabled is True:
            return ARM_ENABLED
        if self.model_specific_arm_enabled is False:
            return ARM_DISABLED_BY_CALIBRATION
        return ARM_UNRESOLVED

    @property
    def runs_model_specific_arm(self) -> bool:
        return self.arm_state == ARM_ENABLED

    @property
    def condition_set(self) -> tuple[str, ...]:
        """The conditions actually generated for this model (§22).

        A disabled arm yields the common arm only -- no placeholder M1/M2.
        """
        return (
            CONDITIONS_WITH_MODEL_SPECIFIC_ARM
            if self.runs_model_specific_arm
            else CONDITIONS_COMMON_ARM_ONLY
        )

    @property
    def resolved(self) -> bool:
        return self.hf_model_id is not None and self.revision is not None

    @property
    def source_roles_resolved(self) -> bool:
        return (
            self.preferred_source is not None
            and self.dispreferred_source is not None
        )


@dataclasses.dataclass(frozen=True)
class Phase3Config:
    seed: int
    dataset: dict[str, Any]
    common_source_a: str
    common_source_b: str
    models: dict[str, Phase3ModelEntry]
    screening: dict[str, Any]
    cohorts: dict[str, Any]
    paths: dict[str, str]
    prompts_config: str
    sources_config: str
    ready_for_real_run: bool
    raw: dict[str, Any]

    def model(self, key: str) -> Phase3ModelEntry:
        if key not in self.models:
            raise Phase3ConfigError(
                f"Unknown Phase 3 model key {key!r}. Known: {sorted(self.models)}"
            )
        return self.models[key]

    def unresolved_models(self) -> list[str]:
        return sorted(k for k, m in self.models.items() if not m.resolved)

    def unresolved_source_roles(self) -> list[str]:
        """Models whose arm state is UNRESOLVED -- neither calibrated into an
        enabled pair nor explicitly disabled under §34. These still block."""
        return sorted(
            k for k, m in self.models.items() if m.arm_state == ARM_UNRESOLVED
        )

    def model_specific_arm_models(self) -> list[str]:
        """Models that will actually generate M1/M2 (§22)."""
        return sorted(k for k, m in self.models.items() if m.runs_model_specific_arm)

    def common_arm_only_models(self) -> list[str]:
        """Models disabled by the frozen §34 calibration rule."""
        return sorted(
            k for k, m in self.models.items()
            if m.arm_state == ARM_DISABLED_BY_CALIBRATION
        )

    def calibration_provenance_gaps(self) -> dict[str, list[str]]:
        """§36-required calibration fields still missing, per NEW model.

        Only `role == "new"` models are checked. Qwen and Llama carry their
        frozen Phase 2 pairs and are explicitly exempt from fresh calibration
        (§20.1); the frozen design asks nothing further of them, so demanding
        Phase 3 calibration artifacts from them would be an invented rule.

        A non-empty result is a FREEZE BLOCKER, not a load error: Phase 3C
        assembles provenance incrementally, but §36 must be satisfied before
        the manifest is sealed.
        """
        gaps: dict[str, list[str]] = {}
        for key, entry in sorted(self.models.items()):
            if entry.role != "new":
                continue
            missing = missing_required_fields(entry.calibration_provenance)
            if missing:
                gaps[key] = missing
        return gaps


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise Phase3ConfigError(f"Phase 3 config not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise Phase3ConfigError(f"Phase 3 config {path} did not parse to a mapping")
    return data


def load_phase3_config(path: str | Path) -> Phase3Config:
    raw = _load_yaml(path)
    required = {"seed", "dataset", "sources", "models", "screening", "cohorts", "paths"}
    missing = required - raw.keys()
    if missing:
        raise Phase3ConfigError(f"{path}: missing top-level keys {sorted(missing)}")

    sources = raw["sources"]
    common_a = sources.get("common_source_a")
    common_b = sources.get("common_source_b")
    # The common pair is FROZEN by the design (§19); a config may not
    # silently redefine it.
    if common_a != COMMON_SOURCE_A or common_b != COMMON_SOURCE_B:
        raise Phase3ConfigError(
            f"{path}: the common source pair is frozen as "
            f"({COMMON_SOURCE_A!r}, {COMMON_SOURCE_B!r}) by "
            "docs/phase3_scaled_study_design.md §19; got "
            f"({common_a!r}, {common_b!r})"
        )

    models: dict[str, Phase3ModelEntry] = {}
    for key, entry in raw["models"].items():
        role = entry.get("role")
        if role not in ("replication", "new"):
            raise Phase3ConfigError(
                f"{path}: model {key!r} role must be 'replication' or 'new', "
                f"got {role!r}"
            )
        hf_model_id = entry.get("hf_model_id")
        revision = entry.get("revision")
        preferred = entry.get("preferred_source")
        dispreferred = entry.get("dispreferred_source")

        if role == "replication":
            # Qwen and Llama reuse their exact frozen Phase 2 artifacts and
            # frozen source pairs (§7, §20.1). A config that disagrees is a
            # design change, so it is rejected rather than accepted.
            frozen_rev = FROZEN_MODEL_REVISIONS.get(key)
            frozen_pair = FROZEN_MODEL_SOURCE_PAIRS.get(key)
            if frozen_rev is None or frozen_pair is None:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} is declared 'replication' but has no "
                    "frozen Phase 2 revision/source pair on record"
                )
            if hf_model_id != frozen_rev["hf_model_id"] or revision != frozen_rev["revision"]:
                raise Phase3ConfigError(
                    f"{path}: replication model {key!r} must pin the exact frozen "
                    f"Phase 2 artifact {frozen_rev['hf_model_id']}@"
                    f"{frozen_rev['revision']} (§7); got {hf_model_id}@{revision}"
                )
            if (
                preferred != frozen_pair["preferred_source"]
                or dispreferred != frozen_pair["dispreferred_source"]
            ):
                raise Phase3ConfigError(
                    f"{path}: replication model {key!r} must use its frozen Phase 2 "
                    f"source pair {frozen_pair} (§20.1); got "
                    f"{{'preferred_source': {preferred!r}, "
                    f"'dispreferred_source': {dispreferred!r}}}"
                )
        else:
            # New families. Before Phase 3C these are fully unresolved; at
            # Phase 3C the researcher resolves the exact id/revision and
            # records the calibration outcome. Identity must be all-or-
            # nothing so a half-resolved model cannot slip through.
            if (hf_model_id is None) != (revision is None):
                raise Phase3ConfigError(
                    f"{path}: model {key!r} has a partially resolved identity "
                    f"(hf_model_id={hf_model_id!r}, revision={revision!r}). Both "
                    "must be resolved together at Phase 3C, or both left null "
                    "(§7, §42.1)."
                )
            if revision is not None and revision.strip().lower() in SYMBOLIC_REVISIONS:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} revision must be an exact immutable "
                    f"commit SHA, never {revision!r} (§35)."
                )

        # --- model-specific arm state (§20.2, §34), all roles ------------
        arm_enabled = entry.get("model_specific_arm_enabled")
        arm_reason = entry.get("model_specific_arm_reason")
        calibration = entry.get("calibration_provenance")
        if arm_enabled is not None and not isinstance(arm_enabled, bool):
            raise Phase3ConfigError(
                f"{path}: model {key!r} model_specific_arm_enabled must be a "
                f"boolean or absent, got {arm_enabled!r}"
            )

        # Shape-check whatever calibration provenance is recorded (§20.2,
        # §36). Malformed hashes, placeholder text, inconsistent trial counts
        # and a matrix that disagrees with its own summary are rejected here.
        # Completeness is a FREEZE requirement, checked by the manifest and
        # the real-run gate, so an in-progress Phase 3C record stays loadable
        # while an incorrect one never does.
        try:
            validate_calibration_provenance(
                f"{path}: model {key!r} calibration_provenance", calibration
            )
        except CalibrationProvenanceError as exc:
            raise Phase3ConfigError(str(exc)) from exc

        if arm_enabled is True:
            # ENABLED: both roles are mandatory. A half-specified pair is
            # not a contrast.
            if preferred is None or dispreferred is None:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} enables the model-specific arm but "
                    f"has preferred_source={preferred!r}, "
                    f"dispreferred_source={dispreferred!r}. An enabled arm "
                    "requires BOTH roles (§20, §22)."
                )
        elif arm_enabled is False:
            # DISABLED by the frozen §34 calibration rule. Roles MUST stay
            # null -- disabling the arm is precisely the refusal to invent a
            # pair -- and an explicit reason plus calibration provenance is
            # required so this can never be confused with UNRESOLVED.
            for field_name, value in (
                ("preferred_source", preferred),
                ("dispreferred_source", dispreferred),
            ):
                if value is not None:
                    raise Phase3ConfigError(
                        f"{path}: model {key!r} disables the model-specific arm "
                        f"but still sets {field_name}={value!r}. A disabled arm "
                        "must leave both source roles null; the frozen rule is "
                        "to run the common arm only, never to force a pair "
                        "(§20.2, §34)."
                    )
            if not (isinstance(arm_reason, str) and arm_reason.strip()):
                raise Phase3ConfigError(
                    f"{path}: model {key!r} disables the model-specific arm but "
                    "records no model_specific_arm_reason. The frozen §34 "
                    "fallback must be documented explicitly, otherwise a "
                    "not-yet-calibrated model is indistinguishable from a "
                    "deliberate common-arm-only model (§20.2, §34)."
                )
            if not isinstance(calibration, dict) or not calibration:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} disables the model-specific arm but "
                    "records no calibration_provenance. The calibration that "
                    "justified the fallback must be recorded (§34, §36)."
                )
        else:
            # UNRESOLVED: not calibrated. Roles must be null and no reason
            # may be claimed; the gate keeps blocking.
            if preferred is not None or dispreferred is not None:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} has source roles but does not declare "
                    "model_specific_arm_enabled. Declare the arm state "
                    "explicitly (§20.2)."
                )
            if arm_reason is not None:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} records a model_specific_arm_reason "
                    "without declaring model_specific_arm_enabled. An "
                    "uncalibrated model may not claim the §34 fallback."
                )

        if role == "replication" and arm_enabled is not True:
            # Qwen/Llama carry frozen Phase 2 pairs; their arm is the direct
            # replication contrast and cannot be switched off here.
            raise Phase3ConfigError(
                f"{path}: replication model {key!r} must keep its "
                "model-specific arm enabled -- it carries the frozen Phase 2 "
                "pair used for the direct replication (§20.1)."
            )

        family = entry.get("family")
        if not family:
            raise Phase3ConfigError(f"{path}: model {key!r} missing 'family'")

        models[key] = Phase3ModelEntry(
            key=key,
            family=family,
            hf_model_id=hf_model_id,
            revision=revision,
            role=role,
            preferred_source=preferred,
            dispreferred_source=dispreferred,
            model_specific_arm_enabled=arm_enabled,
            model_specific_arm_reason=arm_reason,
            calibration_provenance=calibration,
        )

    return Phase3Config(
        seed=int(raw["seed"]),
        dataset=raw["dataset"],
        common_source_a=common_a,
        common_source_b=common_b,
        models=models,
        screening=raw["screening"],
        cohorts=raw["cohorts"],
        paths=raw["paths"],
        prompts_config=raw.get("prompts_config", "configs/prompts.yaml"),
        sources_config=raw.get("sources_config", "configs/sources.yaml"),
        # Never true in a committed Phase 3B config; the gate re-checks it.
        ready_for_real_run=bool(raw.get("ready_for_real_run", False)),
        raw=raw,
    )

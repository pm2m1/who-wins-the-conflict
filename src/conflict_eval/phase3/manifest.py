"""Phase 3 provenance / pre-run freeze manifest.

Implements the manifest contents frozen in
`docs/phase3_scaled_study_design.md` §36. Phase 3B can build only
**synthetic** manifests: no real revision, hash, cohort, or environment
value exists yet, and inventing one is forbidden.

Two safety properties:

- A manifest built here is stamped `synthetic: True` and `frozen: False`
  unless a caller explicitly freezes a fully-resolved one, and
  `real_run_gate` rejects any synthetic manifest outright.
- `validate_manifest` refuses `ready_for_real_run` while any required
  Phase 3C field is unresolved (§36), so the flag cannot be set optimistically.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from conflict_eval.phase3.analysis_status import AnalysisRegistry, default_registry
from conflict_eval.phase3.calibration_provenance import (
    SHA256_PATTERN,
    CalibrationProvenanceError,
    missing_required_fields,
    validate_calibration_provenance,
)
from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_SUPPLY,
    COHORT_A_PER_STRATUM_TARGET,
    COHORT_A_TOTAL_TARGET,
    COHORT_B_CELL_MINIMUM,
    COHORT_B_CELL_TARGET,
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
    CONDITIONS_COMMON_ARM_ONLY,
    CONDITIONS_WITH_MODEL_SPECIFIC_ARM,
    MARGIN_STRATA,
    MODEL_SPECIFIC_ARM_CONDITIONS,
    NOMINAL_CONDITIONS,
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
    SYMBOLIC_REVISIONS,
)

# Fields §36 requires before the manifest may be frozen for a real run.
REQUIRED_FREEZE_FIELDS = (
    "repository_commit",
    "dataset",
    "models",
    "prompt_version",
    "sources",
    "cohorts",
    "cohort_membership_map",
    "condition_specification",
    "deduplication_alias_map",
    # §36 "margin-bin edges" -- the frozen bin boundaries per model/group.
    "final_margin_strata",
    "screening",
    "analysis_status",
    "seed",
    "artifact_hashes",
    "environment",
    "hardware",
)

# --- §36 artifact digests (top level, inside `artifact_hashes`) -----------
# "Repository commit SHA; Phase 3 config file contents and SHA256."
# "Dataset id, split, resolved revision; candidate file SHA256 and IDs."
# "The full condition specification and trial file SHA256."
REQUIRED_ARTIFACT_HASHES = (
    "phase3_config",
    "candidate_file",
    "trial_file",
)

# --- §36 per-model provenance --------------------------------------------
# "Every model: id, requested revision, resolved revision, precision,
#  quantization status (none), device_map, max_memory actually used."
REQUIRED_MODEL_RUNTIME_FIELDS = (
    "dtype",
    "quantization",
    "device_map",
    "max_memory",
)
# "Per model: baseline and exclusion file SHA256; KC/KW membership; margins;
#  margin-bin edges; manual-review decisions."
REQUIRED_MODEL_SHA_FIELDS = (
    "baseline_file_sha256",
    "exclusion_file_sha256",
)
REQUIRED_MODEL_SCREENING_FIELDS = (
    "knowledge_membership",
    "margins",
)

# §36 "Dataset id, split, resolved revision; candidate file SHA256 and IDs."
REQUIRED_DATASET_FIELDS = ("hf_dataset_id", "split", "revision", "candidate_item_ids")

# Per-cohort provenance §36 requires *inside* `cohorts`. Validated by
# content, not merely by key presence, so an incomplete real manifest
# cannot be frozen.
REQUIRED_COHORT_A_FIELDS = (
    # Planned/frozen design (§15.1) -- validated against constants.py.
    "planned_total_target",
    "planned_per_stratum_target",
    "planned_supply_per_stratum",
    # Realized outcome.
    "selected_item_ids",
    "realized_total",
    "excluded_phase2_item_ids",
    "realized_relation_distribution",
    "relation_dominance_share",
    "relation_dominance_flag",
    "per_stratum_selected",
    "status",
)
REQUIRED_COHORT_B_GROUP_FIELDS = (
    "target_cell_count",
    "minimum_cell_count",
    "original_cell_counts",
    "qualifying_relations",
    "excluded_short_relations",
    "realized_relations",
    "realized_cell_count",
    "status",
    "confirmatory_eligible",
    "selected_item_ids",
    "reduction_reason",
)
REQUIRED_COHORT_C_ITEM_FIELDS = ("item_id", "relation", "per_model")
REQUIRED_COHORT_C_PER_MODEL_FIELDS = (
    "knowledge_group",
    "parametric_margin",
    "margin_stratum",
)


class ManifestError(ValueError):
    """Raised when a manifest is internally inconsistent."""


def model_arm_provenance(entry) -> dict[str, Any]:
    """Serialize one model's identity + model-specific arm state (§20.2, §34,
    §36).

    Records the tri-state arm explicitly so a manifest can never leave the
    reader guessing whether a null source pair means "disabled by the frozen
    calibration rule" or "nobody calibrated this model yet".

    The full `calibration_provenance` record is carried through verbatim --
    artifact hashes, trial counts, preference matrix and the researcher's
    stated reason -- because §36 requires those *in the manifest*, not merely
    in the config that produced it. For `role == "new"` models the missing
    §36 fields are listed alongside it, so an incomplete record is visible in
    the artifact itself rather than only at the gate.
    """
    record = {
        "hf_model_id": entry.hf_model_id,
        "requested_revision": entry.revision,
        "resolved_revision": entry.revision,
        "family": entry.family,
        "role": entry.role,
        "dtype": "float16",
        "quantization": "none",
        "model_specific_arm_enabled": entry.runs_model_specific_arm,
        "model_specific_arm_state": entry.arm_state,
        "model_specific_arm_reason": entry.model_specific_arm_reason,
        "preferred_source": entry.preferred_source,
        "dispreferred_source": entry.dispreferred_source,
        "calibration_provenance": entry.calibration_provenance,
        # The conditions this model ACTUALLY runs. The study-level
        # `condition_specification` is the full seven-condition design
        # vocabulary; this per-model field is the realized subset (§22, §34).
        "condition_set": list(entry.condition_set),
    }
    if entry.role == "new":
        record["calibration_provenance_missing_fields"] = missing_required_fields(
            entry.calibration_provenance
        )
    return record


def _sha_problem(label: str, value: Any) -> str | None:
    """Validate one §36 digest against the SINGLE shared SHA256 pattern.

    Deliberately reuses `calibration_provenance.SHA256_PATTERN` rather than
    defining a second regex: one definition means a digest that is valid in
    one part of the freeze manifest can never be invalid in another.
    """
    if value is None or value == "":
        return f"{label} is missing; §36 requires this SHA256"
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        return (
            f"{label} must be a lowercase 64-character hex SHA256, got "
            f"{value!r}. Placeholder or prose values are not provenance (§36)."
        )
    return None


def _populated_capture_problems(label: str, capture: Any) -> list[str]:
    """§36 "Environment and hardware capture".

    Presence is not capture. The Phase 3B builder seeds these with `None`
    placeholders precisely so they are visibly unfilled; a frozen manifest
    that still carries them has recorded nothing, so every value must be
    populated.
    """
    if capture is None:
        return [f"{label} capture is null; §36 requires an actual capture"]
    if not isinstance(capture, dict) or not capture:
        return [
            (
                f"{label} capture must be a non-empty mapping, got "
                f"{type(capture).__name__} (§36)"
            )
        ]
    unfilled = sorted(k for k, v in capture.items() if v in (None, "", {}, []))
    if unfilled:
        return [
            (
                f"{label} capture still has unpopulated placeholder field(s) "
                f"{unfilled}; §36 requires an actual capture, not the unfilled "
                "template"
            )
        ]
    return []


def _artifact_hash_problems(data: dict[str, Any]) -> list[str]:
    """§36 top-level artifact digests."""
    problems: list[str] = []
    hashes = data.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        return [
            (
                "artifact_hashes is missing or empty; §36 requires the Phase 3 "
                "config, candidate file and trial file SHA256 (§36)"
            )
        ]
    for field in REQUIRED_ARTIFACT_HASHES:
        problem = _sha_problem(f"artifact_hashes[{field!r}]", hashes.get(field))
        if problem:
            problems.append(problem)
    return problems


def _dataset_problems(data: dict[str, Any]) -> list[str]:
    """§36 "Dataset id, split, resolved revision; candidate file SHA256 and
    IDs." The digest lives in `artifact_hashes['candidate_file']`; the IDs
    live here."""
    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or not dataset:
        return ["dataset provenance is missing (§36)"]
    problems = []
    for field in REQUIRED_DATASET_FIELDS:
        if dataset.get(field) in (None, "", [], {}):
            problems.append(
                f"dataset provenance missing {field!r}; §36 requires the "
                "dataset id, split, resolved revision and the candidate ids"
            )
    return problems


def _model_screening_provenance_problems(key: str, entry: dict[str, Any]) -> list[str]:
    """§36 per-model runtime + screening provenance.

    Covers "precision, quantization status (none), device_map, max_memory
    actually used" and "baseline and exclusion file SHA256; KC/KW
    membership; margins; manual-review decisions".
    """
    problems: list[str] = []
    for field in REQUIRED_MODEL_RUNTIME_FIELDS:
        if entry.get(field) in (None, "", {}, []):
            problems.append(
                f"model {key!r} is missing runtime provenance {field!r}; §36 "
                "requires precision, quantization, device_map and max_memory "
                "as actually used"
            )
    quantization = entry.get("quantization")
    if quantization is not None and str(quantization).lower() not in ("none", "false"):
        problems.append(
            f"model {key!r} records quantization={quantization!r}; the frozen "
            "design runs unquantized (§7, §36)"
        )
    for field in REQUIRED_MODEL_SHA_FIELDS:
        problem = _sha_problem(f"model {key!r} {field}", entry.get(field))
        if problem:
            problems.append(problem)
    for field in REQUIRED_MODEL_SCREENING_FIELDS:
        if entry.get(field) in (None, "", {}, []):
            problems.append(
                f"model {key!r} is missing {field!r}; §36 requires per-model "
                "KC/KW membership and margins in the freeze manifest"
            )
    # Manual-review decisions must be RECORDED. An explicit empty list is a
    # legitimate finding ("no manual overrides were made"); an absent key is
    # an unfinished record, exactly as with `preference_matrix`.
    if "manual_review_decisions" not in entry:
        problems.append(
            f"model {key!r} does not record manual_review_decisions; §36 "
            "requires them (record an explicit empty list if none were made)"
        )
    return problems


def _model_set_problems(
    data: dict[str, Any], expected_model_keys: Iterable[str] | None
) -> list[str]:
    """§36 "**Every model**: id, requested revision, resolved revision, ..."

    Every per-model rule below is applied per ENTRY, so a manifest that
    simply omits a model would skip all of them -- including the §34
    disabled-arm rules and the §36 new-model calibration requirements. The
    manifest must therefore describe exactly the configured model set: no
    omissions, and no extras that the config never declared.
    """
    if expected_model_keys is None:
        return []
    expected = set(expected_model_keys)
    found = set(data.get("models") or {})
    problems = []
    missing = sorted(expected - found)
    if missing:
        problems.append(
            f"freeze manifest omits configured model(s) {missing}; §36 requires "
            "EVERY model, and an omitted model silently skips its arm-state, "
            "condition-set and calibration validation"
        )
    extra = sorted(found - expected)
    if extra:
        problems.append(
            f"freeze manifest describes model(s) {extra} that the Phase 3 "
            "config does not declare; the manifest and config must agree "
            "exactly (§36)"
        )
    return problems


def _model_arm_problems(key: str, entry: dict[str, Any]) -> list[str]:
    """Validate one manifest model entry's arm state (§20.2, §34)."""
    problems: list[str] = []
    enabled = entry.get("model_specific_arm_enabled")
    preferred = entry.get("preferred_source")
    dispreferred = entry.get("dispreferred_source")
    conditions = entry.get("condition_set")

    if enabled is None:
        problems.append(
            f"model {key!r} does not declare model_specific_arm_enabled; the "
            "arm state must be explicit so a disabled arm is distinguishable "
            "from an uncalibrated one (§20.2, §34)"
        )
        return problems

    if enabled:
        if not preferred or not dispreferred:
            problems.append(
                f"model {key!r} enables the model-specific arm but is missing a "
                f"source role (preferred={preferred!r}, "
                f"dispreferred={dispreferred!r}) (§20, §22)"
            )
        if conditions is not None and list(conditions) != list(
            CONDITIONS_WITH_MODEL_SPECIFIC_ARM
        ):
            problems.append(
                f"model {key!r} enables the model-specific arm but its "
                f"condition_set is {list(conditions)!r}; expected "
                f"{list(CONDITIONS_WITH_MODEL_SPECIFIC_ARM)} (§22)"
            )
    else:
        for field_name, value in (
            ("preferred_source", preferred),
            ("dispreferred_source", dispreferred),
        ):
            if value is not None:
                problems.append(
                    f"model {key!r} disables the model-specific arm but records "
                    f"{field_name}={value!r}; the frozen §34 fallback runs the "
                    "common arm only and never forces a pair"
                )
        reason = entry.get("model_specific_arm_reason")
        if not (isinstance(reason, str) and reason.strip()):
            problems.append(
                f"model {key!r} disables the model-specific arm without a "
                "model_specific_arm_reason; the §34 fallback must be documented "
                "or it is indistinguishable from an uncalibrated model"
            )
        calibration = entry.get("calibration_provenance")
        if not isinstance(calibration, dict) or not calibration:
            problems.append(
                f"model {key!r} disables the model-specific arm without "
                "calibration_provenance; the calibration that justified the "
                "fallback must be recorded (§34, §36)"
            )
        if conditions is not None and list(conditions) != list(
            CONDITIONS_COMMON_ARM_ONLY
        ):
            problems.append(
                f"model {key!r} disables the model-specific arm but its "
                f"condition_set is {list(conditions)!r}; a disabled arm runs the "
                f"common arm only, {list(CONDITIONS_COMMON_ARM_ONLY)} (§22, §34)"
            )

    # --- §36 calibration provenance, NEW models only ---------------------
    # Qwen and Llama are exempt: §20.1 freezes their Phase 2 pairs and asks
    # for no Phase 3 calibration, so requiring artifacts from them would be
    # an invented rule rather than a frozen one.
    if entry.get("role") == "new":
        calibration = entry.get("calibration_provenance")
        try:
            validate_calibration_provenance(
                f"model {key!r} calibration_provenance", calibration
            )
        except CalibrationProvenanceError as exc:
            problems.append(str(exc))
        else:
            missing = missing_required_fields(calibration)
            if missing:
                problems.append(
                    f"model {key!r} is a new model but its calibration "
                    f"provenance is missing {missing}; §36 requires the "
                    "calibration output SHA256, preference matrix and the "
                    "researcher's stated reason for every new model"
                )
    return problems


def _disabled_arm_observation_problems(data: dict[str, Any]) -> list[str]:
    """Refuse a manifest that claims M1/M2 observations for a disabled arm.

    Those generations never happened. Recording them -- even as empty
    placeholders -- would later be indistinguishable from a measured null,
    which §34 forbids ("this is an informative measurement about that model,
    not a failure", and never a null result).
    """
    problems: list[str] = []
    models = data.get("models") or {}
    disabled = {
        key for key, entry in models.items()
        if entry.get("model_specific_arm_enabled") is False
    }
    if not disabled:
        return problems
    model_specific = set(MODEL_SPECIFIC_ARM_CONDITIONS)
    for alias_key in (data.get("deduplication_alias_map") or {}):
        parts = str(alias_key).split("|")
        if len(parts) == 3 and parts[0] in disabled and parts[2] in model_specific:
            problems.append(
                f"model {parts[0]!r} has a disabled model-specific arm but the "
                f"deduplication alias map claims a {parts[2]!r} observation "
                f"({alias_key!r}); those generations were never run (§22, §34)"
            )
    return problems


def planned_cohort_a_design() -> dict[str, Any]:
    """The FROZEN planned Cohort A design (§15.1, §11).

    Emitted into every manifest so planned targets are recorded explicitly
    and can never be inferred from whatever was realized. Values come from
    `constants.py`, the single authoritative representation.
    """
    return {
        "planned_total_target": COHORT_A_TOTAL_TARGET,
        "planned_per_stratum_target": {
            stratum: COHORT_A_PER_STRATUM_TARGET for stratum in MARGIN_STRATA
        },
        "planned_supply_per_stratum": COHORT_A_PER_STRATUM_SUPPLY,
    }


def planned_screening_design() -> dict[str, Any]:
    """The FROZEN acquisition constants (§11)."""
    return {
        "block_size": SCREENING_BLOCK_SIZE,
        "ceiling_per_model": SCREENING_CEILING_PER_MODEL,
        "cohort_a_supply_per_stratum": COHORT_A_PER_STRATUM_SUPPLY,
    }


def _frozen_design_problems(data: dict[str, Any]) -> list[str]:
    """Reject a manifest whose PLANNED/FROZEN design constants differ from
    the frozen protocol.

    This is the single source of truth shared by `validate_manifest` and the
    real-run gate, so the check cannot be bypassed by calling one and not
    the other.

    **Planned vs. realized.** This function validates only the *planned*
    design. The frozen protocol legitimately permits realized reductions --
    a Cohort B cell realized at 6 or 7, a three-relation fallback, or an
    eligibility-limited Cohort A (§32, §15.1) -- and those are checked
    separately in `validate_manifest`. What is rejected here is a manifest
    claiming the *protocol itself* was something other than 96 / 32-32-32 /
    8 / 6 / 250 / 2000.

    Comparison is against `constants.py`, never against a copy stored
    inside the manifest, so editing both does not defeat the check.
    """
    problems: list[str] = []
    cohorts = data.get("cohorts") or {}

    # --- Cohort A planned design (§15.1) ---------------------------------
    cohort_a = cohorts.get("A")
    if isinstance(cohort_a, dict):
        planned_total = cohort_a.get("planned_total_target")
        if planned_total != COHORT_A_TOTAL_TARGET:
            problems.append(
                f"Cohort A planned target must be {COHORT_A_TOTAL_TARGET}; "
                f"found {planned_total!r} (frozen design §15.1 -- this is the "
                "PLANNED target, not the realized cohort size)"
            )
        planned_strata = cohort_a.get("planned_per_stratum_target")
        expected_strata = {s: COHORT_A_PER_STRATUM_TARGET for s in MARGIN_STRATA}
        if planned_strata != expected_strata:
            problems.append(
                f"Cohort A planned per-stratum target must be {expected_strata}; "
                f"found {planned_strata!r} (frozen design §15.1)"
            )
        planned_supply = cohort_a.get("planned_supply_per_stratum")
        if planned_supply != COHORT_A_PER_STRATUM_SUPPLY:
            problems.append(
                f"Cohort A planned supply criterion must be "
                f"{COHORT_A_PER_STRATUM_SUPPLY}/stratum; found {planned_supply!r} "
                "(frozen design §11)"
            )

    # --- Cohort B planned design (§15.2, §32) ----------------------------
    cohort_b = cohorts.get("B")
    if isinstance(cohort_b, dict):
        for group_key, group in cohort_b.items():
            if not isinstance(group, dict):
                continue
            target = group.get("target_cell_count")
            if target != COHORT_B_CELL_TARGET:
                problems.append(
                    f"Cohort B {group_key!r} target_cell_count must be "
                    f"{COHORT_B_CELL_TARGET}; found {target!r} (frozen design "
                    "§15.2 -- the planned target may not be rewritten to match "
                    "a realized reduction)"
                )
            minimum = group.get("minimum_cell_count")
            if minimum != COHORT_B_CELL_MINIMUM:
                problems.append(
                    f"Cohort B {group_key!r} minimum_cell_count must be "
                    f"{COHORT_B_CELL_MINIMUM}; found {minimum!r} (frozen design "
                    "§15.2, §32)"
                )

    # --- Acquisition constants (§11) -------------------------------------
    screening = data.get("screening")
    if isinstance(screening, dict):
        expected_screening = planned_screening_design()
        for field, expected in expected_screening.items():
            if field not in screening:
                problems.append(
                    f"screening provenance missing frozen constant {field!r} "
                    f"(must be {expected}; frozen design §11)"
                )
            elif screening[field] != expected:
                problems.append(
                    f"screening {field} must be {expected}; found "
                    f"{screening[field]!r} (frozen design §11)"
                )

    # --- Condition design (§22) ------------------------------------------
    # `condition_specification` is the study-wide NOMINAL DESIGN VOCABULARY:
    # all seven conditions the design defines. It is NOT a claim that every
    # model ran all seven. What each model actually runs is its own
    # `models[key]["condition_set"]`:
    #
    #     ENABLED arm                 C0, K1, K2, K3, K4, M1, M2
    #     DISABLED_BY_CALIBRATION     C0, K1, K2, K3, K4   (§34 common arm only)
    #
    # The two are validated against each other by `_model_arm_problems`, and
    # no placeholder M1/M2 record is ever created for a disabled arm.
    conditions = data.get("condition_specification")
    if conditions is not None and list(conditions) != list(NOMINAL_CONDITIONS):
        problems.append(
            f"condition specification must be the full seven-condition design "
            f"vocabulary {list(NOMINAL_CONDITIONS)}; found {list(conditions)!r}. "
            "Per-model realized conditions belong in models[...]['condition_set'], "
            "not here (frozen design §22)"
        )

    # --- Common source identities (§19) ----------------------------------
    sources = data.get("sources")
    if isinstance(sources, dict):
        if sources.get("common_source_a") != COMMON_SOURCE_A:
            problems.append(
                f"common_source_a must be {COMMON_SOURCE_A!r}; found "
                f"{sources.get('common_source_a')!r} (frozen design §19)"
            )
        if sources.get("common_source_b") != COMMON_SOURCE_B:
            problems.append(
                f"common_source_b must be {COMMON_SOURCE_B!r}; found "
                f"{sources.get('common_source_b')!r} (frozen design §19)"
            )

    return problems


def cohort_a_provenance(
    result, excluded_phase2_item_ids: frozenset[str] | set[str]
) -> dict[str, Any]:
    """Serialize Cohort A provenance exactly as §36 requires.

    Records the freshness exclusion list supplied at 3C, the realized
    relation distribution, and the dominance diagnostic (which stays
    DIAGNOSTIC and never gates anything, §15.1).
    """
    from conflict_eval.phase3.screening import item_id_of

    selected = [item_id_of(r) for r in result.items]
    return {
        # PLANNED / FROZEN design (§15.1) -- never derived from what was
        # realized, so a reduced realized cohort can never redefine the target.
        **planned_cohort_a_design(),
        # REALIZED outcome after eligibility screening.
        "selected_item_ids": selected,
        "realized_total": len(selected),
        "per_stratum_selected": dict(result.per_stratum_selected),
        "per_stratum_available": dict(result.per_stratum_available),
        "excluded_phase2_item_ids": sorted(str(i) for i in excluded_phase2_item_ids),
        "excluded_phase2_count": result.excluded_phase2_count,
        "realized_relation_distribution": dict(result.relation_distribution),
        "relation_dominance_share": result.relation_dominance_share,
        "relation_dominance_flag": result.relation_dominance_flag,
        "status": result.state,
        "eligibility_limited": result.is_eligibility_limited,
        "shortfall": dict(result.shortfall),
    }


def cohort_b_provenance(result) -> dict[str, Any]:
    """Serialize one Cohort B (model x group) result, including the §32
    realized reduction: which relations qualified, which were excluded as
    short, the realized cell count, and why."""
    from conflict_eval.phase3.screening import item_id_of

    return {
        "model_key": result.model_key,
        "knowledge_group": result.knowledge_group,
        "status": result.state,
        "confirmatory_eligible": result.confirmatory_eligible,
        "target_cell_count": result.target_cell_count,
        "minimum_cell_count": result.minimum_cell_count,
        "original_cell_counts": {
            f"{relation}|{stratum}": count
            for (relation, stratum), count in sorted(result.per_cell_available.items())
        },
        "qualifying_relations": list(result.qualifying_relations),
        "excluded_short_relations": list(result.excluded_short_relations),
        "realized_relations": list(result.realized_relations),
        "realized_cell_count": result.realized_cell_count,
        "deficient_cells": [f"{r}|{s}" for r, s in result.deficient_cells],
        "selected_item_ids": [item_id_of(r) for r in result.items],
        "reduction_reason": result.reduction_reason,
    }


def cohort_c_provenance(result) -> dict[str, Any]:
    """Serialize Cohort C with each model's own knowledge state per shared
    item -- §36 forbids reducing Cohort C to a bare item-id list, because
    KC/KW never transfers across models (§16)."""
    return {
        "status": result.state,
        "candidates_considered": result.candidates_considered,
        "relation_distribution": dict(result.relation_distribution),
        "label_agreement_count": result.label_agreement_count,
        "label_disagreement_count": result.label_disagreement_count,
        "items": [
            {
                "item_id": item.item_id,
                "relation": item.relation,
                "per_model": {
                    model: dict(state) for model, state in sorted(item.per_model.items())
                },
            }
            for item in result.items
        ],
    }


@dataclasses.dataclass
class Phase3Manifest:
    """A Phase 3 provenance manifest under construction."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.data.get("synthetic"))

    @property
    def is_frozen(self) -> bool:
        return bool(self.data.get("frozen"))


def build_manifest(
    *,
    seed: int,
    repository_commit: str | None,
    dataset: dict[str, Any] | None,
    models: dict[str, dict[str, Any]],
    prompt_version: str | None,
    cohorts: dict[str, Any],
    cohort_membership_map: dict[str, list[str]],
    deduplication_alias_map: dict[str, str],
    final_margin_strata: dict[str, Any],
    screening: dict[str, Any],
    nominal_condition_slots: int,
    unique_observations: int,
    artifact_hashes: dict[str, str] | None = None,
    registry: AnalysisRegistry | None = None,
    synthetic: bool = True,
    environment: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
) -> Phase3Manifest:
    """Assemble a Phase 3 manifest.

    `synthetic=True` (the default, and the only correct value in Phase 3B)
    marks the manifest as incapable of authorizing a real run. Unresolved
    Phase 3C values are recorded as `None` rather than filled in.
    """
    registry = registry or default_registry()
    data: dict[str, Any] = {
        "phase": "3",
        "synthetic": bool(synthetic),
        "frozen": False,
        "ready_for_real_run": False,
        "seed": seed,
        "repository_commit": repository_commit,
        "dataset": dataset,
        "models": models,
        "prompt_version": prompt_version,
        "sources": {
            # Fixed source IDENTITIES, not roles (§19).
            "common_source_a": COMMON_SOURCE_A,
            "common_source_b": COMMON_SOURCE_B,
            "model_specific_roles": {
                key: {
                    "preferred_source": entry.get("preferred_source"),
                    "dispreferred_source": entry.get("dispreferred_source"),
                    "provenance": entry.get("source_provenance"),
                }
                for key, entry in models.items()
            },
        },
        "cohorts": cohorts,
        "cohort_membership_map": cohort_membership_map,
        # Study-wide nominal design vocabulary (all seven conditions), NOT a
        # per-model claim -- see the note in `validate_manifest`. Each model's
        # realized subset lives in models[key]["condition_set"].
        "condition_specification": list(NOMINAL_CONDITIONS),
        "condition_specification_note": (
            "Full design vocabulary. ENABLED model-specific arms run "
            f"{list(CONDITIONS_WITH_MODEL_SPECIFIC_ARM)}; arms DISABLED under "
            f"the frozen §34 calibration rule run {list(CONDITIONS_COMMON_ARM_ONLY)} "
            "and generate no M1/M2 observations at all."
        ),
        "deduplication_alias_map": deduplication_alias_map,
        "final_margin_strata": final_margin_strata,
        "screening": screening,
        "compute": {
            # §23: these are different quantities and are recorded separately.
            "nominal_condition_slots": nominal_condition_slots,
            "unique_observations": unique_observations,
            "collapsed_by_deduplication": nominal_condition_slots - unique_observations,
        },
        "analysis_status": [
            {
                "name": e.name,
                "cohort": e.cohort,
                "outcome": e.outcome,
                "contrast": e.contrast,
                "status": e.status,
                "multiplicity_family": e.family,
            }
            for e in registry.entries
        ],
        "artifact_hashes": artifact_hashes or {},
        # Phase 3C fills these; never fabricated here. The `None` values are
        # an UNFILLED TEMPLATE, and `validate_manifest` rejects a frozen
        # manifest that still carries them -- presence is not capture (§36).
        "environment": environment
        or {
            "python": None,
            "torch": None,
            "cuda": None,
            "transformers": None,
            "datasets": None,
            "accelerate": None,
        },
        "hardware": hardware or {"gpu_name": None, "vram": None},
    }
    return Phase3Manifest(data=data)


def validate_manifest(
    manifest: Phase3Manifest, expected_model_keys: Iterable[str] | None = None
) -> list[str]:
    """Return the list of reasons this manifest cannot authorize a real run.

    An empty list means every §36 requirement is satisfied. Callers should
    still go through `real_run_gate.assert_ready_for_real_run`, which also
    checks the config -- and which always supplies `expected_model_keys`
    from that config, so the model-set invariant is never skipped on the
    path that actually authorizes a run.

    `expected_model_keys` is optional only because a manifest can be
    inspected without a config in hand; when it is omitted the model-set
    check cannot be performed and every other rule still applies.
    """
    problems: list[str] = []
    data = manifest.data

    if data.get("synthetic"):
        problems.append(
            "manifest is SYNTHETIC and can never authorize a real Phase 3 run"
        )

    # Frozen PLANNED design constants (§11, §15.1, §15.2, §19, §22). Checked
    # against constants.py, so a manifest cannot redefine the protocol to
    # match whatever it happened to realize.
    problems.extend(_frozen_design_problems(data))

    for field in REQUIRED_FREEZE_FIELDS:
        if data.get(field) in (None, {}, [], ""):
            problems.append(f"required freeze field {field!r} is unresolved (§36)")

    # --- §36 minimum content that key-presence alone cannot guarantee ----
    problems.extend(_artifact_hash_problems(data))
    problems.extend(_dataset_problems(data))
    problems.extend(_populated_capture_problems("environment", data.get("environment")))
    problems.extend(_populated_capture_problems("hardware", data.get("hardware")))
    problems.extend(_model_set_problems(data, expected_model_keys))

    for key, entry in (data.get("models") or {}).items():
        if not entry.get("hf_model_id") or not entry.get("revision"):
            problems.append(
                f"model {key!r} has unresolved hf_model_id/revision (Phase 3C, §7)"
            )
        # A branch/tag pointer moves. Pinning to one would silently change the
        # artifact under a re-run, so the manifest rejects it here as well as
        # the config loader -- a manifest can be hand-assembled, and the
        # freeze artifact is what a replicator actually reads (§35).
        for field in ("revision", "requested_revision", "resolved_revision"):
            value = entry.get(field)
            if isinstance(value, str) and value.strip().lower() in SYMBOLIC_REVISIONS:
                problems.append(
                    f"model {key!r} {field} is the mutable pointer {value!r}; a "
                    "frozen manifest must record an exact immutable commit SHA "
                    "(§35, §36)"
                )
        problems.extend(_model_arm_problems(key, entry))
        problems.extend(_model_screening_provenance_problems(key, entry))

    # A disabled model-specific arm generates no M1/M2, so the manifest must
    # not carry observations for conditions that were never run (§22, §34).
    problems.extend(_disabled_arm_observation_problems(data))

    # --- per-cohort CONTENT validation (§36) ------------------------------
    # Key presence is not enough: a real manifest must actually carry the
    # provenance the frozen design requires, or the freeze is non-compliant.
    cohorts = data.get("cohorts") or {}

    cohort_a = cohorts.get("A")
    if not cohort_a:
        problems.append("Cohort A membership is not finalized (§36)")
    elif not isinstance(cohort_a, dict):
        problems.append(
            "Cohort A must be a provenance mapping, not a bare item-id list (§36)"
        )
    else:
        for field in REQUIRED_COHORT_A_FIELDS:
            if field not in cohort_a or cohort_a[field] is None:
                problems.append(f"Cohort A provenance missing {field!r} (§36, §15.1)")
        if not cohort_a.get("selected_item_ids"):
            problems.append("Cohort A has no selected item ids (§36)")

        # REALIZED vs PLANNED (§15.1). A COMPLETE Cohort A must actually
        # hit the frozen target; a shortfall is permitted ONLY when the
        # manifest says so explicitly and records why.
        status = cohort_a.get("status")
        realized_total = cohort_a.get("realized_total")
        realized_strata = cohort_a.get("per_stratum_selected") or {}
        expected_strata = {s: COHORT_A_PER_STRATUM_TARGET for s in MARGIN_STRATA}
        if status == "COMPLETE":
            if realized_total != COHORT_A_TOTAL_TARGET:
                problems.append(
                    f"Cohort A is marked COMPLETE but realized {realized_total!r} "
                    f"items instead of the frozen target {COHORT_A_TOTAL_TARGET} "
                    "(§15.1). A short cohort must be reported as "
                    "eligibility-limited, not complete."
                )
            if realized_strata != expected_strata:
                problems.append(
                    "Cohort A is marked COMPLETE but its realized strata "
                    f"{realized_strata!r} are not the frozen {expected_strata} "
                    "(§15.1)"
                )
        elif status:
            # Eligibility-limited: the realized cohort may fall short, but
            # only with acquisition provenance proving why (§15.1, §34).
            if not any((cohort_a.get("shortfall") or {}).values()):
                problems.append(
                    f"Cohort A status {status!r} reports no shortfall; an "
                    "eligibility-limited cohort must record which strata fell "
                    "short (§15.1, §34)"
                )
            stopped = (data.get("screening") or {}).get("stopped_reason")
            if stopped not in ("ceiling_reached", "supply_not_met"):
                problems.append(
                    f"Cohort A status {status!r} but screening stopped_reason is "
                    f"{stopped!r}; the frozen protocol permits a short Cohort A "
                    "only as an acquisition failure at the screening ceiling "
                    "(§15.1, §11)"
                )
        # Freshness provenance must be present AND non-empty for a real run:
        # the frozen design excludes the Phase 2 Qwen KW items (§15.1).
        if cohort_a.get("excluded_phase2_item_ids") == []:
            problems.append(
                "Cohort A freshness exclusion list is empty; the frozen design "
                "excludes the Phase 2 Qwen KW items (§15.1). Phase 3C must supply "
                "the real exclusion list."
            )

    cohort_b = cohorts.get("B")
    if cohort_b is None:
        problems.append("Cohort B provenance absent (§36)")
    elif not isinstance(cohort_b, dict):
        problems.append("Cohort B must map model x group -> provenance (§36)")
    else:
        for group_key, group in cohort_b.items():
            if not isinstance(group, dict):
                problems.append(
                    f"Cohort B entry {group_key!r} must be a provenance mapping, "
                    "not a bare item-id list (§36, §32)"
                )
                continue
            for field in REQUIRED_COHORT_B_GROUP_FIELDS:
                if field not in group:
                    problems.append(
                        f"Cohort B {group_key!r} missing {field!r} (§36, §32)"
                    )
            # REALIZED cell count: legitimately 6, 7 or 8 under §32, but
            # never below the frozen minimum and never above the target.
            realized_cell = group.get("realized_cell_count")
            if realized_cell is not None:
                if realized_cell > COHORT_B_CELL_TARGET:
                    problems.append(
                        f"Cohort B {group_key!r} realized_cell_count "
                        f"{realized_cell} exceeds the frozen target "
                        f"{COHORT_B_CELL_TARGET} (§32 rule 2 caps every cell at "
                        "the target)"
                    )
                elif realized_cell < COHORT_B_CELL_MINIMUM and group.get(
                    "qualifying_relations"
                ):
                    problems.append(
                        f"Cohort B {group_key!r} realized_cell_count "
                        f"{realized_cell} is below the frozen minimum "
                        f"{COHORT_B_CELL_MINIMUM} while relations still qualify "
                        "(§32 rule 3/4)"
                    )
            # Confirmatory eligibility follows the frozen 3-relation rule.
            qualifying = group.get("qualifying_relations")
            confirmatory = group.get("confirmatory_eligible")
            if qualifying is not None and confirmatory is not None:
                expected_confirmatory = len(qualifying) >= 3
                if confirmatory != expected_confirmatory:
                    problems.append(
                        f"Cohort B {group_key!r} has {len(qualifying)} qualifying "
                        f"relation(s) but confirmatory_eligible={confirmatory!r}; "
                        "the frozen rule admits a model x group to the "
                        "confirmatory family only with at least three (§32 rule 4)"
                    )
            # A reduction must be explainable: reduced/limited groups need
            # both the original counts and a stated reason (§32 rule 6).
            if group.get("realized_relations") is not None and group.get(
                "excluded_short_relations"
            ):
                if not group.get("original_cell_counts"):
                    problems.append(
                        f"Cohort B {group_key!r} reports excluded relations but no "
                        "original_cell_counts (§32 rule 4)"
                    )
                if not group.get("reduction_reason"):
                    problems.append(
                        f"Cohort B {group_key!r} reports a reduction but no "
                        "reduction_reason (§32 rule 6)"
                    )

    cohort_c = cohorts.get("C")
    if cohort_c is None:
        problems.append("Cohort C provenance absent (§36)")
    elif not isinstance(cohort_c, dict) or "items" not in cohort_c:
        problems.append(
            "Cohort C must carry per-item provenance with per-model knowledge "
            "state, not a bare item-id list (§36, §16)"
        )
    else:
        for entry in cohort_c.get("items") or []:
            for field in REQUIRED_COHORT_C_ITEM_FIELDS:
                if field not in entry:
                    problems.append(f"Cohort C item missing {field!r} (§36)")
            per_model = entry.get("per_model") or {}
            if not per_model:
                problems.append(
                    f"Cohort C item {entry.get('item_id')!r} has no per-model "
                    "knowledge state; KC/KW never transfers across models (§16)"
                )
            for model_key, state in per_model.items():
                for field in REQUIRED_COHORT_C_PER_MODEL_FIELDS:
                    if field not in state or state[field] is None:
                        problems.append(
                            f"Cohort C item {entry.get('item_id')!r} model "
                            f"{model_key!r} missing {field!r} (§36, §16)"
                        )

    compute = data.get("compute") or {}
    nominal = compute.get("nominal_condition_slots")
    unique = compute.get("unique_observations")
    if nominal is not None and unique is not None and unique > nominal:
        problems.append(
            "unique_observations exceeds nominal_condition_slots, which is "
            "impossible under deduplication (§23)"
        )

    return problems


def freeze_manifest(manifest: Phase3Manifest) -> Phase3Manifest:
    """Mark a fully-resolved manifest frozen (a Phase 3C action).

    Refuses while `validate_manifest` reports any problem, so a synthetic or
    incomplete manifest can never be promoted to a freeze.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError(
            "Cannot freeze this Phase 3 manifest; unresolved requirements:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    data = dict(manifest.data)
    data["frozen"] = True
    data["ready_for_real_run"] = True
    return Phase3Manifest(data=data)

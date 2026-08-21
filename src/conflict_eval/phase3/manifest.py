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
from typing import Any

from conflict_eval.phase3.analysis_status import AnalysisRegistry, default_registry
from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_SUPPLY,
    COHORT_A_PER_STRATUM_TARGET,
    COHORT_A_TOTAL_TARGET,
    COHORT_B_CELL_MINIMUM,
    COHORT_B_CELL_TARGET,
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
    MARGIN_STRATA,
    NOMINAL_CONDITIONS,
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
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
    "final_margin_strata",
    "screening",
    "analysis_status",
    "seed",
)

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
    conditions = data.get("condition_specification")
    if conditions is not None and list(conditions) != list(NOMINAL_CONDITIONS):
        problems.append(
            f"condition specification must be {list(NOMINAL_CONDITIONS)}; found "
            f"{list(conditions)!r} (frozen design §22)"
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
        "condition_specification": list(NOMINAL_CONDITIONS),
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
        # Phase 3C fills these; never fabricated here.
        "environment": {
            "python": None,
            "torch": None,
            "cuda": None,
            "transformers": None,
            "datasets": None,
            "accelerate": None,
        },
        "hardware": {"gpu_name": None, "vram": None},
    }
    return Phase3Manifest(data=data)


def validate_manifest(manifest: Phase3Manifest) -> list[str]:
    """Return the list of reasons this manifest cannot authorize a real run.

    An empty list means every §36 requirement is satisfied. Callers should
    still go through `real_run_gate.assert_ready_for_real_run`, which also
    checks the config.
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

    for key, entry in (data.get("models") or {}).items():
        if not entry.get("hf_model_id") or not entry.get("revision"):
            problems.append(
                f"model {key!r} has unresolved hf_model_id/revision (Phase 3C, §7)"
            )

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

"""Phase 3C pre-run freeze construction.

Turns the **verified** outcome-blind baseline screening artifacts returned
from the GPU host into the frozen Phase 3C study state: derived per-model
artifacts, the frozen margin strata, Cohorts A/B/C, the cross-cohort
membership map, the pre-run trial specification, the realized
deduplication map, and the sealed §36 manifest.

Nothing here loads a model, downloads a dataset, or produces a model
answer. The only model outputs it ever reads are the already-completed
baseline screening records, which carry no evidence condition by
construction (`artifact_verification.EVIDENCE_LEAK_FIELDS`).

Three properties are structural rather than conventional:

- **Selection never sees an outcome.** Cohorts are built by the committed
  `cohorts` module from `FinalizedScreening`, whose input `add_block`
  rejects any record carrying a Phase 3 outcome field (§11, §16).
- **Rendering is not running.** The trial specification renders the exact
  prompt text each planned condition *would* send, because prompt identity
  is what §22 deduplicates on. No adapter is constructed and no generation
  is requested, so a trial specification can never become a trial.
- **Derived artifacts never overwrite raw ones.** The returned package is
  opened read-only; every derived file is written under a separate root,
  so a re-derivation can be compared against the raw evidence rather than
  silently replacing it.

Two artifact classes, matching the repository's existing policy
(`configs/frozen/README.md`): large empirical outputs stay outside Git and
are referenced by immutable SHA256, while the small provenance records that
constitute the freeze itself are committed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from conflict_eval.experiment.evidence import render_evidence
from conflict_eval.experiment.prompts import render_experiment_prompt
from conflict_eval.models.base import GenerationConfig
from conflict_eval.phase3.analysis_status import (
    default_registry,
    mark_not_applicable,
)
from conflict_eval.phase3.artifact_verification import combine_blocks
from conflict_eval.phase3.cohorts import (
    build_cohort_a,
    build_cohort_b_group,
    build_cohort_c,
)
from conflict_eval.phase3.conditions import build_phase3_conditions
from conflict_eval.phase3.constants import MARGIN_STRATA
from conflict_eval.phase3.dedup import (
    ConditionRequest,
    GenerationIdentity,
    deduplicate_requests,
)
from conflict_eval.phase3.manifest import (
    build_manifest,
    cohort_a_provenance,
    cohort_b_provenance,
    cohort_c_provenance,
    freeze_manifest,
    model_arm_provenance,
    planned_screening_design,
    validate_manifest,
)
from conflict_eval.phase3.runtime_capture import sha256_file, sha256_text
from conflict_eval.phase3.screening import ScreeningState, item_id_of

KNOWLEDGE_GROUPS = ("KC", "KW")

#: Written by `write_json`/`write_jsonl` and hashed from disk, so a recorded
#: digest always describes the bytes that actually exist.
_JSON_KWARGS: dict[str, Any] = {"indent": 2, "sort_keys": True, "ensure_ascii": False}


class FreezeBuildError(RuntimeError):
    """Raised when the returned artifacts cannot support a lawful freeze.

    Always carries one of the allowed terminal blocker labels, because a
    Phase 3C failure is a finding to report, never a rule to relax.
    """


# ---------------------------------------------------------------------------
# Deterministic artifact writing
# ---------------------------------------------------------------------------


def write_json(path: str | Path, payload: Any) -> tuple[Path, str]:
    """Write canonical JSON and return `(path, sha256_of_file)`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `newline="\n"` is load-bearing, not cosmetic: without it Python
    # translates to CRLF on Windows, so the same freeze derived on two
    # platforms would produce different digests for identical content --
    # and a digest that depends on the host is not provenance (§36).
    path.write_text(
        json.dumps(payload, **_JSON_KWARGS) + "\n", encoding="utf-8", newline="\n"
    )
    return path, sha256_file(path)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> tuple[Path, str]:
    """Write one canonical JSON object per line and return `(path, sha256)`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    path.write_text(body, encoding="utf-8", newline="\n")
    return path, sha256_file(path)


# ---------------------------------------------------------------------------
# Part XVII -- derived baseline artifacts
# ---------------------------------------------------------------------------


def replay_screening(
    return_root: str | Path,
    model_key: str,
    *,
    phase2_excluded_ids: frozenset[str] | set[str] | None = None,
):
    """Rebuild `FinalizedScreening` from the returned blocks, in block order.

    The blocks are replayed through the same `ScreeningState` the design
    specifies rather than concatenated, so the frozen block size, the
    ceiling, and the outcome-blindness rejection all apply to the real
    artifacts exactly as they applied on the GPU host. A returned block that
    violated any of them would raise here rather than quietly become a
    cohort.
    """
    root = Path(return_root)
    block_dir = root / model_key / "blocks"
    paths = sorted(block_dir.glob("block_*.jsonl"))
    if not paths:
        raise FreezeBuildError(
            f"EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE: no returned blocks for "
            f"{model_key!r} under {block_dir}"
        )
    state = ScreeningState(model_key, phase2_excluded_ids=phase2_excluded_ids)
    for path in paths:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        state.add_block(rows)
    return state.finalize()


def _eligible(record: dict[str, Any]) -> bool:
    return record.get("knowledge_group") in KNOWLEDGE_GROUPS and bool(
        record.get("primary_conflict_eligible")
    )


def derive_model_artifacts(
    finalized, out_dir: str | Path, *, raw_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write and hash one model's derived Phase 3C artifacts (§36, Part XVII).

    Every file is a deterministic function of the verified raw blocks; none
    of them is edited by hand, and the raw blocks are never modified.
    """
    model_key = finalized.model_key
    out = Path(out_dir) / model_key
    records = list(finalized.records)

    baseline_path, baseline_sha = write_jsonl(out / "baseline_records.jsonl", records)

    eligibility_rows = [
        {
            "item_id": item_id_of(r),
            "relation": r.get("relation"),
            "knowledge_group": r.get("knowledge_group"),
            "baseline_correct": r.get("baseline_correct"),
            "primary_conflict_eligible": r.get("primary_conflict_eligible"),
            "conflict_eligibility_reason": r.get("conflict_eligibility_reason"),
            "exclusion_reason": r.get("exclusion_reason"),
            "manual_review": bool(r.get("manual_review")),
            "eligible": _eligible(r),
        }
        for r in records
    ]
    eligibility_path, eligibility_sha = write_jsonl(
        out / "eligibility_records.jsonl", eligibility_rows
    )

    # The per-model EXCLUSION file (§36): every screened item that did not
    # enter the eligible KC/KW pool, with the reason it did not. Distinct
    # from Cohort A's §15.1 freshness exclusion, which is Qwen-only and
    # recorded on the cohort.
    excluded_rows = [
        {
            "item_id": item_id_of(r),
            "knowledge_group": r.get("knowledge_group"),
            "reason": (
                r.get("exclusion_reason")
                or ("manual_review" if r.get("knowledge_group") == "manual_review" else None)
                or (
                    r.get("conflict_eligibility_reason")
                    if r.get("primary_conflict_eligible") is False
                    else None
                )
                or "not_primary_conflict_eligible"
            ),
        }
        for r in records
        if not _eligible(r)
    ]
    exclusion_path, exclusion_sha = write_json(
        out / "screening_exclusions.json",
        {
            "model_key": model_key,
            "screened_total": len(records),
            "excluded_total": len(excluded_rows),
            "eligible_total": len(records) - len(excluded_rows),
            "excluded": excluded_rows,
        },
    )

    membership = {
        group: sorted(
            item_id_of(r)
            for r in records
            if _eligible(r) and r.get("knowledge_group") == group
        )
        for group in KNOWLEDGE_GROUPS
    }
    membership_path, membership_sha = write_json(
        out / "knowledge_membership.json",
        {"model_key": model_key, **{g: membership[g] for g in KNOWLEDGE_GROUPS}},
    )

    margins = {
        item_id_of(r): {
            "parametric_margin": r.get("parametric_margin"),
            "knowledge_group": r.get("knowledge_group"),
            "relation": r.get("relation"),
            "margin_stratum": finalized.stratum_of(item_id_of(r)),
        }
        for r in records
        if _eligible(r)
    }
    margins_path, margins_sha = write_json(
        out / "margins.json", {"model_key": model_key, "margins": margins}
    )

    manual_rows = [
        {
            "item_id": item_id_of(r),
            "knowledge_group": r.get("knowledge_group"),
            "relation": r.get("relation"),
            "reason": r.get("conflict_eligibility_reason") or r.get("exclusion_reason"),
            # Recorded as an explicit empty decision set: the frozen screen
            # flags items FOR review and never adjudicates them itself, so
            # no override has been made (§36 requires the record either way).
            "decision": None,
        }
        for r in records
        if r.get("manual_review")
    ]
    manual_path, manual_sha = write_json(
        out / "manual_review.json",
        {
            "model_key": model_key,
            "flagged_total": len(manual_rows),
            "decisions_made": 0,
            "flagged": manual_rows,
        },
    )

    relation_counts: dict[str, dict[str, int]] = {}
    for r in records:
        relation = r.get("relation")
        group = r.get("knowledge_group")
        bucket = relation_counts.setdefault(str(relation), dict.fromkeys(
            (*KNOWLEDGE_GROUPS, "excluded", "manual_review", "eligible"), 0
        ))
        if group in bucket:
            bucket[group] += 1
        if _eligible(r):
            bucket["eligible"] += 1

    summary = {
        "model_key": model_key,
        "model_id": records[0]["model_id"] if records else None,
        "requested_revision": records[0].get("requested_revision") if records else None,
        "resolved_revision": records[0].get("model_revision") if records else None,
        "blocks_screened": finalized.blocks_screened,
        "screened_total": finalized.screened_total,
        "raw_returned_records": len(raw_records),
        "stopped_reason": finalized.stopped_reason,
        "cohort_a_supply_met": finalized.cohort_a_supply_met,
        "cohort_b_supply_met": finalized.cohort_b_supply_met,
        "cohort_a_per_stratum_supply": dict(
            finalized.final_supply.cohort_a_per_stratum
        ),
        "eligible_total": sum(len(v) for v in membership.values()),
        "knowledge_counts": {g: len(membership[g]) for g in KNOWLEDGE_GROUPS},
        "excluded_total": len(excluded_rows),
        "manual_review_flagged": len(manual_rows),
        "relation_counts": relation_counts,
        "stratum_edges": {
            f"{key[0]}|{key[1]}": list(edges)
            for key, edges in sorted(finalized.stratum_edges.items())
        },
    }
    summary_path, summary_sha = write_json(out / "screening_summary.json", summary)

    return {
        "model_key": model_key,
        "summary": summary,
        "artifacts": {
            "baseline_records": {"path": str(baseline_path), "sha256": baseline_sha},
            "eligibility_records": {
                "path": str(eligibility_path),
                "sha256": eligibility_sha,
            },
            "screening_exclusions": {
                "path": str(exclusion_path),
                "sha256": exclusion_sha,
            },
            "knowledge_membership": {
                "path": str(membership_path),
                "sha256": membership_sha,
            },
            "margins": {"path": str(margins_path), "sha256": margins_sha},
            "manual_review": {"path": str(manual_path), "sha256": manual_sha},
            "screening_summary": {"path": str(summary_path), "sha256": summary_sha},
        },
        "knowledge_membership": membership,
        "margins": margins,
        "manual_review_decisions": [],
        "manual_review_flagged": manual_rows,
    }


# ---------------------------------------------------------------------------
# Part XIX -- frozen margin boundaries
# ---------------------------------------------------------------------------


def frozen_margin_strata(finalized_by_model: dict[str, Any]) -> dict[str, Any]:
    """The frozen LOW/MEDIUM/HIGH boundaries per (model x knowledge group).

    Taken from `FinalizedScreening`, which computed them once over the
    complete screened eligible pool and froze them. They are recorded here
    and never recomputed downstream -- no later step rebuckets an item.
    """
    strata: dict[str, Any] = {}
    for model_key in sorted(finalized_by_model):
        finalized = finalized_by_model[model_key]
        by_group: dict[str, list[str]] = {group: [] for group in KNOWLEDGE_GROUPS}
        for record in finalized.records:
            if _eligible(record):
                by_group[record["knowledge_group"]].append(item_id_of(record))
        for (key_model, group), edges in sorted(finalized.stratum_edges.items()):
            assigned = by_group[group]
            strata[f"{key_model}|{group}"] = {
                "model_key": key_model,
                "knowledge_group": group,
                "strata": list(MARGIN_STRATA),
                "edges": list(edges),
                "assigned_items": len(assigned),
                "per_stratum_counts": {
                    stratum: sum(
                        1 for item in assigned if finalized.stratum_of(item) == stratum
                    )
                    for stratum in MARGIN_STRATA
                },
                "method": (
                    "empirical tertiles of parametric_margin over the complete "
                    "screened eligible pool within model x knowledge group "
                    "(§11 step 1, §14); frozen before any trial was specified"
                ),
            }
    return strata


# ---------------------------------------------------------------------------
# Parts XX-XXIII -- cohorts and membership
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CohortBundle:
    cohort_a: Any
    cohort_b: dict[str, Any]
    cohort_c: Any


def build_cohorts(
    finalized_by_model: dict[str, Any],
    seed: int,
    *,
    phase2_excluded_ids: frozenset[str],
    cohort_c_target: int,
) -> CohortBundle:
    """Build Cohorts A, B and C from the frozen screening (§15, §16, §32)."""
    if "qwen" not in finalized_by_model:
        raise FreezeBuildError(
            "EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE: Cohort A requires Qwen "
            "screening artifacts, which are absent"
        )
    cohort_a = build_cohort_a(
        finalized_by_model["qwen"], seed, phase2_excluded_ids=phase2_excluded_ids
    )
    cohort_b: dict[str, Any] = {}
    for model_key in sorted(finalized_by_model):
        for group in KNOWLEDGE_GROUPS:
            cohort_b[f"{model_key}|{group}"] = build_cohort_b_group(
                finalized_by_model[model_key], group, seed
            )
    cohort_c = build_cohort_c(finalized_by_model, seed, cohort_c_target)
    return CohortBundle(cohort_a=cohort_a, cohort_b=cohort_b, cohort_c=cohort_c)


def cross_cohort_membership(bundle: CohortBundle) -> dict[str, list[str]]:
    """`"<model>|<item_id>" -> sorted cohort labels` (Part XXIII, §36).

    Keyed by model AND item because the same question can be selected for
    several models with different knowledge states, and because a single
    item reused across cohorts for one model is exactly the reuse the map
    has to make explicit (§22, §16).
    """
    membership: dict[str, set[str]] = {}

    def add(model_key: str, item_id: str, label: str) -> None:
        membership.setdefault(f"{model_key}|{item_id}", set()).add(label)

    for record in bundle.cohort_a.items:
        add("qwen", item_id_of(record), "A")
    for group_key, result in bundle.cohort_b.items():
        model_key, group = group_key.split("|")
        for record in result.items:
            add(model_key, item_id_of(record), f"B:{group}")
    for item in bundle.cohort_c.items:
        for model_key in item.per_model:
            add(model_key, item.item_id, "C")

    return {key: sorted(labels) for key, labels in sorted(membership.items())}


# ---------------------------------------------------------------------------
# Part XXIV -- pre-run trial specification (no execution)
# ---------------------------------------------------------------------------


def _planned_items(
    membership: dict[str, list[str]], model_key: str
) -> list[str]:
    prefix = f"{model_key}|"
    return sorted(
        key[len(prefix):] for key in membership if key.startswith(prefix)
    )


def build_trial_specification(
    finalized_by_model: dict[str, Any],
    membership: dict[str, list[str]],
    config,
    *,
    baseline_template: str,
    evidence_template: str,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], dict[str, list[ConditionRequest]]]:
    """Enumerate every planned condition, with its exact rendered prompt.

    Rendering a prompt is not running one: no adapter is created and no
    generation is requested anywhere in this function. The rendered text is
    required because §22 deduplicates on exact prompt identity, and the
    freeze has to record the realized dedup map before the run, not
    discover it afterwards.
    """
    rows: list[dict[str, Any]] = []
    requests_by_model: dict[str, list[ConditionRequest]] = {}

    for model_key in sorted(finalized_by_model):
        entry = config.model(model_key)
        finalized = finalized_by_model[model_key]
        by_id = {item_id_of(r): r for r in finalized.eligible_records()}
        requests: list[ConditionRequest] = []

        for item_id in _planned_items(membership, model_key):
            record = by_id.get(item_id)
            if record is None:
                raise FreezeBuildError(
                    f"EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE: item {item_id!r} was "
                    f"selected into a cohort for {model_key!r} but is not in that "
                    "model's eligible screened pool"
                )
            group = record["knowledge_group"]
            specs = build_phase3_conditions(
                knowledge_group=group,
                gold_answer=record["gold_answer"],
                baseline_answer=record["memory_answer"],
                foil_answer=record.get("foil_answer"),
                model_preferred_source=entry.preferred_source,
                model_dispreferred_source=entry.dispreferred_source,
                model_specific_arm_enabled=entry.runs_model_specific_arm,
            )
            cohorts = tuple(membership[f"{model_key}|{item_id}"])
            for spec in specs:
                if spec.source_label is None or spec.asserted_answer is None:
                    evidence_text = None
                else:
                    evidence_text = render_evidence(
                        evidence_template,
                        spec.source_label,
                        record["question"],
                        spec.asserted_answer,
                    )
                rendered = render_experiment_prompt(
                    baseline_template, record["question"], evidence_text
                )
                requests.append(
                    ConditionRequest(
                        model_key=model_key,
                        item_id=item_id,
                        condition=spec.condition,
                        arm=spec.arm,
                        source_role=spec.source_role,
                        source_label=spec.source_label,
                        evidence_truth=spec.evidence_truth,
                        conflict_status=spec.conflict_status,
                        knowledge_group=group,
                        rendered_prompt=rendered,
                        cohorts=cohorts,
                    )
                )
                rows.append(
                    {
                        "model_key": model_key,
                        "model_id": entry.hf_model_id,
                        "model_revision": entry.revision,
                        "item_id": item_id,
                        "relation": record.get("relation"),
                        "knowledge_group": group,
                        "margin_stratum": finalized.stratum_of(item_id),
                        "cohorts": list(cohorts),
                        "condition": spec.condition,
                        "arm": spec.arm,
                        "source_role": spec.source_role,
                        "source_label": spec.source_label,
                        "asserted_answer": spec.asserted_answer,
                        "evidence_truth": spec.evidence_truth,
                        "conflict_status": spec.conflict_status,
                        "prompt_version": prompt_version,
                        "rendered_prompt_sha256": sha256_text(rendered),
                    }
                )
        requests_by_model[model_key] = requests

    rows.sort(key=lambda r: (r["model_key"], r["item_id"], r["condition"]))
    return rows, requests_by_model


# ---------------------------------------------------------------------------
# Part XXV -- realized deduplication map
# ---------------------------------------------------------------------------


def build_dedup_map(
    requests_by_model: dict[str, list[ConditionRequest]],
    config,
    *,
    prompt_version: str,
    gen_config: GenerationConfig,
) -> dict[str, Any]:
    """Apply the §22 canonical-generation identity, per model.

    Deduplication is run **per model**, with that model's own exact
    revision in the identity, so two different models can never collapse
    into one observation even if their prompts are textually identical --
    they are different model artifacts and therefore different generations.
    """
    alias_map: dict[str, str] = {}
    per_model: dict[str, Any] = {}
    observations: list[dict[str, Any]] = []
    nominal_total = 0
    unique_total = 0

    for model_key in sorted(requests_by_model):
        entry = config.model(model_key)
        identity = GenerationIdentity(
            model_key=model_key,
            model_revision=entry.revision,
            prompt_version=prompt_version,
            do_sample=gen_config.do_sample,
            num_beams=gen_config.num_beams,
            max_new_tokens=gen_config.max_new_tokens,
            seed=config.seed,
        )
        result = deduplicate_requests(
            requests_by_model[model_key], config.seed, identity
        )
        nominal_total += result.nominal_slots
        unique_total += result.unique_observations
        for (m_key, item_id, condition), obs_id in sorted(result.alias_map.items()):
            alias_map[f"{m_key}|{item_id}|{condition}"] = obs_id
        for obs in result.observations:
            observations.append(
                {
                    "observation_id": obs.observation_id,
                    "model_key": obs.model_key,
                    "model_revision": obs.model_revision,
                    "item_id": obs.item_id,
                    "prompt_hash": obs.prompt_hash,
                    "prompt_version": obs.prompt_version,
                    "generation_settings_fingerprint": (
                        obs.generation_settings_fingerprint
                    ),
                    "aliased_conditions": [
                        dict(a) for a in obs.aliased_conditions
                    ],
                    "is_aliased": obs.is_aliased,
                    "cohorts": sorted(obs.cohorts),
                }
            )
        per_model[model_key] = {
            "condition_set": list(entry.condition_set),
            "items": len({r.item_id for r in requests_by_model[model_key]}),
            "nominal_condition_slots": result.nominal_slots,
            "unique_planned_generations": result.unique_observations,
            "collapsed_by_deduplication": result.collapsed_slots,
            "aliased_observations": sum(
                1 for o in result.observations if o.is_aliased
            ),
        }

    return {
        "alias_map": alias_map,
        "observations": observations,
        "per_model": per_model,
        "totals": {
            "nominal_condition_slots": nominal_total,
            "unique_planned_generations": unique_total,
            "collapsed_by_deduplication": nominal_total - unique_total,
        },
    }


# ---------------------------------------------------------------------------
# Part XXVI -- analysis-status realization
# ---------------------------------------------------------------------------


def realize_analysis_status(config, bundle: CohortBundle) -> dict[str, Any]:
    """Finalize applicability from arm availability and cohort eligibility.

    Deliberately stops short of a Holm family. §28/§44 membership also
    depends on §30 saturation, which cannot be evaluated without evidence
    outcomes that do not and must not exist yet -- so the realized family
    is recorded as NOT YET DETERMINED rather than invented.
    """
    disabled = config.common_arm_only_models()
    registry = mark_not_applicable(default_registry(), disabled)

    non_confirmatory = sorted(
        key for key, result in bundle.cohort_b.items() if not result.confirmatory_eligible
    )
    return {
        "registry": registry,
        "provenance": {
            "model_specific_arm_disabled": sorted(disabled),
            "not_applicable_analyses": [
                {
                    "name": e.name,
                    "model_key": e.model_key,
                    "cohort": e.cohort,
                    "reason": e.not_applicable_reason,
                }
                for e in registry.entries
                if e.status == "NOT APPLICABLE"
            ],
            "cohort_b_confirmatory_eligible": {
                key: result.confirmatory_eligible
                for key, result in sorted(bundle.cohort_b.items())
            },
            "cohort_b_outside_confirmatory_family": non_confirmatory,
            "primary_family": (
                [registry.primary().name] if registry.primary() else []
            ),
            "declared_secondary_family": list(registry.secondary_family()),
            "realized_secondary_family": None,
            "realized_secondary_family_note": (
                "NOT YET DETERMINED. Realized Holm membership additionally "
                "depends on the §30 saturation diagnostic, which is a function "
                "of evidence-condition adoption rates. No Phase 3 evidence "
                "condition has been generated, so the realized family is "
                "resolved in Phase 3E and is deliberately not computed, "
                "asserted, or hard-coded here (§28, §30, §44)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# §36 manifest assembly
# ---------------------------------------------------------------------------


def assemble_manifest(
    *,
    config,
    config_sha256: str,
    repository_commit: str,
    candidate_ids: list[str],
    candidate_file_sha256: str,
    trial_file_sha256: str,
    prompt_version: str,
    finalized_by_model: dict[str, Any],
    derived_by_model: dict[str, Any],
    bundle: CohortBundle,
    membership: dict[str, list[str]],
    dedup: dict[str, Any],
    analysis: dict[str, Any],
    phase2_excluded_ids: frozenset[str],
    phase2_exclusion_sha256: str,
    environment: dict[str, Any],
    hardware: dict[str, Any],
    device_map: str,
    max_memory: Any,
    screening_extras: dict[str, Any],
):
    """Build the fully-populated §36 pre-run manifest (Part XXVII).

    Every value is read from a verified artifact or from `constants.py`.
    Nothing is defaulted, guessed, or stubbed: a field that cannot be filled
    from real provenance is left for `validate_manifest` to reject.
    """
    models: dict[str, dict[str, Any]] = {}
    for model_key in sorted(config.models):
        entry = config.model(model_key)
        derived = derived_by_model[model_key]
        record = model_arm_provenance(entry)
        record.update(
            {
                # `validate_manifest` also checks a plain `revision`; the
                # requested/resolved pair is recorded alongside it and the
                # returned artifacts proved all three identical.
                "revision": entry.revision,
                "resolved_revision": derived["summary"]["resolved_revision"],
                "device_map": device_map,
                "max_memory": max_memory,
                "baseline_file_sha256": derived["artifacts"]["baseline_records"][
                    "sha256"
                ],
                "exclusion_file_sha256": derived["artifacts"]["screening_exclusions"][
                    "sha256"
                ],
                "knowledge_membership": derived["knowledge_membership"],
                "margins": derived["margins"],
                "manual_review_decisions": derived["manual_review_decisions"],
                "manual_review_flagged": derived["manual_review_flagged"],
                "derived_artifact_sha256": {
                    name: meta["sha256"]
                    for name, meta in sorted(derived["artifacts"].items())
                },
                "screened_total": derived["summary"]["screened_total"],
                "blocks_screened": derived["summary"]["blocks_screened"],
                "stopped_reason": derived["summary"]["stopped_reason"],
            }
        )
        if model_key == "qwen":
            record["phase2_freshness_exclusion_sha256"] = phase2_exclusion_sha256
        models[model_key] = record

    cohorts = {
        "A": cohort_a_provenance(bundle.cohort_a, phase2_excluded_ids),
        "B": {
            key: cohort_b_provenance(result)
            for key, result in sorted(bundle.cohort_b.items())
        },
        "C": cohort_c_provenance(bundle.cohort_c),
    }

    screening = {
        **planned_screening_design(),
        **screening_extras,
        "per_model": {
            key: {
                "blocks_screened": derived_by_model[key]["summary"]["blocks_screened"],
                "screened_total": derived_by_model[key]["summary"]["screened_total"],
                "stopped_reason": derived_by_model[key]["summary"]["stopped_reason"],
                "eligible_total": derived_by_model[key]["summary"]["eligible_total"],
                "knowledge_counts": derived_by_model[key]["summary"][
                    "knowledge_counts"
                ],
                "cohort_a_supply_met": derived_by_model[key]["summary"][
                    "cohort_a_supply_met"
                ],
                "cohort_b_supply_met": derived_by_model[key]["summary"][
                    "cohort_b_supply_met"
                ],
            }
            for key in sorted(derived_by_model)
        },
    }

    manifest = build_manifest(
        seed=config.seed,
        repository_commit=repository_commit,
        dataset={
            **{k: v for k, v in config.dataset.items()},
            "candidate_item_ids": candidate_ids,
            "candidate_item_count": len(candidate_ids),
        },
        models=models,
        prompt_version=prompt_version,
        cohorts=cohorts,
        cohort_membership_map=membership,
        deduplication_alias_map=dedup["alias_map"],
        final_margin_strata=frozen_margin_strata(finalized_by_model),
        screening=screening,
        nominal_condition_slots=dedup["totals"]["nominal_condition_slots"],
        unique_observations=dedup["totals"]["unique_planned_generations"],
        artifact_hashes={
            "phase3_config": config_sha256,
            "candidate_file": candidate_file_sha256,
            "trial_file": trial_file_sha256,
        },
        registry=analysis["registry"],
        synthetic=False,
        environment=environment,
        hardware=hardware,
    )
    manifest.data["deduplication_provenance"] = {
        "per_model": dedup["per_model"],
        "totals": dedup["totals"],
        "observation_count": len(dedup["observations"]),
        # The full per-observation records stay outside Git under the
        # repository's artifact policy; this digest makes the reference
        # immutable (configs/frozen/README.md, §36).
        "observations_file": dedup.get("observations_file"),
        "observations_file_sha256": dedup.get("observations_file_sha256"),
        "identity_rule": (
            "model_key + exact model revision + item_id + exact rendered-prompt "
            "SHA256 + prompt version + generation-settings fingerprint (§22). "
            "Different item ids never collapse on identical text, and different "
            "models never deduplicate against each other."
        ),
    }
    manifest.data["analysis_status_realization"] = analysis["provenance"]
    return manifest


__all__ = [
    "CohortBundle",
    "FreezeBuildError",
    "assemble_manifest",
    "build_cohorts",
    "build_dedup_map",
    "build_trial_specification",
    "combine_blocks",
    "cross_cohort_membership",
    "derive_model_artifacts",
    "freeze_manifest",
    "frozen_margin_strata",
    "realize_analysis_status",
    "replay_screening",
    "validate_manifest",
    "write_json",
    "write_jsonl",
]

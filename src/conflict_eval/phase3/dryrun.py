"""Phase 3 SYNTHETIC dry-run pipeline.

Exercises the whole Phase 3 chain without a real model:

    synthetic baseline records
      -> blockwise screening state
      -> margin-stratum freeze
      -> Cohort A / B / C construction
      -> seven nominal condition requests
      -> prompt rendering
      -> prompt-identical deduplication
      -> dummy generations (DummyModelAdapter)
      -> canonical observation reuse
      -> synthetic classification
      -> paired summary (frozen procedures)
      -> synthetic Phase 3 manifest

`DummyModelAdapter` never loads a real model and never makes a network
call (`models/dummy.py`), so this path satisfies the Phase 3B rule that no
real model may run. Every output produced here is stamped synthetic, and
`real_run_gate` refuses any manifest carrying that stamp.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from conflict_eval.experiment.evidence import render_evidence
from conflict_eval.experiment.prompts import render_experiment_prompt
from conflict_eval.models.base import GenerationConfig
from conflict_eval.models.dummy import DummyModelAdapter
from conflict_eval.phase3 import synthetic
from conflict_eval.phase3.cohorts import (
    build_cohort_a,
    build_cohort_b_group,
    build_cohort_c,
)
from conflict_eval.phase3.conditions import build_phase3_conditions
from conflict_eval.phase3.constants import (
    DRYRUN_PREFIX,
    FROZEN_MODEL_SOURCE_PAIRS,
    SCREENING_BLOCK_SIZE,
)
from conflict_eval.phase3.dedup import (
    ConditionRequest,
    GenerationIdentity,
    collect_paired_outcomes,
    deduplicate_requests,
)
from conflict_eval.phase3.manifest import (
    build_manifest,
    cohort_a_provenance,
    cohort_b_provenance,
    cohort_c_provenance,
    planned_screening_design,
)
from conflict_eval.phase3.paired_stats import paired_source_result
from conflict_eval.phase3.screening import ScreeningState, item_id_of

SYNTHETIC_BANNER = "*** SYNTHETIC/DRY-RUN OUTPUT - NOT A PHASE 3 RESULT ***"

# Minimal stand-in templates so the dry run does not depend on reading the
# committed prompt files; the real run uses prompts/ unchanged.
_DRYRUN_BASELINE_TEMPLATE = (
    "Evidence:\n{evidence_or_none}\n\nQuestion:\n{question}\n\n"
    "Answer: <short answer>\nDecision: <answer or uncertain>\nConfidence: <0-100>"
)
_DRYRUN_EVIDENCE_TEMPLATE = (
    'Source: {source}\n\nStatement:\nThe answer to the question "{question}" '
    'is "{asserted_answer}".'
)


@dataclasses.dataclass
class DryRunReport:
    banner: str
    screening_stopped_reason: str
    screened_total: int
    cohort_a_state: str
    cohort_a_size: int
    cohort_a_relations: dict[str, int]
    cohort_b_states: dict[str, str]
    cohort_c_size: int
    nominal_slots: int
    unique_observations: int
    collapsed_slots: int
    paired_summary: dict
    manifest: dict


def _render(question: str, source: str | None, asserted: str | None) -> str:
    evidence = (
        render_evidence(_DRYRUN_EVIDENCE_TEMPLATE, source, question, asserted)
        if source is not None and asserted is not None
        else None
    )
    return render_experiment_prompt(_DRYRUN_BASELINE_TEMPLATE, question, evidence)


def run_synthetic_dryrun(
    seed: int = 42,
    model_key: str = "qwen",
    n_per_cell: int = 36,
    cohort_a_per_stratum: int = 6,
    output_dir: str | Path | None = None,
) -> DryRunReport:
    """Run the full synthetic Phase 3 chain.

    The synthetic pool and per-stratum target are scaled down from the
    frozen 32/32/32 and 8-per-cell targets so the dry run stays fast. The
    *logic* exercised is identical and the frozen constants in
    `constants.py` are untouched -- scaling a synthetic rehearsal is not a
    design change, because no scientific quantity is produced here. Defaults
    are chosen so the dry run demonstrates the SUCCESS path (Cohort A and
    both Cohort B groups complete); tests drive the failure paths directly.
    """
    pairs = FROZEN_MODEL_SOURCE_PAIRS[model_key]

    # --- 1. synthetic baseline records, fed blockwise ---------------------
    pool = synthetic.make_baseline_pool(n_per_cell=n_per_cell)
    state = ScreeningState(model_key, phase2_excluded_ids=set(), require_cohort_a=False)
    for start in range(0, len(pool), SCREENING_BLOCK_SIZE):
        block = pool[start : start + SCREENING_BLOCK_SIZE]
        state.add_block(block)
        if state.should_stop():
            break
    finalized = state.finalize()

    # --- 2. cohorts -------------------------------------------------------
    # Explicit SYNTHETIC exclusion set. The argument is required (no
    # default), so a dry run cannot silently skip freshness; the real frozen
    # Phase 2 Qwen KW ids are supplied only at Phase 3C and are never
    # invented here.
    excluded_phase2_ids = {
        item_id_of(r)
        for r in finalized.eligible_records()
        if r.get("knowledge_group") == "KW"
    }
    excluded_phase2_ids = set(sorted(excluded_phase2_ids)[:2])
    cohort_a = build_cohort_a(
        finalized,
        seed=seed,
        phase2_excluded_ids=excluded_phase2_ids,
        per_stratum_target=cohort_a_per_stratum,
    )
    cohort_b = {
        group: build_cohort_b_group(finalized, group, seed=seed)
        for group in ("KC", "KW")
    }
    cohort_c = build_cohort_c({model_key: finalized}, seed=seed, target_size=8)

    # --- 3. seven conditions, with cohort membership ---------------------
    membership: dict[str, set[str]] = {}
    for record in cohort_a.items:
        membership.setdefault(item_id_of(record), set()).add("A")
    for group_result in cohort_b.values():
        for record in group_result.items:
            membership.setdefault(item_id_of(record), set()).add("B")
    for shared in cohort_c.items:
        membership.setdefault(shared.item_id, set()).add("C")

    by_id = {item_id_of(r): r for r in finalized.eligible_records()}
    requests: list[ConditionRequest] = []
    for item_id, cohorts in sorted(membership.items()):
        record = by_id[item_id]
        specs = build_phase3_conditions(
            knowledge_group=record["knowledge_group"],
            gold_answer=record["gold_answer"],
            baseline_answer=record["baseline_answer"],
            foil_answer=record["foil_answer"],
            model_preferred_source=pairs["preferred_source"],
            model_dispreferred_source=pairs["dispreferred_source"],
            model_specific_arm_enabled=True,
        )
        for spec in specs:
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
                    knowledge_group=record["knowledge_group"],
                    rendered_prompt=_render(
                        record["question"], spec.source_label, spec.asserted_answer
                    ),
                    cohorts=tuple(sorted(cohorts)),
                )
            )

    # --- 4. deduplication -------------------------------------------------
    generation_config = GenerationConfig(do_sample=False, max_new_tokens=32, num_beams=1)
    # SYNTHETIC generation identity: an obviously fake revision, plus the
    # frozen deterministic generation settings. A real run supplies the
    # resolved revision; the gate blocks until it does.
    identity = GenerationIdentity(
        model_key=model_key,
        model_revision="synthetic-dryrun-revision",
        prompt_version="phase3-dryrun-v1",
        do_sample=generation_config.do_sample,
        num_beams=generation_config.num_beams,
        max_new_tokens=generation_config.max_new_tokens,
        seed=seed,
    )
    dedup = deduplicate_requests(requests, seed=seed, identity=identity)

    # --- 5. dummy generations, once per canonical observation ------------
    adapter = DummyModelAdapter(seed=seed, context_adoption_rate=0.6)
    outcomes: dict[str, bool] = {}
    for observation in dedup.observations:
        raw = adapter.generate(
            [{"role": "user", "content": observation.rendered_prompt}],
            generation_config,
        )
        # SYNTHETIC classification: the dummy's committed answer is compared
        # against the asserted context answer only to exercise the pipeline.
        committed = "Decision: answer" in raw
        outcomes[observation.observation_id] = committed and synthetic.synthetic_outcome(
            observation.observation_id, adoption_rate=0.6
        )

    # --- 6. paired summary on Cohort A KW items (M1 vs M2) ---------------
    cohort_a_ids = [item_id_of(r) for r in cohort_a.items]
    paired = collect_paired_outcomes(
        dedup, outcomes, model_key, cohort_a_ids, "M1", "M2"
    )
    summary = paired_source_result(paired) if paired else None

    # --- 7. synthetic manifest -------------------------------------------
    manifest = build_manifest(
        seed=seed,
        repository_commit=None,
        dataset={"hf_dataset_id": "akariasai/PopQA", "revision": None},
        models={
            model_key: {
                "hf_model_id": None,
                "revision": None,
                "preferred_source": pairs["preferred_source"],
                "dispreferred_source": pairs["dispreferred_source"],
                "source_provenance": "frozen Phase 2 pair (§20.1)",
            }
        },
        prompt_version="phase3-dryrun-v1",
        cohorts={
            "A": cohort_a_provenance(cohort_a, excluded_phase2_ids),
            "B": {
                f"{model_key}|{group}": cohort_b_provenance(res)
                for group, res in cohort_b.items()
            },
            "C": cohort_c_provenance(cohort_c),
        },
        cohort_membership_map=dedup.cohort_membership_map(),
        deduplication_alias_map={
            f"{m}|{i}|{c}": obs for (m, i, c), obs in dedup.alias_map.items()
        },
        final_margin_strata={
            f"{k[0]}|{k[1]}": list(v) for k, v in finalized.stratum_edges.items()
        },
        screening={
            # Frozen acquisition constants (§11), recorded so a real freeze
            # cannot silently redefine them. The synthetic dry run does not
            # physically screen 2,000 records -- it records the frozen design
            # and stays synthetic, which the gate rejects anyway.
            **planned_screening_design(),
            "stopped_reason": finalized.stopped_reason,
            "blocks": finalized.blocks_screened,
            "screened_total": finalized.screened_total,
        },
        nominal_condition_slots=dedup.nominal_slots,
        unique_observations=dedup.unique_observations,
        synthetic=True,
    )

    report = DryRunReport(
        banner=SYNTHETIC_BANNER,
        screening_stopped_reason=finalized.stopped_reason,
        screened_total=finalized.screened_total,
        cohort_a_state=cohort_a.state,
        cohort_a_size=len(cohort_a.items),
        cohort_a_relations=cohort_a.relation_distribution,
        cohort_b_states={g: r.state for g, r in cohort_b.items()},
        cohort_c_size=len(cohort_c.items),
        nominal_slots=dedup.nominal_slots,
        unique_observations=dedup.unique_observations,
        collapsed_slots=dedup.collapsed_slots,
        paired_summary=dataclasses.asdict(summary) if summary else {},
        manifest=manifest.to_dict(),
    )

    if output_dir is not None:
        _write_dryrun_outputs(Path(output_dir), report)
    return report


def _write_dryrun_outputs(output_dir: Path, report: DryRunReport) -> None:
    """Write dry-run output under a clearly-marked `dryrun_` filename.

    Mirrors the Phase 2 convention that synthetic output is never written to
    a path that could be mistaken for real results (README.md, "Dry run").
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{DRYRUN_PREFIX}_phase3_synthetic_report.json"
    payload = dataclasses.asdict(report)
    payload["_warning"] = SYNTHETIC_BANNER
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

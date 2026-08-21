"""Deterministic construction of the three Phase 3 cohorts.

Implements `docs/phase3_scaled_study_design.md` §15 (Cohort A / Cohort B),
§16 (Cohort C), §17 (relation balancing) and §32 (missing-cell ladder).

The three cohorts answer different questions and are never interchangeable
(§15). Two structural properties of the frozen design are enforced here:

- **Cohort A imposes no relation quota** and is balanced by margin stratum
  only, so a scarce PRIMARY relation can never make the sole primary
  replication untestable (§15.1).
- **A Cohort B eligibility failure is returned as state, not raised**, and
  Cohort A is constructed independently, so B's failure cannot terminate or
  invalidate A (§15.2, §34).

All selection uses baseline/eligibility information only; nothing here may
consult a Phase 3 outcome (§16 "selection integrity").
"""

from __future__ import annotations

import dataclasses
import random

from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_TARGET,
    COHORT_A_RELATION_DOMINANCE_FLAG,
    COHORT_B_CELL_MINIMUM,
    COHORT_B_CELL_TARGET,
    MARGIN_STRATA,
    PHASE3_RELATIONS,
)
from conflict_eval.phase3.screening import FinalizedScreening, item_id_of

COHORT_A = "A"
COHORT_B = "B"
COHORT_C = "C"

STATE_COMPLETE = "COMPLETE"
STATE_DOWNSAMPLED = "DOWNSAMPLED"
STATE_ELIGIBILITY_LIMITED = "ELIGIBILITY_LIMITED"
# Fewer than three of four PRIMARY relations qualify: removed from the
# confirmatory families but still reported descriptively (frozen §32 rule 4).
STATE_ELIGIBILITY_LIMITED_EXPLORATORY = "ELIGIBILITY_LIMITED_EXPLORATORY"


def _sorted_pool(records: list[dict]) -> list[dict]:
    """Deterministic ordering independent of input order, matching the
    Phase 2 convention in `data/sampling.py`."""
    return sorted(records, key=lambda r: item_id_of(r))


def _take(pool: list[dict], count: int, seed: int, salt: str) -> list[dict]:
    """Deterministic seeded selection of `count` items from `pool`."""
    ordered = _sorted_pool(pool)
    if count >= len(ordered):
        return ordered
    rng = random.Random(f"{seed}:{salt}")
    return sorted(rng.sample(ordered, count), key=lambda r: item_id_of(r))


# ---------------------------------------------------------------------------
# Cohort A -- direct Qwen replication (PRIMARY)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CohortAResult:
    state: str
    items: tuple[dict, ...]
    per_stratum_selected: dict[str, int]
    per_stratum_available: dict[str, int]
    relation_distribution: dict[str, int]
    relation_dominance_share: float | None
    relation_dominance_flag: bool
    excluded_phase2_count: int
    shortfall: dict[str, int]

    @property
    def is_eligibility_limited(self) -> bool:
        return self.state == STATE_ELIGIBILITY_LIMITED


def build_cohort_a(
    finalized: FinalizedScreening,
    seed: int,
    *,
    phase2_excluded_ids: frozenset[str] | set[str],
    per_stratum_target: int = COHORT_A_PER_STRATUM_TARGET,
) -> CohortAResult:
    """Select the 96 fresh Qwen KW items, 32 per margin stratum (§15.1).

    `phase2_excluded_ids` is **required and keyword-only**: item freshness
    is a defining property of the direct replication (§15.1), so a caller
    must consciously supply the exclusion set. There is deliberately no
    default -- an omitted argument raises `TypeError` rather than silently
    producing a "replication" that re-measures the original Phase 2 items.
    Phase 3C supplies the real frozen Phase 2 Qwen KW id list; it is never
    hardcoded or invented here, and tests pass synthetic ids.

    Passing an explicitly empty set is permitted (a caller stating there is
    nothing to exclude), but the real-run gate additionally requires a
    non-empty exclusion provenance before a real Phase 3 run.

    Cohort A is ELIGIBILITY_LIMITED only when the screened supply cannot
    fill 32/32/32 -- never because a relation is scarce and never because
    Cohort B failed (§15.1, §34).
    """
    excluded = frozenset(str(i) for i in phase2_excluded_ids)

    by_stratum: dict[str, list[dict]] = {s: [] for s in MARGIN_STRATA}
    excluded_count = 0
    for record in finalized.eligible_records():
        if record.get("knowledge_group") != "KW":
            continue
        # PRIMARY_RELATIONS only (§15.1); no relation quota is applied.
        if record.get("relation") not in PHASE3_RELATIONS:
            continue
        item_id = item_id_of(record)
        if item_id in excluded:
            excluded_count += 1
            continue
        stratum = finalized.stratum_of(item_id)
        if stratum in by_stratum:
            by_stratum[stratum].append(record)

    available = {s: len(items) for s, items in by_stratum.items()}
    shortfall = {
        s: max(0, per_stratum_target - available[s]) for s in MARGIN_STRATA
    }

    selected: list[dict] = []
    for stratum in MARGIN_STRATA:
        selected.extend(
            _take(by_stratum[stratum], per_stratum_target, seed, f"cohortA:{stratum}")
        )

    per_stratum_selected = {
        s: sum(1 for r in selected if finalized.stratum_of(item_id_of(r)) == s)
        for s in MARGIN_STRATA
    }
    relation_distribution: dict[str, int] = {}
    for record in selected:
        relation_distribution[record["relation"]] = (
            relation_distribution.get(record["relation"], 0) + 1
        )

    dominance_share = (
        max(relation_distribution.values()) / len(selected) if selected else None
    )
    state = (
        STATE_ELIGIBILITY_LIMITED
        if any(shortfall.values())
        else STATE_COMPLETE
    )

    return CohortAResult(
        state=state,
        items=tuple(selected),
        per_stratum_selected=per_stratum_selected,
        per_stratum_available=available,
        relation_distribution=relation_distribution,
        relation_dominance_share=dominance_share,
        # DIAGNOSTIC and explicitly non-gating: it never changes `state`
        # and never triggers re-sampling (§15.1).
        relation_dominance_flag=(
            dominance_share is not None
            and dominance_share > COHORT_A_RELATION_DOMINANCE_FLAG
        ),
        excluded_phase2_count=excluded_count,
        shortfall=shortfall,
    )


# ---------------------------------------------------------------------------
# Cohort B -- relation-balanced generalization (SECONDARY)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CohortBGroupResult:
    """Realized Cohort B cohort for one (model x knowledge group).

    Rich enough to record everything §32 rule 4 requires to be reported and
    frozen in the manifest: original per-cell availability, which relations
    qualified, which were excluded as short, the realized relation set and
    cell count, the status, whether the group may enter the secondary
    confirmatory family, and why any reduction happened.
    """

    model_key: str
    knowledge_group: str
    state: str
    items: tuple[dict, ...]
    target_cell_count: int
    minimum_cell_count: int
    per_cell_available: dict[tuple[str, str], int]
    qualifying_relations: tuple[str, ...]
    excluded_short_relations: tuple[str, ...]
    realized_relations: tuple[str, ...]
    realized_cell_count: int | None
    deficient_cells: tuple[tuple[str, str], ...]
    confirmatory_eligible: bool
    reduction_reason: str | None

    @property
    def is_eligibility_limited(self) -> bool:
        return self.state in (STATE_ELIGIBILITY_LIMITED, STATE_ELIGIBILITY_LIMITED_EXPLORATORY)


def build_cohort_b_group(
    finalized: FinalizedScreening,
    knowledge_group: str,
    seed: int,
    target: int = COHORT_B_CELL_TARGET,
    minimum: int = COHORT_B_CELL_MINIMUM,
) -> CohortBGroupResult:
    """Relation-balanced selection for one (model x knowledge group) (§15.2).

    Implements the frozen §32 ladder exactly. A relation QUALIFIES only if
    **all three** of its margin cells reach the minimum; the balanced
    estimate is then computed over the qualifying relations, preserving
    exact relation x stratum balance:

    1. all four relations qualify and every cell >= target -> `target`/cell;
    2. all four qualify but some cell is below target -> downsample **every**
       cell to the common realized minimum (>= minimum), keeping balance;
    3. exactly one relation fails -> **the balanced estimate is still
       computed over the remaining three relations** (§32 rule 4, second
       bullet). The group is ELIGIBILITY_LIMITED but remains confirmatory-
       eligible, and items are NOT discarded;
    4. two or more relations fail (fewer than three of four qualify) -> the
       whole model x group is ELIGIBILITY_LIMITED / EXPLORATORY, removed
       from the confirmatory families (§28), still selected and reported
       descriptively over whatever relations qualify.

    Never backfills a short relation from a full one, and never selects
    based on any evidence outcome.
    """
    by_cell: dict[tuple[str, str], list[dict]] = {
        (relation, stratum): []
        for relation in PHASE3_RELATIONS
        for stratum in MARGIN_STRATA
    }
    for record in finalized.eligible_records():
        if record.get("knowledge_group") != knowledge_group:
            continue
        key = (record.get("relation"), finalized.stratum_of(item_id_of(record)))
        if key in by_cell:
            by_cell[key].append(record)

    available = {key: len(items) for key, items in by_cell.items()}
    deficient = tuple(sorted(key for key, n in available.items() if n < minimum))

    # A relation qualifies only if ALL THREE of its margin cells reach the
    # minimum -- a relation with one starved stratum cannot support a
    # balanced relation x stratum grid.
    qualifying = tuple(
        relation
        for relation in PHASE3_RELATIONS
        if all(available[(relation, s)] >= minimum for s in MARGIN_STRATA)
    )
    excluded_short = tuple(r for r in PHASE3_RELATIONS if r not in qualifying)

    if not qualifying:
        return CohortBGroupResult(
            model_key=finalized.model_key,
            knowledge_group=knowledge_group,
            state=STATE_ELIGIBILITY_LIMITED_EXPLORATORY,
            items=(),
            target_cell_count=target,
            minimum_cell_count=minimum,
            per_cell_available=available,
            qualifying_relations=(),
            excluded_short_relations=excluded_short,
            realized_relations=(),
            realized_cell_count=None,
            deficient_cells=deficient,
            confirmatory_eligible=False,
            reduction_reason="no relation meets the minimum in all three strata",
        )

    # Realized cell count: the smallest availability across the QUALIFYING
    # cells, capped at the target. Balance is preserved by applying it to
    # every retained cell.
    realized_cell_count = min(
        target,
        min(available[(relation, s)] for relation in qualifying for s in MARGIN_STRATA),
    )

    selected: list[dict] = []
    for relation in qualifying:
        for stratum in MARGIN_STRATA:
            selected.extend(
                _take(
                    by_cell[(relation, stratum)],
                    realized_cell_count,
                    seed,
                    f"cohortB:{finalized.model_key}:{knowledge_group}:{relation}:{stratum}",
                )
            )

    if len(qualifying) == len(PHASE3_RELATIONS):
        # All four relations retained.
        if realized_cell_count >= target:
            state, reason = STATE_COMPLETE, None
        else:
            state = STATE_DOWNSAMPLED
            reason = (
                f"all {len(PHASE3_RELATIONS)} relations qualify; smallest cell "
                f"{realized_cell_count} < target {target}, so every cell was "
                "downsampled to the common realized count"
            )
        confirmatory = True
    elif len(qualifying) >= 3:
        # §32 rule 4: balanced estimate over the remaining relations.
        state = STATE_ELIGIBILITY_LIMITED
        confirmatory = True
        reason = (
            f"relation(s) {list(excluded_short)} did not reach the minimum "
            f"{minimum} in all three strata and were excluded; the balanced "
            f"estimate is computed over {list(qualifying)} at "
            f"{realized_cell_count} items/cell"
        )
    else:
        # Fewer than three of four qualify -> EXPLORATORY, out of the
        # confirmatory families, still reported descriptively.
        state = STATE_ELIGIBILITY_LIMITED_EXPLORATORY
        confirmatory = False
        reason = (
            f"only {len(qualifying)} of {len(PHASE3_RELATIONS)} relations qualify "
            f"({list(qualifying)}); fewer than three, so this model x group is "
            "removed from the confirmatory families and reported descriptively"
        )

    return CohortBGroupResult(
        model_key=finalized.model_key,
        knowledge_group=knowledge_group,
        state=state,
        items=tuple(selected),
        target_cell_count=target,
        minimum_cell_count=minimum,
        per_cell_available=available,
        qualifying_relations=qualifying,
        excluded_short_relations=excluded_short,
        realized_relations=qualifying,
        realized_cell_count=realized_cell_count,
        deficient_cells=deficient,
        confirmatory_eligible=confirmatory,
        reduction_reason=reason,
    )


# ---------------------------------------------------------------------------
# Cohort C -- shared cross-model cohort (SECONDARY)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SharedItem:
    """One shared question plus each model's own knowledge state.

    KC/KW labels are never forced to agree across models: what is shared is
    the *question*, not the knowledge label (§16). `per_model` therefore
    keeps each model's own group, margin, and stratum separately.
    """

    item_id: str
    relation: str
    per_model: dict[str, dict]

    def knowledge_group_for(self, model_key: str) -> str:
        return self.per_model[model_key]["knowledge_group"]


@dataclasses.dataclass(frozen=True)
class CohortCResult:
    state: str
    items: tuple[SharedItem, ...]
    candidates_considered: int
    relation_distribution: dict[str, int]
    label_agreement_count: int
    label_disagreement_count: int


def build_cohort_c(
    finalized_by_model: dict[str, FinalizedScreening],
    seed: int,
    target_size: int,
) -> CohortCResult:
    """Deterministic shared cohort across all models (§16).

    Retains items that are primary-conflict eligible and KC-or-KW for
    **every** model, then applies a seeded relation-balanced quota. Each
    model's own knowledge state is preserved per item.
    """
    if not finalized_by_model:
        raise ValueError("Cohort C requires at least one finalized screening")

    models = sorted(finalized_by_model)
    per_model_index: dict[str, dict[str, dict]] = {}
    for model_key in models:
        finalized = finalized_by_model[model_key]
        per_model_index[model_key] = {
            item_id_of(r): r for r in finalized.eligible_records()
        }

    shared_ids = set(per_model_index[models[0]])
    for model_key in models[1:]:
        shared_ids &= set(per_model_index[model_key])

    candidates: list[SharedItem] = []
    for item_id in sorted(shared_ids):
        per_model: dict[str, dict] = {}
        relations = set()
        for model_key in models:
            record = per_model_index[model_key][item_id]
            relations.add(record.get("relation"))
            per_model[model_key] = {
                "knowledge_group": record["knowledge_group"],
                "parametric_margin": record.get("parametric_margin"),
                "margin_stratum": finalized_by_model[model_key].stratum_of(item_id),
                "baseline_answer": record.get("baseline_answer"),
            }
        relation = relations.pop() if len(relations) == 1 else None
        if relation not in PHASE3_RELATIONS:
            continue
        candidates.append(
            SharedItem(item_id=item_id, relation=relation, per_model=per_model)
        )

    # Relation-balanced quota, deterministic and seeded (§16 step 4).
    by_relation: dict[str, list[SharedItem]] = {r: [] for r in PHASE3_RELATIONS}
    for item in candidates:
        by_relation[item.relation].append(item)

    per_relation_quota = target_size // len(PHASE3_RELATIONS)
    selected: list[SharedItem] = []
    for relation in PHASE3_RELATIONS:
        pool = sorted(by_relation[relation], key=lambda i: i.item_id)
        if per_relation_quota >= len(pool):
            chosen = pool
        else:
            rng = random.Random(f"{seed}:cohortC:{relation}")
            chosen = sorted(rng.sample(pool, per_relation_quota), key=lambda i: i.item_id)
        selected.extend(chosen)

    relation_distribution: dict[str, int] = {}
    agreement = 0
    for item in selected:
        relation_distribution[item.relation] = relation_distribution.get(item.relation, 0) + 1
        labels = {item.knowledge_group_for(m) for m in models}
        if len(labels) == 1:
            agreement += 1

    state = (
        STATE_COMPLETE if len(selected) >= target_size else STATE_ELIGIBILITY_LIMITED
    )
    return CohortCResult(
        state=state,
        items=tuple(selected),
        candidates_considered=len(candidates),
        relation_distribution=relation_distribution,
        label_agreement_count=agreement,
        label_disagreement_count=len(selected) - agreement,
    )

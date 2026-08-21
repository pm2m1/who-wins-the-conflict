"""Blockwise Phase 3 screening state and margin-stratum freeze.

Implements the frozen adaptive pre-outcome screening procedure
(`docs/phase3_scaled_study_design.md`, §11) and the margin-stratum
stability rule in the same section.

**This module never screens anything itself.** It accepts already-produced
baseline records block by block and tracks supply. Acquiring real baseline
records is Phase 3D work and is forbidden in Phase 3B.

Two invariants are enforced structurally rather than by convention:

1. **Outcome-blindness.** `add_block` accepts only baseline/eligibility
   fields. Any record carrying a Phase 3 evidence-condition outcome field
   (`context_adopted`, `condition`, ...) is rejected outright, so the
   prohibited stopping information in §11 cannot reach the stopping rule
   even by accident.
2. **Freeze immutability.** Once `finalize()` is called, the stratum
   boundaries and per-item assignments are captured in a frozen
   `FinalizedScreening` object. Adding further blocks to the mutable state
   afterwards cannot alter an already-finalized result.
"""

from __future__ import annotations

import dataclasses

from conflict_eval.data.sampling import assign_margin_bin, compute_margin_bin_edges
from conflict_eval.phase3.constants import (
    COHORT_A_PER_STRATUM_SUPPLY,
    COHORT_B_CELL_SUPPLY,
    ITEM_ID_FIELD,
    MARGIN_STRATA,
    PHASE3_RELATIONS,
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
)

# Fields that only exist once an evidence condition has been generated.
# Their presence means outcome information is leaking into a pre-outcome
# decision, which §11 prohibits absolutely.
_PROHIBITED_OUTCOME_FIELDS = frozenset(
    {
        "context_adopted",
        "condition",
        "evidence_truth",
        "source_role",
        "source_label",
        "conflict_status",
        "final_correct",
        "parsed_answer_accuracy",
    }
)

STOP_SUPPLY_MET = "supply_criteria_met"
STOP_CEILING = "ceiling_reached"
# Screening ended with supply unmet but without reaching the ceiling --
# i.e. the candidate frame ran out. Kept distinct from STOP_CEILING so the
# manifest records why screening actually stopped (§11).
STOP_SUPPLY_NOT_MET = "supply_not_met"


class ScreeningError(ValueError):
    """Raised when screening input or sequencing violates the frozen rule."""


def item_id_of(record: dict) -> str:
    """Canonical item identity for a baseline record.

    The repository's real baseline records key each item as `item_id`
    (`src/conflict_eval/cli.py`, `cmd_screen`), so Phase 3 requires exactly
    that field rather than accepting a second competing key. A record
    without it fails loudly. An earlier Phase 3B draft read `id`, which
    passed only because the synthetic fixtures happened to match it and
    would have raised `KeyError` on every real record in Phase 3D.
    """
    if ITEM_ID_FIELD not in record:
        raise ScreeningError(
            f"Baseline record is missing the canonical {ITEM_ID_FIELD!r} field "
            f"(got keys {sorted(record)[:8]}). Phase 3 uses the repository's real "
            "baseline schema; see src/conflict_eval/cli.py, cmd_screen."
        )
    value = record[ITEM_ID_FIELD]
    if value is None or str(value) == "":
        raise ScreeningError(
            f"Baseline record has an empty {ITEM_ID_FIELD!r}; item identity is "
            "required for deterministic selection and deduplication."
        )
    return str(value)


def _is_eligible(record: dict) -> bool:
    """A record contributes supply only if it is a usable KC/KW item that
    is primary-conflict eligible (§11, §12). Abstentions, manual-review
    items, and malformed records never count -- unchanged from Phase 2.
    """
    return (
        record.get("knowledge_group") in ("KC", "KW")
        and bool(record.get("primary_conflict_eligible"))
    )


@dataclasses.dataclass(frozen=True)
class SupplyReport:
    """Supply counts after one block, evaluated against freshly recomputed
    strata (§11 step 2)."""

    block_index: int
    screened_total: int
    cohort_a_per_stratum: dict[str, int]
    cohort_a_supply_met: bool
    cohort_b_per_cell: dict[tuple[str, str, str], int]
    cohort_b_supply_met: bool
    stratum_edges: dict[tuple[str, str], list[float]]


@dataclasses.dataclass(frozen=True)
class FinalizedScreening:
    """Immutable screening result. Created only by `ScreeningState.finalize`.

    The stratum edges and per-item assignments recorded here are the frozen
    ones (§11 step 3) and are what the cohort samplers and the manifest
    must use. Nothing can recompute them afterwards.
    """

    model_key: str
    stopped_reason: str
    blocks_screened: int
    screened_total: int
    records: tuple[dict, ...]
    stratum_edges: dict[tuple[str, str], tuple[float, ...]]
    stratum_assignments: dict[str, str]
    cohort_a_supply_met: bool
    cohort_b_supply_met: bool
    final_supply: SupplyReport

    def eligible_records(self) -> list[dict]:
        return [dict(r) for r in self.records if _is_eligible(r)]

    def stratum_of(self, item_id: str) -> str | None:
        return self.stratum_assignments.get(str(item_id))


class ScreeningState:
    """Accumulates baseline records blockwise for one model.

    Usage (Phase 3D; in Phase 3B exercised only with synthetic records)::

        state = ScreeningState("qwen", phase2_excluded_ids=frozen_ids)
        while not state.should_stop():
            state.add_block(next_block_of_baseline_records)
        finalized = state.finalize()
    """

    def __init__(
        self,
        model_key: str,
        phase2_excluded_ids: frozenset[str] | set[str] | None = None,
        block_size: int = SCREENING_BLOCK_SIZE,
        ceiling: int = SCREENING_CEILING_PER_MODEL,
        require_cohort_a: bool | None = None,
    ) -> None:
        self.model_key = model_key
        # The real Phase 2 exclusion id set is supplied by Phase 3C; it is
        # never hardcoded or invented here (§15.1).
        self.phase2_excluded_ids = frozenset(str(i) for i in (phase2_excluded_ids or ()))
        self.block_size = block_size
        self.ceiling = ceiling
        # Cohort A's supply criterion applies to Qwen only (§11).
        self.require_cohort_a = (
            (model_key == "qwen") if require_cohort_a is None else require_cohort_a
        )
        self._records: list[dict] = []
        self._blocks = 0
        self._finalized = False

    # -- input ------------------------------------------------------------
    def add_block(self, records: list[dict]) -> SupplyReport:
        """Add one screened block and recompute supply against freshly
        recomputed strata (§11 steps 1-2)."""
        if self._finalized:
            raise ScreeningError(
                "Cannot add a block after finalize(): the frozen strata and "
                "stopping decision are immutable (§11 step 3)."
            )
        if len(records) > self.block_size:
            raise ScreeningError(
                f"Block of {len(records)} exceeds the frozen block size "
                f"{self.block_size} (§11)."
            )
        if self._blocks * self.block_size >= self.ceiling:
            raise ScreeningError(
                f"Screening ceiling of {self.ceiling} candidates already reached "
                "for this model; screening is never extended past the ceiling (§11)."
            )
        for record in records:
            leaked = _PROHIBITED_OUTCOME_FIELDS.intersection(record.keys())
            if leaked:
                raise ScreeningError(
                    "Screening input carries prohibited outcome field(s) "
                    f"{sorted(leaked)}. Screening may use baseline/eligibility "
                    "information only (§11)."
                )
            self._records.append(dict(record))
        self._blocks += 1
        return self.supply_report()

    # -- strata -----------------------------------------------------------
    def _compute_strata(
        self, records: list[dict]
    ) -> tuple[dict[tuple[str, str], list[float]], dict[str, str]]:
        """Recompute strata deterministically from the COMPLETE currently
        screened eligible pool, within (model x knowledge group) (§11 step 1,
        §14). Reuses the committed Phase 2 tertile helpers unchanged.
        """
        edges: dict[tuple[str, str], list[float]] = {}
        assignments: dict[str, str] = {}
        for group in ("KC", "KW"):
            pool = [
                r
                for r in records
                if _is_eligible(r) and r.get("knowledge_group") == group
            ]
            if not pool:
                continue
            margins = [float(r["parametric_margin"]) for r in pool]
            group_edges = compute_margin_bin_edges(margins, n_bins=len(MARGIN_STRATA))
            edges[(self.model_key, group)] = group_edges
            for record in pool:
                assignments[item_id_of(record)] = assign_margin_bin(
                    float(record["parametric_margin"]), group_edges, MARGIN_STRATA
                )
        return edges, assignments

    # -- supply -----------------------------------------------------------
    def supply_report(self) -> SupplyReport:
        edges, assignments = self._compute_strata(self._records)

        # Cohort A supply: FRESH Qwen KW items only; Phase 2 items never
        # count toward this supply (§11, §15.1). No relation quota.
        per_stratum = {stratum: 0 for stratum in MARGIN_STRATA}
        for record in self._records:
            if not _is_eligible(record) or record.get("knowledge_group") != "KW":
                continue
            item_id = item_id_of(record)
            if item_id in self.phase2_excluded_ids:
                continue
            stratum = assignments.get(item_id)
            if stratum is not None:
                per_stratum[stratum] += 1
        cohort_a_met = all(
            count >= COHORT_A_PER_STRATUM_SUPPLY for count in per_stratum.values()
        )

        # Cohort B supply: every (relation x group x stratum) cell needs
        # target + reserve = 10 (§11). Phase 2 items are NOT excluded here;
        # freshness is a Cohort A requirement only (§15.1).
        per_cell: dict[tuple[str, str, str], int] = {
            (relation, group, stratum): 0
            for relation in PHASE3_RELATIONS
            for group in ("KC", "KW")
            for stratum in MARGIN_STRATA
        }
        for record in self._records:
            if not _is_eligible(record):
                continue
            relation = record.get("relation")
            group = record.get("knowledge_group")
            stratum = assignments.get(item_id_of(record))
            key = (relation, group, stratum)
            if key in per_cell:
                per_cell[key] += 1
        cohort_b_met = all(count >= COHORT_B_CELL_SUPPLY for count in per_cell.values())

        return SupplyReport(
            block_index=self._blocks,
            screened_total=len(self._records),
            cohort_a_per_stratum=per_stratum,
            cohort_a_supply_met=cohort_a_met,
            cohort_b_per_cell=per_cell,
            cohort_b_supply_met=cohort_b_met,
            stratum_edges=edges,
        )

    # -- stopping ---------------------------------------------------------
    def ceiling_reached(self) -> bool:
        return self._blocks * self.block_size >= self.ceiling

    def should_stop(self) -> bool:
        """Stop when all applicable supply criteria are met, or at the
        ceiling (§11). Uses supply information only."""
        if self.ceiling_reached():
            return True
        report = self.supply_report()
        criteria = [report.cohort_b_supply_met]
        if self.require_cohort_a:
            criteria.append(report.cohort_a_supply_met)
        return all(criteria)

    def finalize(self) -> FinalizedScreening:
        """Freeze the strata and the stopping decision (§11 step 3)."""
        report = self.supply_report()
        edges, assignments = self._compute_strata(self._records)
        criteria_met = report.cohort_b_supply_met and (
            report.cohort_a_supply_met or not self.require_cohort_a
        )
        if criteria_met:
            reason = STOP_SUPPLY_MET
        elif self.ceiling_reached():
            reason = STOP_CEILING
        else:
            reason = STOP_SUPPLY_NOT_MET
        self._finalized = True
        return FinalizedScreening(
            model_key=self.model_key,
            stopped_reason=reason,
            blocks_screened=self._blocks,
            screened_total=len(self._records),
            records=tuple(dict(r) for r in self._records),
            stratum_edges={k: tuple(v) for k, v in edges.items()},
            stratum_assignments=dict(assignments),
            cohort_a_supply_met=report.cohort_a_supply_met,
            cohort_b_supply_met=report.cohort_b_supply_met,
            final_supply=report,
        )

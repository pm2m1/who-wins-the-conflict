"""Primary pilot metrics.

Definitions match docs/phase2_research_design.md exactly. Every metric
returns a `RateEstimate` carrying both the rate and the subset size `n`,
rather than a bare float, so an empty or tiny subset is visible rather
than silently reported as 0.0 or NaN.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class RateEstimate:
    rate: float | None
    n: int


def _rate(records: list[dict], predicate: Callable[[dict], bool]) -> RateEstimate:
    if not records:
        return RateEstimate(rate=None, n=0)
    hits = sum(1 for r in records if predicate(r))
    return RateEstimate(rate=hits / len(records), n=len(records))


def context_adoption_rate(records: list[dict]) -> RateEstimate:
    """CAR = P(final answer == contextual conflicting answer).

    `records` must already be restricted to conflict trials
    (conflict_status == "conflict") — this is checked, not assumed, since
    silently including agreement trials would make CAR uninterpretable
    (docs/phase2_research_design.md, "Conflict vs. agreement").
    """
    for r in records:
        if r.get("conflict_status") != "conflict":
            raise ValueError("context_adoption_rate requires conflict trials only")
    return _rate(records, lambda r: bool(r["context_adopted"]))


def harmful_override_rate(records: list[dict]) -> RateEstimate:
    """HOR: KC items, false conflicting context (C3/C4 conflict trials).
    Lower is better.
    """
    subset = [
        r
        for r in records
        if r.get("knowledge_group") == "KC"
        and r.get("evidence_truth") == "false"
        and r.get("conflict_status") == "conflict"
    ]
    return _rate(subset, lambda r: bool(r["context_adopted"]))


def corrective_override_rate(records: list[dict]) -> RateEstimate:
    """COR: KW items, correct conflicting context (C1/C2 conflict
    trials). Higher is better.
    """
    subset = [
        r
        for r in records
        if r.get("knowledge_group") == "KW"
        and r.get("evidence_truth") == "true"
        and r.get("conflict_status") == "conflict"
    ]
    return _rate(subset, lambda r: bool(r["context_adopted"]))


def _split_by_source_role(records: list[dict]) -> tuple[list[dict], list[dict]]:
    preferred = [r for r in records if r.get("source_role") == "preferred"]
    dispreferred = [r for r in records if r.get("source_role") == "dispreferred"]
    return preferred, dispreferred


def source_effect_on_harmful_override(records: list[dict]) -> dict:
    """Delta_harm = HOR_preferred - HOR_dispreferred."""
    preferred, dispreferred = _split_by_source_role(records)
    hor_preferred = harmful_override_rate(preferred)
    hor_dispreferred = harmful_override_rate(dispreferred)
    delta = None
    if hor_preferred.rate is not None and hor_dispreferred.rate is not None:
        delta = hor_preferred.rate - hor_dispreferred.rate
    return {
        "hor_preferred": hor_preferred,
        "hor_dispreferred": hor_dispreferred,
        "delta_harm": delta,
    }


def source_effect_on_corrective_override(records: list[dict]) -> dict:
    """Delta_correct = COR_preferred - COR_dispreferred."""
    preferred, dispreferred = _split_by_source_role(records)
    cor_preferred = corrective_override_rate(preferred)
    cor_dispreferred = corrective_override_rate(dispreferred)
    delta = None
    if cor_preferred.rate is not None and cor_dispreferred.rate is not None:
        delta = cor_preferred.rate - cor_dispreferred.rate
    return {
        "cor_preferred": cor_preferred,
        "cor_dispreferred": cor_dispreferred,
        "delta_correct": delta,
    }


def abstention_rate(records: list[dict]) -> RateEstimate:
    """AR = P(Decision == uncertain). Exploratory."""
    return _rate(records, lambda r: r.get("decision") == "uncertain")

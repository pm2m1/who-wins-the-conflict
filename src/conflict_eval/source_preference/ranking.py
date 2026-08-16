"""Aggregate calibration trials into pairwise statistics and a preference
matrix, plus an explicitly-labeled pilot heuristic for recommending
candidate preferred/dispreferred pairs to the researcher.

Per docs/phase2_research_design.md, this module never selects a
preferred/dispreferred source pair automatically — build-pilot requires
the researcher to set those in configs/pilot.yaml after inspecting this
output.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict


@dataclasses.dataclass(frozen=True)
class PairwiseStat:
    source_a: str
    source_b: str
    n_trials: int
    n_valid: int
    a_wins: int
    b_wins: int
    p_a_preferred: float | None  # None when no valid (parseable) trials exist


def compute_pairwise_stats(trials: list[dict]) -> list[PairwiseStat]:
    """`trials` are calibration-trial records (dicts with at least
    source_a, source_b, selected_source) — both presentation orders (AB
    and BA) of a pair are pooled into one statistic per pair.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for trial in trials:
        key = (trial["source_a"], trial["source_b"])
        groups[key].append(trial)

    stats = []
    for (source_a, source_b), group in groups.items():
        valid = [t for t in group if t["selected_source"] is not None]
        n_valid = len(valid)
        a_wins = sum(1 for t in valid if t["selected_source"] == source_a)
        b_wins = sum(1 for t in valid if t["selected_source"] == source_b)
        p_a_preferred = a_wins / n_valid if n_valid > 0 else None
        stats.append(
            PairwiseStat(
                source_a=source_a,
                source_b=source_b,
                n_trials=len(group),
                n_valid=n_valid,
                a_wins=a_wins,
                b_wins=b_wins,
                p_a_preferred=p_a_preferred,
            )
        )
    return stats


def build_preference_matrix(stats: list[PairwiseStat]) -> dict[str, dict[str, float | None]]:
    labels = sorted({s.source_a for s in stats} | {s.source_b for s in stats})
    matrix: dict[str, dict[str, float | None]] = {a: {b: None for b in labels} for a in labels}
    for s in stats:
        matrix[s.source_a][s.source_b] = s.p_a_preferred
        matrix[s.source_b][s.source_a] = None if s.p_a_preferred is None else 1 - s.p_a_preferred
    return matrix


def rank_sources_pilot_heuristic(stats: list[PairwiseStat]) -> list[dict]:
    """PILOT HEURISTIC, not a statistically validated ranking method.

    Ranks each source by its mean pairwise preference rate across the
    pairs it participated in (pairs with zero valid trials excluded).
    Reports underlying win/total counts alongside the mean so the
    researcher can judge stability before treating any source as
    "preferred" — see docs/phase2_research_design.md, "Source pair
    selection".
    """
    rates: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [wins, total_valid]

    for s in stats:
        if s.p_a_preferred is None:
            continue
        rates[s.source_a].append(s.p_a_preferred)
        rates[s.source_b].append(1 - s.p_a_preferred)
        counts[s.source_a][0] += s.a_wins
        counts[s.source_a][1] += s.n_valid
        counts[s.source_b][0] += s.b_wins
        counts[s.source_b][1] += s.n_valid

    ranking = []
    for source, source_rates in rates.items():
        wins, total = counts[source]
        ranking.append(
            {
                "source": source,
                "mean_pairwise_preference_rate": sum(source_rates) / len(source_rates),
                "wins": wins,
                "total_valid_trials": total,
            }
        )
    ranking.sort(key=lambda r: r["mean_pairwise_preference_rate"], reverse=True)
    return ranking

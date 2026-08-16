"""Deterministic sampling utilities.

Two distinct sampling steps use this module: (1) subsampling PopQA down
to a manageable screening pool, and (2) spreading pilot item selection
across parametric-margin bins so H3 (interaction hypothesis) has items to
observe at more than one strength level, per docs/phase2_research_design.md.
"""

from __future__ import annotations

import random


def sample_candidates(items: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministically subsample `n` items (by id) for baseline
    screening. Sorting before sampling makes the result independent of
    the input list's original order.
    """
    rng = random.Random(seed)
    sorted_items = sorted(items, key=lambda x: str(x["id"]))
    if n >= len(sorted_items):
        return sorted_items
    return rng.sample(sorted_items, n)


def compute_margin_bin_edges(margins: list[float], n_bins: int = 3) -> list[float]:
    """Quantile edges splitting `margins` into `n_bins` roughly equal
    groups. Bins are a pilot sampling convenience, not a claim that
    parametric strength is categorical (docs/phase2_research_design.md).
    """
    if not margins:
        raise ValueError("Cannot compute quantile edges from an empty margin list")
    sorted_margins = sorted(margins)
    edges = []
    for i in range(1, n_bins):
        idx = round(i / n_bins * (len(sorted_margins) - 1))
        edges.append(sorted_margins[idx])
    return edges


def assign_margin_bin(margin: float, edges: list[float], labels: tuple[str, ...] = ("low", "medium", "high")) -> str:
    if len(edges) != len(labels) - 1:
        raise ValueError(f"Expected {len(labels) - 1} edges for {len(labels)} labels, got {len(edges)}")
    for edge, label in zip(edges, labels[:-1]):
        if margin <= edge:
            return label
    return labels[-1]


def sample_balanced_across_bins(
    items: list[dict],
    target_n: int,
    seed: int,
    bin_key: str = "margin_bin",
    id_key: str = "id",
) -> list[dict]:
    """Sample approximately `target_n` items, drawing as evenly as
    possible across the bins present in `items[bin_key]`.

    If a bin has fewer eligible items than its even share, all of that
    bin's items are taken and the shortfall is redistributed to the other
    bins, so the function always returns min(target_n, len(items)) items
    rather than silently under-sampling.
    """
    rng = random.Random(seed)
    by_bin: dict[str, list[dict]] = {}
    for item in items:
        by_bin.setdefault(item[bin_key], []).append(item)
    for bucket in by_bin.values():
        bucket.sort(key=lambda x: str(x[id_key]))

    bins = sorted(by_bin.keys())
    remaining_target = min(target_n, len(items))
    selected: list[dict] = []
    active_bins = list(bins)

    while remaining_target > 0 and active_bins:
        share = max(1, remaining_target // len(active_bins))
        next_active_bins = []
        for b in active_bins:
            available = by_bin[b]
            take = min(share, len(available))
            if take > 0:
                chosen = rng.sample(available, take)
                chosen_ids = {id(x) for x in chosen}
                by_bin[b] = [x for x in available if id(x) not in chosen_ids]
                selected.extend(chosen)
                remaining_target -= take
            if by_bin[b]:
                next_active_bins.append(b)
            if remaining_target <= 0:
                break
        active_bins = next_active_bins

    return selected

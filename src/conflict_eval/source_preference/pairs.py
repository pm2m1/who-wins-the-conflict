"""Enumerate unordered source pairs for direct pairwise calibration."""

from __future__ import annotations

import itertools


def enumerate_unordered_pairs(source_labels: list[str]) -> list[tuple[str, str]]:
    """All unordered pairs (S_A, S_B) from the candidate source list, in a
    fixed, deterministic order (itertools.combinations preserves input
    order). Duplicate labels are rejected rather than silently collapsed.
    """
    if len(set(source_labels)) != len(source_labels):
        raise ValueError(f"Duplicate source labels are not allowed: {source_labels}")
    return list(itertools.combinations(source_labels, 2))

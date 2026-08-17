"""Generic paired preferred-vs-dispreferred source comparison on
`context_adopted`.

Each factual item is measured under both a "preferred-source" condition
and a "dispreferred-source" condition (e.g. C1 vs. C2 for KW corrective
override, or C3 vs. C4 for KC harmful override), so a paired exact test
on the discordant pairs is a natural SECONDARY check alongside the
primary aggregate rates (HOR/COR, evaluation/metrics.py) — see
docs/decisions.md, "Freeze the first Qwen pilot after validated
analysis". This module does not decide which conditions to compare or
what counts as "primary"; it is a small, reusable statistical utility,
not a new primary metric.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PairedSourceComparison:
    n_items: int
    preferred_rate: float | None
    dispreferred_rate: float | None
    delta: float | None
    both: int
    preferred_only: int
    dispreferred_only: int
    neither: int
    p_value: float | None  # exact two-sided paired binomial (McNemar-equivalent)


def paired_source_comparison(
    records: list[dict],
    knowledge_group: str,
    preferred_condition: str,
    dispreferred_condition: str,
    relation: str | None = None,
) -> PairedSourceComparison:
    """Pair records by `item_id` within `knowledge_group` (and `relation`,
    if given), one under `preferred_condition` and one under
    `dispreferred_condition`, and compute the discordance counts and an
    exact two-sided paired binomial test on `context_adopted`.

    Raises `ValueError` if more than one record shares the same `item_id`
    under the same condition — this function assumes exactly one record
    per item per condition and does not silently average duplicates.
    Items missing either condition are excluded from the paired count
    (incomplete pairs cannot contribute to a paired comparison).
    """
    from scipy import stats

    filtered = [
        r
        for r in records
        if r.get("knowledge_group") == knowledge_group
        and (relation is None or r.get("relation") == relation)
        and r.get("condition") in (preferred_condition, dispreferred_condition)
    ]

    by_item: dict[str, dict[str, bool]] = {}
    for r in filtered:
        item_id = r["item_id"]
        role = "preferred" if r["condition"] == preferred_condition else "dispreferred"
        entry = by_item.setdefault(item_id, {})
        if role in entry:
            raise ValueError(
                f"Duplicate {r['condition']!r} record for item_id={item_id!r}; "
                "paired_source_comparison requires exactly one record per "
                "item per condition."
            )
        entry[role] = bool(r["context_adopted"])

    complete_pairs = {
        item_id: entry
        for item_id, entry in by_item.items()
        if "preferred" in entry and "dispreferred" in entry
    }

    n_items = len(complete_pairs)
    if n_items == 0:
        return PairedSourceComparison(
            n_items=0,
            preferred_rate=None,
            dispreferred_rate=None,
            delta=None,
            both=0,
            preferred_only=0,
            dispreferred_only=0,
            neither=0,
            p_value=None,
        )

    both = sum(1 for e in complete_pairs.values() if e["preferred"] and e["dispreferred"])
    preferred_only = sum(1 for e in complete_pairs.values() if e["preferred"] and not e["dispreferred"])
    dispreferred_only = sum(1 for e in complete_pairs.values() if not e["preferred"] and e["dispreferred"])
    neither = sum(1 for e in complete_pairs.values() if not e["preferred"] and not e["dispreferred"])

    preferred_rate = (both + preferred_only) / n_items
    dispreferred_rate = (both + dispreferred_only) / n_items
    delta = preferred_rate - dispreferred_rate

    discordant = preferred_only + dispreferred_only
    if discordant == 0:
        # No discordant pairs: preferred and dispreferred agreed on every
        # item, so there is nothing to test — not evidence of "no effect".
        p_value = 1.0
    else:
        result = stats.binomtest(preferred_only, discordant, 0.5, alternative="two-sided")
        p_value = float(result.pvalue)

    return PairedSourceComparison(
        n_items=n_items,
        preferred_rate=preferred_rate,
        dispreferred_rate=dispreferred_rate,
        delta=delta,
        both=both,
        preferred_only=preferred_only,
        dispreferred_only=dispreferred_only,
        neither=neither,
        p_value=p_value,
    )

"""Plausible false-answer (foil) construction for Known-Correct items.

Foils are used as the false, conflicting evidence answer for KC conflict
trials (C3/C4). Per docs/phase2_research_design.md, foils are sampled from
another PopQA item sharing the same relation, which is the minimum
defensible type-compatibility control (e.g., a "capital of" foil is drawn
from another "capital of" item, so it is at least the right answer type).
No LLM is used to invent foils.
"""

from __future__ import annotations

import dataclasses
import random

from conflict_eval.data.normalize import is_match


@dataclasses.dataclass(frozen=True)
class Foil:
    foil_answer: str
    source_item_id: str
    relation: str
    generation_method: str = "same_relation_sample"


def build_relation_index(items: list[dict]) -> dict[str, list[dict]]:
    """Group items by relation (`prop`) for same-relation foil sampling."""
    index: dict[str, list[dict]] = {}
    for item in items:
        index.setdefault(item["prop"], []).append(item)
    return index


def sample_foil(
    item: dict,
    relation_index: dict[str, list[dict]],
    rng: random.Random,
) -> Foil | None:
    """Deterministically sample a same-relation foil for `item`.

    Returns None if no defensible foil exists (e.g., `item` is the only
    member of its relation, or every same-relation object happens to match
    this item's gold/aliases) — callers must exclude the item rather than
    forcing a mismatched-type foil, per docs/decisions.md.
    """
    relation = item["prop"]
    gold = item["obj"]
    aliases = item.get("aliases", [])

    candidates = [
        other
        for other in relation_index.get(relation, [])
        if other["id"] != item["id"] and not is_match(other["obj"], gold, aliases)
    ]
    if not candidates:
        return None

    # Sort before drawing so the rng draw is deterministic regardless of
    # the input pool's original ordering.
    candidates_sorted = sorted(candidates, key=lambda c: str(c["id"]))
    chosen = rng.choice(candidates_sorted)
    return Foil(
        foil_answer=chosen["obj"],
        source_item_id=str(chosen["id"]),
        relation=relation,
    )

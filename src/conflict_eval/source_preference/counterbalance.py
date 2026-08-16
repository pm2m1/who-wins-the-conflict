"""A/B vs. B/A presentation-order counterbalancing.

Presenting every source pair in both orders prevents position preference
(e.g., a model systematically favoring whichever source is listed first)
from being mistaken for genuine source preference
(docs/phase2_research_design.md, "Source-preference calibration").
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Presentation:
    source_a: str
    source_b: str
    displayed_source_1: str
    displayed_source_2: str
    presentation_order: str  # "AB" or "BA"


def expand_pair_to_presentations(source_a: str, source_b: str) -> list[Presentation]:
    """Build both presentation orders for one unordered pair. The pair
    identity (source_a, source_b) is preserved on both records so trials
    from the same underlying pair can be grouped later, regardless of
    which order was displayed.
    """
    return [
        Presentation(
            source_a=source_a,
            source_b=source_b,
            displayed_source_1=source_a,
            displayed_source_2=source_b,
            presentation_order="AB",
        ),
        Presentation(
            source_a=source_a,
            source_b=source_b,
            displayed_source_1=source_b,
            displayed_source_2=source_a,
            presentation_order="BA",
        ),
    ]


def expand_pairs_to_presentations(pairs: list[tuple[str, str]]) -> list[Presentation]:
    presentations: list[Presentation] = []
    for source_a, source_b in pairs:
        presentations.extend(expand_pair_to_presentations(source_a, source_b))
    return presentations

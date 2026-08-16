"""Conflict-specific parametric preference margin.

Per docs/phase2_research_design.md ("Primary conflict trials"), there is
deliberately no single universal gold-vs-foil margin used for every trial.
The margin is always defined between the trial's `memory_answer` (the
model's own baseline answer, or gold for KW items) and the specific
`conflicting_context_answer` relevant to that trial (a foil for KC items,
gold for KW items).
"""

from __future__ import annotations


def compute_parametric_margin(memory_score: float, conflicting_score: float) -> float:
    """B(q) = score(memory_answer) - score(conflicting_context_answer).

    Both scores must be length-normalized log probabilities computed under
    the identical no-evidence prompt prefix (docs/methodology.md, section
    4) — this function does not itself enforce that; callers are
    responsible for passing in comparable scores.
    """
    return memory_score - conflicting_score

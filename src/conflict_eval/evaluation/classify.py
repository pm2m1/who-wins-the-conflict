"""Primary outcome classification.

Every generated response is classified into exactly one of: gold, memory,
context, other, uncertain, manual_review (docs/phase2_research_design.md,
"Primary outcome classification"). On agreement trials, memory and
context labels can coincide; causal attribution to source is never
inferred from those trials (see metrics.py, which restricts CAR/HOR/COR to
conflict trials only).
"""

from __future__ import annotations

from conflict_eval.evaluation.answer_match import is_match

VALID_CLASSES = ("gold", "memory", "context", "other", "uncertain", "manual_review")


def classify_answer(
    parsed_answer: str | None,
    decision: str | None,
    malformed: bool,
    gold_answer: str,
    gold_aliases: list[str],
    memory_answer: str,
    context_answer: str | None,
) -> str:
    if malformed or parsed_answer is None:
        return "manual_review"
    if decision == "uncertain":
        return "uncertain"
    if decision != "answer":
        return "manual_review"

    matches_memory = is_match(parsed_answer, memory_answer, [])
    matches_context = context_answer is not None and is_match(parsed_answer, context_answer, [])

    if matches_context and matches_memory:
        # Agreement trial (or memory happens to equal context for this
        # item): the two hypotheses are not separable, so we do not force
        # a causal label. Uniformly reported as "context" — the record's
        # conflict_status field (set independently in
        # experiment/conditions.py) is what downstream metrics use to
        # exclude these trials from CAR/HOR/COR.
        return "context"
    if matches_context:
        return "context"
    if matches_memory:
        return "memory"

    if is_match(parsed_answer, gold_answer, gold_aliases):
        return "gold"
    return "other"


def is_context_adopted(answer_class: str) -> bool:
    return answer_class == "context"


def is_final_correct(parsed_answer: str | None, gold_answer: str, gold_aliases: list[str]) -> bool:
    if parsed_answer is None:
        return False
    return is_match(parsed_answer, gold_answer, gold_aliases)

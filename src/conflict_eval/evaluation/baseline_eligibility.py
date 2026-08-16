"""Baseline-screening eligibility rules for the KC/KW knowledge groups.

Centralizes "is this parsed baseline (no-evidence) response usable as a
KC/KW memory candidate at all?" — deliberately separate from
evaluation/classify.py, which classifies C0-C4 EXPERIMENTAL trial
responses (a different question, applied after evidence has been shown).

This exists because a 20-item real-model smoke screen
(Qwen/Qwen2.5-3B-Instruct) found that baseline abstentions ("Answer:
uncertain", sometimes with an inconsistent "Decision: answer") were being
treated as valid KW memory answers, producing meaningless parametric
margins such as score("uncertain") - score(gold). See docs/decisions.md,
"Baseline abstentions must not become KC/KW memory candidates".
"""

from __future__ import annotations

import dataclasses

from conflict_eval.data.normalize import normalize_answer

# Explicit uncertainty/refusal markers, checked even when the model
# (inconsistently) emits "Decision: answer" alongside one of these as the
# answer text. Deliberately a short, explicit list — not a speculative
# blacklist. Anything not on this list, but that still fails the
# clean-candidate checks below, is routed to manual_review rather than
# guessed at (docs/phase2_research_design.md).
_UNCERTAINTY_PHRASES = (
    "uncertain",
    "unknown",
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
)
_UNCERTAINTY_MARKERS = frozenset(normalize_answer(phrase) for phrase in _UNCERTAINTY_PHRASES)

MAX_CLEAN_ANSWER_WORDS = 6


@dataclasses.dataclass(frozen=True)
class BaselineEligibility:
    eligible: bool
    reason: str | None  # populated when not eligible; None when eligible


def classify_baseline_eligibility(
    parsed_answer: str | None, decision: str | None, malformed: bool
) -> BaselineEligibility:
    """Decide whether a baseline response is eligible to become a KC/KW
    memory candidate at all — independent of whether it matches gold.

    This check must run, and must fail, BEFORE a response is allowed to
    become KC or KW: an abstention that happens to restate the gold
    answer (or a wrong-looking answer) is still not a usable parametric
    memory candidate, so KC/KW eligibility, not just KW cleanliness, is
    gated here.
    """
    if malformed or parsed_answer is None:
        return BaselineEligibility(False, "malformed")
    if decision != "answer":
        # Covers "Decision: uncertain" and any other non-"answer" value
        # parse_response accepts.
        return BaselineEligibility(False, "baseline_uncertain")

    if normalize_answer(parsed_answer) in _UNCERTAINTY_MARKERS:
        # The model emitted "Decision: answer" but the answer text
        # itself is an explicit abstention/refusal — the real-model
        # failure mode this module exists to catch.
        return BaselineEligibility(False, "baseline_uncertain")

    return BaselineEligibility(True, None)


def is_clean_factual_candidate(parsed_answer: str) -> bool:
    """Whether an eligible (Decision == "answer", not an explicit
    uncertainty marker) parsed answer is additionally clean enough to
    serve as a KW memory candidate: short, unambiguous, not a list of
    alternatives. Borderline cases should be routed to manual_review by
    the caller rather than forced into KW.

    This does not attempt to catch every conceivable hedge or refusal
    phrasing (docs/decisions.md) — only the specific markers in
    `classify_baseline_eligibility` are checked explicitly; anything else
    is judged solely on this shape-based cleanliness check.

    A real 7B PopQA screen produced "Eric Paul Friedmann and Christophe
    Beck" as a screenwriter answer — a two-name conjunction that the
    original comma/" or " checks did not catch, and which is not a single
    factual candidate (docs/decisions.md, "Restrict primary trials to
    defensible conflicts"). " and " is therefore also rejected as a
    word-level conjunction/list marker; this is a small, explicit
    addition, not a broad speculative blacklist.
    """
    if len(parsed_answer.split()) > MAX_CLEAN_ANSWER_WORDS:
        return False
    if "," in parsed_answer:
        return False
    lowered = parsed_answer.lower()
    if " or " in lowered:
        return False
    return " and " not in lowered

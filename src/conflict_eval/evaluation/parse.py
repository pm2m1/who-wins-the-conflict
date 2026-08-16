"""Deterministic parsing of the fixed three-field response format
(Answer / Decision / Confidence) requested by prompts/baseline.txt.

Malformed responses are marked, not forced into a class — downstream
classification routes them to `manual_review` (evaluation/classify.py).
"""

from __future__ import annotations

import dataclasses
import re

_ANSWER_RE = re.compile(r"Answer:\s*(.+)")
# Line-anchored and exact: the ENTIRE Decision line (surrounding
# whitespace aside) must be exactly "answer" or "uncertain" — a prefix
# match here previously accepted trailing junk such as
# "Decision: answer | certain" or "Decision: answer blah" as decision =
# "answer", which a real Qwen2.5-7B-Instruct generation reproduced
# (docs/decisions.md, "Decision output format made strict"). MULTILINE so
# ^/$ anchor to individual lines within the full multi-line response,
# not just the whole string.
_DECISION_RE = re.compile(r"^Decision:\s*(answer|uncertain)\s*$", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"Confidence:\s*(-?\d+)")


@dataclasses.dataclass(frozen=True)
class ParsedResponse:
    raw: str
    answer: str | None
    decision: str | None
    confidence: int | None
    malformed: bool


def parse_response(raw_text: str) -> ParsedResponse:
    answer_match = _ANSWER_RE.search(raw_text)
    decision_match = _DECISION_RE.search(raw_text)
    confidence_match = _CONFIDENCE_RE.search(raw_text)

    answer = answer_match.group(1).strip() if answer_match else None
    if answer == "":
        answer = None
    decision = decision_match.group(1).lower() if decision_match else None

    confidence = None
    if confidence_match:
        confidence = max(0, min(100, int(confidence_match.group(1))))

    # A response is malformed if we could not locate a non-empty answer or
    # a valid decision field — confidence is exploratory only (per
    # docs/phase2_research_design.md) and its absence alone does not make
    # a response malformed.
    malformed = answer is None or decision is None

    return ParsedResponse(
        raw=raw_text, answer=answer, decision=decision, confidence=confidence, malformed=malformed
    )

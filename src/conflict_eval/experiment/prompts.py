"""Render the full experimental prompt for one trial.

Uses the single canonical template (prompts/baseline.txt) for every
condition, C0 through C4 — the only thing that changes between conditions
is what gets substituted into {evidence_or_none}, per
docs/phase2_research_design.md ("Prompt design").
"""

from __future__ import annotations

# The literal field label that precedes the answer value in the fixed
# response format requested by prompts/baseline.txt ("Answer: <short
# answer>\nDecision: ...\nConfidence: ..."). Candidate-answer scoring
# (models/*.py: score_candidate) must include this text in the scoring
# prefix, ahead of the candidate itself — otherwise the candidate would be
# scored as if it were the literal first thing the assistant says, not as
# the value of the Answer field the model is instructed to produce
# (docs/decisions.md, "Scoring prefix must include the Answer: field
# label").
ANSWER_FIELD_PREFIX = "Answer: "


def render_experiment_prompt(template: str, question: str, evidence_text: str | None) -> str:
    evidence_or_none = evidence_text if evidence_text is not None else "None"
    return template.format(question=question, evidence_or_none=evidence_or_none)

"""Controlled evidence rendering.

Renders the fixed evidence template (prompts/evidence.txt) so that
changing source identity changes only {source} and changing factual
content changes only {asserted_answer} — no other text differs between
conditions. Evidence is never LLM-generated (docs/phase2_research_design.md).
"""

from __future__ import annotations


def render_evidence(template: str, source: str, question: str, asserted_answer: str) -> str:
    return template.format(source=source, question=question, asserted_answer=asserted_answer)

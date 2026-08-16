"""Verifies the committed baseline prompt matches the strict Decision
output format required by evaluation/parse.py's line-anchored regex
(docs/decisions.md, "Decision output format made strict").

A real Qwen2.5-7B-Instruct generation under the old prompt wording
("Decision: answer | uncertain") partially reproduced the pipe syntax
literally ("Decision: answer | certain"); the prompt was made explicit
about writing exactly one word.
"""

from __future__ import annotations

from pathlib import Path

BASELINE_PROMPT = Path("prompts/baseline.txt").read_text(encoding="utf-8")


def test_prompt_no_longer_contains_the_pipe_syntax_decision_line():
    assert "Decision: answer | uncertain" not in BASELINE_PROMPT


def test_prompt_instructs_exactly_one_decision_word():
    assert "exactly one word" in BASELINE_PROMPT
    assert "answer or uncertain" in BASELINE_PROMPT
    assert "| symbol" in BASELINE_PROMPT or "the | symbol" in BASELINE_PROMPT


def test_prompt_still_requests_the_three_field_structure():
    assert "Answer: <short answer>" in BASELINE_PROMPT
    assert "Decision:" in BASELINE_PROMPT
    assert "Confidence: <integer from 0 to 100>" in BASELINE_PROMPT


def test_prompt_does_not_request_chain_of_thought():
    lowered = BASELINE_PROMPT.lower()
    for forbidden in ("explain", "reasoning", "step by step", "think through"):
        assert forbidden not in lowered

"""Deterministic mock model adapter.

Used for unit tests and the labeled synthetic dry run described in
docs/decisions.md and README.md. DummyModelAdapter never loads a real
model and never makes a network call. Its outputs must never be written
into results/ as if they were real pilot data — dry-run output paths are
kept clearly separate (see scripts/build_pilot.py and README.md).
"""

from __future__ import annotations

import hashlib
import re

from conflict_eval.data.normalize import normalize_answer
from conflict_eval.models.base import BaseModelAdapter, GenerationConfig, Message
from conflict_eval.scoring.sequence_logprob import (
    DetailedScore,
    ScoredSequence,
    score_from_token_logprobs,
)

_QUESTION_RE = re.compile(r"Question:\s*(.+)")
_ASSERTED_ANSWER_RE = re.compile(r'is "([^"]+)"')
_CALIBRATION_RE = re.compile(r"Source 1:\s*(.+)\nSource 2:\s*(.+)")


class DummyModelAdapter(BaseModelAdapter):
    def __init__(
        self,
        seed: int = 0,
        context_adoption_rate: float = 0.5,
        model_id: str = "dummy",
    ) -> None:
        self.model_id = model_id
        self.model_revision = "dummy"
        # No real model is loaded, so there is no requested/resolved
        # revision distinction to make — both are explicitly None rather
        # than a fabricated value (docs/decisions.md, "Exact model
        # revision recording").
        self.requested_revision = None
        self.resolved_revision = None
        self.seed = seed
        self.context_adoption_rate = context_adoption_rate

    def _extract_question(self, prompt_text: str) -> str:
        match = _QUESTION_RE.search(prompt_text)
        return match.group(1).strip() if match else prompt_text.strip()

    def _extract_asserted_answer(self, prompt_text: str) -> str | None:
        match = _ASSERTED_ANSWER_RE.search(prompt_text)
        return match.group(1) if match else None

    def _memory_answer_for(self, question: str) -> str:
        # A fixed pseudo-answer derived only from the question (and the
        # adapter seed), so repeated calls for the same question return
        # the same "memory" answer across C0-C4 regardless of condition.
        digest = hashlib.sha256(f"{self.seed}:{question}".encode()).hexdigest()
        return f"dummy-answer-{digest[:6]}"

    def _prompt_text(self, messages: list[Message]) -> str:
        return messages[-1]["content"]

    def generate(self, messages: list[Message], generation_config: GenerationConfig) -> str:
        prompt_text = self._prompt_text(messages)

        # Source-calibration prompts (prompts/source_calibration.txt) ask
        # for a structured Choice: 1|2, not an Answer/Decision/Confidence
        # triple. Recognizing this format lets the dummy adapter also
        # exercise the calibration pathway during a dry run, not only the
        # baseline/evidence experimental prompts.
        if _CALIBRATION_RE.search(prompt_text):
            digest = hashlib.sha256(f"{self.seed}:{prompt_text}".encode()).hexdigest()
            choice = 1 if int(digest[:8], 16) % 2 == 0 else 2
            return f"Choice: {choice}"

        question = self._extract_question(prompt_text)
        asserted = self._extract_asserted_answer(prompt_text)
        memory_answer = self._memory_answer_for(question)

        if asserted is None:
            answer = memory_answer
        else:
            digest = hashlib.sha256(f"{self.seed}:{prompt_text}".encode()).hexdigest()
            draw = int(digest[:8], 16) / 0xFFFFFFFF
            answer = asserted if draw < self.context_adoption_rate else memory_answer

        return f"Answer: {answer}\nDecision: answer\nConfidence: 70"

    def _token_logprobs_for_candidate(
        self, messages: list[Message], candidate_text: str, answer_prefix: str
    ) -> list[float]:
        prompt_text = self._prompt_text(messages)
        question = self._extract_question(prompt_text)
        memory_answer = self._memory_answer_for(question)

        # The dummy's "memory" answer always scores higher than any other
        # candidate, with small deterministic jitter, so the pipeline has
        # a well-defined and testable margin structure to exercise.
        # answer_prefix is folded into the jitter digest (not into
        # token_count) purely so the real adapter's answer_prefix
        # parameter is exercised end to end by the dummy too, per the
        # shared BaseModelAdapter interface.
        token_count = max(1, len(candidate_text.split()))
        base = -1.0 if normalize_answer(candidate_text) == normalize_answer(memory_answer) else -3.0
        digest = hashlib.sha256(
            f"{self.seed}:{answer_prefix}:{prompt_text}:{candidate_text}".encode()
        ).hexdigest()
        jitter = (int(digest[:4], 16) / 0xFFFF) * 0.1
        return [base - jitter] * token_count

    def score_candidate(
        self, messages: list[Message], candidate_text: str, answer_prefix: str = ""
    ) -> ScoredSequence:
        token_logprobs = self._token_logprobs_for_candidate(messages, candidate_text, answer_prefix)
        return score_from_token_logprobs(token_logprobs)

    def score_candidate_detailed(
        self, messages: list[Message], candidate_text: str, answer_prefix: str = ""
    ) -> DetailedScore:
        token_logprobs = self._token_logprobs_for_candidate(messages, candidate_text, answer_prefix)
        # The dummy has no real tokenizer; whitespace-split words are used
        # as a SYNTHETIC stand-in for tokens, purely so diagnose-score has
        # something to display in dry-run mode. Never treat these as real
        # subword tokens.
        answer_tokens = candidate_text.split() or [candidate_text]
        return DetailedScore(
            scored=score_from_token_logprobs(token_logprobs),
            answer_tokens=answer_tokens,
            token_logprobs=token_logprobs,
        )

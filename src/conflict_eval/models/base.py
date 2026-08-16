"""Model adapter interface.

Every model used by this project (real Hugging Face causal LMs, or the
deterministic dummy used in tests/dry-run) implements this interface, so
the rest of the pipeline (experiment runner, scoring, calibration) never
depends on a specific model library directly.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

from conflict_eval.scoring.sequence_logprob import DetailedScore, ScoredSequence

Message = dict[str, str]  # {"role": "user"|"system"|"assistant", "content": str}


@dataclasses.dataclass
class GenerationConfig:
    do_sample: bool = False
    max_new_tokens: int = 32
    num_beams: int = 1

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class BaseModelAdapter(abc.ABC):
    """Minimal interface required by the experiment pipeline.

    `model_id` and `model_revision` are stored on every result record for
    reproducibility (docs/methodology.md, "Reproducibility notes").

    `model_revision` is the field existing result records already use; it
    holds the best available revision identifier — the resolved commit SHA
    when one could be determined, otherwise whatever `requested_revision`
    was (which may itself be `None`). Adapters that can distinguish the
    two also expose `requested_revision` (what configuration asked for)
    and `resolved_revision` (the concrete snapshot actually loaded, or
    `None` if that could not be determined) — see
    `docs/decisions.md`, "Exact model revision recording".
    """

    model_id: str
    model_revision: str | None
    requested_revision: str | None
    resolved_revision: str | None

    @abc.abstractmethod
    def generate(self, messages: list[Message], generation_config: GenerationConfig) -> str:
        """Deterministic (or near-deterministic, per generation_config)
        chat generation. Returns raw decoded text only — parsing into
        Answer/Decision/Confidence happens in evaluation/parse.py.
        """

    @abc.abstractmethod
    def score_candidate(
        self, messages: list[Message], candidate_text: str, answer_prefix: str = ""
    ) -> ScoredSequence:
        """Teacher-forced length-normalized log-probability score of
        `candidate_text` as the continuation of the chat-formatted
        `messages` prefix. See docs/methodology.md, section 3.

        `answer_prefix` is literal text the model is expected to produce,
        as part of its own turn, immediately before `candidate_text` —
        for this project's fixed response format that is the field label
        "Answer: " (see experiment/prompts.py:ANSWER_FIELD_PREFIX). It is
        appended to the scoring prefix but excluded from the score itself:
        without it, `candidate_text` would be scored as if it were the
        literal first thing the assistant says, which is a different (and
        for a model instructed to always start with "Answer: ", far less
        meaningful) quantity than "the probability the Answer field takes
        this value."
        """

    @abc.abstractmethod
    def score_candidate_detailed(
        self, messages: list[Message], candidate_text: str, answer_prefix: str = ""
    ) -> DetailedScore:
        """Same computation as score_candidate, but also returns the
        decoded answer tokens and their individual log probabilities.
        Diagnostic only — see scoring/sequence_logprob.py:DetailedScore.
        """

"""Teacher-forced sequence log-probability scoring.

Implements the scoring described in docs/methodology.md, section 3:

    score(a | prompt) = (1/N) * sum_{t=1..N} log P(a_t | prompt, a_<t))

The two pieces that are easy to get subtly wrong — locating exactly where
answer tokens start inside a tokenized (prefix + answer) sequence, and
normalizing a raw summed log probability by answer length — are isolated
here as pure functions so they can be unit tested without loading a real
model (tests/test_logprob.py).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class ScoredSequence:
    logprob_sum: float
    token_count: int
    logprob_normalized: float


@dataclasses.dataclass(frozen=True)
class DetailedScore:
    """Token-level view of a ScoredSequence, for diagnostics only.

    Not written to result records (docs/methodology.md: "Store only
    scalar quantities necessary for analysis") — used exclusively by the
    `diagnose-score` CLI command to make prefix/boundary/masking behavior
    inspectable on a real model.
    """

    scored: ScoredSequence
    answer_tokens: list[str]
    token_logprobs: list[float]


def normalize_logprob(logprob_sum: float, token_count: int) -> float:
    if token_count <= 0:
        raise ValueError("token_count must be positive to normalize a log probability")
    return logprob_sum / token_count


def score_from_token_logprobs(token_logprobs: Sequence[float]) -> ScoredSequence:
    """Build a ScoredSequence from per-answer-token log probabilities.

    `token_logprobs[i]` must already be P(a_i | prompt, a_<i) for the i-th
    answer token only — prompt-token log probabilities must be excluded by
    the caller before this function is used.
    """
    if len(token_logprobs) == 0:
        raise ValueError("Cannot score an empty answer (zero tokens)")
    logprob_sum = float(sum(token_logprobs))
    token_count = len(token_logprobs)
    return ScoredSequence(
        logprob_sum=logprob_sum,
        token_count=token_count,
        logprob_normalized=normalize_logprob(logprob_sum, token_count),
    )


def answer_token_boundary(prefix_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """Return the index in `full_ids` where answer tokens begin.

    `full_ids` is the tokenization of (prefix_text + answer_text) as a
    single string, which is not guaranteed to equal
    tokenize(prefix_text) + tokenize(answer_text) token-for-token — a
    leading whitespace token can merge into the first answer token
    depending on the tokenizer. We therefore find the longest common
    prefix between `prefix_ids` and `full_ids` directly, rather than
    assuming `len(prefix_ids)` is always correct.
    """
    boundary = 0
    for p, f in zip(prefix_ids, full_ids):
        if p != f:
            break
        boundary += 1
    if boundary == len(full_ids):
        raise ValueError(
            "Tokenized (prefix + answer) is identical to tokenized prefix — "
            "the answer contributed no tokens; check for an empty candidate answer."
        )
    return boundary

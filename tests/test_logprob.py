import pytest

from conflict_eval.scoring.sequence_logprob import (
    answer_token_boundary,
    normalize_logprob,
    score_from_token_logprobs,
)


def test_normalize_logprob_divides_by_token_count():
    assert normalize_logprob(-6.0, 3) == -2.0


def test_normalize_logprob_rejects_zero_tokens():
    with pytest.raises(ValueError):
        normalize_logprob(-6.0, 0)


def test_score_from_token_logprobs_sums_and_normalizes():
    scored = score_from_token_logprobs([-1.0, -2.0, -3.0])
    assert scored.token_count == 3
    assert scored.logprob_sum == -6.0
    assert scored.logprob_normalized == -2.0


def test_score_from_token_logprobs_rejects_empty_answer():
    with pytest.raises(ValueError):
        score_from_token_logprobs([])


def test_answer_token_boundary_simple_case():
    # prefix = [1, 2, 3], full = prefix + answer tokens [4, 5]
    prefix_ids = [1, 2, 3]
    full_ids = [1, 2, 3, 4, 5]
    assert answer_token_boundary(prefix_ids, full_ids) == 3


def test_answer_token_boundary_handles_whitespace_merge():
    # A tokenizer can merge the answer's leading whitespace into a
    # different token id than the standalone prefix would suggest — the
    # boundary must be found by comparing actual token ids, not just
    # trusting len(prefix_ids).
    prefix_ids = [1, 2, 3]
    full_ids = [1, 2, 3, 99, 5]  # token 4 became 99 once merged with " answer"
    assert answer_token_boundary(prefix_ids, full_ids) == 3


def test_answer_token_boundary_rejects_empty_answer_contribution():
    prefix_ids = [1, 2, 3]
    full_ids = [1, 2, 3]
    with pytest.raises(ValueError):
        answer_token_boundary(prefix_ids, full_ids)


def test_masking_excludes_prompt_tokens_from_score():
    # Simulate per-position log-probabilities for a 5-token full sequence
    # (3 prompt tokens + 2 answer tokens). Only the answer-token log
    # probabilities should enter the score.
    prefix_ids = [10, 11, 12]
    full_ids = [10, 11, 12, 20, 21]
    boundary = answer_token_boundary(prefix_ids, full_ids)

    # log P(token_t | ..., token_<t) is conventionally read from the
    # logits at position t-1; here we hand-construct that mapping.
    per_position_logprob_of_next_token = {2: -0.5, 3: -0.7, 4: -0.9}
    token_logprobs = [
        per_position_logprob_of_next_token[t - 1] for t in range(boundary, len(full_ids))
    ]
    scored = score_from_token_logprobs(token_logprobs)
    assert scored.token_count == 2
    assert scored.logprob_sum == pytest.approx(-0.5 + -0.7)

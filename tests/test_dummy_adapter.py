"""Tests for the deterministic mock adapter used in dry runs
(docs/decisions.md). Never exercises a real model or the network.
"""

from conflict_eval.models.base import GenerationConfig
from conflict_eval.models.dummy import DummyModelAdapter

GEN_CONFIG = GenerationConfig()


def test_generate_is_deterministic_given_same_seed():
    adapter_a = DummyModelAdapter(seed=1)
    adapter_b = DummyModelAdapter(seed=1)
    messages = [{"role": "user", "content": "Question: What is the capital of France?"}]
    assert adapter_a.generate(messages, GEN_CONFIG) == adapter_b.generate(messages, GEN_CONFIG)


def test_generate_returns_well_formed_response():
    adapter = DummyModelAdapter(seed=1)
    messages = [{"role": "user", "content": "Question: What is the capital of France?"}]
    raw = adapter.generate(messages, GEN_CONFIG)
    assert "Answer:" in raw
    assert "Decision:" in raw
    assert "Confidence:" in raw


def test_memory_answer_consistent_across_conditions_for_same_question():
    adapter = DummyModelAdapter(seed=1)
    # Regardless of condition/evidence, the underlying "memory" answer for
    # a fixed question and seed must stay fixed.
    first = adapter._memory_answer_for("What is the capital of France?")
    second = adapter._memory_answer_for("What is the capital of France?")
    assert first == second


def test_score_candidate_prefers_memory_answer():
    adapter = DummyModelAdapter(seed=1)
    messages = [{"role": "user", "content": "Question: What is the capital of France?"}]
    memory_answer = adapter._memory_answer_for("What is the capital of France?")

    memory_score = adapter.score_candidate(messages, memory_answer)
    other_score = adapter.score_candidate(messages, "some other answer")
    assert memory_score.logprob_normalized > other_score.logprob_normalized


def test_generate_answers_source_calibration_prompts_with_a_choice():
    adapter = DummyModelAdapter(seed=1)
    messages = [
        {
            "role": "user",
            "content": (
                "For answering general factual questions, which of these two sources "
                "would you prefer to rely on if both were available?\n\n"
                "Source 1: Wikipedia\nSource 2: a personal blog\n\n"
                "Respond using exactly this format:\n\nChoice: 1 | 2\n"
            ),
        }
    ]
    raw = adapter.generate(messages, GEN_CONFIG)
    assert raw in ("Choice: 1", "Choice: 2")


def test_score_candidate_accepts_and_uses_answer_prefix():
    # answer_prefix must actually be threaded into the computation (see
    # models/base.py: score_candidate's answer_prefix parameter) rather
    # than silently ignored — otherwise a real bug in HFCausalAdapter
    # (scoring the candidate as if it were the assistant's literal first
    # tokens, omitting the "Answer: " field label) could go unnoticed
    # because the interface would still "work" without it.
    adapter = DummyModelAdapter(seed=1)
    messages = [{"role": "user", "content": "Question: What is the capital of France?"}]
    without_prefix = adapter.score_candidate(messages, "Paris", answer_prefix="")
    with_prefix = adapter.score_candidate(messages, "Paris", answer_prefix="Answer: ")
    assert without_prefix.logprob_normalized != with_prefix.logprob_normalized


def test_score_candidate_detailed_matches_score_candidate():
    adapter = DummyModelAdapter(seed=1)
    messages = [{"role": "user", "content": "Question: What is the capital of France?"}]
    plain = adapter.score_candidate(messages, "Paris", answer_prefix="Answer: ")
    detailed = adapter.score_candidate_detailed(messages, "Paris", answer_prefix="Answer: ")
    assert detailed.scored == plain
    assert len(detailed.answer_tokens) == len(detailed.token_logprobs)


def test_context_adoption_rate_parameter_changes_behavior():
    messages = [
        {
            "role": "user",
            "content": (
                'Evidence:\nSource: Wikipedia\n\nStatement:\nThe answer to the question '
                '"Q" is "asserted-value".\n\nQuestion: Q'
            ),
        }
    ]
    always_context = DummyModelAdapter(seed=1, context_adoption_rate=1.0)
    never_context = DummyModelAdapter(seed=1, context_adoption_rate=0.0)

    assert "asserted-value" in always_context.generate(messages, GEN_CONFIG)
    assert "asserted-value" not in never_context.generate(messages, GEN_CONFIG)

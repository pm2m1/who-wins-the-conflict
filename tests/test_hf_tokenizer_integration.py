"""Real-tokenizer validation of chat-template rendering and answer-token
boundary/masking logic (models/hf_causal.py, scoring/sequence_logprob.py),
plus real-world validation of the model-revision resolution mechanism
(docs/decisions.md, "Exact model revision recording").

This downloads ONLY the Qwen2.5-7B-Instruct tokenizer and config files (a
few MB combined — vocab/merges/tokenizer_config/config.json, not the
~15GB model weights) so the exact chat template, BPE tokenizer, and
commit-hash metadata this project targets can be exercised for real,
without the compute cost of loading and running the model itself.

Skipped by default per project convention (no test triggers a network call
or download unless explicitly opted in): set
CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1 to run it.
"""

from __future__ import annotations

import os

import pytest

from conflict_eval.experiment.prompts import ANSWER_FIELD_PREFIX
from conflict_eval.scoring.sequence_logprob import answer_token_boundary

pytestmark = pytest.mark.skipif(
    os.environ.get("CONFLICT_EVAL_RUN_TOKENIZER_TESTS") != "1",
    reason="Downloads a real tokenizer from Hugging Face; opt in with CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1",
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def _render_prefix(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "You are answering a factual question.\n\nEvidence:\nNone\n\n"
                f"Question:\n{question}\n\nRespond using exactly this format:\n\n"
                "Answer: <short answer>\nDecision: <answer or uncertain>\n"
                "Confidence: <integer from 0 to 100>\n"
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def test_chat_template_renders_and_ends_with_assistant_marker(tokenizer):
    prefix = _render_prefix(tokenizer, "What is the capital of France?")
    assert prefix.strip() != ""
    # Qwen2.5's ChatML template ends the generation prompt with the
    # assistant turn opener and no trailing assistant content.
    assert prefix.rstrip().endswith("<|im_start|>assistant")


def test_boundary_recovers_exact_single_token_answer(tokenizer):
    prefix = _render_prefix(tokenizer, "What is the capital of France?") + ANSWER_FIELD_PREFIX
    candidate = "Paris"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + candidate, add_special_tokens=False)["input_ids"]

    boundary = answer_token_boundary(prefix_ids, full_ids)
    answer_ids = full_ids[boundary:]
    decoded = tokenizer.decode(answer_ids)

    # The decoded answer-token span should reconstruct the candidate text
    # (allowing for a possible leading space merged in at the boundary).
    assert decoded.strip() == candidate


def test_boundary_recovers_exact_multi_token_answer(tokenizer):
    prefix = _render_prefix(tokenizer, "What is the capital of France?") + ANSWER_FIELD_PREFIX
    candidate = "The United Kingdom of Great Britain"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + candidate, add_special_tokens=False)["input_ids"]

    boundary = answer_token_boundary(prefix_ids, full_ids)
    assert len(full_ids) - boundary > 1  # genuinely multi-token
    decoded = tokenizer.decode(full_ids[boundary:])
    assert decoded.strip() == candidate


def test_boundary_handles_punctuation_candidate(tokenizer):
    prefix = _render_prefix(tokenizer, "What is the capital of France?") + ANSWER_FIELD_PREFIX
    candidate = "Washington, D.C."
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + candidate, add_special_tokens=False)["input_ids"]

    boundary = answer_token_boundary(prefix_ids, full_ids)
    decoded = tokenizer.decode(full_ids[boundary:])
    assert decoded.strip() == candidate


def test_prefix_tokenization_is_deterministic(tokenizer):
    prefix = _render_prefix(tokenizer, "What is the capital of France?") + ANSWER_FIELD_PREFIX
    ids_a = tokenizer(prefix + "Paris", add_special_tokens=False)["input_ids"]
    ids_b = tokenizer(prefix + "Paris", add_special_tokens=False)["input_ids"]
    assert ids_a == ids_b


def test_candidate_order_does_not_affect_independently_computed_tokenization(tokenizer):
    # Scoring candidate A then candidate B (or vice versa) must tokenize
    # each independently against the same prefix — order of evaluation
    # must not leak state between the two candidate encodings.
    prefix = _render_prefix(tokenizer, "What is the capital of France?") + ANSWER_FIELD_PREFIX
    paris_first = tokenizer(prefix + "Paris", add_special_tokens=False)["input_ids"]
    london_first = tokenizer(prefix + "London", add_special_tokens=False)["input_ids"]
    paris_second = tokenizer(prefix + "Paris", add_special_tokens=False)["input_ids"]
    assert paris_first == paris_second
    assert paris_first != london_first


def test_answer_prefix_changes_real_tokenization_at_the_boundary(tokenizer):
    # This is the direct real-tokenizer confirmation of the bug fixed in
    # models/hf_causal.py (docs/decisions.md, "Scoring prefix must include
    # the Answer: field label"): scoring "Paris" directly after the bare
    # chat-template marker happens under a DIFFERENT (shorter, unlabeled)
    # prefix than scoring it after "...assistant\nAnswer: ".
    bare_prefix = _render_prefix(tokenizer, "What is the capital of France?")
    with_field_label = bare_prefix + ANSWER_FIELD_PREFIX

    bare_prefix_ids = tokenizer(bare_prefix, add_special_tokens=False)["input_ids"]
    labeled_prefix_ids = tokenizer(with_field_label, add_special_tokens=False)["input_ids"]

    # Appending the literal "Answer: " field label must add real tokens
    # to the scoring prefix — this is the whole point of the fix, and
    # confirms it is not silently a no-op against the real tokenizer.
    assert len(labeled_prefix_ids) > len(bare_prefix_ids)

    bare_ids = tokenizer(bare_prefix + "Paris", add_special_tokens=False)["input_ids"]
    labeled_ids = tokenizer(with_field_label + "Paris", add_special_tokens=False)["input_ids"]
    bare_boundary = answer_token_boundary(bare_prefix_ids, bare_ids)
    labeled_boundary = answer_token_boundary(labeled_prefix_ids, labeled_ids)

    # Regardless of which prefix is used, boundary detection still
    # recovers exactly "Paris" as the answer span — the fix changes what
    # context precedes the candidate, not whether masking still works.
    assert tokenizer.decode(bare_ids[bare_boundary:]).strip() == "Paris"
    assert tokenizer.decode(labeled_ids[labeled_boundary:]).strip() == "Paris"


def test_config_exposes_resolved_commit_hash():
    # Real-world confirmation of the mechanism HFCausalAdapter relies on
    # (docs/decisions.md, "Exact model revision recording"): loading only
    # the config (a few KB, not the ~15GB weights) is enough for
    # transformers to populate config._commit_hash with the actual
    # resolved snapshot SHA, extracted from the local HF Hub cache path
    # with no extra network call beyond the config load itself.
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID)
    commit_hash = getattr(config, "_commit_hash", None)
    assert commit_hash is not None
    assert len(commit_hash) >= 7  # short SHAs are at least this long
    assert all(c in "0123456789abcdef" for c in commit_hash)

"""Hugging Face causal LM adapter.

Loads a real model (e.g. meta-llama/Llama-3.1-8B-Instruct or
Qwen/Qwen2.5-7B-Instruct) and implements the generation and teacher-forced
scoring interface defined in models/base.py. This module is never imported
by unit tests in a way that triggers a model download — see tests/ for the
dummy-adapter-only test suite.
"""

from __future__ import annotations

from conflict_eval.models.base import BaseModelAdapter, GenerationConfig, Message
from conflict_eval.scoring.sequence_logprob import (
    DetailedScore,
    ScoredSequence,
    answer_token_boundary,
    score_from_token_logprobs,
)


class HFCausalAdapter(BaseModelAdapter):
    def __init__(
        self,
        hf_model_id: str,
        revision: str | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = hf_model_id
        self.requested_revision = revision
        self._torch = torch

        torch_dtype = getattr(torch, dtype) if dtype else None
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

        # transformers records the exact resolved commit SHA for the
        # snapshot it loaded on `config._commit_hash`, extracted from the
        # local Hugging Face Hub cache path (.../snapshots/<sha>/...) —
        # this requires no extra network call beyond the load that already
        # happened above (docs/decisions.md, "Exact model revision
        # recording"). It is populated even when `revision` was left
        # `None` ("main"), which is the common case that most needs a
        # concrete, citable snapshot identifier. If the attribute is
        # missing or empty (e.g. loading from a plain local directory, or
        # a transformers version that does not expose it), this is left
        # as None rather than guessed.
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or None
        # model_revision is the field already used throughout result
        # records (docs/methodology.md); it now prefers the resolved
        # commit SHA and only falls back to the requested revision string
        # when a concrete SHA could not be determined, so existing
        # consumers of this field keep working without a schema change.
        self.model_revision = self.resolved_revision or self.requested_revision

    def _render_prefix(self, messages: list[Message]) -> str:
        # add_generation_prompt=True appends the assistant-turn-start
        # marker without any assistant content, which is exactly the
        # scoring prefix described in docs/methodology.md, section 3 —
        # before any answer_prefix (e.g. "Answer: ") is appended on top.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, messages: list[Message], generation_config: GenerationConfig) -> str:
        prefix = self._render_prefix(messages)
        inputs = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=generation_config.do_sample,
                max_new_tokens=generation_config.max_new_tokens,
                num_beams=generation_config.num_beams,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _score_tokens(
        self, messages: list[Message], candidate_text: str, answer_prefix: str
    ) -> tuple[list[int], int, list[float]]:
        # answer_prefix (e.g. "Answer: ") is part of the scoring prefix,
        # not the candidate: the candidate is expected to appear
        # immediately after it in the model's own turn, per the
        # Answer/Decision/Confidence response format
        # (docs/methodology.md, section 3; docs/decisions.md, "Scoring
        # prefix must include the Answer: field label"). Scoring the
        # candidate right after the bare chat-template marker instead
        # would measure a different, far less meaningful quantity: the
        # probability the assistant's very first tokens are the candidate
        # text verbatim, with no "Answer:" label at all.
        prefix = self._render_prefix(messages) + answer_prefix
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prefix + candidate_text, add_special_tokens=False)["input_ids"]

        boundary = answer_token_boundary(prefix_ids, full_ids)

        input_ids = self._torch.tensor([full_ids]).to(self.model.device)
        with self._torch.no_grad():
            logits = self.model(input_ids).logits[0]
        log_probs = self._torch.log_softmax(logits.float(), dim=-1)

        # logits[t-1] holds the model's prediction distribution for the
        # token at position t (standard next-token-prediction shift).
        token_logprobs = [
            log_probs[t - 1, full_ids[t]].item() for t in range(boundary, len(full_ids))
        ]
        return full_ids, boundary, token_logprobs

    def score_candidate(
        self, messages: list[Message], candidate_text: str, answer_prefix: str = ""
    ) -> ScoredSequence:
        _, _, token_logprobs = self._score_tokens(messages, candidate_text, answer_prefix)
        return score_from_token_logprobs(token_logprobs)

    def score_candidate_detailed(
        self, messages: list[Message], candidate_text: str, answer_prefix: str = ""
    ) -> DetailedScore:
        full_ids, boundary, token_logprobs = self._score_tokens(
            messages, candidate_text, answer_prefix
        )
        answer_tokens = [
            self.tokenizer.decode([token_id]) for token_id in full_ids[boundary:]
        ]
        return DetailedScore(
            scored=score_from_token_logprobs(token_logprobs),
            answer_tokens=answer_tokens,
            token_logprobs=list(token_logprobs),
        )

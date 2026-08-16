"""Hugging Face causal LM adapter.

Loads a real model (e.g. meta-llama/Llama-3.1-8B-Instruct or
Qwen/Qwen2.5-7B-Instruct) and implements the generation and teacher-forced
scoring interface defined in models/base.py. This module is never imported
by unit tests in a way that triggers a model download — see tests/ for the
dummy-adapter-only test suite.
"""

from __future__ import annotations

from typing import Any

from conflict_eval.models.base import BaseModelAdapter, GenerationConfig, Message
from conflict_eval.scoring.sequence_logprob import (
    DetailedScore,
    ScoredSequence,
    answer_token_boundary,
    score_from_token_logprobs,
)


class ModelRevisionResolutionError(RuntimeError):
    """Raised when a real model load cannot be pinned to a verified, exact
    Hugging Face commit SHA before loading. Real experimental runs must
    fail clearly here rather than silently proceeding with an unknown or
    unverified model snapshot (docs/decisions.md, "Resolve, pin, load,
    record").
    """


def resolve_model_revision(hf_model_id: str, requested_revision: str | None) -> str | None:
    """Resolve `requested_revision` (or "main" if `None`) to the exact,
    immutable Hugging Face commit SHA for `hf_model_id`, via a single
    `huggingface_hub` metadata request — no file/weight download.

    This is the SELECTION mechanism (used to decide which revision to
    pass to `from_pretrained`), not just a post-load inference — see
    docs/decisions.md, "Resolve, pin, load, record". Returns `None`,
    never a guessed value, if Hub access is unavailable (offline, no
    credentials for a gated repo, etc.) or the lookup otherwise fails.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        info = HfApi().model_info(hf_model_id, revision=requested_revision or "main")
    except Exception:  # noqa: BLE001 — any Hub/network failure must degrade to "unavailable", not crash resolution
        return None
    return info.sha


class HFCausalAdapter(BaseModelAdapter):
    def __init__(
        self,
        hf_model_id: str,
        revision: str | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
        max_memory: dict[int | str, str] | None = None,
        require_pinned_revision: bool = True,
    ) -> None:
        """`require_pinned_revision` defaults to True: if the exact commit
        SHA cannot be resolved before loading, construction raises
        `ModelRevisionResolutionError` rather than silently loading an
        unpinned/unverified snapshot. Passing `require_pinned_revision=False`
        is an explicit, documented opt-out (e.g. offline exploratory use)
        that falls back to loading with whatever `revision` was requested
        — the resulting run cannot claim exact revision pinning, and
        `resolved_revision` will be `None` unless a post-load consistency
        check (below) is still able to recover a commit hash.

        `max_memory` is passed through unchanged to
        `AutoModelForCausalLM.from_pretrained` (Accelerate's
        `device_map="auto"` memory-limited placement — e.g.
        `{0: "12.0GiB", "cpu": "5GiB"}`) when configured; `None` (the
        default) preserves prior behavior exactly, since the kwarg is
        then simply not passed at all rather than passed as `None`
        (docs/decisions.md, "Support reproducible model memory limits").
        This does not change quantization, dtype, or introduce disk
        offloading — those remain whatever `dtype`/`device_map` already
        configure.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = hf_model_id
        self.requested_revision = revision
        self.max_memory = max_memory
        self._torch = torch

        # Resolve BEFORE loading, so the revision determines what gets
        # loaded rather than being inferred afterward (docs/decisions.md,
        # "Resolve, pin, load, record").
        self.resolved_revision = resolve_model_revision(hf_model_id, revision)

        if self.resolved_revision is None:
            if require_pinned_revision:
                raise ModelRevisionResolutionError(
                    f"Could not resolve an exact Hugging Face commit SHA for "
                    f"{hf_model_id!r} (requested revision: {revision!r}). A real "
                    "experimental run must not proceed with an unpinned/unverified "
                    "model snapshot. Check Hub access (network connectivity, "
                    "authentication for gated repos) and retry, or construct this "
                    "adapter with require_pinned_revision=False to explicitly "
                    "accept an unpinned load."
                )
            load_revision = revision
        else:
            load_revision = self.resolved_revision

        torch_dtype = getattr(torch, dtype) if dtype else None
        # Both calls use the SAME load_revision value, so the tokenizer
        # and model are guaranteed to come from the identical snapshot.
        # max_memory is deliberately NOT passed to the tokenizer: it is a
        # model-weight placement concern (device_map/max_memory govern
        # where tensors live across GPU/CPU), not a vocabulary/text one.
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, revision=load_revision)

        model_kwargs: dict[str, Any] = {
            "revision": load_revision,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs)
        self.model.eval()

        # Post-load CONSISTENCY CHECK only, not the selection mechanism:
        # transformers exposes the commit hash it actually resolved
        # locally on config._commit_hash. If we deliberately pinned a
        # specific SHA above, this must agree with it — a mismatch would
        # mean something is wrong (a moved ref, a caching bug, ...) and
        # must not be silently accepted.
        post_load_commit_hash = getattr(self.model.config, "_commit_hash", None) or None
        if (
            self.resolved_revision is not None
            and post_load_commit_hash is not None
            and post_load_commit_hash != self.resolved_revision
        ):
            raise ModelRevisionResolutionError(
                f"Post-load commit hash {post_load_commit_hash!r} does not match "
                f"the resolved revision {self.resolved_revision!r} that was "
                f"explicitly requested for {hf_model_id!r}. Refusing to continue "
                "with a model snapshot that cannot be verified."
            )

        # model_revision is the field already used throughout result
        # records (docs/methodology.md). It equals resolved_revision
        # whenever exact resolution succeeded; in the explicit
        # require_pinned_revision=False opt-out path, it falls back to
        # the post-load commit hash (if the local cache still yields one)
        # and finally to the bare requested revision string.
        self.model_revision = self.resolved_revision or post_load_commit_hash or self.requested_revision

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

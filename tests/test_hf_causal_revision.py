"""Tests for HFCausalAdapter's requested/resolved model revision handling
(docs/decisions.md, "Exact model revision recording").

Uses mocked `transformers.AutoTokenizer`/`AutoModelForCausalLM` — no
network call, no real model weights loaded or downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from conflict_eval.models.hf_causal import HFCausalAdapter


def _fake_tokenizer(*args, **kwargs):
    return SimpleNamespace(eos_token_id=0)


def _fake_model_with_commit_hash(commit_hash):
    def _from_pretrained(*args, **kwargs):
        model = SimpleNamespace()
        model.config = SimpleNamespace(_commit_hash=commit_hash)
        model.eval = lambda: model
        return model

    return _from_pretrained


def _build_adapter(commit_hash, requested_revision=None):
    with (
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_tokenizer),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_model_with_commit_hash(commit_hash),
        ),
    ):
        return HFCausalAdapter("fake/model-id", revision=requested_revision, dtype=None, device_map=None)


def test_resolved_revision_taken_from_model_config_commit_hash():
    adapter = _build_adapter(commit_hash="a1b2c3d4e5f6")
    assert adapter.resolved_revision == "a1b2c3d4e5f6"


def test_model_revision_prefers_resolved_over_requested():
    adapter = _build_adapter(commit_hash="a1b2c3d4e5f6", requested_revision="main")
    assert adapter.requested_revision == "main"
    assert adapter.resolved_revision == "a1b2c3d4e5f6"
    assert adapter.model_revision == "a1b2c3d4e5f6"


def test_model_revision_falls_back_to_requested_when_unresolvable():
    # config._commit_hash is None (attribute exists but empty) — must not
    # fabricate a resolved SHA; fall back to the explicitly requested
    # revision string instead.
    adapter = _build_adapter(commit_hash=None, requested_revision="v1.0-pinned")
    assert adapter.resolved_revision is None
    assert adapter.model_revision == "v1.0-pinned"


def test_model_revision_is_none_when_nothing_is_available():
    # Neither a resolvable commit hash nor an explicitly requested
    # revision — model_revision must be None, not an invented value.
    adapter = _build_adapter(commit_hash=None, requested_revision=None)
    assert adapter.resolved_revision is None
    assert adapter.requested_revision is None
    assert adapter.model_revision is None


def test_missing_commit_hash_attribute_does_not_crash():
    # A transformers version (or non-standard config object) that does
    # not expose `_commit_hash` at all must be handled gracefully via
    # getattr's default, not an AttributeError.
    def _fake_model_no_attr(*args, **kwargs):
        model = SimpleNamespace()
        model.config = SimpleNamespace()  # no _commit_hash attribute
        model.eval = lambda: model
        return model

    with (
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_fake_model_no_attr),
    ):
        adapter = HFCausalAdapter("fake/model-id", revision="main", dtype=None, device_map=None)

    assert adapter.resolved_revision is None
    assert adapter.model_revision == "main"

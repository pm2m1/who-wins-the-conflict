"""Tests for HFCausalAdapter's requested/resolved model revision handling
(docs/decisions.md, "Resolve, pin, load, record").

Follows resolve -> pin -> load -> record: the exact commit SHA is
resolved via a single mocked huggingface_hub metadata request BEFORE
`AutoTokenizer`/`AutoModelForCausalLM.from_pretrained` are called, and
that same SHA is what gets passed to both. Uses mocked
`transformers`/`huggingface_hub` — no network call, no real model weights
loaded or downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from conflict_eval.models.hf_causal import (
    HFCausalAdapter,
    ModelRevisionResolutionError,
    resolve_model_revision,
)


def _fake_hf_api(sha):
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(sha=sha)
    return api


def _fake_tokenizer(*args, **kwargs):
    return SimpleNamespace(eos_token_id=0)


def _fake_model_with_commit_hash(commit_hash):
    def _from_pretrained(*args, **kwargs):
        model = SimpleNamespace()
        model.config = SimpleNamespace(_commit_hash=commit_hash)
        model.eval = lambda: model
        return model

    return _from_pretrained


def _build_adapter(resolved_sha, post_load_commit_hash, requested_revision=None, require_pinned_revision=True):
    with (
        patch("huggingface_hub.HfApi", return_value=_fake_hf_api(resolved_sha)),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_tokenizer),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_model_with_commit_hash(post_load_commit_hash),
        ),
    ):
        return HFCausalAdapter(
            "fake/model-id",
            revision=requested_revision,
            dtype=None,
            device_map=None,
            require_pinned_revision=require_pinned_revision,
        )


# --- resolve_model_revision: the selection mechanism -------------------------


def test_resolve_model_revision_returns_resolved_sha():
    with patch("huggingface_hub.HfApi", return_value=_fake_hf_api("a09a35458c702")):
        assert resolve_model_revision("Qwen/Qwen2.5-7B-Instruct", None) == "a09a35458c702"


def test_resolve_model_revision_defaults_requested_revision_to_main():
    api = _fake_hf_api("main-sha")
    with patch("huggingface_hub.HfApi", return_value=api):
        resolve_model_revision("some/model", None)
    api.model_info.assert_called_once_with("some/model", revision="main")


def test_resolve_model_revision_passes_requested_revision_through():
    api = _fake_hf_api("pinned-sha")
    with patch("huggingface_hub.HfApi", return_value=api):
        resolve_model_revision("some/model", "v1.0-tag")
    api.model_info.assert_called_once_with("some/model", revision="v1.0-tag")


def test_resolve_model_revision_returns_none_on_lookup_failure():
    api = MagicMock()
    api.model_info.side_effect = OSError("offline")
    with patch("huggingface_hub.HfApi", return_value=api):
        assert resolve_model_revision("some/model", None) is None


# --- HFCausalAdapter: resolve BEFORE load, same SHA to both calls ------------


def test_resolution_happens_before_tokenizer_and_model_construction():
    call_order = []

    def _tracking_hf_api(*args, **kwargs):
        call_order.append("resolve")
        return _fake_hf_api("a1b2c3d4e5f6")

    def _tracking_tokenizer(*args, **kwargs):
        call_order.append("tokenizer")
        return SimpleNamespace(eos_token_id=0)

    def _tracking_model(*args, **kwargs):
        call_order.append("model")
        model = SimpleNamespace()
        model.config = SimpleNamespace(_commit_hash="a1b2c3d4e5f6")
        model.eval = lambda: model
        return model

    with (
        patch("huggingface_hub.HfApi", side_effect=_tracking_hf_api),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_tracking_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_tracking_model),
    ):
        HFCausalAdapter("fake/model-id", revision=None, dtype=None, device_map=None)

    assert call_order == ["resolve", "tokenizer", "model"]


def test_tokenizer_and_model_receive_the_same_resolved_sha():
    captured_revisions = {}

    def _tracking_tokenizer(model_id, revision=None, **kwargs):
        captured_revisions["tokenizer"] = revision
        return SimpleNamespace(eos_token_id=0)

    def _tracking_model(model_id, revision=None, **kwargs):
        captured_revisions["model"] = revision
        model = SimpleNamespace()
        model.config = SimpleNamespace(_commit_hash="a1b2c3d4e5f6")
        model.eval = lambda: model
        return model

    with (
        patch("huggingface_hub.HfApi", return_value=_fake_hf_api("a1b2c3d4e5f6")),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_tracking_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_tracking_model),
    ):
        HFCausalAdapter("fake/model-id", revision=None, dtype=None, device_map=None)

    assert captured_revisions["tokenizer"] == "a1b2c3d4e5f6"
    assert captured_revisions["model"] == "a1b2c3d4e5f6"
    assert captured_revisions["tokenizer"] == captured_revisions["model"]


def test_requested_revision_preserved_separately_from_resolved():
    adapter = _build_adapter(
        resolved_sha="a1b2c3d4e5f6", post_load_commit_hash="a1b2c3d4e5f6", requested_revision="main"
    )
    assert adapter.requested_revision == "main"
    assert adapter.resolved_revision == "a1b2c3d4e5f6"


def test_model_revision_equals_resolved_revision_on_successful_pin():
    adapter = _build_adapter(resolved_sha="a1b2c3d4e5f6", post_load_commit_hash="a1b2c3d4e5f6")
    assert adapter.model_revision == adapter.resolved_revision == "a1b2c3d4e5f6"


# --- strict-by-default: fail clearly when resolution is impossible ----------


def test_raises_when_revision_cannot_be_resolved_and_strict_by_default():
    api = MagicMock()
    api.model_info.side_effect = OSError("offline")
    with (
        patch("huggingface_hub.HfApi", return_value=api),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_tokenizer) as mock_tok,
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_model_with_commit_hash("irrelevant"),
        ) as mock_model,
        pytest.raises(ModelRevisionResolutionError),
    ):
        HFCausalAdapter("fake/model-id", revision=None, dtype=None, device_map=None)

    # Must fail BEFORE attempting to load anything.
    mock_tok.assert_not_called()
    mock_model.assert_not_called()


def test_explicit_opt_out_allows_unpinned_load():
    api = MagicMock()
    api.model_info.side_effect = OSError("offline")
    with (
        patch("huggingface_hub.HfApi", return_value=api),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_tokenizer),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_model_with_commit_hash(None),
        ),
    ):
        adapter = HFCausalAdapter(
            "fake/model-id",
            revision="main",
            dtype=None,
            device_map=None,
            require_pinned_revision=False,
        )

    assert adapter.resolved_revision is None
    # Falls back to the requested revision string — not fabricated.
    assert adapter.model_revision == "main"


# --- post-load consistency check ---------------------------------------------


def test_post_load_commit_hash_mismatch_raises():
    with pytest.raises(ModelRevisionResolutionError):
        _build_adapter(resolved_sha="a1b2c3d4e5f6", post_load_commit_hash="DIFFERENT-HASH")


def test_post_load_commit_hash_agreement_does_not_raise():
    adapter = _build_adapter(resolved_sha="a1b2c3d4e5f6", post_load_commit_hash="a1b2c3d4e5f6")
    assert adapter.model_revision == "a1b2c3d4e5f6"


def test_missing_post_load_commit_hash_does_not_raise():
    # transformers not exposing config._commit_hash at all must not be
    # treated as a mismatch — only an actual disagreement is an error.
    adapter = _build_adapter(resolved_sha="a1b2c3d4e5f6", post_load_commit_hash=None)
    assert adapter.model_revision == "a1b2c3d4e5f6"

"""Tests that _build_model_adapter propagates ModelSpec.max_memory to
HFCausalAdapter unchanged (docs/decisions.md, "Support reproducible model
memory limits").

HFCausalAdapter itself is mocked out entirely — no model is loaded, no
network call is made.
"""

from __future__ import annotations

from unittest.mock import patch

from conflict_eval.cli import _build_model_adapter
from conflict_eval.config import ModelSpec


def _hf_causal_spec(max_memory):
    return ModelSpec(
        key="qwen",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        revision=None,
        adapter="hf_causal",
        requires_gated_access=False,
        dtype="float16",
        device_map="auto",
        max_memory=max_memory,
    )


def test_build_model_adapter_propagates_configured_max_memory():
    configured = {0: "12.0GiB", "cpu": "5GiB"}
    spec = _hf_causal_spec(configured)

    with patch("conflict_eval.models.hf_causal.HFCausalAdapter") as mock_adapter_cls:
        _build_model_adapter(spec)

    mock_adapter_cls.assert_called_once_with(
        "Qwen/Qwen2.5-7B-Instruct",
        revision=None,
        dtype="float16",
        device_map="auto",
        max_memory=configured,
    )


def test_build_model_adapter_propagates_none_max_memory():
    spec = _hf_causal_spec(None)

    with patch("conflict_eval.models.hf_causal.HFCausalAdapter") as mock_adapter_cls:
        _build_model_adapter(spec)

    mock_adapter_cls.assert_called_once_with(
        "Qwen/Qwen2.5-7B-Instruct",
        revision=None,
        dtype="float16",
        device_map="auto",
        max_memory=None,
    )

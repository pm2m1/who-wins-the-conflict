"""Tests for exact PopQA dataset revision resolution
(conflict_eval.data.popqa.resolve_dataset_revision, download_raw,
build_manifest).

Follows resolve -> pin -> load -> record (docs/decisions.md): the exact
commit SHA is resolved via a single mocked huggingface_hub metadata
request BEFORE `datasets.load_dataset` is called, and that same SHA is
what gets passed to it. No network call, no real dataset download.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from conflict_eval.data.popqa import (
    DatasetRevisionResolutionError,
    build_manifest,
    download_raw,
    resolve_dataset_revision,
)


def _fake_hf_api(sha):
    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha=sha)
    return api


def test_resolve_dataset_revision_returns_resolved_sha():
    with patch("huggingface_hub.HfApi", return_value=_fake_hf_api("098765c79ea10a2cb19c828324e33281b8336ec0")):
        assert (
            resolve_dataset_revision("akariasai/PopQA")
            == "098765c79ea10a2cb19c828324e33281b8336ec0"
        )


def test_resolve_dataset_revision_passes_requested_revision_through():
    api = _fake_hf_api("pinned-sha")
    with patch("huggingface_hub.HfApi", return_value=api):
        resolve_dataset_revision("akariasai/PopQA", requested_revision="v2-tag")
    api.dataset_info.assert_called_once_with("akariasai/PopQA", revision="v2-tag")


def test_resolve_dataset_revision_defaults_requested_revision_to_main():
    api = _fake_hf_api("main-sha")
    with patch("huggingface_hub.HfApi", return_value=api):
        resolve_dataset_revision("akariasai/PopQA")
    api.dataset_info.assert_called_once_with("akariasai/PopQA", revision="main")


def test_resolve_dataset_revision_returns_none_on_lookup_failure():
    api = MagicMock()
    api.dataset_info.side_effect = OSError("offline")
    with patch("huggingface_hub.HfApi", return_value=api):
        assert resolve_dataset_revision("akariasai/PopQA") is None


def test_resolve_dataset_revision_returns_none_when_huggingface_hub_missing():
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        assert resolve_dataset_revision("akariasai/PopQA") is None


def test_build_manifest_includes_resolved_revision_field():
    manifest = build_manifest(
        hf_dataset_id="akariasai/PopQA",
        split="test",
        num_rows=14267,
        fields=["id", "question", "obj"],
        resolved_revision="098765c79ea10a2cb19c828324e33281b8336ec0",
    )
    assert manifest["resolved_revision"] == "098765c79ea10a2cb19c828324e33281b8336ec0"


def test_build_manifest_preserves_none_when_revision_unavailable():
    # Must record null/unavailable explicitly rather than omitting the
    # field or fabricating a value.
    manifest = build_manifest(
        hf_dataset_id="akariasai/PopQA",
        split="test",
        num_rows=14267,
        fields=["id", "question", "obj"],
        resolved_revision=None,
    )
    assert "resolved_revision" in manifest
    assert manifest["resolved_revision"] is None


# --- download_raw: resolve -> pin -> load -> record -------------------------


def _fake_dataset(rows):
    ds = MagicMock()
    ds.__iter__.return_value = iter(rows)
    ds.__len__.return_value = len(rows)
    ds.column_names = list(rows[0].keys()) if rows else []
    return ds


def test_download_raw_passes_resolved_sha_to_load_dataset(tmp_path):
    rows = [{"id": "1", "question": "Q?", "obj": "A"}]
    fake_datasets_module = SimpleNamespace(load_dataset=MagicMock(return_value=_fake_dataset(rows)))

    with (
        patch.dict("sys.modules", {"datasets": fake_datasets_module}),
        patch(
            "conflict_eval.data.popqa.resolve_dataset_revision",
            return_value="098765c79ea10a2cb19c828324e33281b8336ec0",
        ),
    ):
        download_raw("akariasai/PopQA", "test", tmp_path)

    fake_datasets_module.load_dataset.assert_called_once_with(
        "akariasai/PopQA", split="test", revision="098765c79ea10a2cb19c828324e33281b8336ec0"
    )


def test_download_raw_manifest_records_exact_resolved_sha(tmp_path):
    import json

    rows = [{"id": "1", "question": "Q?", "obj": "A"}]
    fake_datasets_module = SimpleNamespace(load_dataset=MagicMock(return_value=_fake_dataset(rows)))

    with (
        patch.dict("sys.modules", {"datasets": fake_datasets_module}),
        patch(
            "conflict_eval.data.popqa.resolve_dataset_revision",
            return_value="098765c79ea10a2cb19c828324e33281b8336ec0",
        ),
    ):
        download_raw("akariasai/PopQA", "test", tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_revision"] == "098765c79ea10a2cb19c828324e33281b8336ec0"


def test_download_raw_raises_when_revision_cannot_be_resolved(tmp_path):
    # Must not proceed to load_dataset at all, and must not fabricate a
    # SHA for the manifest — a real data-preparation run fails clearly.
    fake_datasets_module = SimpleNamespace(load_dataset=MagicMock())

    with (
        patch.dict("sys.modules", {"datasets": fake_datasets_module}),
        patch("conflict_eval.data.popqa.resolve_dataset_revision", return_value=None),
        pytest.raises(DatasetRevisionResolutionError),
    ):
        download_raw("akariasai/PopQA", "test", tmp_path)

    fake_datasets_module.load_dataset.assert_not_called()
    assert not (tmp_path / "manifest.json").exists()

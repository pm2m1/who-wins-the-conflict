"""Tests for exact PopQA dataset revision resolution
(conflict_eval.data.popqa.resolve_dataset_revision, build_manifest).

Uses mocked huggingface_hub cache-scan results — no network call, no real
dataset download (docs/decisions.md, "Exact PopQA dataset revision
recording").
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

from conflict_eval.data.popqa import build_manifest, resolve_dataset_revision


@dataclasses.dataclass
class _FakeRevision:
    commit_hash: str
    refs: frozenset[str]
    last_modified: float


@dataclasses.dataclass
class _FakeRepo:
    repo_id: str
    repo_type: str
    revisions: list[_FakeRevision]


@dataclasses.dataclass
class _FakeCacheInfo:
    repos: list[_FakeRepo]


def test_resolve_dataset_revision_finds_main_ref():
    fake_cache = _FakeCacheInfo(
        repos=[
            _FakeRepo(
                repo_id="akariasai/PopQA",
                repo_type="dataset",
                revisions=[_FakeRevision(commit_hash="abc123", refs=frozenset({"main"}), last_modified=1.0)],
            )
        ]
    )
    with patch("huggingface_hub.scan_cache_dir", return_value=fake_cache):
        assert resolve_dataset_revision("akariasai/PopQA") == "abc123"


def test_resolve_dataset_revision_falls_back_to_most_recent_when_no_main_ref():
    fake_cache = _FakeCacheInfo(
        repos=[
            _FakeRepo(
                repo_id="akariasai/PopQA",
                repo_type="dataset",
                revisions=[
                    _FakeRevision(commit_hash="older", refs=frozenset(), last_modified=1.0),
                    _FakeRevision(commit_hash="newer", refs=frozenset(), last_modified=2.0),
                ],
            )
        ]
    )
    with patch("huggingface_hub.scan_cache_dir", return_value=fake_cache):
        assert resolve_dataset_revision("akariasai/PopQA") == "newer"


def test_resolve_dataset_revision_returns_none_when_repo_not_cached():
    fake_cache = _FakeCacheInfo(repos=[])
    with patch("huggingface_hub.scan_cache_dir", return_value=fake_cache):
        assert resolve_dataset_revision("akariasai/PopQA") is None


def test_resolve_dataset_revision_ignores_model_repos_with_same_id():
    # A model repo happening to share the dataset's repo_id string must
    # not be mistaken for the dataset cache entry.
    fake_cache = _FakeCacheInfo(
        repos=[
            _FakeRepo(
                repo_id="akariasai/PopQA",
                repo_type="model",
                revisions=[_FakeRevision(commit_hash="wrong-type", refs=frozenset({"main"}), last_modified=1.0)],
            )
        ]
    )
    with patch("huggingface_hub.scan_cache_dir", return_value=fake_cache):
        assert resolve_dataset_revision("akariasai/PopQA") is None


def test_resolve_dataset_revision_returns_none_on_scan_failure():
    with patch("huggingface_hub.scan_cache_dir", side_effect=OSError("cache dir unreadable")):
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

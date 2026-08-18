"""Tests for `dataset.revision` config semantics in `load_pilot_config`
and its propagation through `cmd_prepare_data` (docs/decisions.md, "Add
generic dataset revision propagation").

This is a reproducibility feature that lets a pilot config pin the exact
immutable PopQA snapshot passed to `download_raw`. It is generic — not
specific to any one model or replication — and mirrors the existing
`dataset.candidate_pool` validation pattern in
tests/test_pilot_config.py. No network call, no real dataset download.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from conflict_eval.cli import cmd_prepare_data
from conflict_eval.config import ConfigError, load_pilot_config

BASE_CONFIG = {
    "seed": 42,
    "dataset": {"hf_dataset_id": "akariasai/PopQA", "split": "test", "screening_candidates": 10},
    "paths": {
        "raw_dir": "raw",
        "interim_dir": "interim",
        "processed_dir": "processed",
        "results_dir": "results",
        "figures_dir": "figures",
    },
    "sampling": {"target_kc_items": 5, "target_kw_items": 5, "margin_bins": ["low", "medium", "high"]},
    "models": ["dummy"],
    "source_roles": {"dummy": {"preferred_source": None, "dispreferred_source": None}},
    "prompts_config": "configs/prompts.yaml",
    "sources_config": "configs/sources.yaml",
    "models_config": "configs/models.yaml",
}


def _write_config(tmp_path, revision=None, *, include_key=False):
    config = {"seed": 42, "dataset": dict(BASE_CONFIG["dataset"])}
    config.update({k: v for k, v in BASE_CONFIG.items() if k not in ("seed", "dataset")})
    if include_key:
        config["dataset"]["revision"] = revision
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


# 1. omitted revision defaults to "main"
def test_omitted_revision_defaults_to_main(tmp_path):
    path = _write_config(tmp_path, include_key=False)
    config = load_pilot_config(path)
    assert config.dataset["revision"] == "main"


# 2. explicit revision is preserved unchanged
def test_explicit_revision_is_preserved_unchanged(tmp_path):
    sha = "098765c79ea10a2cb19c828324e33281b8336ec0"
    path = _write_config(tmp_path, sha, include_key=True)
    config = load_pilot_config(path)
    assert config.dataset["revision"] == sha


# 3. empty explicit string raises ConfigError
def test_empty_explicit_revision_raises_config_error(tmp_path):
    path = _write_config(tmp_path, "", include_key=True)
    with pytest.raises(ConfigError):
        load_pilot_config(path)


def test_whitespace_only_revision_raises_config_error(tmp_path):
    path = _write_config(tmp_path, "   ", include_key=True)
    with pytest.raises(ConfigError):
        load_pilot_config(path)


# 4. non-string explicit revision raises ConfigError
def test_non_string_revision_raises_config_error(tmp_path):
    path = _write_config(tmp_path, 12345, include_key=True)
    with pytest.raises(ConfigError):
        load_pilot_config(path)


# explicit null is treated the same as omitted (both mean "unset")
def test_explicit_null_revision_defaults_to_main(tmp_path):
    path = _write_config(tmp_path, None, include_key=True)
    config = load_pilot_config(path)
    assert config.dataset["revision"] == "main"


# 6. historical config with no revision preserves old behavior
def test_real_project_pilot_config_defaults_to_main():
    # configs/pilot.yaml is checked in and predates this option, so it
    # must keep resolving to "main" with no edits required.
    config = load_pilot_config("configs/pilot.yaml")
    assert config.dataset["revision"] == "main"


# 7. exact SHA-shaped strings are not rewritten/reformatted
def test_sha_shaped_revision_is_not_reformatted(tmp_path):
    sha = "A09A35458c702B33eeacc393d103063234E8BC28"  # deliberately mixed case
    path = _write_config(tmp_path, sha, include_key=True)
    config = load_pilot_config(path)
    assert config.dataset["revision"] == sha


# 5. cmd_prepare_data passes the exact configured revision to download_raw
def _build_prepare_data_config(tmp_path, revision=None, *, include_key=False):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = {"hf_dataset_id": "akariasai/PopQA", "split": "test", "screening_candidates": 2}
    if include_key:
        dataset["revision"] = revision
    config = {
        "seed": 42,
        "dataset": dataset,
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "results_dir": str(tmp_path / "results"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "sampling": {"target_kc_items": 1, "target_kw_items": 1, "margin_bins": ["low", "medium", "high"]},
        "models": ["dummy"],
        "source_roles": {"dummy": {"preferred_source": None, "dispreferred_source": None}},
        "prompts_config": "configs/prompts.yaml",
        "sources_config": "configs/sources.yaml",
        "models_config": "configs/models.yaml",
    }
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _fake_download_raw(hf_dataset_id, split, raw_dir, revision="main"):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "popqa_raw.jsonl"
    row = {"id": "1", "subj": "S", "prop": "sport", "obj": "hockey", "question": "Q?"}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return out_path


def test_cmd_prepare_data_propagates_omitted_revision_as_main(tmp_path):
    config_path = _build_prepare_data_config(tmp_path, include_key=False)
    with patch("conflict_eval.cli.download_raw", side_effect=_fake_download_raw) as mock_download_raw:
        cmd_prepare_data(str(config_path))
    _, kwargs = mock_download_raw.call_args
    assert kwargs["revision"] == "main"


def test_cmd_prepare_data_propagates_explicit_revision_unchanged(tmp_path):
    sha = "098765c79ea10a2cb19c828324e33281b8336ec0"
    config_path = _build_prepare_data_config(tmp_path, sha, include_key=True)
    with patch("conflict_eval.cli.download_raw", side_effect=_fake_download_raw) as mock_download_raw:
        cmd_prepare_data(str(config_path))
    _, kwargs = mock_download_raw.call_args
    assert kwargs["revision"] == sha

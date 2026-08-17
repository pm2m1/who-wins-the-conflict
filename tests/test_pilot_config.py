"""Tests for dataset.candidate_pool validation in load_pilot_config
(docs/decisions.md, "Support targeted primary conflict screening").
"""

from __future__ import annotations

import pytest
import yaml

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


def _write_config(tmp_path, candidate_pool=None, *, include_key=False):
    config = {"seed": 42, "dataset": dict(BASE_CONFIG["dataset"])}
    config.update({k: v for k, v in BASE_CONFIG.items() if k not in ("seed", "dataset")})
    if include_key:
        config["dataset"]["candidate_pool"] = candidate_pool
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_omitted_candidate_pool_defaults_to_all(tmp_path):
    path = _write_config(tmp_path, include_key=False)
    config = load_pilot_config(path)
    assert config.dataset["candidate_pool"] == "all"


def test_explicit_all_is_accepted(tmp_path):
    path = _write_config(tmp_path, "all", include_key=True)
    config = load_pilot_config(path)
    assert config.dataset["candidate_pool"] == "all"


def test_primary_conflict_relations_is_accepted(tmp_path):
    path = _write_config(tmp_path, "primary_conflict_relations", include_key=True)
    config = load_pilot_config(path)
    assert config.dataset["candidate_pool"] == "primary_conflict_relations"


def test_invalid_candidate_pool_raises_config_error(tmp_path):
    path = _write_config(tmp_path, "not_a_real_option", include_key=True)
    with pytest.raises(ConfigError):
        load_pilot_config(path)


def test_real_project_pilot_config_defaults_to_all():
    # configs/pilot.yaml is checked in and must always default to "all",
    # so historical runs do not silently change scope.
    config = load_pilot_config("configs/pilot.yaml")
    assert config.dataset["candidate_pool"] == "all"

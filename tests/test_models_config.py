"""Tests for optional per-model `max_memory` loading and validation
(conflict_eval.config.load_models_config, ModelSpec.max_memory).

Reproducibility infrastructure only — no model is loaded here
(docs/decisions.md, "Support reproducible model memory limits").
"""

from __future__ import annotations

import pytest
import yaml

from conflict_eval.config import ConfigError, load_models_config

BASE_MODEL_ENTRY = {
    "hf_model_id": "some/model",
    "revision": None,
    "adapter": "hf_causal",
    "requires_gated_access": False,
    "dtype": "float16",
    "device_map": "auto",
}


def _write_models_config(tmp_path, max_memory_value, *, include_key=True):
    entry = dict(BASE_MODEL_ENTRY)
    if include_key:
        entry["max_memory"] = max_memory_value
    raw = {
        "models": {"test_model": entry},
        "generation": {"do_sample": False, "max_new_tokens": 32, "num_beams": 1},
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# 1. max_memory omitted -> None
def test_max_memory_omitted_defaults_to_none(tmp_path):
    path = _write_models_config(tmp_path, None, include_key=False)
    config = load_models_config(path)
    assert config.get("test_model").max_memory is None


# 2. max_memory: null -> None
def test_max_memory_explicit_null_is_none(tmp_path):
    path = _write_models_config(tmp_path, None, include_key=True)
    config = load_models_config(path)
    assert config.get("test_model").max_memory is None


# 3. valid mapping preserved exactly
def test_valid_max_memory_preserved_exactly(tmp_path):
    path = _write_models_config(tmp_path, {0: "12.0GiB", "cpu": "5GiB"}, include_key=True)
    config = load_models_config(path)
    assert config.get("test_model").max_memory == {0: "12.0GiB", "cpu": "5GiB"}


# 4. multiple GPU integer keys preserved
def test_multiple_gpu_keys_preserved(tmp_path):
    path = _write_models_config(
        tmp_path, {0: "10GiB", 1: "10GiB", "cpu": "8GiB"}, include_key=True
    )
    config = load_models_config(path)
    assert config.get("test_model").max_memory == {0: "10GiB", 1: "10GiB", "cpu": "8GiB"}


# 5. invalid non-mapping max_memory -> ConfigError
def test_non_mapping_max_memory_raises(tmp_path):
    path = _write_models_config(tmp_path, "12GiB", include_key=True)
    with pytest.raises(ConfigError):
        load_models_config(path)


# 6. invalid negative GPU key -> ConfigError
def test_negative_gpu_key_raises(tmp_path):
    path = _write_models_config(tmp_path, {-1: "12GiB"}, include_key=True)
    with pytest.raises(ConfigError):
        load_models_config(path)


# 7. invalid unsupported string device key -> ConfigError
def test_unsupported_string_device_key_raises(tmp_path):
    path = _write_models_config(tmp_path, {"gpu": "12GiB"}, include_key=True)
    with pytest.raises(ConfigError):
        load_models_config(path)


# 8. empty/non-string memory value -> ConfigError
def test_empty_memory_value_raises(tmp_path):
    path = _write_models_config(tmp_path, {0: ""}, include_key=True)
    with pytest.raises(ConfigError):
        load_models_config(path)


def test_non_string_memory_value_raises(tmp_path):
    path = _write_models_config(tmp_path, {0: 12}, include_key=True)
    with pytest.raises(ConfigError):
        load_models_config(path)


def test_bool_key_is_rejected_even_though_bool_is_an_int_subclass(tmp_path):
    path = _write_models_config(tmp_path, {True: "12GiB"}, include_key=True)
    with pytest.raises(ConfigError):
        load_models_config(path)


def test_real_project_models_config_has_max_memory_null_by_default():
    # configs/models.yaml is checked in and must always be loadable, with
    # no machine-specific memory limits baked into the committed config.
    config = load_models_config("configs/models.yaml")
    assert config.get("qwen").max_memory is None
    assert config.get("llama").max_memory is None
    assert config.get("qwen").hf_model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.get("llama").hf_model_id == "meta-llama/Llama-3.1-8B-Instruct"

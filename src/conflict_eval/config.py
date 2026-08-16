"""Configuration loading and validation.

Every experimental choice that could change the scientific meaning of a
run (seeds, counts, model ids, source roles, prompt versions, output
paths) lives in YAML under configs/, not in Python literals, per
docs/phase2_research_design.md. This module loads and validates that YAML
so mistakes fail loudly instead of silently changing an experiment.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing required fields."""


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping")
    return data


@dataclasses.dataclass
class ModelSpec:
    key: str
    hf_model_id: str | None
    revision: str | None
    adapter: str
    requires_gated_access: bool
    dtype: str | None
    device_map: str | None


@dataclasses.dataclass
class ModelsConfig:
    models: dict[str, ModelSpec]
    generation: dict[str, Any]

    def get(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise ConfigError(
                f"Unknown model key '{key}'. Known: {sorted(self.models)}"
            )
        return self.models[key]


def load_models_config(path: str | Path) -> ModelsConfig:
    raw = _load_yaml(path)
    required_top = {"models", "generation"}
    missing = required_top - raw.keys()
    if missing:
        raise ConfigError(f"{path}: missing top-level keys {missing}")

    models: dict[str, ModelSpec] = {}
    required_fields = {"hf_model_id", "revision", "adapter", "requires_gated_access"}
    for key, entry in raw["models"].items():
        missing_fields = required_fields - entry.keys()
        if missing_fields:
            raise ConfigError(f"{path}: model '{key}' missing fields {missing_fields}")
        models[key] = ModelSpec(
            key=key,
            hf_model_id=entry.get("hf_model_id"),
            revision=entry.get("revision"),
            adapter=entry["adapter"],
            requires_gated_access=bool(entry["requires_gated_access"]),
            dtype=entry.get("dtype"),
            device_map=entry.get("device_map"),
        )
    return ModelsConfig(models=models, generation=raw["generation"])


@dataclasses.dataclass
class SourceRoleConfig:
    preferred_source: str | None
    dispreferred_source: str | None

    def is_set(self) -> bool:
        return self.preferred_source is not None and self.dispreferred_source is not None


@dataclasses.dataclass
class PilotConfig:
    seed: int
    dataset: dict[str, Any]
    paths: dict[str, str]
    sampling: dict[str, Any]
    models: list[str]
    source_roles: dict[str, SourceRoleConfig]
    prompts_config: str
    sources_config: str
    models_config: str
    raw: dict[str, Any]

    def source_roles_for(self, model_key: str) -> SourceRoleConfig:
        if model_key not in self.source_roles:
            raise ConfigError(
                f"No source_roles entry for model '{model_key}' in pilot config"
            )
        return self.source_roles[model_key]


def load_pilot_config(path: str | Path) -> PilotConfig:
    raw = _load_yaml(path)
    required_top = {
        "seed",
        "dataset",
        "paths",
        "sampling",
        "models",
        "source_roles",
        "prompts_config",
        "sources_config",
        "models_config",
    }
    missing = required_top - raw.keys()
    if missing:
        raise ConfigError(f"{path}: missing top-level keys {missing}")

    source_roles = {
        key: SourceRoleConfig(
            preferred_source=entry.get("preferred_source"),
            dispreferred_source=entry.get("dispreferred_source"),
        )
        for key, entry in raw["source_roles"].items()
    }

    return PilotConfig(
        seed=int(raw["seed"]),
        dataset=raw["dataset"],
        paths=raw["paths"],
        sampling=raw["sampling"],
        models=list(raw["models"]),
        source_roles=source_roles,
        prompts_config=raw["prompts_config"],
        sources_config=raw["sources_config"],
        models_config=raw["models_config"],
        raw=raw,
    )


def load_sources_config(path: str | Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    if "source_labels" not in raw:
        raise ConfigError(f"{path}: missing 'source_labels'")
    if len(raw["source_labels"]) < 2:
        raise ConfigError(f"{path}: need at least 2 source labels to form a pair")
    return raw


def load_prompts_config(path: str | Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    required = {"baseline", "evidence", "source_calibration", "evidence_template_version"}
    missing = required - raw.keys()
    if missing:
        raise ConfigError(f"{path}: missing top-level keys {missing}")
    return raw

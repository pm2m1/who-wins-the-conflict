"""Phase 3 study configuration schema and loader.

Separate from `conflict_eval.config` (the Phase 2 pilot schema), which is
left completely unchanged: Phase 2 must continue to behave exactly as
before (Phase 3B brief §19). This module adds a Phase 3 namespace rather
than retrofitting the Phase 2 loader.

The loader deliberately **accepts** unresolved fields -- the two approved
new model families have no exact release, repository id, or commit SHA
until Phase 3C (`docs/phase3_scaled_study_design.md`, §7, §42.1), and
representing that state honestly is a requirement, not a defect. What it
refuses is *executing* while those fields are unresolved; see
`real_run_gate.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from conflict_eval.phase3.constants import (
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
    FROZEN_MODEL_REVISIONS,
    FROZEN_MODEL_SOURCE_PAIRS,
)


class Phase3ConfigError(ValueError):
    """Raised when a Phase 3 configuration is structurally invalid."""


@dataclasses.dataclass(frozen=True)
class Phase3ModelEntry:
    """One model in the Phase 3 study.

    `resolved` is False for the approved-but-unresolved families. Their
    `hf_model_id` and `revision` stay `None` until Phase 3C; the loader
    never fills them in and never guesses.
    """

    key: str
    family: str
    hf_model_id: str | None
    revision: str | None
    role: str  # "replication" | "new"
    preferred_source: str | None
    dispreferred_source: str | None

    @property
    def resolved(self) -> bool:
        return self.hf_model_id is not None and self.revision is not None

    @property
    def source_roles_resolved(self) -> bool:
        return (
            self.preferred_source is not None
            and self.dispreferred_source is not None
        )


@dataclasses.dataclass(frozen=True)
class Phase3Config:
    seed: int
    dataset: dict[str, Any]
    common_source_a: str
    common_source_b: str
    models: dict[str, Phase3ModelEntry]
    screening: dict[str, Any]
    cohorts: dict[str, Any]
    paths: dict[str, str]
    prompts_config: str
    sources_config: str
    ready_for_real_run: bool
    raw: dict[str, Any]

    def model(self, key: str) -> Phase3ModelEntry:
        if key not in self.models:
            raise Phase3ConfigError(
                f"Unknown Phase 3 model key {key!r}. Known: {sorted(self.models)}"
            )
        return self.models[key]

    def unresolved_models(self) -> list[str]:
        return sorted(k for k, m in self.models.items() if not m.resolved)

    def unresolved_source_roles(self) -> list[str]:
        return sorted(k for k, m in self.models.items() if not m.source_roles_resolved)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise Phase3ConfigError(f"Phase 3 config not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise Phase3ConfigError(f"Phase 3 config {path} did not parse to a mapping")
    return data


def load_phase3_config(path: str | Path) -> Phase3Config:
    raw = _load_yaml(path)
    required = {"seed", "dataset", "sources", "models", "screening", "cohorts", "paths"}
    missing = required - raw.keys()
    if missing:
        raise Phase3ConfigError(f"{path}: missing top-level keys {sorted(missing)}")

    sources = raw["sources"]
    common_a = sources.get("common_source_a")
    common_b = sources.get("common_source_b")
    # The common pair is FROZEN by the design (§19); a config may not
    # silently redefine it.
    if common_a != COMMON_SOURCE_A or common_b != COMMON_SOURCE_B:
        raise Phase3ConfigError(
            f"{path}: the common source pair is frozen as "
            f"({COMMON_SOURCE_A!r}, {COMMON_SOURCE_B!r}) by "
            "docs/phase3_scaled_study_design.md §19; got "
            f"({common_a!r}, {common_b!r})"
        )

    models: dict[str, Phase3ModelEntry] = {}
    for key, entry in raw["models"].items():
        role = entry.get("role")
        if role not in ("replication", "new"):
            raise Phase3ConfigError(
                f"{path}: model {key!r} role must be 'replication' or 'new', "
                f"got {role!r}"
            )
        hf_model_id = entry.get("hf_model_id")
        revision = entry.get("revision")
        preferred = entry.get("preferred_source")
        dispreferred = entry.get("dispreferred_source")

        if role == "replication":
            # Qwen and Llama reuse their exact frozen Phase 2 artifacts and
            # frozen source pairs (§7, §20.1). A config that disagrees is a
            # design change, so it is rejected rather than accepted.
            frozen_rev = FROZEN_MODEL_REVISIONS.get(key)
            frozen_pair = FROZEN_MODEL_SOURCE_PAIRS.get(key)
            if frozen_rev is None or frozen_pair is None:
                raise Phase3ConfigError(
                    f"{path}: model {key!r} is declared 'replication' but has no "
                    "frozen Phase 2 revision/source pair on record"
                )
            if hf_model_id != frozen_rev["hf_model_id"] or revision != frozen_rev["revision"]:
                raise Phase3ConfigError(
                    f"{path}: replication model {key!r} must pin the exact frozen "
                    f"Phase 2 artifact {frozen_rev['hf_model_id']}@"
                    f"{frozen_rev['revision']} (§7); got {hf_model_id}@{revision}"
                )
            if (
                preferred != frozen_pair["preferred_source"]
                or dispreferred != frozen_pair["dispreferred_source"]
            ):
                raise Phase3ConfigError(
                    f"{path}: replication model {key!r} must use its frozen Phase 2 "
                    f"source pair {frozen_pair} (§20.1); got "
                    f"{{'preferred_source': {preferred!r}, "
                    f"'dispreferred_source': {dispreferred!r}}}"
                )
        else:
            # New families: unresolved is the CORRECT state in Phase 3B.
            # Reject any attempt to invent an id/SHA/source role now.
            for field_name, value in (
                ("hf_model_id", hf_model_id),
                ("revision", revision),
                ("preferred_source", preferred),
                ("dispreferred_source", dispreferred),
            ):
                if value is not None:
                    raise Phase3ConfigError(
                        f"{path}: model {key!r} is an approved-but-unresolved family; "
                        f"{field_name} must remain null until Phase 3C resolves and "
                        f"freezes it (§7, §20.2, §42.1). Got {value!r}."
                    )

        family = entry.get("family")
        if not family:
            raise Phase3ConfigError(f"{path}: model {key!r} missing 'family'")

        models[key] = Phase3ModelEntry(
            key=key,
            family=family,
            hf_model_id=hf_model_id,
            revision=revision,
            role=role,
            preferred_source=preferred,
            dispreferred_source=dispreferred,
        )

    return Phase3Config(
        seed=int(raw["seed"]),
        dataset=raw["dataset"],
        common_source_a=common_a,
        common_source_b=common_b,
        models=models,
        screening=raw["screening"],
        cohorts=raw["cohorts"],
        paths=raw["paths"],
        prompts_config=raw.get("prompts_config", "configs/prompts.yaml"),
        sources_config=raw.get("sources_config", "configs/sources.yaml"),
        # Never true in a committed Phase 3B config; the gate re-checks it.
        ready_for_real_run=bool(raw.get("ready_for_real_run", False)),
        raw=raw,
    )

"""Invariant tests for the precommitted Llama replication configs
(configs/replication/llama_pilot.yaml, configs/replication/models_llama.yaml).

These configs are a PRE-RUN specification only (docs/decisions.md,
"Precommit the Llama second-model replication";
docs/llama_replication_protocol.md). This file only loads and asserts on
the committed YAML through the real config loaders — it never loads a
model, calls the Hugging Face Hub, or touches network/GPU state.
"""

from __future__ import annotations

from conflict_eval.config import load_models_config, load_pilot_config

PILOT_PATH = "configs/replication/llama_pilot.yaml"
MODELS_PATH = "configs/replication/models_llama.yaml"


def test_pilot_models_list_is_llama_only():
    config = load_pilot_config(PILOT_PATH)
    assert config.models == ["llama"]


def test_pilot_seed_is_42():
    config = load_pilot_config(PILOT_PATH)
    assert config.seed == 42


def test_pilot_dataset_identity_and_pinned_revision():
    config = load_pilot_config(PILOT_PATH)
    assert config.dataset["hf_dataset_id"] == "akariasai/PopQA"
    assert config.dataset["split"] == "test"
    # Same immutable PopQA snapshot as the frozen Qwen pilot
    # (docs/qwen_pilot_results.md, "Frozen provenance").
    assert config.dataset["revision"] == "098765c79ea10a2cb19c828324e33281b8336ec0"
    assert config.dataset["screening_candidates"] == 500
    assert config.dataset["candidate_pool"] == "primary_conflict_relations"


def test_pilot_sampling_targets_match_qwen_method():
    config = load_pilot_config(PILOT_PATH)
    assert config.sampling["target_kc_items"] == 30
    assert config.sampling["target_kw_items"] == 30
    assert config.sampling["margin_bins"] == ["low", "medium", "high"]


def test_pilot_llama_source_roles_are_unset():
    config = load_pilot_config(PILOT_PATH)
    roles = config.source_roles_for("llama")
    assert roles.preferred_source is None
    assert roles.dispreferred_source is None
    assert not roles.is_set()


def test_pilot_llama_source_roles_do_not_copy_qwen_measured_pair():
    # Qwen's measured pair (docs/qwen_pilot_results.md) must never be
    # assigned to Llama's precommitted config — Llama is calibrated
    # independently.
    config = load_pilot_config(PILOT_PATH)
    roles = config.source_roles_for("llama")
    assert roles.preferred_source != "a government website"
    assert roles.dispreferred_source != "an anonymous online forum post"


def test_pilot_paths_are_isolated_under_llama_namespace():
    config = load_pilot_config(PILOT_PATH)
    for value in config.paths.values():
        assert value.startswith("runs/llama/")
        # Must not point at the committed shared results/figures paths
        # used by the frozen Qwen run.
        assert not value.startswith("results/")
        assert not value.startswith("figures/")
        assert not value.startswith("data/")


def test_pilot_references_the_llama_models_config():
    config = load_pilot_config(PILOT_PATH)
    assert config.models_config == MODELS_PATH


def test_pilot_reuses_shared_prompts_and_sources_configs():
    # Prompts and source labels are part of the replicated METHOD, not
    # something re-specified per model (docs/llama_replication_protocol.md).
    config = load_pilot_config(PILOT_PATH)
    assert config.prompts_config == "configs/prompts.yaml"
    assert config.sources_config == "configs/sources.yaml"


def test_models_config_llama_identity_and_access_flag():
    config = load_models_config(MODELS_PATH)
    spec = config.get("llama")
    assert spec.hf_model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec.requires_gated_access is True
    assert spec.adapter == "hf_causal"


def test_models_config_revision_is_not_guessed():
    # The exact Llama commit SHA is not known until a future researcher
    # resolves it against the live Hub with gated access
    # (docs/llama_replication_protocol.md, "Exact revision lock"). This
    # committed template must never contain a guessed value.
    config = load_models_config(MODELS_PATH)
    assert config.get("llama").revision is None


def test_models_config_precision_and_hardware_gate():
    config = load_models_config(MODELS_PATH)
    spec = config.get("llama")
    # Matches the frozen Qwen pilot's real-run precision for
    # comparability (docs/qwen_pilot_results.md, "Frozen provenance").
    assert spec.dtype == "float16"
    assert spec.device_map == "auto"
    # Machine-specific memory caps belong only in an ungitted scratch
    # config, never in this committed template.
    assert spec.max_memory is None


def test_models_config_generation_settings_match_project_defaults():
    config = load_models_config(MODELS_PATH)
    assert config.generation == {
        "do_sample": False,
        "max_new_tokens": 32,
        "num_beams": 1,
    }

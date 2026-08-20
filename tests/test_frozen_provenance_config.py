"""Invariant tests for the post-run provenance configs under
configs/frozen/ (qwen_pilot_frozen.yaml, qwen_model_frozen.yaml,
llama_pilot_frozen.yaml, llama_model_frozen.yaml).

These files are POST-RUN provenance copies, not preregistrations
(configs/frozen/README.md). This file only loads and asserts on the
committed YAML through the real config loaders — it never loads a model,
calls the Hugging Face Hub, or touches network/GPU state. It also checks
that the historical pre-run templates these were copied from were not
themselves modified.
"""

from __future__ import annotations

from conflict_eval.config import load_models_config, load_pilot_config

QWEN_PILOT_PATH = "configs/frozen/qwen_pilot_frozen.yaml"
QWEN_MODELS_PATH = "configs/frozen/qwen_model_frozen.yaml"
LLAMA_PILOT_PATH = "configs/frozen/llama_pilot_frozen.yaml"
LLAMA_MODELS_PATH = "configs/frozen/llama_model_frozen.yaml"


def test_qwen_frozen_pilot_identity_and_provenance():
    config = load_pilot_config(QWEN_PILOT_PATH)
    assert config.models == ["qwen"]
    assert config.seed == 42
    assert config.dataset["hf_dataset_id"] == "akariasai/PopQA"
    assert config.dataset["revision"] == "098765c79ea10a2cb19c828324e33281b8336ec0"
    assert config.dataset["screening_candidates"] == 500
    assert config.dataset["candidate_pool"] == "primary_conflict_relations"
    assert config.sampling["target_kc_items"] == 30
    assert config.sampling["target_kw_items"] == 30
    assert config.sampling["margin_bins"] == ["low", "medium", "high"]


def test_qwen_frozen_pilot_paths_match_original_shared_dirs():
    # The real Qwen run predates the runs/llama/ isolation namespace and
    # used the repository's original shared paths.
    config = load_pilot_config(QWEN_PILOT_PATH)
    assert config.paths["raw_dir"] == "data/raw"
    assert config.paths["results_dir"] == "results"
    assert config.paths["figures_dir"] == "figures"


def test_qwen_frozen_pilot_source_roles_match_frozen_report():
    config = load_pilot_config(QWEN_PILOT_PATH)
    roles = config.source_roles_for("qwen")
    assert roles.preferred_source == "a government website"
    assert roles.dispreferred_source == "an anonymous online forum post"
    assert roles.is_set()


def test_qwen_frozen_model_revision_and_precision():
    config = load_models_config(QWEN_MODELS_PATH)
    spec = config.get("qwen")
    assert spec.hf_model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert spec.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert spec.dtype == "float16"
    assert spec.device_map == "auto"
    # Machine-specific cap actually recorded for this run
    # (docs/qwen_pilot_results.md, "Frozen provenance").
    assert spec.max_memory == {0: "22GiB"}


def test_llama_frozen_pilot_identity_and_provenance():
    config = load_pilot_config(LLAMA_PILOT_PATH)
    assert config.models == ["llama"]
    assert config.seed == 42
    assert config.dataset["hf_dataset_id"] == "akariasai/PopQA"
    assert config.dataset["revision"] == "098765c79ea10a2cb19c828324e33281b8336ec0"
    assert config.dataset["candidate_pool"] == "primary_conflict_relations"
    assert config.sampling["target_kc_items"] == 30
    assert config.sampling["target_kw_items"] == 30


def test_llama_frozen_pilot_paths_are_isolated_under_llama_namespace():
    config = load_pilot_config(LLAMA_PILOT_PATH)
    for value in config.paths.values():
        assert value.startswith("runs/llama/")


def test_llama_frozen_pilot_source_roles_match_frozen_report():
    config = load_pilot_config(LLAMA_PILOT_PATH)
    roles = config.source_roles_for("llama")
    assert roles.preferred_source == "a government website"
    assert roles.dispreferred_source == "a social media post"
    assert roles.is_set()


def test_llama_frozen_pilot_dispreferred_source_differs_from_qwen():
    # Independently calibrated per model — never copied
    # (docs/cross_model_pilot_results.md, "What is comparable, and what
    # is not").
    llama_roles = load_pilot_config(LLAMA_PILOT_PATH).source_roles_for("llama")
    qwen_roles = load_pilot_config(QWEN_PILOT_PATH).source_roles_for("qwen")
    assert llama_roles.preferred_source == qwen_roles.preferred_source
    assert llama_roles.dispreferred_source != qwen_roles.dispreferred_source


def test_llama_frozen_model_revision_and_precision():
    config = load_models_config(LLAMA_MODELS_PATH)
    spec = config.get("llama")
    assert spec.hf_model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec.revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert spec.dtype == "float16"
    assert spec.device_map == "auto"
    # No machine-specific cap is recorded for Llama in
    # docs/cross_model_pilot_results.md — must not be guessed as equal to
    # Qwen's.
    assert spec.max_memory is None


def test_frozen_configs_are_distinct_files_from_historical_pre_run_templates():
    # These provenance copies must never be the same path as the
    # historical pre-run configs they were derived from.
    assert QWEN_PILOT_PATH != "configs/pilot.yaml"
    assert LLAMA_PILOT_PATH != "configs/replication/llama_pilot.yaml"
    assert LLAMA_MODELS_PATH != "configs/replication/models_llama.yaml"


def test_historical_pre_run_llama_template_still_has_null_revision():
    # configs/replication/models_llama.yaml is the historical pre-run
    # template and must remain untouched by adding frozen/ copies
    # alongside it (docs/decisions.md, "Precommit the Llama second-model
    # replication").
    config = load_models_config("configs/replication/models_llama.yaml")
    assert config.get("llama").revision is None


def test_historical_pre_run_llama_pilot_still_has_null_source_roles():
    config = load_pilot_config("configs/replication/llama_pilot.yaml")
    roles = config.source_roles_for("llama")
    assert not roles.is_set()

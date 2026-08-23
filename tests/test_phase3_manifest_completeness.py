"""Adversarial tests for §36 freeze-manifest completeness and the gate.

A read-only freeze audit found three ways a Phase 3 run could be authorized
by an incomplete freeze:

1. mandatory §36 content (candidate/baseline/exclusion/trial digests,
   margin-bin edges, manual-review decisions, environment and hardware
   capture) could be omitted entirely;
2. a manifest could describe only *some* of the configured models, silently
   skipping every per-model rule for the rest -- including the §34
   disabled-arm rules and §36 new-model calibration provenance;
3. `check_readiness` validated less than `validate_manifest`, so "the gate
   passed" could mean something weaker than "the manifest is valid".

These tests pin all three shut. The invariant that matters most is the
last one: **if `validate_manifest` reports any problem, `check_readiness`
must not return READY.**

No real artifact is created here. The digests below are structurally valid
but obviously synthetic, because what is under test is the validator, not
any real baseline, exclusion, candidate or trial file -- none of which
exists yet.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from conflict_eval.phase3.config import load_phase3_config
from conflict_eval.phase3.manifest import (
    REQUIRED_ARTIFACT_HASHES,
    REQUIRED_MODEL_RUNTIME_FIELDS,
    REQUIRED_MODEL_SHA_FIELDS,
    Phase3Manifest,
    validate_manifest,
)
from conflict_eval.phase3.real_run_gate import (
    Phase3NotReadyError,
    assert_ready_for_real_run,
    check_readiness,
)

COMMITTED_CONFIG = "configs/phase3/phase3_study.yaml"
SHA = "a" * 64
ALL_MODELS = ("qwen", "llama", "mistral", "gemma")


def _ready_config(tmp_path):
    """The real config with `ready_for_real_run` flipped true.

    Flipping the flag is exactly the shortcut the audit warned about, so
    every test here starts from the most permissive config a researcher
    could produce. The committed config on disk is never modified.
    """
    with open(COMMITTED_CONFIG, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["ready_for_real_run"] = True
    path = tmp_path / "ready.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_phase3_config(path)


def _model_entry(key, config):
    entry = config.model(key)
    return {
        "hf_model_id": entry.hf_model_id,
        "revision": entry.revision,
        "requested_revision": entry.revision,
        "resolved_revision": entry.revision,
        "role": entry.role,
        "dtype": "float16",
        "quantization": "none",
        "device_map": {"": 0},
        "max_memory": {0: "23GiB"},
        "baseline_file_sha256": SHA,
        "exclusion_file_sha256": "b" * 64,
        "knowledge_membership": {"KC": ["syn-1"], "KW": ["syn-2"]},
        "margins": {"syn-1": 0.4},
        "manual_review_decisions": [],
        "model_specific_arm_enabled": entry.runs_model_specific_arm,
        "model_specific_arm_reason": entry.model_specific_arm_reason,
        "preferred_source": entry.preferred_source,
        "dispreferred_source": entry.dispreferred_source,
        "calibration_provenance": entry.calibration_provenance,
        "condition_set": list(entry.condition_set),
    }


def _manifest(config, models=ALL_MODELS):
    """A manifest carrying the §36 minimum for the given model set.

    Deliberately NOT a plausible production manifest: the cohorts here are
    skeletal, so this never becomes READY. It exists to isolate the
    completeness rules under test.
    """
    return {
        "frozen": True,
        "synthetic": False,
        "repository_commit": "38afd32",
        "seed": 42,
        "dataset": {
            "hf_dataset_id": "akariasai/PopQA",
            "split": "test",
            "revision": "0" * 40,
            "candidate_item_ids": ["syn-1"],
        },
        "models": {k: _model_entry(k, config) for k in models},
        "prompt_version": "v1",
        "sources": {"common_source_a": "a government website"},
        "cohorts": {"A": {}, "B": {}, "C": {}},
        "cohort_membership_map": {"o": ["A"]},
        "condition_specification": ["C0", "K1", "K2", "K3", "K4", "M1", "M2"],
        "deduplication_alias_map": {"qwen|syn-1|K1": "o"},
        "final_margin_strata": {"qwen|KW": [1.0, 2.0]},
        "screening": {"block_size": 250},
        "analysis_status": [{"name": "x"}],
        "artifact_hashes": dict.fromkeys(REQUIRED_ARTIFACT_HASHES, SHA),
        "environment": {"python": "3.10.13", "torch": "2.3.1", "cuda": "12.1"},
        "hardware": {"gpu_name": "RTX 3090", "vram": "24GiB"},
    }


def _problems(data, config=None):
    keys = sorted(config.models) if config else None
    return validate_manifest(Phase3Manifest(data=data), expected_model_keys=keys)


# --- 1/2: mandatory §36 content cannot be omitted --------------------------


@pytest.mark.parametrize("field", ["artifact_hashes", "environment", "hardware"])
def test_missing_top_level_36_field_is_rejected(tmp_path, field):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data[field]
    assert any(field in p for p in _problems(data, cfg))


@pytest.mark.parametrize("digest", REQUIRED_ARTIFACT_HASHES)
def test_missing_artifact_digest_is_rejected(tmp_path, digest):
    """§36: config, candidate file and trial file SHA256."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data["artifact_hashes"][digest]
    assert any(digest in p and "SHA256" in p for p in _problems(data, cfg))


@pytest.mark.parametrize(
    "field", ["hf_dataset_id", "split", "revision", "candidate_item_ids"]
)
def test_missing_dataset_provenance_is_rejected(tmp_path, field):
    """§36: dataset id, split, resolved revision, and the candidate IDs."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data["dataset"][field]
    assert any(field in p for p in _problems(data, cfg))


@pytest.mark.parametrize("field", REQUIRED_MODEL_SHA_FIELDS)
def test_missing_per_model_baseline_or_exclusion_digest_is_rejected(tmp_path, field):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data["models"]["qwen"][field]
    assert any(field in p and "SHA256" in p for p in _problems(data, cfg))


@pytest.mark.parametrize("field", REQUIRED_MODEL_RUNTIME_FIELDS)
def test_missing_per_model_runtime_provenance_is_rejected(tmp_path, field):
    """§36: precision, quantization, device_map and max_memory as used."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data["models"]["llama"][field]
    assert any(field in p for p in _problems(data, cfg))


@pytest.mark.parametrize("field", ["knowledge_membership", "margins"])
def test_missing_per_model_screening_provenance_is_rejected(tmp_path, field):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data["models"]["mistral"][field]
    assert any(field in p for p in _problems(data, cfg))


def test_missing_manual_review_decisions_is_rejected(tmp_path):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    del data["models"]["gemma"]["manual_review_decisions"]
    assert any("manual_review_decisions" in p for p in _problems(data, cfg))


def test_an_explicit_empty_manual_review_list_is_accepted(tmp_path):
    """"No manual overrides were made" is a finding; an absent key is an
    unfinished record. The two must stay distinguishable."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data["models"]["gemma"]["manual_review_decisions"] = []
    assert not any("manual_review_decisions" in p for p in _problems(data, cfg))


def test_missing_margin_bin_edges_is_rejected(tmp_path):
    """§36 "margin-bin edges" -- carried by `final_margin_strata`."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data["final_margin_strata"] = {}
    assert any("final_margin_strata" in p for p in _problems(data, cfg))


# --- 3/4: presence is not capture -----------------------------------------


@pytest.mark.parametrize("field", ["environment", "hardware"])
def test_null_capture_is_rejected(tmp_path, field):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data[field] = None
    assert any(field in p and "capture" in p for p in _problems(data, cfg))


@pytest.mark.parametrize("field", ["environment", "hardware"])
def test_unfilled_placeholder_capture_is_rejected(tmp_path, field):
    """The Phase 3B builder seeds these with None. A frozen manifest still
    carrying the unfilled template has captured nothing."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data[field] = dict.fromkeys(data[field], None)
    assert any("placeholder" in p and field in p for p in _problems(data, cfg))


# --- 5: digest shape, via the single shared SHA256 pattern -----------------


@pytest.mark.parametrize(
    "bad", ["TBD", "pending", "not-a-hash", "A" * 64, "a" * 63, "a" * 65, "g" * 64, ""]
)
def test_invalid_digest_is_rejected_everywhere(tmp_path, bad):
    cfg = _ready_config(tmp_path)
    for setter in (
        lambda d: d["artifact_hashes"].update(candidate_file=bad),
        lambda d: d["artifact_hashes"].update(trial_file=bad),
        lambda d: d["models"]["qwen"].update(baseline_file_sha256=bad),
        lambda d: d["models"]["qwen"].update(exclusion_file_sha256=bad),
    ):
        data = _manifest(cfg)
        setter(data)
        assert _problems(data, cfg), f"{bad!r} accepted"


def test_quantization_must_be_none(tmp_path):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data["models"]["qwen"]["quantization"] = "int8"
    assert any("unquantized" in p for p in _problems(data, cfg))


# --- 6-11: the manifest must describe exactly the configured model set -----


@pytest.mark.parametrize("omitted", ALL_MODELS)
def test_omitting_any_configured_model_is_rejected(tmp_path, omitted):
    """An omitted model silently skips every per-model rule -- arm state,
    condition set and calibration provenance included."""
    cfg = _ready_config(tmp_path)
    kept = [m for m in ALL_MODELS if m != omitted]
    problems = _problems(_manifest(cfg, models=kept), cfg)
    assert any("omits configured model" in p and omitted in p for p in problems)


def test_an_unexpected_extra_model_is_rejected(tmp_path):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data["models"]["phi"] = copy.deepcopy(data["models"]["qwen"])
    problems = _problems(data, cfg)
    assert any("does not declare" in p and "phi" in p for p in problems)


def test_the_exact_four_model_set_passes_the_model_set_check(tmp_path):
    cfg = _ready_config(tmp_path)
    problems = _problems(_manifest(cfg), cfg)
    assert not any("omits configured model" in p for p in problems)
    assert not any("does not declare" in p for p in problems)


def test_disabled_arm_with_complete_provenance_raises_no_arm_problem(tmp_path):
    """§34: a disabled arm is a legitimate pre-specified state. It must not
    generate an arm-state or calibration problem of its own."""
    cfg = _ready_config(tmp_path)
    problems = _problems(_manifest(cfg), cfg)
    for model in ("mistral", "gemma"):
        assert not any(model in p and "arm" in p for p in problems)
        assert not any(model in p and "calibration" in p for p in problems)


@pytest.mark.parametrize("model", ["mistral", "gemma"])
def test_disabled_models_still_need_no_m_conditions(tmp_path, model):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    assert data["models"][model]["condition_set"] == ["C0", "K1", "K2", "K3", "K4"]
    assert data["models"][model]["preferred_source"] is None
    assert data["models"][model]["dispreferred_source"] is None
    assert not any(model in p and "condition_set" in p for p in _problems(data, cfg))


@pytest.mark.parametrize("model", ["qwen", "llama"])
def test_replication_models_are_unchanged(tmp_path, model):
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    entry = data["models"][model]
    assert entry["condition_set"] == ["C0", "K1", "K2", "K3", "K4", "M1", "M2"]
    assert entry["preferred_source"] == "a government website"
    assert not any(model in p and "arm" in p for p in _problems(data, cfg))


# --- 12/13: the gate is a strict superset of manifest validation -----------


def test_gate_is_never_ready_while_validate_manifest_reports_a_problem(tmp_path):
    """THE invariant. Probed across many independent mutilations so it
    cannot pass by coincidence on one shape."""
    cfg = _ready_config(tmp_path)
    mutations = [
        ("drop artifact_hashes", lambda d: d.pop("artifact_hashes")),
        ("drop environment", lambda d: d.pop("environment")),
        ("null hardware", lambda d: d.update(hardware=None)),
        ("placeholder env", lambda d: d.update(environment={"python": None})),
        ("bad candidate digest", lambda d: d["artifact_hashes"].update(candidate_file="TBD")),
        ("drop baseline sha", lambda d: d["models"]["qwen"].pop("baseline_file_sha256")),
        ("drop margins", lambda d: d["models"]["llama"].pop("margins")),
        ("drop manual review", lambda d: d["models"]["gemma"].pop("manual_review_decisions")),
        ("drop candidate ids", lambda d: d["dataset"].pop("candidate_item_ids")),
        ("empty margin bins", lambda d: d.update(final_margin_strata={})),
        ("only qwen", lambda d: d.update(models={"qwen": d["models"]["qwen"]})),
        ("symbolic revision", lambda d: d["models"]["qwen"].update(revision="main")),
    ]
    for label, mutate in mutations:
        data = _manifest(cfg)
        mutate(data)
        problems = _problems(data, cfg)
        assert problems, f"{label}: expected validate_manifest to object"
        report = check_readiness(cfg, manifest=data)
        assert report.ready is False, f"{label}: gate went READY despite {problems}"
        with pytest.raises(Phase3NotReadyError):
            assert_ready_for_real_run(cfg, manifest=data)


def test_gate_surfaces_manifest_problems_rather_than_a_weaker_subset(tmp_path):
    """The gate must report what the validator reports, not a subset."""
    cfg = _ready_config(tmp_path)
    data = _manifest(cfg)
    data["models"]["gemma"].pop("manual_review_decisions")
    data["artifact_hashes"].pop("trial_file")
    problems = set(_problems(data, cfg))
    blockers = set(check_readiness(cfg, manifest=data).blockers)
    assert problems <= blockers, f"gate dropped: {problems - blockers}"


def test_ready_flag_plus_incomplete_manifest_is_still_not_ready(tmp_path):
    """Flipping `ready_for_real_run` cannot buy readiness."""
    cfg = _ready_config(tmp_path)
    assert cfg.ready_for_real_run is True
    data = _manifest(cfg)
    data.pop("artifact_hashes")
    assert check_readiness(cfg, manifest=data).ready is False


# --- 20: the real committed config stays closed ----------------------------


def test_the_committed_config_remains_not_ready():
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.ready_for_real_run is False
    report = check_readiness(config, manifest=None)
    assert report.ready is False
    with pytest.raises(Phase3NotReadyError):
        assert_ready_for_real_run(config, manifest=None)

"""Tests for the Phase 3 config schema, manifest, and real-run safety gate.

Covers `docs/phase3_scaled_study_design.md` §7/§20 (frozen artifacts and
source pairs; unresolved new families), §36 (freeze manifest) and §41
("NO REAL PHASE 3 MODEL MAY RUN BEFORE 3C IS FROZEN").
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from conflict_eval.phase3.analysis_status import (
    AnalysisEntry,
    AnalysisRegistry,
    AnalysisStatusError,
    default_registry,
)
from conflict_eval.phase3.config import Phase3ConfigError, load_phase3_config
from conflict_eval.phase3.constants import (
    ARM_DISABLED_BY_CALIBRATION,
    ARM_UNRESOLVED,
    FROZEN_MODEL_REVISIONS,
    STATUS_DIAGNOSTIC,
    STATUS_EXPLORATORY,
    STATUS_PRIMARY,
    STATUS_SECONDARY,
)
from conflict_eval.phase3.manifest import (
    ManifestError,
    build_manifest,
    freeze_manifest,
    planned_cohort_a_design,
    planned_screening_design,
    validate_manifest,
)
from conflict_eval.phase3.real_run_gate import (
    Phase3NotReadyError,
    assert_ready_for_real_run,
    check_readiness,
)

COMMITTED_CONFIG = "configs/phase3/phase3_study.yaml"
FREEZE_MANIFEST = Path("configs/phase3/freeze/phase3c_pre_run_manifest.json")


def _write_config(tmp_path, mutate=None):
    with open(COMMITTED_CONFIG, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if mutate:
        mutate(data)
    path = tmp_path / "phase3.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- committed config -----------------------------------------------------


def test_committed_config_loads():
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.seed == 42
    assert set(config.models) == {"qwen", "llama", "mistral", "gemma"}


def test_committed_config_pins_frozen_replication_artifacts():
    config = load_phase3_config(COMMITTED_CONFIG)
    for key in ("qwen", "llama"):
        entry = config.model(key)
        assert entry.hf_model_id == FROZEN_MODEL_REVISIONS[key]["hf_model_id"]
        assert entry.revision == FROZEN_MODEL_REVISIONS[key]["revision"]
        assert entry.resolved is True


def test_committed_config_resolves_the_new_families_at_phase3c():
    """Phase 3C resolved both approved families to exact releases/SHAs."""
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.unresolved_models() == []
    assert config.model("mistral").hf_model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert (
        config.model("mistral").revision
        == "c170c708c41dac9275d15a8fff4eca08d52bab71"
    )
    assert config.model("gemma").hf_model_id == "google/gemma-2-9b-it"
    assert (
        config.model("gemma").revision == "11c9b309abf73637e4b6f9a3fa1e92e615547819"
    )


def test_committed_config_disables_new_model_arms_under_the_frozen_rule():
    """Both new models had calibration tied/heavily malformed, so §34 puts
    them on the common arm only with null roles -- by design, not omission."""
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.common_arm_only_models() == ["gemma", "mistral"]
    assert config.model_specific_arm_models() == ["llama", "qwen"]
    for key in ("mistral", "gemma"):
        entry = config.model(key)
        assert entry.arm_state == ARM_DISABLED_BY_CALIBRATION
        assert entry.preferred_source is None
        assert entry.dispreferred_source is None
        assert entry.model_specific_arm_reason
        assert entry.calibration_provenance
        assert list(entry.condition_set) == ["C0", "K1", "K2", "K3", "K4"]
    # No calibration result is fabricated: the recorded counts are the ones
    # actually observed.
    assert config.model("gemma").calibration_provenance["parser_valid_trials"] == 0
    assert config.model("mistral").calibration_provenance["parser_valid_trials"] == 30


def test_committed_config_is_ready_only_alongside_the_sealed_freeze_manifest():
    """Phase 3C is frozen, so `ready_for_real_run` is now legitimately true.

    The safety property this test guards did NOT go away when the freeze
    happened -- it got stricter. The flag is not self-certifying, so what
    must hold is that it is true *only* because a sealed, fully-valid §36
    manifest exists beside it. Both halves are asserted here: the manifest
    opens the gate, and removing it closes the gate again even though the
    flag stays true.
    """
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.ready_for_real_run is True

    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["frozen"] is True
    assert manifest["synthetic"] is False
    assert check_readiness(config, manifest=manifest).ready is True

    # The flag alone buys nothing.
    assert check_readiness(config, manifest=None).ready is False


# --- config rejects design changes ---------------------------------------


def test_config_rejects_a_changed_common_source_pair(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["sources"].update(common_source_b="a personal blog")
    )
    with pytest.raises(Phase3ConfigError, match="common source pair is frozen"):
        load_phase3_config(path)


def test_config_rejects_a_re_resolved_replication_revision(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["models"]["qwen"].update(revision="0" * 40)
    )
    with pytest.raises(Phase3ConfigError, match="frozen"):
        load_phase3_config(path)


def test_config_rejects_a_changed_replication_source_pair(tmp_path):
    path = _write_config(
        tmp_path,
        lambda d: d["models"]["llama"].update(dispreferred_source="a personal blog"),
    )
    with pytest.raises(Phase3ConfigError, match="frozen Phase 2"):
        load_phase3_config(path)


def test_config_rejects_a_partially_resolved_new_model(tmp_path):
    """Identity is all-or-nothing: a half-resolved model cannot slip through."""
    path = _write_config(
        tmp_path, lambda d: d["models"]["mistral"].update(revision=None)
    )
    with pytest.raises(Phase3ConfigError, match="partially resolved identity"):
        load_phase3_config(path)


def test_config_rejects_a_mutable_new_model_revision(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["models"]["gemma"].update(revision="main")
    )
    with pytest.raises(Phase3ConfigError, match="exact immutable"):
        load_phase3_config(path)


def test_config_rejects_a_source_role_on_a_disabled_arm(tmp_path):
    """Disabling the arm IS the refusal to invent a pair (§20.2, §34)."""
    path = _write_config(
        tmp_path,
        lambda d: d["models"]["mistral"].update(
            preferred_source="a government website"
        ),
    )
    with pytest.raises(Phase3ConfigError, match="must leave both source roles null"):
        load_phase3_config(path)


def test_config_rejects_a_disabled_arm_without_a_reason(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["models"]["gemma"].pop("model_specific_arm_reason")
    )
    with pytest.raises(Phase3ConfigError, match="no model_specific_arm_reason"):
        load_phase3_config(path)


def test_config_rejects_a_disabled_arm_without_calibration_provenance(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["models"]["mistral"].pop("calibration_provenance")
    )
    with pytest.raises(Phase3ConfigError, match="no calibration_provenance"):
        load_phase3_config(path)


def test_config_rejects_an_enabled_arm_missing_a_role(tmp_path):
    def mutate(d):
        d["models"]["mistral"].update(
            model_specific_arm_enabled=True,
            preferred_source="a government website",
        )

    path = _write_config(tmp_path, mutate)
    with pytest.raises(Phase3ConfigError, match="requires BOTH roles"):
        load_phase3_config(path)


def test_config_rejects_an_unresolved_arm_state(tmp_path):
    """A model that never declares its arm state stays UNRESOLVED and blocks."""
    def mutate(d):
        entry = d["models"]["mistral"]
        entry.pop("model_specific_arm_enabled")
        entry.pop("model_specific_arm_reason")

    path = _write_config(tmp_path, mutate)
    config = load_phase3_config(path)
    assert config.model("mistral").arm_state == ARM_UNRESOLVED
    assert "mistral" in config.unresolved_source_roles()
    with pytest.raises(Phase3NotReadyError):
        assert_ready_for_real_run(config, manifest=None)


def test_config_rejects_disabling_a_replication_arm(tmp_path):
    """Qwen/Llama carry the frozen Phase 2 replication pair; their arm
    cannot be switched off."""
    def mutate(d):
        # Clear the roles too, so this isolates the replication rule rather
        # than tripping the disabled-arm-must-have-null-roles check first.
        d["models"]["qwen"].update(
            model_specific_arm_enabled=False,
            model_specific_arm_reason="attempted downgrade",
            preferred_source=None,
            dispreferred_source=None,
        )

    path = _write_config(tmp_path, mutate)
    # Rejected either by the frozen-pair guard (§20.1) or by the
    # replication-arm guard -- both are layers of the same protection, and
    # which one fires first does not matter. What matters is that a
    # replication model can never be downgraded to common-arm-only.
    with pytest.raises(
        Phase3ConfigError, match="frozen Phase 2 source pair|must keep its"
    ):
        load_phase3_config(path)


def test_config_rejects_an_unknown_role(tmp_path):
    path = _write_config(tmp_path, lambda d: d["models"]["qwen"].update(role="other"))
    with pytest.raises(Phase3ConfigError, match="role must be"):
        load_phase3_config(path)


def test_config_rejects_a_missing_top_level_key(tmp_path):
    path = _write_config(tmp_path, lambda d: d.pop("cohorts"))
    with pytest.raises(Phase3ConfigError, match="missing top-level keys"):
        load_phase3_config(path)


# --- real-run gate --------------------------------------------------------


def test_gate_blocks_the_committed_phase3b_config():
    config = load_phase3_config(COMMITTED_CONFIG)
    with pytest.raises(Phase3NotReadyError, match="NO REAL PHASE 3 MODEL MAY RUN"):
        assert_ready_for_real_run(config, manifest=None)


def test_gate_reports_every_blocker_not_just_the_first(tmp_path):
    """The gate must accumulate blockers, not short-circuit on the first.

    Exercised against a config with several independent faults at once, so
    the property is pinned by construction rather than by whatever the
    committed config happens to be missing at a given phase.
    """
    def mutate(d):
        d["ready_for_real_run"] = False
        d["dataset"]["revision"] = None
        d["models"]["gemma"]["calibration_provenance"].pop("calibration_output_sha256")

    config = load_phase3_config(_write_config(tmp_path, mutate))
    report = check_readiness(config, manifest=None)
    assert report.ready is False
    joined = " ".join(report.blockers)
    assert "ready_for_real_run" in joined
    assert "freeze manifest" in joined
    assert "dataset.revision" in joined
    assert "calibration provenance" in joined
    # Every independent fault is reported together, not one at a time.
    assert len(report.blockers) >= 4


def test_gate_blocks_on_incomplete_new_model_calibration_provenance(tmp_path):
    """§36 requires each new model's calibration artifacts before the freeze.

    A missing artifact hash must surface as a NAMED blocker, so nobody is
    tempted to invent one to clear the gate.
    """
    def mutate(d):
        d["models"]["gemma"]["calibration_provenance"].pop(
            "calibration_archive_sha256"
        )

    config = load_phase3_config(_write_config(tmp_path, mutate))
    assert config.calibration_provenance_gaps() == {
        "gemma": ["calibration_archive_sha256"]
    }
    blockers = check_readiness(config, manifest=None).blockers
    assert any(
        "calibration provenance for new model 'gemma'" in b
        and "calibration_archive_sha256" in b
        for b in blockers
    )


def test_replication_models_need_no_phase3_calibration_artifacts():
    """Qwen/Llama carry frozen Phase 2 pairs (§20.1); demanding Phase 3
    calibration artifacts from them would be an invented rule."""
    config = load_phase3_config(COMMITTED_CONFIG)
    assert set(config.calibration_provenance_gaps()) <= {"mistral", "gemma"}


def test_gate_rejects_a_synthetic_manifest_even_if_otherwise_complete(tmp_path):
    """A synthetic Phase 3B artifact can never authorize a real run."""
    config = load_phase3_config(
        _write_config(tmp_path, lambda d: d.update(ready_for_real_run=True))
    )
    manifest = {
        "frozen": True,
        "synthetic": True,
        "repository_commit": "abc",
        "cohort_membership_map": {"o": ["A"]},
        "final_margin_strata": {"qwen|KW": [1.0, 2.0]},
        "condition_specification": ["C0"],
        "prompt_version": "v1",
        "cohorts": {"A": ["i1"]},
    }
    report = check_readiness(config, manifest)
    assert report.ready is False
    assert any("SYNTHETIC" in b for b in report.blockers)


def test_gate_still_blocks_when_the_ready_flag_is_set_but_fields_are_unresolved(tmp_path):
    """The flag is not self-certifying; state is re-derived."""
    config = load_phase3_config(
        _write_config(tmp_path, lambda d: d.update(ready_for_real_run=True))
    )
    with pytest.raises(Phase3NotReadyError):
        assert_ready_for_real_run(config, manifest={"frozen": True, "synthetic": False})


# --- manifest -------------------------------------------------------------


# --- §36 minimum provenance shared by the manifest fixtures ---------------
# Digest values are obviously synthetic but structurally valid (lowercase
# 64-hex), because these fixtures exercise the VALIDATOR, not real
# artifacts. No real baseline, exclusion, candidate or trial file exists
# yet, and none is created by these tests.
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

ARTIFACT_HASHES_36 = {
    "phase3_config": SHA_A,
    "candidate_file": SHA_B,
    "trial_file": SHA_C,
}
ENVIRONMENT_36 = {
    "python": "3.10.13",
    "torch": "2.3.1",
    "cuda": "12.1",
    "transformers": "4.44.0",
    "datasets": "2.20.0",
    "accelerate": "0.33.0",
}
HARDWARE_36 = {"gpu_name": "NVIDIA GeForce RTX 3090", "vram": "24GiB"}
DATASET_36 = {
    "hf_dataset_id": "akariasai/PopQA",
    "split": "test",
    "revision": "0" * 40,
    "candidate_item_ids": ["syn-1"],
}


def model_provenance_36(**overrides):
    """The §36 per-model runtime + screening provenance every model needs."""
    entry = {
        "dtype": "float16",
        "quantization": "none",
        "device_map": {"": 0},
        "max_memory": {0: "23GiB"},
        "baseline_file_sha256": SHA_A,
        "exclusion_file_sha256": SHA_B,
        "knowledge_membership": {"KC": ["syn-1"], "KW": ["syn-2"]},
        "margins": {"syn-1": 0.4},
        "manual_review_decisions": [],
    }
    entry.update(overrides)
    return entry


def _manifest(**overrides):
    kwargs = {
        "seed": 42,
        "repository_commit": None,
        "dataset": {"hf_dataset_id": "akariasai/PopQA", "revision": None},
        "models": {
            "qwen": {
                "hf_model_id": None,
                "revision": None,
                "model_specific_arm_enabled": True,
                "preferred_source": "a government website",
                "dispreferred_source": "an anonymous online forum post",
                "condition_set": ["C0", "K1", "K2", "K3", "K4", "M1", "M2"],
            }
        },
        "prompt_version": "v1",
        "cohorts": {
            "A": {
                **planned_cohort_a_design(),
                "selected_item_ids": ["syn-1"],
                "realized_total": 1,
                "status": "ELIGIBILITY_LIMITED",
                "eligibility_limited": True,
                "shortfall": {"low": 31, "medium": 32, "high": 32},
                "excluded_phase2_item_ids": [f"syn-old-{i}" for i in range(30)],
                "realized_relation_distribution": {"country": 1},
                "relation_dominance_share": 1.0,
                "relation_dominance_flag": True,
                "per_stratum_selected": {"low": 1, "medium": 0, "high": 0},
            },
            "B": {
                "qwen|KW": {
                    "target_cell_count": 8,
                    "minimum_cell_count": 6,
                    "original_cell_counts": {"country|low": 9},
                    "qualifying_relations": [
                        "country", "sport", "place of birth", "mother",
                    ],
                    "excluded_short_relations": [],
                    "realized_relations": [
                        "country", "sport", "place of birth", "mother",
                    ],
                    "realized_cell_count": 8,
                    "status": "COMPLETE",
                    "confirmatory_eligible": True,
                    "selected_item_ids": ["syn-1"],
                    "reduction_reason": None,
                }
            },
            "C": {
                "items": [
                    {
                        "item_id": "syn-1",
                        "relation": "country",
                        "per_model": {
                            "qwen": {
                                "knowledge_group": "KW",
                                "parametric_margin": 1.0,
                                "margin_stratum": "low",
                            }
                        },
                    }
                ]
            },
        },
        "cohort_membership_map": {"obs1": ["A"]},
        "deduplication_alias_map": {"qwen|syn-1|M1": "obs1"},
        "final_margin_strata": {"qwen|KW": [1.0, 2.0]},
        "screening": {**planned_screening_design(), "stopped_reason": "ceiling_reached"},
        "nominal_condition_slots": 7,
        "unique_observations": 5,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_manifest_is_synthetic_and_unfrozen_by_default():
    manifest = _manifest()
    assert manifest.is_synthetic is True
    assert manifest.is_frozen is False
    assert manifest.data["ready_for_real_run"] is False


def test_manifest_contains_the_cohort_membership_map():
    manifest = _manifest()
    assert manifest.data["cohort_membership_map"] == {"obs1": ["A"]}
    assert manifest.data["deduplication_alias_map"]


def test_manifest_records_nominal_and_unique_counts_separately():
    """§23: nominal condition slots and unique generations are different
    quantities and are recorded separately."""
    compute = _manifest().data["compute"]
    assert compute["nominal_condition_slots"] == 7
    assert compute["unique_observations"] == 5
    assert compute["collapsed_by_deduplication"] == 2


def test_manifest_records_unresolved_phase3c_fields_as_null():
    manifest = _manifest()
    assert manifest.data["environment"]["torch"] is None
    assert manifest.data["hardware"]["gpu_name"] is None
    assert manifest.data["repository_commit"] is None


def test_manifest_carries_the_analysis_status_table():
    statuses = {e["status"] for e in _manifest().data["analysis_status"]}
    assert STATUS_PRIMARY in statuses
    assert STATUS_SECONDARY in statuses
    primary = [
        e for e in _manifest().data["analysis_status"] if e["status"] == STATUS_PRIMARY
    ]
    assert len(primary) == 1
    assert primary[0]["cohort"] == "A"


def test_validate_refuses_a_synthetic_manifest():
    problems = validate_manifest(_manifest())
    assert any("SYNTHETIC" in p for p in problems)


def test_validate_flags_unresolved_model_revisions():
    problems = validate_manifest(_manifest(synthetic=False))
    assert any("unresolved hf_model_id/revision" in p for p in problems)


def test_validate_flags_impossible_compute_counts():
    problems = validate_manifest(
        _manifest(synthetic=False, nominal_condition_slots=3, unique_observations=9)
    )
    assert any("exceeds nominal_condition_slots" in p for p in problems)


def test_freeze_refuses_an_unresolved_manifest():
    with pytest.raises(ManifestError, match="unresolved requirements"):
        freeze_manifest(_manifest())


def test_freeze_succeeds_only_when_everything_is_resolved():
    manifest = _manifest(
        synthetic=False,
        repository_commit="d684f39",
        dataset=dict(DATASET_36),
        artifact_hashes=dict(ARTIFACT_HASHES_36),
        environment=dict(ENVIRONMENT_36),
        hardware=dict(HARDWARE_36),
        models={
            "qwen": {
                "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": FROZEN_MODEL_REVISIONS["qwen"]["revision"],
                "model_specific_arm_enabled": True,
                "preferred_source": "a government website",
                "dispreferred_source": "an anonymous online forum post",
                "condition_set": ["C0", "K1", "K2", "K3", "K4", "M1", "M2"],
                **model_provenance_36(),
            }
        },
    )
    frozen = freeze_manifest(manifest)
    assert frozen.is_frozen is True
    assert frozen.data["ready_for_real_run"] is True


# --- analysis status ------------------------------------------------------


def test_default_registry_has_exactly_one_primary_test():
    registry = default_registry()
    assert len(registry.by_status(STATUS_PRIMARY)) == 1
    assert registry.primary().name == "cohort_a_qwen_corrective_frozen_pair"
    assert registry.primary().cohort == "A"


def test_primary_family_requires_no_multiplicity_correction():
    registry = default_registry()
    assert registry.requires_multiplicity_correction("primary") is False
    assert registry.requires_multiplicity_correction("secondary") is True


def test_registry_rejects_a_second_primary_analysis():
    # Both in cohort A, so this isolates the one-primary rule rather than
    # tripping the "primary must be Cohort A" guard first.
    with pytest.raises(AnalysisStatusError, match="exactly ONE test"):
        AnalysisRegistry(
            [
                AnalysisEntry("a", "A", "o", "c", STATUS_PRIMARY, "primary"),
                AnalysisEntry("b", "A", "o", "c", STATUS_PRIMARY, "primary"),
            ]
        )


def test_primary_must_live_in_cohort_a():
    """The sole primary test is the Cohort A replication (§28, §44)."""
    with pytest.raises(AnalysisStatusError, match="but sits in cohort"):
        AnalysisEntry("b", "B", "o", "c", STATUS_PRIMARY, "primary")


def test_cohort_a_never_enters_the_secondary_family():
    """§28: 'Cohort A never enters the secondary family.'"""
    with pytest.raises(AnalysisStatusError, match="never enters the secondary"):
        AnalysisEntry("a2", "A", "o", "c", STATUS_SECONDARY, "secondary")
    registry = default_registry()
    for name in registry.secondary_family():
        entry = next(e for e in registry.entries if e.name == name)
        assert entry.cohort != "A"


def test_status_and_family_must_agree():
    """An exploratory analysis cannot be filed into a confirmatory family --
    the mechanism by which promotion would happen."""
    with pytest.raises(AnalysisStatusError, match="frozen design pairs it with"):
        AnalysisEntry("x", "B", "o", "c", STATUS_EXPLORATORY, "secondary")
    with pytest.raises(AnalysisStatusError):
        AnalysisEntry("y", "B", "o", "c", STATUS_DIAGNOSTIC, "primary")


def test_registry_entries_are_immutable():
    """A declared status cannot be reassigned, so an exploratory analysis
    cannot be promoted to confirmatory after results exist (§28, §44)."""
    entry = default_registry().entries[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.status = STATUS_PRIMARY


def test_unknown_status_is_rejected():
    with pytest.raises(AnalysisStatusError, match="Unknown analysis status"):
        AnalysisEntry("z", "A", "o", "c", "SOMEWHAT CONFIRMATORY", "primary")


# --- §36 provenance: each omission independently blocks a real freeze ------


def _real_ready_manifest(mutate=None):
    """A manifest that would otherwise be freezable, so each test can remove
    exactly one required field and prove that alone blocks the freeze."""
    manifest = _manifest(
        synthetic=False,
        repository_commit="d684f39",
        dataset=dict(DATASET_36),
        artifact_hashes=dict(ARTIFACT_HASHES_36),
        environment=dict(ENVIRONMENT_36),
        hardware=dict(HARDWARE_36),
        models={
            "qwen": {
                "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": FROZEN_MODEL_REVISIONS["qwen"]["revision"],
                "model_specific_arm_enabled": True,
                "preferred_source": "a government website",
                "dispreferred_source": "an anonymous online forum post",
                "condition_set": ["C0", "K1", "K2", "K3", "K4", "M1", "M2"],
                **model_provenance_36(),
            }
        },
    )
    if mutate:
        mutate(manifest.data)
    return manifest


def test_baseline_real_ready_manifest_freezes():
    assert freeze_manifest(_real_ready_manifest()).is_frozen is True


def test_missing_cohort_a_exclusion_list_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["A"].pop("excluded_phase2_item_ids")
    )
    assert any("excluded_phase2_item_ids" in p for p in validate_manifest(manifest))
    with pytest.raises(ManifestError):
        freeze_manifest(manifest)


def test_empty_cohort_a_exclusion_list_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["A"].update(excluded_phase2_item_ids=[])
    )
    assert any("exclusion list is empty" in p for p in validate_manifest(manifest))


def test_missing_cohort_a_relation_distribution_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["A"].pop("realized_relation_distribution")
    )
    assert any("realized_relation_distribution" in p for p in validate_manifest(manifest))


def test_missing_cohort_a_dominance_diagnostic_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["A"].pop("relation_dominance_flag")
    )
    assert any("relation_dominance_flag" in p for p in validate_manifest(manifest))


def test_cohort_a_as_a_bare_id_list_blocks_freeze():
    manifest = _real_ready_manifest(lambda d: d["cohorts"].update(A=["syn-1"]))
    assert any("provenance mapping" in p for p in validate_manifest(manifest))


def test_cohort_b_reduction_without_original_cell_counts_blocks_freeze():
    def mutate(d):
        group = d["cohorts"]["B"]["qwen|KW"]
        group["excluded_short_relations"] = ["mother"]
        group["original_cell_counts"] = {}

    manifest = _real_ready_manifest(mutate)
    assert any("original_cell_counts" in p for p in validate_manifest(manifest))


def test_cohort_b_reduction_without_a_reason_blocks_freeze():
    def mutate(d):
        group = d["cohorts"]["B"]["qwen|KW"]
        group["excluded_short_relations"] = ["mother"]
        group["reduction_reason"] = None

    manifest = _real_ready_manifest(mutate)
    assert any("reduction_reason" in p for p in validate_manifest(manifest))


def test_missing_cohort_b_field_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["B"]["qwen|KW"].pop("confirmatory_eligible")
    )
    assert any("confirmatory_eligible" in p for p in validate_manifest(manifest))


def test_cohort_c_without_per_model_knowledge_labels_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["C"]["items"][0].update(per_model={})
    )
    assert any("per-model knowledge state" in p for p in validate_manifest(manifest))


def test_cohort_c_missing_a_per_model_field_blocks_freeze():
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["C"]["items"][0]["per_model"]["qwen"].pop(
            "knowledge_group"
        )
    )
    assert any("knowledge_group" in p for p in validate_manifest(manifest))


def test_cohort_c_as_a_bare_id_list_blocks_freeze():
    manifest = _real_ready_manifest(lambda d: d["cohorts"].update(C=["syn-1"]))
    assert any("bare item-id list" in p for p in validate_manifest(manifest))


def test_setting_ready_for_real_run_by_hand_does_not_bypass_validation():
    """Boolean spoofing: the flag is not self-certifying."""
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["A"].pop("excluded_phase2_item_ids")
    )
    manifest.data["ready_for_real_run"] = True
    manifest.data["frozen"] = True
    assert validate_manifest(manifest)  # still reports problems
    with pytest.raises(ManifestError):
        freeze_manifest(manifest)


def test_gate_requires_cohort_a_exclusion_provenance(tmp_path):
    config = load_phase3_config(
        _write_config(tmp_path, lambda d: d.update(ready_for_real_run=True))
    )
    manifest = _real_ready_manifest(
        lambda d: d["cohorts"]["A"].update(excluded_phase2_item_ids=[])
    ).data
    manifest["frozen"] = True
    report = check_readiness(config, manifest)
    assert any("exclusion list" in b for b in report.blockers)

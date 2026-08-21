"""Tests for the Phase 3 config schema, manifest, and real-run safety gate.

Covers `docs/phase3_scaled_study_design.md` §7/§20 (frozen artifacts and
source pairs; unresolved new families), §36 (freeze manifest) and §41
("NO REAL PHASE 3 MODEL MAY RUN BEFORE 3C IS FROZEN").
"""

from __future__ import annotations

import dataclasses

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
    assert set(config.models) == {"qwen", "llama", "model_c", "model_d"}


def test_committed_config_pins_frozen_replication_artifacts():
    config = load_phase3_config(COMMITTED_CONFIG)
    for key in ("qwen", "llama"):
        entry = config.model(key)
        assert entry.hf_model_id == FROZEN_MODEL_REVISIONS[key]["hf_model_id"]
        assert entry.revision == FROZEN_MODEL_REVISIONS[key]["revision"]
        assert entry.resolved is True


def test_committed_config_leaves_new_families_unresolved():
    """The approved families have no exact release/SHA until Phase 3C."""
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.unresolved_models() == ["model_c", "model_d"]
    for key in ("model_c", "model_d"):
        entry = config.model(key)
        assert entry.hf_model_id is None
        assert entry.revision is None
        assert entry.preferred_source is None
        assert entry.dispreferred_source is None
    assert {config.model("model_c").family, config.model("model_d").family} == {
        "Mistral-7B-Instruct",
        "Gemma-2-9B-it",
    }


def test_committed_config_is_not_marked_ready_for_a_real_run():
    assert load_phase3_config(COMMITTED_CONFIG).ready_for_real_run is False


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


def test_config_rejects_an_invented_new_model_id(tmp_path):
    """Inventing a Mistral/Gemma repository id now is explicitly forbidden."""
    path = _write_config(
        tmp_path,
        lambda d: d["models"]["model_c"].update(
            hf_model_id="mistralai/Mistral-7B-Instruct-v0.3"
        ),
    )
    with pytest.raises(Phase3ConfigError, match="must remain null until Phase 3C"):
        load_phase3_config(path)


def test_config_rejects_an_invented_new_model_revision(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["models"]["model_d"].update(revision="a" * 40)
    )
    with pytest.raises(Phase3ConfigError, match="must remain null until Phase 3C"):
        load_phase3_config(path)


def test_config_rejects_invented_new_model_source_roles(tmp_path):
    path = _write_config(
        tmp_path,
        lambda d: d["models"]["model_c"].update(
            preferred_source="a government website"
        ),
    )
    with pytest.raises(Phase3ConfigError, match="must remain null until Phase 3C"):
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


def test_gate_reports_every_blocker_not_just_the_first():
    config = load_phase3_config(COMMITTED_CONFIG)
    report = check_readiness(config, manifest=None)
    assert report.ready is False
    joined = " ".join(report.blockers)
    assert "model id/revision unresolved" in joined
    assert "source roles" in joined
    assert "freeze manifest" in joined


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


def _manifest(**overrides):
    kwargs = {
        "seed": 42,
        "repository_commit": None,
        "dataset": {"hf_dataset_id": "akariasai/PopQA", "revision": None},
        "models": {"qwen": {"hf_model_id": None, "revision": None}},
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
        dataset={"hf_dataset_id": "akariasai/PopQA", "revision": "0" * 40},
        models={
            "qwen": {
                "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": FROZEN_MODEL_REVISIONS["qwen"]["revision"],
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
    with pytest.raises(AnalysisStatusError, match="exactly ONE test"):
        AnalysisRegistry(
            [
                AnalysisEntry("a", "A", "o", "c", STATUS_PRIMARY, "primary"),
                AnalysisEntry("b", "B", "o", "c", STATUS_PRIMARY, "primary"),
            ]
        )


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
        dataset={"hf_dataset_id": "akariasai/PopQA", "revision": "0" * 40},
        models={
            "qwen": {
                "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": FROZEN_MODEL_REVISIONS["qwen"]["revision"],
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

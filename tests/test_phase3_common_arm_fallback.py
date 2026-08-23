"""Adversarial tests for the frozen §34 common-arm-only fallback.

`docs/phase3_scaled_study_design.md` §20.2 and §34 require that a new
model whose calibration is tied, unstable, or heavily malformed runs the
**common arm only**: its `M1`/`M2` arm is disabled, its source roles stay
null, and it contributes to H2a but not to the model-specific family.

The property these tests pin down is that **three states never collapse**:

    ENABLED                  roles present, M1/M2 generated
    DISABLED_BY_CALIBRATION  roles null + explicit reason + provenance
    UNRESOLVED               roles null, no reason -- still blocks

Without the reason/provenance requirement an uncalibrated model would be
indistinguishable from a deliberate common-arm-only one, and a contrast
that was never measured could later be read as a measured null.
"""

from __future__ import annotations

import inspect

import pytest
import yaml

from conflict_eval.phase3.analysis_status import (
    FAMILY_NONE,
    default_registry,
    mark_not_applicable,
)
from conflict_eval.phase3.conditions import (
    DisabledArmWithSourceRoleError,
    UnresolvedSourceRolesError,
    build_phase3_conditions,
)
from conflict_eval.phase3.config import (
    ARM_DISABLED_BY_CALIBRATION,
    ARM_ENABLED,
    ARM_UNRESOLVED,
    Phase3ConfigError,
    load_phase3_config,
)
from conflict_eval.phase3.constants import (
    CONDITIONS_COMMON_ARM_ONLY,
    CONDITIONS_WITH_MODEL_SPECIFIC_ARM,
    FROZEN_MODEL_SOURCE_PAIRS,
    STATUS_DIAGNOSTIC,
    STATUS_EXPLORATORY,
    STATUS_NOT_APPLICABLE,
    STATUS_PRIMARY,
    STATUS_SECONDARY,
)
from conflict_eval.phase3.dedup import (
    ConditionRequest,
    GenerationIdentity,
    deduplicate_requests,
)
from conflict_eval.phase3.manifest import model_arm_provenance, validate_manifest
from conflict_eval.phase3.real_run_gate import (
    Phase3NotReadyError,
    assert_ready_for_real_run,
    check_readiness,
)

COMMITTED_CONFIG = "configs/phase3/phase3_study.yaml"
QWEN = FROZEN_MODEL_SOURCE_PAIRS["qwen"]


def _write_config(tmp_path, mutate=None):
    with open(COMMITTED_CONFIG, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if mutate:
        mutate(data)
    path = tmp_path / "phase3.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- config-level arm state -----------------------------------------------


def test_disabled_arm_with_null_roles_and_valid_reason_is_accepted():
    config = load_phase3_config(COMMITTED_CONFIG)
    for key in ("mistral", "gemma"):
        entry = config.model(key)
        assert entry.arm_state == ARM_DISABLED_BY_CALIBRATION
        assert entry.preferred_source is None
        assert entry.dispreferred_source is None
        assert entry.model_specific_arm_reason
        assert entry.calibration_provenance


def test_disabled_arm_without_a_reason_is_rejected(tmp_path):
    path = _write_config(
        tmp_path, lambda d: d["models"]["mistral"].pop("model_specific_arm_reason")
    )
    with pytest.raises(Phase3ConfigError, match="no model_specific_arm_reason"):
        load_phase3_config(path)


def test_disabled_arm_with_a_non_null_role_is_rejected(tmp_path):
    path = _write_config(
        tmp_path,
        lambda d: d["models"]["gemma"].update(dispreferred_source="a personal blog"),
    )
    with pytest.raises(Phase3ConfigError, match="must leave both source roles null"):
        load_phase3_config(path)


def test_enabled_arm_missing_a_role_is_rejected(tmp_path):
    def mutate(d):
        d["models"]["gemma"].update(
            model_specific_arm_enabled=True,
            dispreferred_source="a personal blog",
        )

    path = _write_config(tmp_path, mutate)
    with pytest.raises(Phase3ConfigError, match="requires BOTH roles"):
        load_phase3_config(path)


def test_unresolved_arm_state_still_blocks_a_real_run(tmp_path):
    """The discriminator: no reason means UNRESOLVED, which keeps blocking."""

    def mutate(d):
        entry = d["models"]["gemma"]
        entry.pop("model_specific_arm_enabled")
        entry.pop("model_specific_arm_reason")

    config = load_phase3_config(_write_config(tmp_path, mutate))
    assert config.model("gemma").arm_state == ARM_UNRESOLVED
    assert "gemma" in config.unresolved_source_roles()
    report = check_readiness(config, manifest=None)
    assert any("arm state unresolved" in b for b in report.blockers)
    with pytest.raises(Phase3NotReadyError):
        assert_ready_for_real_run(config, manifest=None)


def test_disabled_arm_does_not_block_on_arm_state(tmp_path):
    """A legitimately disabled arm is not an unresolved one: it must not
    appear as an arm-state blocker (other freeze gates still apply)."""
    config = load_phase3_config(COMMITTED_CONFIG)
    blockers = check_readiness(config, manifest=None).blockers
    assert not any("arm state unresolved" in b for b in blockers)


def test_qwen_and_llama_remain_enabled_with_frozen_pairs():
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.model("qwen").arm_state == ARM_ENABLED
    assert config.model("llama").arm_state == ARM_ENABLED
    assert config.model("qwen").preferred_source == "a government website"
    assert config.model("qwen").dispreferred_source == "an anonymous online forum post"
    assert config.model("llama").dispreferred_source == "a social media post"


def test_common_pair_is_still_frozen_exactly():
    config = load_phase3_config(COMMITTED_CONFIG)
    assert config.common_source_a == "a government website"
    assert config.common_source_b == "an anonymous online forum post"


# --- condition construction -----------------------------------------------


@pytest.mark.parametrize("group", ["KC", "KW"])
def test_disabled_arm_builds_common_conditions_only(group):
    specs = build_phase3_conditions(
        knowledge_group=group,
        gold_answer="gold",
        baseline_answer="memory",
        foil_answer="foil",
        model_preferred_source=None,
        model_dispreferred_source=None,
        model_specific_arm_enabled=False,
    )
    assert [s.condition for s in specs] == list(CONDITIONS_COMMON_ARM_ONLY)
    # No placeholder M records: those generations never happen.
    assert not any(s.condition.startswith("M") for s in specs)
    assert not any(s.arm == "model_specific" for s in specs)


def test_disabled_arm_still_builds_the_full_common_arm():
    specs = build_phase3_conditions(
        "KW", "gold", "memory", "foil", None, None, model_specific_arm_enabled=False
    )
    by_condition = {s.condition: s for s in specs}
    assert by_condition["K1"].source_label == "a government website"
    assert by_condition["K2"].source_label == "an anonymous online forum post"
    assert by_condition["K1"].conflict_status == "conflict"  # KW + gold
    assert by_condition["K3"].conflict_status == "agreement"


def test_enabled_arm_still_builds_all_seven_conditions():
    specs = build_phase3_conditions(
        "KW", "gold", "memory", "foil",
        QWEN["preferred_source"], QWEN["dispreferred_source"],
    )
    assert [s.condition for s in specs] == list(CONDITIONS_WITH_MODEL_SPECIFIC_ARM)


def test_disabled_arm_carrying_a_source_role_is_refused():
    with pytest.raises(DisabledArmWithSourceRoleError):
        build_phase3_conditions(
            "KW", "gold", "memory", "foil",
            "a government website", None,
            model_specific_arm_enabled=False,
        )


def test_enabled_arm_without_roles_is_still_refused():
    with pytest.raises(UnresolvedSourceRolesError):
        build_phase3_conditions("KW", "gold", "memory", "foil", None, None)


# --- workload / dedup accounting ------------------------------------------


def _request(model, item, condition, prompt):
    return ConditionRequest(
        model_key=model, item_id=item, condition=condition, arm="common",
        source_role="identity_a", source_label="a government website",
        evidence_truth="true", conflict_status="conflict", knowledge_group="KW",
        rendered_prompt=prompt, cohorts=("B",),
    )


def test_dedup_accounting_uses_the_actual_enabled_condition_set():
    """A common-arm-only model contributes 5 slots per item, not 7."""
    identity = GenerationIdentity("gemma", "rev", "v1", False, 1, 32, seed=42)
    specs = build_phase3_conditions(
        "KW", "gold", "memory", "foil", None, None, model_specific_arm_enabled=False
    )
    requests = [
        _request("gemma", "syn-1", s.condition, f"P|{s.condition}") for s in specs
    ]
    result = deduplicate_requests(requests, seed=1, identity=identity)
    assert result.nominal_slots == len(CONDITIONS_COMMON_ARM_ONLY) == 5
    assert result.unique_observations == 5
    assert result.collapsed_slots == 0


def test_common_arm_analyses_remain_available_for_a_disabled_model():
    identity = GenerationIdentity("mistral", "rev", "v1", False, 1, 32, seed=42)
    specs = build_phase3_conditions(
        "KW", "gold", "memory", "foil", None, None, model_specific_arm_enabled=False
    )
    requests = [
        _request("mistral", "syn-1", s.condition, f"P|{s.condition}") for s in specs
    ]
    result = deduplicate_requests(requests, seed=1, identity=identity)
    for condition in ("K1", "K2", "K3", "K4"):
        assert ("mistral", "syn-1", condition) in result.alias_map
    for condition in ("M1", "M2"):
        assert ("mistral", "syn-1", condition) not in result.alias_map


# --- manifest provenance ---------------------------------------------------


def test_manifest_records_the_arm_state_for_every_model():
    config = load_phase3_config(COMMITTED_CONFIG)
    for key in sorted(config.models):
        record = model_arm_provenance(config.model(key))
        assert record["hf_model_id"]
        assert record["requested_revision"] == record["resolved_revision"]
        assert record["dtype"] == "float16"
        assert record["quantization"] == "none"
        assert record["model_specific_arm_enabled"] in (True, False)
        assert record["condition_set"]


def test_manifest_disabled_model_records_null_roles_reason_and_provenance():
    config = load_phase3_config(COMMITTED_CONFIG)
    record = model_arm_provenance(config.model("gemma"))
    assert record["model_specific_arm_enabled"] is False
    assert record["preferred_source"] is None
    assert record["dispreferred_source"] is None
    assert record["model_specific_arm_reason"]
    assert record["calibration_provenance"]["parser_valid_trials"] == 0
    assert record["condition_set"] == list(CONDITIONS_COMMON_ARM_ONLY)
    # §36 artifacts travel INTO the manifest, not just the config.
    assert record["calibration_provenance"]["calibration_output_sha256"]
    assert record["calibration_provenance"]["preference_matrix"] is None


def test_manifest_enabled_model_records_both_roles():
    config = load_phase3_config(COMMITTED_CONFIG)
    record = model_arm_provenance(config.model("qwen"))
    assert record["model_specific_arm_enabled"] is True
    assert record["preferred_source"] == "a government website"
    assert record["condition_set"] == list(CONDITIONS_WITH_MODEL_SPECIFIC_ARM)


def _manifest_with_models(models, alias_map=None):
    class _Fake:
        def __init__(self, data):
            self.data = data

    return _Fake(
        {
            "models": models,
            "deduplication_alias_map": alias_map or {},
        }
    )


def test_manifest_rejects_a_model_without_a_declared_arm_state():
    manifest = _manifest_with_models(
        {"gemma": {"hf_model_id": "google/gemma-2-9b-it", "revision": "abc"}}
    )
    problems = validate_manifest(manifest)
    assert any("does not declare model_specific_arm_enabled" in p for p in problems)


def _disabled_model_entry(**overrides):
    """A manifest model entry for a §34 common-arm-only model.

    `role` defaults to "replication" so the §36 new-model calibration
    requirements do not fire; tests that care about those set it to "new".
    """
    entry = {
        "hf_model_id": "google/gemma-2-9b-it",
        "revision": "abc",
        "role": "replication",
        "model_specific_arm_enabled": False,
        "model_specific_arm_reason": "heavily malformed calibration",
        "calibration_provenance": {"parser_valid_trials": 0},
        "preferred_source": None,
        "dispreferred_source": None,
        "condition_set": list(CONDITIONS_COMMON_ARM_ONLY),
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize("condition", ["M1", "M2"])
def test_manifest_rejects_a_disabled_model_claiming_m_observations(condition):
    """A disabled arm generated no M1 AND no M2, so the manifest may carry
    neither. Both are asserted; the guard is a set membership test and a
    regression could easily cover only one."""
    manifest = _manifest_with_models(
        {"gemma": _disabled_model_entry()},
        alias_map={f"gemma|syn-1|{condition}": "obs-1"},
    )
    problems = validate_manifest(manifest)
    assert any("were never run" in p and condition in p for p in problems)


def test_manifest_rejects_a_disabled_arm_without_a_reason():
    manifest = _manifest_with_models(
        {"gemma": _disabled_model_entry(model_specific_arm_reason=None)}
    )
    problems = validate_manifest(manifest)
    assert any("without a model_specific_arm_reason" in p for p in problems)


def test_manifest_rejects_a_disabled_arm_without_calibration_provenance():
    manifest = _manifest_with_models(
        {"gemma": _disabled_model_entry(calibration_provenance=None)}
    )
    problems = validate_manifest(manifest)
    assert any("without calibration_provenance" in p for p in problems)


@pytest.mark.parametrize("missing_role", ["preferred_source", "dispreferred_source"])
def test_manifest_rejects_an_enabled_arm_missing_a_source_role(missing_role):
    entry = {
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "abc",
        "role": "replication",
        "model_specific_arm_enabled": True,
        "preferred_source": "a government website",
        "dispreferred_source": "an anonymous online forum post",
        "condition_set": list(CONDITIONS_WITH_MODEL_SPECIFIC_ARM),
    }
    entry[missing_role] = None
    problems = validate_manifest(_manifest_with_models({"qwen": entry}))
    assert any("is missing a source role" in p for p in problems)


def test_manifest_accepts_a_disabled_model_with_only_common_observations():
    manifest = _manifest_with_models(
        {"gemma": _disabled_model_entry()},
        alias_map={"gemma|syn-1|K1": "obs-1"},
    )
    problems = validate_manifest(manifest)
    assert not any("were never run" in p for p in problems)
    assert not any("gemma" in p and "arm" in p for p in problems)


def test_manifest_rejects_a_disabled_model_with_a_seven_condition_set():
    manifest = _manifest_with_models(
        {
            "gemma": _disabled_model_entry(
                condition_set=list(CONDITIONS_WITH_MODEL_SPECIFIC_ARM)
            )
        }
    )
    problems = validate_manifest(manifest)
    assert any("runs the common arm only" in p for p in problems)


# --- §36 calibration provenance -------------------------------------------


def _new_model_entry(**provenance_overrides):
    provenance = {
        "calibration_output_sha256": "f" * 64,
        "calibration_summary_sha256": "e" * 64,
        "calibration_archive_sha256": "d" * 64,
        "calibration_prompt_version": "v2",
        "parser_total_trials": 30,
        "parser_valid_trials": 0,
        "malformed_trials": 30,
        "parser_relaxed": False,
        "rerun_performed": False,
        "decision": "NO_MODEL_SPECIFIC_PAIR",
        "decision_reason": "all outputs malformed under the frozen parser",
        "preference_matrix": None,
    }
    provenance.update(provenance_overrides)
    for key, value in list(provenance.items()):
        if value is _ABSENT:
            del provenance[key]
    return _disabled_model_entry(role="new", calibration_provenance=provenance)


_ABSENT = object()


def test_manifest_accepts_a_complete_new_model_calibration_record():
    problems = validate_manifest(_manifest_with_models({"gemma": _new_model_entry()}))
    assert not any("calibration" in p for p in problems)


@pytest.mark.parametrize(
    "field",
    [
        "calibration_output_sha256",
        "calibration_summary_sha256",
        "calibration_archive_sha256",
        "decision_reason",
        "parser_total_trials",
        "calibration_prompt_version",
    ],
)
def test_manifest_rejects_a_new_model_missing_a_required_field(field):
    manifest = _manifest_with_models({"gemma": _new_model_entry(**{field: _ABSENT})})
    problems = validate_manifest(manifest)
    assert any(field in p and "§36 requires" in p for p in problems)


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-hash",
        "TBD",
        "pending",
        "F" * 64,  # uppercase
        "a" * 63,  # too short
        "a" * 65,  # too long -- the exact defect that blocked Mistral
        "g" * 64,  # non-hex
        "",
    ],
)
def test_manifest_rejects_a_placeholder_or_malformed_sha(bad):
    manifest = _manifest_with_models(
        {"gemma": _new_model_entry(calibration_output_sha256=bad)}
    )
    problems = validate_manifest(manifest)
    assert any("64-character hex SHA256" in p for p in problems)


@pytest.mark.parametrize("pointer", ["main", "latest", "master", "HEAD", "Main"])
@pytest.mark.parametrize(
    "field", ["revision", "requested_revision", "resolved_revision"]
)
def test_manifest_rejects_a_symbolic_revision(field, pointer):
    """A branch/tag pointer moves; a frozen manifest must pin a commit SHA.

    Enforced at the manifest layer as well as the config loader, because a
    manifest can be hand-assembled and it is what a replicator reads.
    """
    entry = _new_model_entry()
    entry[field] = pointer
    problems = validate_manifest(_manifest_with_models({"gemma": entry}))
    assert any("mutable pointer" in p and field in p for p in problems)


def test_manifest_accepts_an_exact_commit_sha_revision():
    entry = _new_model_entry()
    entry["revision"] = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
    problems = validate_manifest(_manifest_with_models({"gemma": entry}))
    assert not any("mutable pointer" in p for p in problems)


def test_manifest_rejects_inconsistent_parser_counts():
    manifest = _manifest_with_models(
        {"gemma": _new_model_entry(parser_valid_trials=5, malformed_trials=30)}
    )
    problems = validate_manifest(manifest)
    assert any("must equal parser_total_trials" in p for p in problems)


# --- analysis registry -----------------------------------------------------


def test_disabled_model_specific_contrasts_are_not_applicable_not_nulls():
    registry = mark_not_applicable(default_registry(), ["mistral", "gemma"])
    not_applicable = registry.by_status(STATUS_NOT_APPLICABLE)
    names = {e.name for e in not_applicable}
    # BOTH the corrective (KW) and harmful (KC) model-specific contrasts,
    # plus the Cohort C model-specific contrast: all three need an M1/M2
    # arm that was never generated.
    assert names == {
        "cohort_b_mistral_corrective",
        "cohort_b_gemma_corrective",
        "cohort_b_mistral_harmful",
        "cohort_b_gemma_harmful",
        "cohort_c_model_specific_mistral",
        "cohort_c_model_specific_gemma",
    }
    # Every one must say WHY, so it can never be read as a measured null.
    assert all(e.not_applicable_reason for e in not_applicable)
    assert all("NOT a null result" in e.not_applicable_reason for e in not_applicable)
    assert all(e.family == FAMILY_NONE for e in not_applicable)


def test_harmful_contrasts_are_not_applicable_while_common_rows_survive():
    """The §34 fallback removes the M1/M2 contrasts and keeps H2a."""
    registry = mark_not_applicable(default_registry(), ["mistral", "gemma"])
    for model in ("mistral", "gemma"):
        harmful = next(e for e in registry.by_model(model) if "harmful" in e.name)
        assert harmful.status == STATUS_NOT_APPLICABLE
        common = next(
            e for e in registry.by_model(model)
            if e.name == f"common_fixed_source_{model}"
        )
        assert common.status == STATUS_SECONDARY
        assert common.name in registry.secondary_family()


def test_not_applicable_contrasts_leave_the_holm_family():
    """A never-run contrast must not inflate the secondary family and
    penalise the tests that were actually run.

    Asserted RELATIVELY on purpose: no final Holm family size may be
    hard-coded, because realized membership still depends on Cohort B
    confirmatory eligibility and the other frozen eligibility rules.
    """
    before = default_registry()
    after = mark_not_applicable(before, ["mistral", "gemma"])
    removed = set(before.secondary_family()) - set(after.secondary_family())
    assert removed == {
        "cohort_b_mistral_corrective",
        "cohort_b_gemma_corrective",
        "cohort_b_mistral_harmful",
        "cohort_b_gemma_harmful",
        "cohort_c_model_specific_mistral",
        "cohort_c_model_specific_gemma",
    }
    assert len(after.secondary_family()) == len(before.secondary_family()) - len(
        removed
    )


def test_mark_not_applicable_uses_explicit_model_key_not_substrings():
    """A short or overlapping model key must not reach another model's
    analysis. Substring matching on `name` would do exactly that."""
    registry = default_registry()
    # "a" and "m" are substrings of many analysis names; neither is a
    # declared model_key, so nothing may change.
    for bogus in ("a", "m", "model", "cohort"):
        after = mark_not_applicable(registry, [bogus])
        assert after.by_status(STATUS_NOT_APPLICABLE) == ()
        assert after.secondary_family() == registry.secondary_family()
    # And disabling one real model must not touch the other.
    only_gemma = mark_not_applicable(registry, ["gemma"])
    touched = {e.model_key for e in only_gemma.by_status(STATUS_NOT_APPLICABLE)}
    assert touched == {"gemma"}


def test_mark_not_applicable_cannot_touch_primary_exploratory_or_diagnostic():
    """It may only ever move SECONDARY -> NOT APPLICABLE."""
    before = default_registry()
    after = mark_not_applicable(before, [k for k, _ in (("mistral", 0), ("gemma", 0))])
    assert after.primary().name == before.primary().name
    assert after.primary().status == STATUS_PRIMARY
    for status in (STATUS_EXPLORATORY, STATUS_DIAGNOSTIC):
        assert {e.name for e in after.by_status(status)} == {
            e.name for e in before.by_status(status)
        }
    # Every changed entry was SECONDARY before and NOT APPLICABLE after.
    by_name = {e.name: e for e in before.entries}
    for entry in after.entries:
        if entry.status != by_name[entry.name].status:
            assert by_name[entry.name].status == STATUS_SECONDARY
            assert entry.status == STATUS_NOT_APPLICABLE


def test_qwen_common_arm_contrast_is_counted_once():
    """§28/§44: Qwen's common-arm estimate rests on the same observations as
    its frozen-pair contrast, so it never re-enters as an independent test."""
    registry = default_registry()
    qwen_common = next(
        e for e in registry.entries if e.name == "common_fixed_source_qwen"
    )
    assert qwen_common.counted_once_with == "cohort_b_qwen_corrective"
    assert "common_fixed_source_qwen" not in registry.secondary_family()
    # Other models' common-arm contrasts ARE independent members.
    for model in ("llama", "mistral", "gemma"):
        assert f"common_fixed_source_{model}" in registry.secondary_family()


def test_registry_declares_no_final_holm_family_size():
    """The nominal declaration must not pretend Phase 3E selection is done."""
    import conflict_eval.phase3.analysis_status as module

    assert "pre-outcome nominal" in module.__doc__
    assert "NOT complete" in module.__doc__
    assert "not the realized Phase 3E" in module.AnalysisRegistry.secondary_family.__doc__
    # No literal family size may appear as a constant anywhere in the module.
    source = inspect.getsource(module)
    assert "len(secondary" not in source
    assert "SECONDARY_FAMILY_SIZE" not in source


def test_enabled_models_keep_their_secondary_contrasts():
    registry = mark_not_applicable(default_registry(), ["mistral", "gemma"])
    assert "cohort_b_qwen_corrective" in registry.secondary_family()
    assert "cohort_b_llama_corrective" in registry.secondary_family()
    assert registry.primary().name == "cohort_a_qwen_corrective_frozen_pair"


def test_marking_is_a_no_op_when_no_model_is_disabled():
    before = default_registry()
    after = mark_not_applicable(before, [])
    assert after.secondary_family() == before.secondary_family()
    assert after.by_status(STATUS_NOT_APPLICABLE) == ()


def test_common_arm_contrast_is_never_marked_not_applicable():
    """Disabling the model-specific arm must not remove the common-arm
    analysis -- that is precisely what these models still contribute."""
    registry = mark_not_applicable(default_registry(), ["mistral", "gemma"])
    for model in ("mistral", "gemma"):
        common = next(
            e for e in registry.entries if e.name == f"common_fixed_source_{model}"
        )
        assert common.status == STATUS_SECONDARY
        # H1 rests on margins, not on the M arm, so it survives too.
        h1 = next(
            e for e in registry.entries
            if e.name == f"parametric_strength_h1_{model}"
        )
        assert h1.status == STATUS_SECONDARY

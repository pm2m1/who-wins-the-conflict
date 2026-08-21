"""Integration test for the Phase 3 SYNTHETIC dry run.

Exercises the whole chain -- synthetic baselines -> screening -> margin
freeze -> Cohort A/B/C -> seven conditions -> prompt rendering ->
deduplication -> dummy generations -> reuse -> paired summary -> synthetic
manifest -- without loading a real model or touching the network
(`docs/phase3_scaled_study_design.md` §41; Phase 3B brief §17, §22).
"""

from __future__ import annotations

import json

import pytest

from conflict_eval.phase3.cohorts import STATE_COMPLETE
from conflict_eval.phase3.config import load_phase3_config
from conflict_eval.phase3.constants import DRYRUN_PREFIX
from conflict_eval.phase3.dryrun import SYNTHETIC_BANNER, run_synthetic_dryrun
from conflict_eval.phase3.real_run_gate import (
    Phase3NotReadyError,
    assert_ready_for_real_run,
)


@pytest.fixture(scope="module")
def report():
    return run_synthetic_dryrun(seed=42)


def test_dryrun_completes_the_whole_chain(report):
    assert report.screening_stopped_reason
    assert report.cohort_a_state == STATE_COMPLETE
    assert report.cohort_a_size > 0
    assert set(report.cohort_b_states) == {"KC", "KW"}
    assert report.cohort_c_size > 0
    assert report.nominal_slots > 0
    assert report.paired_summary


def test_dryrun_output_is_clearly_marked_synthetic(report):
    assert "SYNTHETIC" in report.banner
    assert report.banner == SYNTHETIC_BANNER
    assert report.manifest["synthetic"] is True


def test_dryrun_manifest_can_never_authorize_a_real_run(report):
    """Synthetic dry-run output must be rejected as a frozen artifact."""
    config = load_phase3_config("configs/phase3/phase3_study.yaml")
    with pytest.raises(Phase3NotReadyError):
        assert_ready_for_real_run(config, report.manifest)


def test_dryrun_manifest_is_not_frozen_and_not_ready(report):
    assert report.manifest["frozen"] is False
    assert report.manifest["ready_for_real_run"] is False


def test_dryrun_demonstrates_deduplication(report):
    """Qwen's frozen pair is the common pair, so M1/M2 collapse into the
    common arm -- nominal slots must exceed unique observations (§22, §23)."""
    assert report.unique_observations < report.nominal_slots
    assert report.collapsed_slots > 0
    assert (
        report.nominal_slots
        == report.unique_observations + report.collapsed_slots
    )


def test_dryrun_collapses_exactly_two_conditions_per_qwen_item(report):
    """For Qwen every item collapses M1->K and M2->K, i.e. 2 per item."""
    items = report.nominal_slots // 7
    assert report.collapsed_slots == 2 * items


def test_dryrun_paired_summary_reports_all_mandatory_fields(report):
    summary = report.paired_summary
    for field in (
        "n", "both", "a_only", "b_only", "neither", "discordant",
        "risk_difference", "ci_lower", "ci_upper", "exact_p",
    ):
        assert field in summary, field
    assert summary["both"] + summary["a_only"] + summary["b_only"] + summary[
        "neither"
    ] == summary["n"]


def test_dryrun_paired_summary_includes_saturation_diagnostics(report):
    diagnostics = report.paired_summary["diagnostics"]
    assert "saturated_uninformative" in diagnostics
    assert "low_information" in diagnostics
    assert "discordant_pairs" in diagnostics


def test_dryrun_is_deterministic():
    first = run_synthetic_dryrun(seed=7)
    second = run_synthetic_dryrun(seed=7)
    assert first.cohort_a_relations == second.cohort_a_relations
    assert first.unique_observations == second.unique_observations
    assert first.paired_summary == second.paired_summary


def test_dryrun_writes_output_under_a_dryrun_marked_filename(tmp_path):
    run_synthetic_dryrun(seed=42, output_dir=tmp_path)
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].name.startswith(DRYRUN_PREFIX)
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert "SYNTHETIC" in payload["_warning"]
    assert payload["manifest"]["synthetic"] is True


def test_dryrun_uses_no_real_model_and_no_network(monkeypatch):
    """Guard the Phase 3B rule directly: fail loudly if anything tries to
    load a real model or reach the Hub."""
    from conflict_eval.models import hf_causal

    def _explode(*args, **kwargs):
        raise AssertionError("dry run attempted real model/Hub access")

    monkeypatch.setattr(hf_causal, "resolve_model_revision", _explode, raising=False)
    monkeypatch.setattr(hf_causal, "build_hf_adapter", _explode, raising=False)
    run_synthetic_dryrun(seed=1)


def test_synthetic_ids_are_obviously_synthetic(report):
    cohort_a_ids = report.manifest["cohorts"]["A"]["selected_item_ids"]
    assert cohort_a_ids
    assert all(item_id.startswith("syn-") for item_id in cohort_a_ids)


def test_dryrun_manifest_carries_full_cohort_provenance(report):
    """The repaired manifest models §36 provenance rather than bare id
    lists, so a 3C freeze can record what the frozen design requires."""
    cohorts = report.manifest["cohorts"]

    cohort_a = cohorts["A"]
    for field in (
        "selected_item_ids",
        "excluded_phase2_item_ids",
        "realized_relation_distribution",
        "relation_dominance_share",
        "relation_dominance_flag",
        "per_stratum_selected",
    ):
        assert field in cohort_a, field
    # The dry run supplies an explicit (synthetic) exclusion set.
    assert cohort_a["excluded_phase2_item_ids"]

    for group_key, group in cohorts["B"].items():
        for field in (
            "target_cell_count",
            "minimum_cell_count",
            "original_cell_counts",
            "qualifying_relations",
            "excluded_short_relations",
            "realized_relations",
            "realized_cell_count",
            "status",
            "confirmatory_eligible",
            "selected_item_ids",
        ):
            assert field in group, f"{group_key}:{field}"

    assert cohorts["C"]["items"]
    entry = cohorts["C"]["items"][0]
    assert entry["per_model"]
    for state in entry["per_model"].values():
        assert state["knowledge_group"] in ("KC", "KW")
        assert state["margin_stratum"] is not None

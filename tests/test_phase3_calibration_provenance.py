"""Adversarial tests for §36 calibration provenance.

`docs/phase3_scaled_study_design.md` §36 requires, per new model, "its
calibration output SHA256, preference matrix, and the researcher's stated
reason for the selected pair (§20.2)".

The threat these tests guard against is not a hostile actor -- it is
*transcription*. The calibration artifacts live on a separate GPU host and
their summaries are copied into this repository by hand, so a truncated
hash, a placeholder left in place, a miscounted trial total, or a matrix
that disagrees with its own summary are all realistic. Each must fail
loudly rather than reach the freeze manifest.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from conflict_eval.phase3.calibration_provenance import (
    CalibrationProvenanceError,
    missing_required_fields,
    validate_calibration_provenance,
)
from conflict_eval.phase3.config import Phase3ConfigError, load_phase3_config
from conflict_eval.phase3.constants import CALIBRATION_SOURCE_LABELS

COMMITTED_CONFIG = "configs/phase3/phase3_study.yaml"


def _mistral_provenance():
    config = load_phase3_config(COMMITTED_CONFIG)
    return copy.deepcopy(config.model("mistral").calibration_provenance)


def _gemma_provenance():
    config = load_phase3_config(COMMITTED_CONFIG)
    return copy.deepcopy(config.model("gemma").calibration_provenance)


def _write_config(tmp_path, mutate):
    with open(COMMITTED_CONFIG, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    mutate(data)
    path = tmp_path / "phase3.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- source labels stay in step with the frozen file -----------------------


def test_calibration_source_labels_match_the_frozen_sources_file():
    """A drift between the constant and configs/sources.yaml must be a test
    failure, never a silent provenance mismatch."""
    with open("configs/sources.yaml", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert list(CALIBRATION_SOURCE_LABELS) == list(data["source_labels"])
    assert len(CALIBRATION_SOURCE_LABELS) == 6


# --- SHA256 shape ----------------------------------------------------------


def test_valid_lowercase_64_hex_sha_is_accepted():
    record = _gemma_provenance()
    record["calibration_output_sha256"] = "0123456789abcdef" * 4
    validate_calibration_provenance("test", record)


@pytest.mark.parametrize(
    "bad",
    [
        "TBD",
        "pending",
        "<to be supplied>",
        "n/a",
        "calibration tied",
        "A" * 64,
        "abc",
        "0" * 63,
        "0" * 65,
        "z" * 64,
        123,
        None if False else "",
    ],
)
def test_placeholder_or_malformed_sha_is_rejected(bad):
    record = _gemma_provenance()
    record["calibration_archive_sha256"] = bad
    with pytest.raises(CalibrationProvenanceError, match="64-character hex SHA256"):
        validate_calibration_provenance("test", record)


def test_a_65_character_digest_is_rejected_and_reports_its_length():
    """The exact defect that left Mistral's output digest unrecordable."""
    record = _gemma_provenance()
    record["calibration_output_sha256"] = (
        "33c1665cb5ea91447807e7234aba9d7e400e5579761546662623f2d551574e7c2"
    )
    with pytest.raises(CalibrationProvenanceError, match="length 65"):
        validate_calibration_provenance("test", record)


# --- required-field presence ----------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "calibration_output_sha256",
        "calibration_summary_sha256",
        "calibration_archive_sha256",
        "decision",
        "decision_reason",
        "parser_total_trials",
        "parser_valid_trials",
        "malformed_trials",
        "calibration_prompt_version",
    ],
)
def test_missing_required_field_is_reported(field):
    record = _gemma_provenance()
    del record[field]
    assert field in missing_required_fields(record)


def test_a_complete_record_reports_no_missing_fields():
    assert missing_required_fields(_gemma_provenance()) == []


def test_both_new_models_have_complete_calibration_provenance():
    """Restored from the preserved Phase 3C-3 archives; every §36 field is
    present, so calibration no longer blocks the freeze."""
    assert missing_required_fields(_mistral_provenance()) == []
    assert missing_required_fields(_gemma_provenance()) == []


def test_committed_output_digests_match_the_preserved_artifacts():
    """These are the digests recomputed from the restored calibration JSONL
    files. Pinned so a later edit cannot quietly substitute another value.

    The Mistral digest in particular was once supplied as a 65-character
    string and correctly refused; this pins the verified 64-character one.
    """
    mistral = _mistral_provenance()
    gemma = _gemma_provenance()
    assert mistral["calibration_output_sha256"] == (
        "33c1665cb5ea9147807e7234aba9d7e400e5579761546682623f2d551574e7c2"
    )
    assert gemma["calibration_output_sha256"] == (
        "f65188136221392503333b14c2e0ca68b9ceab38cdd696ed990fa81dbc1d9da6"
    )
    for record in (mistral, gemma):
        for field in (
            "calibration_output_sha256",
            "calibration_summary_sha256",
            "calibration_archive_sha256",
        ):
            assert len(record[field]) == 64


def test_config_rejects_a_new_model_with_a_malformed_sha(tmp_path):
    path = _write_config(
        tmp_path,
        lambda d: d["models"]["gemma"]["calibration_provenance"].update(
            calibration_archive_sha256="TBD"
        ),
    )
    with pytest.raises(Phase3ConfigError, match="64-character hex SHA256"):
        load_phase3_config(path)


# --- trial counts ----------------------------------------------------------


def test_parser_counts_must_sum_to_the_total():
    record = _gemma_provenance()
    record["malformed_trials"] = 29
    with pytest.raises(CalibrationProvenanceError, match="must equal"):
        validate_calibration_provenance("test", record)


def test_committed_records_have_consistent_counts():
    for record in (_mistral_provenance(), _gemma_provenance()):
        assert (
            record["parser_valid_trials"] + record["malformed_trials"]
            == record["parser_total_trials"]
        )


def test_negative_trial_counts_are_rejected():
    record = _gemma_provenance()
    record["malformed_trials"] = -1
    with pytest.raises(CalibrationProvenanceError, match="non-negative integer"):
        validate_calibration_provenance("test", record)


def test_boolean_flags_must_be_boolean():
    record = _gemma_provenance()
    record["parser_relaxed"] = "false"
    with pytest.raises(CalibrationProvenanceError, match="must be a boolean"):
        validate_calibration_provenance("test", record)


# --- preference matrix -----------------------------------------------------


def test_zero_valid_trials_with_an_explicit_null_matrix_is_accepted():
    record = _gemma_provenance()
    assert record["parser_valid_trials"] == 0
    assert record["preference_matrix"] is None
    validate_calibration_provenance("test", record)
    assert missing_required_fields(record) == []


def test_valid_trials_with_a_null_matrix_is_rejected():
    record = _mistral_provenance()
    record["preference_matrix"] = None
    with pytest.raises(CalibrationProvenanceError, match="preference_matrix is null"):
        validate_calibration_provenance("test", record)


def test_zero_valid_trials_with_a_matrix_is_rejected():
    """0/30 parser-valid output cannot have produced a matrix."""
    record = _gemma_provenance()
    record["preference_matrix"] = _mistral_provenance()["preference_matrix"]
    with pytest.raises(CalibrationProvenanceError, match="nothing to build a matrix"):
        validate_calibration_provenance("test", record)


def test_an_absent_matrix_is_reported_as_missing_even_at_zero_valid_trials():
    """Absent is not the same as explicitly null: null is a recorded finding,
    absent is an unfinished record."""
    record = _gemma_provenance()
    del record["preference_matrix"]
    assert "preference_matrix" in missing_required_fields(record)


def test_the_committed_mistral_matrix_covers_all_fifteen_pairs():
    matrix = _mistral_provenance()["preference_matrix"]
    assert len(matrix) == 15
    pairs = {frozenset((p["source_a"], p["source_b"])) for p in matrix}
    assert len(pairs) == 15
    labels = set(CALIBRATION_SOURCE_LABELS)
    for pair in matrix:
        assert {pair["source_a"], pair["source_b"]} <= labels
        assert pair["a_wins"] + pair["b_wins"] == 2


def test_matrix_presentations_must_equal_parser_valid_trials():
    record = _mistral_provenance()
    record["parser_valid_trials"] = 28
    record["malformed_trials"] = 2
    with pytest.raises(CalibrationProvenanceError, match="accounts for 30"):
        validate_calibration_provenance("test", record)


def test_a_duplicated_pair_is_rejected():
    record = _mistral_provenance()
    record["preference_matrix"][1] = dict(record["preference_matrix"][0])
    with pytest.raises(CalibrationProvenanceError, match="appears more than once"):
        validate_calibration_provenance("test", record)


def test_a_reversed_duplicate_pair_is_rejected():
    """AB and BA are the SAME unordered pair; recording both would double
    count the presentations."""
    record = _mistral_provenance()
    first = record["preference_matrix"][0]
    record["preference_matrix"][1] = {
        "source_a": first["source_b"],
        "source_b": first["source_a"],
        "a_wins": 1,
        "b_wins": 1,
    }
    with pytest.raises(CalibrationProvenanceError, match="appears more than once"):
        validate_calibration_provenance("test", record)


def test_an_unknown_source_label_is_rejected():
    record = _mistral_provenance()
    record["preference_matrix"][0]["source_b"] = "a peer-reviewed journal"
    with pytest.raises(CalibrationProvenanceError, match="not one of the six frozen"):
        validate_calibration_provenance("test", record)


def test_a_pair_compared_with_itself_is_rejected():
    record = _mistral_provenance()
    record["preference_matrix"][0]["source_b"] = record["preference_matrix"][0][
        "source_a"
    ]
    with pytest.raises(CalibrationProvenanceError, match="with itself"):
        validate_calibration_provenance("test", record)


def test_a_pair_without_both_presentation_orders_is_rejected():
    record = _mistral_provenance()
    record["preference_matrix"][0]["a_wins"] = 1
    with pytest.raises(CalibrationProvenanceError, match="must equal 2"):
        validate_calibration_provenance("test", record)


def test_a_short_matrix_is_rejected():
    record = _mistral_provenance()
    record["preference_matrix"] = record["preference_matrix"][:14]
    record["parser_valid_trials"] = 28
    record["malformed_trials"] = 2
    with pytest.raises(CalibrationProvenanceError, match="enumerates all 15"):
        validate_calibration_provenance("test", record)


# --- summary/matrix agreement ---------------------------------------------


def test_stable_pairs_are_rederived_from_the_matrix():
    """The recorded summary must agree with the raw record it summarizes."""
    record = _mistral_provenance()
    record["stable_pairs"] = 11
    with pytest.raises(CalibrationProvenanceError, match="stable_pairs is recorded"):
        validate_calibration_provenance("test", record)


def test_order_reversal_pairs_are_rederived_from_the_matrix():
    record = _mistral_provenance()
    record["order_reversal_pairs"] = 0
    with pytest.raises(
        CalibrationProvenanceError, match="order_reversal_pairs is recorded"
    ):
        validate_calibration_provenance("test", record)


def test_the_committed_mistral_summary_matches_its_own_matrix():
    """Independent re-derivation of the transcribed 12 / 3."""
    record = _mistral_provenance()
    matrix = record["preference_matrix"]
    stable = sum(1 for p in matrix if p["a_wins"] != p["b_wins"])
    reversals = sum(1 for p in matrix if p["a_wins"] == p["b_wins"])
    assert stable == record["stable_pairs"] == 12
    assert reversals == record["order_reversal_pairs"] == 3


# --- decision vocabulary ---------------------------------------------------


def test_an_unknown_decision_is_rejected():
    record = _gemma_provenance()
    record["decision"] = "probably fine"
    with pytest.raises(CalibrationProvenanceError, match="decision must be one of"):
        validate_calibration_provenance("test", record)


def test_an_empty_decision_reason_is_rejected():
    record = _gemma_provenance()
    record["decision_reason"] = "   "
    with pytest.raises(CalibrationProvenanceError, match="decision_reason"):
        validate_calibration_provenance("test", record)


def test_mistral_heuristic_ranking_records_the_tie_that_forced_the_decision():
    """The bottom three tie at 0.20 -- the measured fact that makes a
    prespecified dispreferred source impossible. Recorded as evidence, never
    as a selection."""
    record = _mistral_provenance()
    ranking = record["heuristic_source_ranking"]
    assert len(ranking) == 6
    assert {r["source"] for r in ranking} == set(CALIBRATION_SOURCE_LABELS)
    assert ranking[0]["source"] == "a government website"
    assert ranking[0]["mean_pairwise_preference_rate"] == 1.0
    bottom = [r for r in ranking if r["mean_pairwise_preference_rate"] == 0.2]
    assert len(bottom) == 3
    assert {r["source"] for r in bottom} == {
        "a personal blog",
        "a social media post",
        "an anonymous online forum post",
    }
    # A measured top source is still not a pair: no role was assigned.
    assert record["decision"] == "NO_MODEL_SPECIFIC_PAIR"


def test_gemma_records_all_fifteen_pairs_as_malformed():
    """0 stable and 0 reversal pairs out of 0 MEASURED pairs -- not out of 15,
    which would imply 15 measured ties."""
    record = _gemma_provenance()
    assert record["malformed_pairs"] == 15
    assert record["stable_pairs"] == 0
    assert record["order_reversal_pairs"] == 0
    assert record["parser_valid_trials"] == 0
    assert record["preference_matrix"] is None


def test_no_rescue_was_applied_to_either_new_model():
    """§34's fallback is triggered, not worked around."""
    for record in (_mistral_provenance(), _gemma_provenance()):
        assert record["parser_relaxed"] is False
        assert record["rerun_performed"] is False
        assert record["calibration_prompt_version"] == "v2"


def test_both_new_models_decided_no_model_specific_pair():
    for record in (_mistral_provenance(), _gemma_provenance()):
        assert record["decision"] == "NO_MODEL_SPECIFIC_PAIR"
        assert record["decision_reason"].strip()
        assert record["parser_relaxed"] is False
        assert record["rerun_performed"] is False
        assert record["calibration_prompt_version"] == "v2"

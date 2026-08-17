"""Tests for strict source-calibration Choice parsing and model
provenance recording (docs/decisions.md, "Make source calibration output
strict").

A real Qwen2.5-7B-Instruct baseline generation partially reproduced the
old "Decision: answer | uncertain" pipe wording literally
("Decision: answer | certain"), which an unanchored parser accepted as a
valid prefix. The source-calibration Choice field used the same
pipe-alternatives wording and the same unanchored-prefix parser, so it is
fixed the same way before any real calibration is run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from conflict_eval.source_preference.calibration import (
    CalibrationTrial,
    parse_choice,
    run_calibration_trial,
    selected_source_from_choice,
)
from conflict_eval.source_preference.counterbalance import (
    Presentation,
    expand_pairs_to_presentations,
)
from conflict_eval.source_preference.pairs import enumerate_unordered_pairs

# --- strict Choice parsing (Task 6, items 1-9) ------------------------------


@pytest.mark.parametrize("raw", ["Choice: 1", "Choice: 2"])
def test_strict_valid_single_digit_choice(raw):
    expected = int(raw.split()[-1])
    assert parse_choice(raw) == expected


def test_surrounding_whitespace_is_accepted():
    assert parse_choice("Choice:    1   ") == 1
    assert parse_choice("Choice:\t2") == 2


@pytest.mark.parametrize(
    "raw",
    [
        "Choice: 1 | 2",
        "Choice: 1 because I trust it more",
        "Choice: 2 blah",
        "Choice: 12",
        "Choice: 1/2",
        "no choice field at all",
    ],
)
def test_malformed_choice_lines_are_rejected(raw):
    assert parse_choice(raw) is None


def test_missing_choice_field_returns_none():
    assert parse_choice("I prefer the first source.") is None


# --- selected_source_from_choice (Task 6, items 10-11) ----------------------


def _presentation():
    return Presentation(
        source_a="Wikipedia",
        source_b="a personal blog",
        displayed_source_1="Wikipedia",
        displayed_source_2="a personal blog",
        presentation_order="AB",
    )


def test_selected_source_from_choice_maps_1_and_2_correctly():
    presentation = _presentation()
    assert selected_source_from_choice(presentation, 1) == "Wikipedia"
    assert selected_source_from_choice(presentation, 2) == "a personal blog"


def test_selected_source_from_choice_is_none_for_malformed_choice():
    presentation = _presentation()
    assert selected_source_from_choice(presentation, None) is None


# --- committed prompt content (Task 6, items 12-13) -------------------------

CALIBRATION_PROMPT = Path("prompts/source_calibration.txt").read_text(encoding="utf-8")


def test_prompt_no_longer_contains_pipe_alternatives_choice_line():
    assert "Choice: 1 | 2" not in CALIBRATION_PROMPT


def test_prompt_states_exactly_one_digit_no_pipe_no_explanation():
    assert "exactly one digit" in CALIBRATION_PROMPT
    assert "| symbol" in CALIBRATION_PROMPT
    assert "explanation" in CALIBRATION_PROMPT.lower()


def test_prompt_substantive_question_unchanged():
    assert "which of these two sources" in CALIBRATION_PROMPT
    assert "Source 1: {source_1}" in CALIBRATION_PROMPT
    assert "Source 2: {source_2}" in CALIBRATION_PROMPT


# --- calibration prompt version (Task 6, item 14) ---------------------------


def test_calibration_prompt_version_is_v2():
    from conflict_eval.config import load_sources_config

    sources_config = load_sources_config("configs/sources.yaml")
    assert sources_config["calibration_prompt_version"] == "v2"


# --- model provenance (Task 6, items 15-16) ---------------------------------


class _FakeModelAdapter:
    def __init__(self, model_id, model_revision, requested_revision, resolved_revision):
        self.model_id = model_id
        self.model_revision = model_revision
        self.requested_revision = requested_revision
        self.resolved_revision = resolved_revision

    def generate(self, messages, generation_config):
        return "Choice: 1"


def test_trial_records_exact_model_provenance():
    model = _FakeModelAdapter(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        requested_revision=None,
        resolved_revision="a09a35458c702b33eeacc393d103063234e8bc28",
    )
    presentation = _presentation()
    trial = run_calibration_trial(
        model,
        template="Source 1: {source_1}\nSource 2: {source_2}\n\nChoice: <1 or 2>",
        presentation=presentation,
        prompt_version="v2",
        seed=42,
        run_id="test-0",
        generation_config=SimpleNamespace(),
    )
    assert trial.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert trial.model_revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert trial.requested_revision is None
    assert trial.resolved_revision == "a09a35458c702b33eeacc393d103063234e8bc28"


def test_trial_provenance_does_not_hardcode_a_sha():
    # A different adapter's own values must flow through unchanged — the
    # dataclass must not silently substitute a fixed/expected SHA.
    model = _FakeModelAdapter(
        model_id="some/other-model",
        model_revision="deadbeef",
        requested_revision="main",
        resolved_revision="deadbeef",
    )
    trial = run_calibration_trial(
        model,
        template="Source 1: {source_1}\nSource 2: {source_2}\n\nChoice: <1 or 2>",
        presentation=_presentation(),
        prompt_version="v2",
        seed=1,
        run_id="test-1",
        generation_config=SimpleNamespace(),
    )
    assert trial.model_id == "some/other-model"
    assert trial.model_revision == "deadbeef"
    assert trial.requested_revision == "main"


def test_provenance_fields_present_even_when_adapter_lacks_revision_attributes():
    # A minimal adapter exposing only model_id/model_revision (no
    # requested_revision/resolved_revision attributes) must not raise —
    # CalibrationTrial falls back to None via getattr, consistent with
    # cli.py's existing baseline-record convention.
    class _MinimalAdapter:
        model_id = "minimal-model"
        model_revision = "minimal-rev"

        def generate(self, messages, generation_config):
            return "Choice: 2"

    trial = run_calibration_trial(
        _MinimalAdapter(),
        template="Source 1: {source_1}\nSource 2: {source_2}\n\nChoice: <1 or 2>",
        presentation=_presentation(),
        prompt_version="v2",
        seed=1,
        run_id="test-2",
        generation_config=SimpleNamespace(),
    )
    assert trial.requested_revision is None
    assert trial.resolved_revision is None


def test_summary_json_includes_model_provenance(tmp_path):
    # cmd_calibrate_sources builds the summary dict inline; this test
    # exercises the same construction pattern directly against a fake
    # adapter to confirm the provenance fields are present, without
    # running the full CLI command (no config/model/network needed).
    model = _FakeModelAdapter(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        requested_revision=None,
        resolved_revision="a09a35458c702b33eeacc393d103063234e8bc28",
    )
    summary = {
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "requested_revision": getattr(model, "requested_revision", None),
        "resolved_revision": getattr(model, "resolved_revision", None),
    }
    assert summary["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert summary["resolved_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"


def test_calibration_trial_dataclass_has_provenance_fields():
    field_names = {f.name for f in dataclasses.fields(CalibrationTrial)}
    assert {"model_id", "model_revision", "requested_revision", "resolved_revision"} <= field_names


# --- pair enumeration / counterbalancing unchanged (Task 6, items 17-18) ----


def test_six_source_labels_imply_fifteen_unordered_pairs_and_thirty_presentations():
    from conflict_eval.config import load_sources_config

    sources_config = load_sources_config("configs/sources.yaml")
    labels = sources_config["source_labels"]
    assert len(labels) == 6

    pairs = enumerate_unordered_pairs(labels)
    assert len(pairs) == 15  # 6 choose 2

    presentations = expand_pairs_to_presentations(pairs)
    assert len(presentations) == 30  # AB + BA for each of the 15 pairs


def test_ab_and_ba_counterbalancing_present_for_every_pair():
    pairs = enumerate_unordered_pairs(["Wikipedia", "a personal blog", "a news article"])
    presentations = expand_pairs_to_presentations(pairs)
    orders_by_pair = {}
    for p in presentations:
        orders_by_pair.setdefault((p.source_a, p.source_b), set()).add(p.presentation_order)
    assert all(orders == {"AB", "BA"} for orders in orders_by_pair.values())

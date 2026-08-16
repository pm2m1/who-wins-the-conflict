import pytest

from conflict_eval.evaluation.metrics import (
    context_adoption_rate,
    corrective_override_rate,
    harmful_override_rate,
    source_effect_on_corrective_override,
    source_effect_on_harmful_override,
)


def _kc_conflict_record(source_role: str, context_adopted: bool) -> dict:
    return {
        "knowledge_group": "KC",
        "evidence_truth": "false",
        "conflict_status": "conflict",
        "source_role": source_role,
        "context_adopted": context_adopted,
    }


def _kw_conflict_record(source_role: str, context_adopted: bool) -> dict:
    return {
        "knowledge_group": "KW",
        "evidence_truth": "true",
        "conflict_status": "conflict",
        "source_role": source_role,
        "context_adopted": context_adopted,
    }


def test_context_adoption_rate_basic():
    records = [
        _kc_conflict_record("preferred", True),
        _kc_conflict_record("preferred", False),
        _kw_conflict_record("dispreferred", True),
        _kw_conflict_record("dispreferred", True),
    ]
    car = context_adoption_rate(records)
    assert car.n == 4
    assert car.rate == 0.75


def test_context_adoption_rate_rejects_agreement_trials():
    agreement_record = _kc_conflict_record("preferred", True)
    agreement_record["conflict_status"] = "agreement"
    with pytest.raises(ValueError):
        context_adoption_rate([agreement_record])


def test_context_adoption_rate_empty_input_returns_none_rate():
    car = context_adoption_rate([])
    assert car.rate is None
    assert car.n == 0


def test_harmful_override_rate_filters_to_kc_false_conflict():
    records = [
        _kc_conflict_record("preferred", True),
        _kc_conflict_record("preferred", False),
        _kw_conflict_record("preferred", True),  # must be excluded (KW, not KC)
    ]
    hor = harmful_override_rate(records)
    assert hor.n == 2
    assert hor.rate == 0.5


def test_corrective_override_rate_filters_to_kw_true_conflict():
    records = [
        _kw_conflict_record("preferred", True),
        _kw_conflict_record("preferred", True),
        _kw_conflict_record("preferred", False),
        _kc_conflict_record("preferred", True),  # must be excluded (KC, not KW)
    ]
    cor = corrective_override_rate(records)
    assert cor.n == 3
    assert cor.rate == pytest.approx(2 / 3)


def test_source_effect_on_harmful_override_delta():
    records = [
        _kc_conflict_record("preferred", True),
        _kc_conflict_record("preferred", True),
        _kc_conflict_record("dispreferred", False),
        _kc_conflict_record("dispreferred", False),
    ]
    effect = source_effect_on_harmful_override(records)
    assert effect["hor_preferred"].rate == 1.0
    assert effect["hor_dispreferred"].rate == 0.0
    assert effect["delta_harm"] == 1.0


def test_source_effect_on_corrective_override_delta():
    records = [
        _kw_conflict_record("preferred", True),
        _kw_conflict_record("preferred", False),
        _kw_conflict_record("dispreferred", False),
        _kw_conflict_record("dispreferred", False),
    ]
    effect = source_effect_on_corrective_override(records)
    assert effect["cor_preferred"].rate == 0.5
    assert effect["cor_dispreferred"].rate == 0.0
    assert effect["delta_correct"] == 0.5


def test_source_effect_delta_is_none_when_a_side_has_no_data():
    records = [_kc_conflict_record("preferred", True)]  # no dispreferred trials at all
    effect = source_effect_on_harmful_override(records)
    assert effect["hor_dispreferred"].rate is None
    assert effect["delta_harm"] is None

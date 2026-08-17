"""Tests for the metric-semantics distinction in
conflict_eval.analysis.summaries.condition_summary — in particular that
`parsed_answer_accuracy` (textual Answer-field correctness) is not
conflated with `context_adopted` (the primary committed outcome) or
`abstention_rate` (docs/decisions.md, "Freeze the first Qwen pilot after
validated analysis"; docs/methodology.md, "Metric semantics").

`final_accuracy` was the old, misleading name for this column: it read as
if it meant "accuracy of a committed final answer," but the prompt format
requires an `Answer:` field even under `Decision: uncertain`
(prompts/baseline.txt), so a record can have `final_correct == True`
while the model never committed to that answer (`context_adopted ==
False`, `decision == "uncertain"`).
"""

from __future__ import annotations

from conflict_eval.analysis.summaries import condition_summary


def _record(**overrides):
    base = {
        "model_id": "qwen",
        "knowledge_group": "KW",
        "condition": "C1",
        "conflict_status": "conflict",
        "context_adopted": False,
        "final_correct": False,
        "decision": "answer",
    }
    base.update(overrides)
    return base


def test_uncertain_but_textually_correct_record_reports_all_three_rates_correctly():
    # The exact scenario this cleanup targets: a "tentative answer" that
    # happens to match gold under an abstention. This must be:
    #   parsed_answer_accuracy = 1.0 (final_correct is True)
    #   abstention_rate = 1.0 (decision is "uncertain")
    #   context_adoption_rate = 0.0 (context_adopted is False)
    records = [
        _record(
            context_adopted=False,
            final_correct=True,
            decision="uncertain",
        )
    ]
    summary = condition_summary(records)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["parsed_answer_accuracy"] == 1.0
    assert row["abstention_rate"] == 1.0
    assert row["context_adoption_rate"] == 0.0


def test_final_accuracy_column_name_no_longer_present():
    records = [_record()]
    summary = condition_summary(records)
    assert "final_accuracy" not in summary.columns
    assert "parsed_answer_accuracy" in summary.columns


def test_committed_correct_adoption_still_reports_expected_rates():
    # A normal committed, correct, context-adopting record: all three
    # rates should independently reflect that.
    records = [
        _record(
            context_adopted=True,
            final_correct=True,
            decision="answer",
        )
    ]
    summary = condition_summary(records)
    row = summary.iloc[0]
    assert row["context_adoption_rate"] == 1.0
    assert row["parsed_answer_accuracy"] == 1.0
    assert row["abstention_rate"] == 0.0


def test_mixed_group_rates_are_computed_independently():
    records = [
        _record(context_adopted=False, final_correct=True, decision="uncertain"),
        _record(context_adopted=True, final_correct=True, decision="answer"),
    ]
    summary = condition_summary(records)
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["parsed_answer_accuracy"] == 1.0  # both textually correct
    assert row["context_adoption_rate"] == 0.5  # only one committed-adopted
    assert row["abstention_rate"] == 0.5  # only one uncertain

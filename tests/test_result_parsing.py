import pytest

from conflict_eval.evaluation.parse import parse_response
from conflict_eval.experiment.resume import make_record_key
from conflict_eval.io.results import ResultWriter


def test_parses_well_formed_response():
    raw = "Answer: Paris\nDecision: answer\nConfidence: 85"
    parsed = parse_response(raw)
    assert parsed.answer == "Paris"
    assert parsed.decision == "answer"
    assert parsed.confidence == 85
    assert not parsed.malformed


def test_parses_uncertain_decision():
    raw = "Answer: unknown\nDecision: uncertain\nConfidence: 20"
    parsed = parse_response(raw)
    assert parsed.decision == "uncertain"
    assert not parsed.malformed


def test_confidence_is_clamped_to_0_100():
    raw = "Answer: Paris\nDecision: answer\nConfidence: 250"
    parsed = parse_response(raw)
    assert parsed.confidence == 100


def test_missing_answer_field_is_malformed():
    raw = "Decision: answer\nConfidence: 50"
    parsed = parse_response(raw)
    assert parsed.malformed
    assert parsed.answer is None


def test_missing_decision_field_is_malformed():
    raw = "Answer: Paris\nConfidence: 50"
    parsed = parse_response(raw)
    assert parsed.malformed


def test_missing_confidence_alone_is_not_malformed():
    # Confidence is exploratory-only; its absence should not force
    # manual_review by itself.
    raw = "Answer: Paris\nDecision: answer"
    parsed = parse_response(raw)
    assert not parsed.malformed
    assert parsed.confidence is None


def test_completely_unstructured_response_is_malformed():
    raw = "I think it might be Paris, but I'm not fully sure."
    parsed = parse_response(raw)
    assert parsed.malformed


# --- strict Decision-line parsing -------------------------------------------
#
# A real Qwen2.5-7B-Instruct generation under the old prompt/parser
# produced "Decision: answer | certain", which the old prefix-matching
# regex accepted as decision="answer" — this matters because parsed.decision
# controls baseline KC/KW eligibility (docs/decisions.md, "Decision output
# format made strict").


def test_decision_answer_is_valid():
    raw = "Answer: Paris\nDecision: answer\nConfidence: 100"
    parsed = parse_response(raw)
    assert parsed.decision == "answer"
    assert not parsed.malformed


def test_decision_uncertain_is_valid():
    raw = "Answer: Paris\nDecision: uncertain\nConfidence: 20"
    parsed = parse_response(raw)
    assert parsed.decision == "uncertain"
    assert not parsed.malformed


def test_decision_is_case_insensitive():
    raw = "Answer: Paris\nDecision: ANSWER\nConfidence: 90"
    parsed = parse_response(raw)
    assert parsed.decision == "answer"
    assert not parsed.malformed


def test_decision_allows_ordinary_surrounding_whitespace():
    raw = "Answer: Paris\nDecision:   answer   \nConfidence: 90"
    parsed = parse_response(raw)
    assert parsed.decision == "answer"
    assert not parsed.malformed


@pytest.mark.parametrize(
    "decision_line",
    [
        "Decision: answer | certain",
        "Decision: answer | uncertain",
        "Decision: answer blah",
        "Decision: uncertain because I do not know",
        "Decision: answer/uncertain",
    ],
)
def test_decision_rejects_trailing_junk_after_a_valid_prefix(decision_line):
    raw = f"Answer: Paris\n{decision_line}\nConfidence: 100"
    parsed = parse_response(raw)
    assert parsed.decision is None
    assert parsed.malformed is True


def test_missing_decision_line_entirely_is_malformed_with_none_decision():
    raw = "Answer: Paris\nConfidence: 100"
    parsed = parse_response(raw)
    assert parsed.decision is None
    assert parsed.malformed is True


# --- resumable record-key behavior -----------------------------------------


def test_record_key_is_deterministic():
    key_a = make_record_key("pilot", "llama", "item-1", "C1", "v1", seed=42)
    key_b = make_record_key("pilot", "llama", "item-1", "C1", "v1", seed=42)
    assert key_a == key_b


def test_record_key_differs_when_any_field_differs():
    base = make_record_key("pilot", "llama", "item-1", "C1", "v1", seed=42)
    assert base != make_record_key("pilot", "qwen", "item-1", "C1", "v1", seed=42)
    assert base != make_record_key("pilot", "llama", "item-2", "C1", "v1", seed=42)
    assert base != make_record_key("pilot", "llama", "item-1", "C2", "v1", seed=42)
    assert base != make_record_key("pilot", "llama", "item-1", "C1", "v2", seed=42)
    assert base != make_record_key("pilot", "llama", "item-1", "C1", "v1", seed=1)


def test_result_writer_skips_already_completed_keys(tmp_path):
    path = tmp_path / "results.jsonl"
    writer = ResultWriter(path)
    key = make_record_key("pilot", "llama", "item-1", "C1", "v1", seed=42)

    assert not writer.is_completed(key)
    writer.write({"record_key": key, "answer": "Paris"})
    assert writer.is_completed(key)

    # A second writer instance opened against the same file (simulating a
    # resumed run) must also recognize the key as completed.
    resumed_writer = ResultWriter(path)
    assert resumed_writer.is_completed(key)


def test_result_writer_does_not_duplicate_records(tmp_path):
    path = tmp_path / "results.jsonl"
    writer = ResultWriter(path)
    key = make_record_key("pilot", "llama", "item-1", "C1", "v1", seed=42)

    writer.write({"record_key": key, "answer": "Paris"})
    writer.write({"record_key": key, "answer": "Paris"})  # duplicate write attempt

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

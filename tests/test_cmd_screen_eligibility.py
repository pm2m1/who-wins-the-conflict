"""Integration test for cmd_screen's KC/KW eligibility wiring
(docs/decisions.md, "Baseline abstentions must not become KC/KW memory
candidates").

Runs the real `cmd_screen` function end to end (file I/O, foil sampling,
margin computation, margin-bin assignment) against a small scripted
model adapter that returns controlled responses in a fixed order — not
DummyModelAdapter, whose baseline behavior cannot be scripted to produce
an abstention. No network call, no real model.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import yaml

from conflict_eval.cli import cmd_screen
from conflict_eval.models.base import BaseModelAdapter
from conflict_eval.scoring.sequence_logprob import DetailedScore, ScoredSequence

PILOT_CONFIG = {
    "seed": 42,
    "dataset": {"hf_dataset_id": "akariasai/PopQA", "split": "test", "screening_candidates": 10},
    "sampling": {"target_kc_items": 5, "target_kw_items": 5, "margin_bins": ["low", "medium", "high"]},
    "models": ["dummy"],
    "source_roles": {"dummy": {"preferred_source": None, "dispreferred_source": None}},
    "prompts_config": "configs/prompts.yaml",
    "sources_config": "configs/sources.yaml",
    "models_config": "configs/models.yaml",
}

# id -> (question, gold, scripted raw model response)
CASES = {
    "A": ("What is A?", "GoldA", "Answer: uncertain\nDecision: uncertain\nConfidence: 10"),
    "B": ("What is B?", "GoldB", "Answer: uncertain\nDecision: answer\nConfidence: 70"),
    "C": ("What is C?", "GoldC", "Answer: GoldC\nDecision: uncertain\nConfidence: 40"),
    "D": ("What is D?", "GoldD", "Answer: WrongD\nDecision: answer\nConfidence: 88"),
    "E": ("What is E?", "GoldE", "Answer: GoldE\nDecision: answer\nConfidence: 95"),
    "F": ("What is F?", "GoldF", "Answer: Paris or London\nDecision: answer\nConfidence: 60"),
    "G": ("What is G?", "GoldG", "Answer: SomeFactualG\nDecision: uncertain\nConfidence: 30"),
    "H": ("What is H?", "GoldH", "Answer: I don't know\nDecision: answer\nConfidence: 20"),
}
CASE_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]


class _ScriptedModelAdapter(BaseModelAdapter):
    """Returns pre-scripted `generate()` responses in call order. Score
    values are constant placeholders — this test does not assert on
    specific margin magnitudes, only on whether a margin was computed at
    all for eligible records.
    """

    def __init__(self, responses: list[str]) -> None:
        self.model_id = "scripted"
        self.model_revision = "scripted"
        self.requested_revision = None
        self.resolved_revision = None
        self._responses = iter(responses)

    def generate(self, messages, generation_config):
        return next(self._responses)

    def score_candidate(self, messages, candidate_text, answer_prefix=""):
        return ScoredSequence(logprob_sum=-1.0, token_count=1, logprob_normalized=-1.0)

    def score_candidate_detailed(self, messages, candidate_text, answer_prefix=""):
        scored = self.score_candidate(messages, candidate_text, answer_prefix)
        return DetailedScore(scored=scored, answer_tokens=[candidate_text], token_logprobs=[-1.0])


@pytest.fixture(scope="module")
def screened_records(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cmd_screen_eligibility")

    interim_items = [
        {
            "id": case_id,
            "subj": f"subject-{case_id}",
            "prop": "relA",
            "obj": gold,
            "question": question,
            "aliases": [],
            "gold_normalized": gold.lower(),
        }
        for case_id, (question, gold, _response) in CASES.items()
    ]

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    interim_dir.mkdir()
    processed_dir.mkdir()

    with open(interim_dir / "popqa_interim.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in interim_items)
    with open(processed_dir / "popqa_candidates.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in interim_items)

    config = dict(PILOT_CONFIG)
    config["paths"] = {
        "raw_dir": str(tmp_path / "raw"),
        "interim_dir": str(interim_dir),
        "processed_dir": str(processed_dir),
        "results_dir": str(results_dir),
        "figures_dir": str(tmp_path / "figures"),
    }
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    scripted_responses = [CASES[case_id][2] for case_id in CASE_ORDER]
    scripted_adapter = _ScriptedModelAdapter(scripted_responses)

    with patch("conflict_eval.cli._build_model_adapter", return_value=scripted_adapter):
        cmd_screen("dummy", str(config_path))

    baseline_path = results_dir / "dryrun_baseline.jsonl"
    records = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines()]
    return {r["item_id"]: r for r in records}


def _record(screened_records, case_id):
    return screened_records[case_id]


# 1. Answer: uncertain / Decision: uncertain -> excluded, baseline_uncertain, no margin.
def test_decision_uncertain_excluded_with_no_margin(screened_records):
    record = _record(screened_records, "A")
    assert record["knowledge_group"] == "excluded"
    assert record["exclusion_reason"] == "baseline_uncertain"
    assert "parametric_margin" not in record
    assert "margin_bin" not in record


# 2. A factual-looking answer / Decision: uncertain -> excluded, baseline_uncertain, no KC/KW.
def test_factual_looking_answer_with_decision_uncertain_excluded(screened_records):
    record = _record(screened_records, "G")
    assert record["parsed_answer"] == "SomeFactualG"
    assert record["knowledge_group"] == "excluded"
    assert record["exclusion_reason"] == "baseline_uncertain"


# 3. Gold-matching answer / Decision: uncertain -> NOT KC.
def test_gold_matching_answer_with_decision_uncertain_is_not_kc(screened_records):
    record = _record(screened_records, "C")
    assert record["parsed_answer"] == "GoldC"
    assert record["baseline_correct"] is True
    assert record["knowledge_group"] == "excluded"
    assert record["exclusion_reason"] == "baseline_uncertain"


# 4. Wrong clean factual answer / Decision: answer -> KW.
def test_wrong_clean_answer_with_decision_answer_is_kw(screened_records):
    record = _record(screened_records, "D")
    assert record["knowledge_group"] == "KW"
    assert record["memory_answer"] == "WrongD"
    assert record["conflicting_context_answer"] == "GoldD"


# 5. Gold answer / Decision: answer -> KC.
def test_gold_answer_with_decision_answer_is_kc(screened_records):
    record = _record(screened_records, "E")
    assert record["knowledge_group"] == "KC"
    assert record["memory_answer"] == "GoldE"
    assert "foil_answer" in record


# 6. Explicit uncertainty phrase with Decision: answer -> not KW.
def test_explicit_uncertainty_phrase_with_decision_answer_is_not_kw(screened_records):
    for case_id in ("B", "H"):
        record = _record(screened_records, case_id)
        assert record["knowledge_group"] != "KW", case_id
        assert record["knowledge_group"] != "KC", case_id
        assert record["exclusion_reason"] == "baseline_uncertain", case_id


# 7. Ambiguous/multi-valued candidate -> manual_review.
def test_multi_valued_candidate_routed_to_manual_review(screened_records):
    record = _record(screened_records, "F")
    assert record["knowledge_group"] == "manual_review"
    assert record["manual_review"] is True
    assert "parametric_margin" not in record


# 8. Existing KC/KW margin behavior remains unchanged for eligible records.
def test_eligible_kc_and_kw_records_receive_margins(screened_records):
    for case_id in ("D", "E"):
        record = _record(screened_records, case_id)
        assert isinstance(record["parametric_margin"], float)
        assert record["margin_bin"] in ("low", "medium", "high")


# 9. parsed_decision and parsed_confidence are persisted on baseline records.
def test_parsed_decision_and_confidence_persisted(screened_records):
    record = _record(screened_records, "D")
    assert record["parsed_decision"] == "answer"
    assert record["parsed_confidence"] == 88
    # Existing fields must still be present (additive change only).
    assert record["parsed_answer"] == "WrongD"
    assert "raw_generation" in record


# 10. Excluded uncertainty records do not enter margin-bin computation.
def test_excluded_uncertainty_records_excluded_from_margin_bins(screened_records):
    for case_id in ("A", "B", "C", "G", "H"):
        record = _record(screened_records, case_id)
        assert "margin_bin" not in record

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

# id -> {question, gold, response, [relation], [subject]}
# relation defaults to "relA" (an unrecognized-but-harmless placeholder)
# and subject to f"subject-{case_id}" when not given, matching cases A-H's
# original behavior; I-L exercise the relation/subject conflict-eligibility
# policy explicitly (docs/decisions.md, "Restrict primary trials to
# defensible conflicts").
CASES = {
    "A": {"question": "What is A?", "gold": "GoldA", "response": "Answer: uncertain\nDecision: uncertain\nConfidence: 10"},
    "B": {"question": "What is B?", "gold": "GoldB", "response": "Answer: uncertain\nDecision: answer\nConfidence: 70"},
    "C": {"question": "What is C?", "gold": "GoldC", "response": "Answer: GoldC\nDecision: uncertain\nConfidence: 40"},
    "D": {"question": "What is D?", "gold": "GoldD", "response": "Answer: WrongD\nDecision: answer\nConfidence: 88"},
    "E": {"question": "What is E?", "gold": "GoldE", "response": "Answer: GoldE\nDecision: answer\nConfidence: 95"},
    "F": {"question": "What is F?", "gold": "GoldF", "response": "Answer: Paris or London\nDecision: answer\nConfidence: 60"},
    "G": {"question": "What is G?", "gold": "GoldG", "response": "Answer: SomeFactualG\nDecision: uncertain\nConfidence: 30"},
    "H": {"question": "What is H?", "gold": "GoldH", "response": "Answer: I don't know\nDecision: answer\nConfidence: 20"},
    # EXCLUDED_PRIMARY_RELATIONS: a settled policy exclusion, not flagged
    # for manual review, but still a valid KW record with a margin.
    "I": {
        "question": "What genre is Album I?",
        "gold": "drama",
        "response": "Answer: comedy\nDecision: answer\nConfidence: 80",
        "relation": "genre",
        "subject": "Album I",
    },
    # REVIEW_RELATIONS: flagged for manual review; still a valid KC record.
    "J": {
        "question": "Who is J's father?",
        "gold": "GoldJ",
        "response": "Answer: GoldJ\nDecision: answer\nConfidence: 90",
        "relation": "father",
        "subject": "Person J",
    },
    # PRIMARY_RELATIONS, single-object subject: primary-conflict eligible KC.
    "K": {
        "question": "What sport does K play?",
        "gold": "hockey",
        "response": "Answer: hockey\nDecision: answer\nConfidence: 99",
        "relation": "sport",
        "subject": "Athlete K",
    },
    # PRIMARY_RELATIONS but multi-object subject (see EXTRA_INTERIM_ONLY_ITEMS
    # below): must be ineligible despite the relation itself being primary.
    "L": {
        "question": "What country is L in?",
        "gold": "USA",
        "response": "Answer: France\nDecision: answer\nConfidence: 85",
        "relation": "country",
        "subject": "Multi Subject L",
    },
}
CASE_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Present only in the full interim pool, never as screening candidates —
# they exist purely to give foil sampling (J, K) and the subject-level
# multi-object check (L) something real to find, matching the requirement
# that the multi-object index is built from the full interim pool, not
# just the sampled candidates.
EXTRA_INTERIM_ONLY_ITEMS = [
    {
        "id": "shadow-father",
        "subj": "Person J2",
        "prop": "father",
        "obj": "ShadowFather",
        "question": "Shadow father question",
        "aliases": [],
        "gold_normalized": "shadowfather",
    },
    {
        "id": "shadow-sport",
        "subj": "Athlete K2",
        "prop": "sport",
        "obj": "basketball",
        "question": "Shadow sport question",
        "aliases": [],
        "gold_normalized": "basketball",
    },
    {
        "id": "shadow-country",
        "subj": "Multi Subject L",
        "prop": "country",
        "obj": "Germany",
        "question": "Shadow country question",
        "aliases": [],
        "gold_normalized": "germany",
    },
]


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

    candidate_items = [
        {
            "id": case_id,
            "subj": case.get("subject", f"subject-{case_id}"),
            "prop": case.get("relation", "relA"),
            "obj": case["gold"],
            "question": case["question"],
            "aliases": [],
            "gold_normalized": case["gold"].lower(),
        }
        for case_id, case in CASES.items()
    ]
    # The full interim pool includes items that are never screened
    # candidates themselves — required for foil sampling and the
    # multi-object check to see the same relation/subject context a real
    # run would (docs/decisions.md, "Restrict primary trials to
    # defensible conflicts").
    interim_items = candidate_items + EXTRA_INTERIM_ONLY_ITEMS

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    interim_dir.mkdir()
    processed_dir.mkdir()

    with open(interim_dir / "popqa_interim.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in interim_items)
    with open(processed_dir / "popqa_candidates.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in candidate_items)

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

    scripted_responses = [CASES[case_id]["response"] for case_id in CASE_ORDER]
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


# --- relation-level / subject-level primary conflict eligibility --------
# (docs/decisions.md, "Restrict primary trials to defensible conflicts")


def test_excluded_relation_kw_is_ineligible_but_not_flagged_for_review(screened_records):
    # A settled policy exclusion (relation_not_primary_conflict) is
    # distinct from a case needing researcher attention — KW assignment
    # and its margin are both preserved regardless (item 17: margin
    # calculations themselves remain unchanged).
    record = _record(screened_records, "I")
    assert record["knowledge_group"] == "KW"
    assert record["primary_conflict_eligible"] is False
    assert record["conflict_eligibility_reason"] == "relation_not_primary_conflict"
    assert record["manual_review"] is False
    assert isinstance(record["parametric_margin"], float)


def test_review_relation_kc_is_ineligible_and_flagged_for_manual_review(screened_records):
    record = _record(screened_records, "J")
    assert record["knowledge_group"] == "KC"
    assert record["primary_conflict_eligible"] is False
    assert record["conflict_eligibility_reason"] == "relation_requires_review"
    assert record["manual_review"] is True
    assert isinstance(record["parametric_margin"], float)


def test_primary_relation_single_object_kc_is_eligible(screened_records):
    record = _record(screened_records, "K")
    assert record["knowledge_group"] == "KC"
    assert record["primary_conflict_eligible"] is True
    assert record["conflict_eligibility_reason"] is None
    assert record["manual_review"] is False


def test_primary_relation_multi_object_subject_is_ineligible_and_flagged(screened_records):
    # "country" is a PRIMARY relation in general, but this specific
    # subject has two distinct known objects in the full interim pool
    # (its own candidate row plus EXTRA_INTERIM_ONLY_ITEMS' shadow row) —
    # subject-level multiplicity overrides the relation's general policy.
    record = _record(screened_records, "L")
    assert record["knowledge_group"] == "KW"
    assert record["primary_conflict_eligible"] is False
    assert record["conflict_eligibility_reason"] == "relation_multi_object"
    assert record["manual_review"] is True

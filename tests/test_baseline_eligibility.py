"""Tests for baseline KC/KW eligibility rules
(conflict_eval.evaluation.baseline_eligibility).

Covers the real-model finding recorded in docs/decisions.md, "Baseline
abstentions must not become KC/KW memory candidates": a 20-item
Qwen2.5-3B-Instruct smoke screen found abstentions ("Answer: uncertain")
being treated as valid KW memory answers.
"""

from __future__ import annotations

from conflict_eval.evaluation.baseline_eligibility import (
    classify_baseline_eligibility,
    is_clean_factual_candidate,
)

# --- classify_baseline_eligibility ------------------------------------------


def test_decision_uncertain_is_never_eligible():
    result = classify_baseline_eligibility("uncertain", "uncertain", malformed=False)
    assert not result.eligible
    assert result.reason == "baseline_uncertain"


def test_factual_looking_answer_with_decision_uncertain_is_not_eligible():
    # A syntactically clean-looking factual answer is still ineligible if
    # the model reports Decision: uncertain.
    result = classify_baseline_eligibility("Jazz", "uncertain", malformed=False)
    assert not result.eligible
    assert result.reason == "baseline_uncertain"


def test_explicit_uncertainty_text_with_decision_answer_is_not_eligible():
    # The real-model failure mode this module exists to catch: the model
    # inconsistently reports Decision: answer alongside an abstention.
    result = classify_baseline_eligibility("uncertain", "answer", malformed=False)
    assert not result.eligible
    assert result.reason == "baseline_uncertain"


def test_other_uncertainty_phrases_are_caught_regardless_of_case_or_punctuation():
    for phrase in ["Unknown", "I don't know", "I do not know", "Cannot determine", "can't determine"]:
        result = classify_baseline_eligibility(phrase, "answer", malformed=False)
        assert not result.eligible, phrase
        assert result.reason == "baseline_uncertain", phrase


def test_clean_factual_answer_with_decision_answer_is_eligible():
    result = classify_baseline_eligibility("Paris", "answer", malformed=False)
    assert result.eligible
    assert result.reason is None


def test_malformed_response_is_not_eligible():
    result = classify_baseline_eligibility(None, None, malformed=True)
    assert not result.eligible
    assert result.reason == "malformed"


def test_missing_answer_is_not_eligible_even_if_not_flagged_malformed():
    result = classify_baseline_eligibility(None, "answer", malformed=False)
    assert not result.eligible


def test_unrecognized_decision_value_is_not_eligible():
    # Anything other than exactly "answer" (parse_response only accepts
    # "answer"/"uncertain" today, but this must not silently admit a
    # future third value).
    result = classify_baseline_eligibility("Paris", "maybe", malformed=False)
    assert not result.eligible
    assert result.reason == "baseline_uncertain"


# --- is_clean_factual_candidate ---------------------------------------------


def test_short_single_word_answer_is_clean():
    assert is_clean_factual_candidate("Paris")


def test_short_multi_word_answer_is_clean():
    assert is_clean_factual_candidate("New York City")


def test_answer_with_comma_is_not_clean():
    assert not is_clean_factual_candidate("Paris, France")


def test_answer_with_or_is_not_clean():
    assert not is_clean_factual_candidate("Paris or London")


def test_answer_with_and_is_not_clean():
    assert not is_clean_factual_candidate("Paris and London")


def test_real_model_multi_name_conjunction_is_not_clean():
    # Regression test: a real 7B PopQA screen produced this exact
    # screenwriter answer, which the original comma/" or " checks did not
    # catch (docs/decisions.md, "Restrict primary trials to defensible
    # conflicts").
    assert not is_clean_factual_candidate("Eric Paul Friedmann and Christophe Beck")


def test_excessively_long_answer_is_not_clean():
    long_answer = " ".join(["word"] * 10)
    assert not is_clean_factual_candidate(long_answer)


def test_boundary_word_count_is_clean():
    from conflict_eval.evaluation.baseline_eligibility import MAX_CLEAN_ANSWER_WORDS

    exactly_at_limit = " ".join(["word"] * MAX_CLEAN_ANSWER_WORDS)
    assert is_clean_factual_candidate(exactly_at_limit)
    one_over_limit = " ".join(["word"] * (MAX_CLEAN_ANSWER_WORDS + 1))
    assert not is_clean_factual_candidate(one_over_limit)

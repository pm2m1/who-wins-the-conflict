"""Tests for the relation-level and subject-level primary conflict
eligibility policy (conflict_eval.data.conflict_eligibility).

docs/decisions.md, "Restrict primary trials to defensible conflicts":
different answer != validated semantic conflict.
"""

from __future__ import annotations

import pytest

from conflict_eval.data.conflict_eligibility import (
    build_relation_subject_object_index,
    check_subject_multiplicity,
    classify_primary_conflict_eligibility,
    classify_relation_policy,
)

# --- relation-level policy ---------------------------------------------


@pytest.mark.parametrize("relation", ["place of birth", "sport", "country", "mother"])
def test_primary_relations_are_eligible(relation):
    result = classify_relation_policy(relation)
    assert result.eligible
    assert result.reason is None


@pytest.mark.parametrize("relation", ["father", "capital", "color"])
def test_review_relations_require_review(relation):
    result = classify_relation_policy(relation)
    assert not result.eligible
    assert result.reason == "relation_requires_review"


@pytest.mark.parametrize(
    "relation",
    [
        "genre",
        "religion",
        "screenwriter",
        "director",
        "producer",
        "composer",
        "author",
        "occupation",
        "capital of",
    ],
)
def test_excluded_relations_are_not_primary_eligible(relation):
    result = classify_relation_policy(relation)
    assert not result.eligible
    assert result.reason == "relation_not_primary_conflict"


def test_unrecognized_relation_defaults_to_review_not_silent_eligibility():
    # A relation outside all three configured sets must not be silently
    # treated as primary-eligible.
    result = classify_relation_policy("some_future_relation")
    assert not result.eligible
    assert result.reason == "relation_unrecognized"


def test_relation_policy_lists_do_not_overlap():
    from conflict_eval.data.conflict_eligibility import (
        EXCLUDED_PRIMARY_RELATIONS,
        PRIMARY_RELATIONS,
        REVIEW_RELATIONS,
    )

    assert PRIMARY_RELATIONS.isdisjoint(REVIEW_RELATIONS)
    assert PRIMARY_RELATIONS.isdisjoint(EXCLUDED_PRIMARY_RELATIONS)
    assert REVIEW_RELATIONS.isdisjoint(EXCLUDED_PRIMARY_RELATIONS)


# --- subject-level multiplicity -----------------------------------------


def test_single_object_subject_is_eligible():
    interim_items = [
        {"prop": "country", "subj": "Brown University", "obj": "United States of America"},
    ]
    index = build_relation_subject_object_index(interim_items)
    result = check_subject_multiplicity("country", "Brown University", index)
    assert result.eligible
    assert result.reason is None


def test_two_distinct_objects_for_same_subject_relation_is_ineligible():
    interim_items = [
        {"prop": "country", "subj": "Ambiguous University", "obj": "United States of America"},
        {"prop": "country", "subj": "Ambiguous University", "obj": "Canada"},
    ]
    index = build_relation_subject_object_index(interim_items)
    result = check_subject_multiplicity("country", "Ambiguous University", index)
    assert not result.eligible
    assert result.reason == "relation_multi_object"


def test_duplicate_rows_with_the_same_object_are_not_multi_object():
    # Normalized-equal duplicates (e.g. re-scraped rows) must not be
    # mistaken for genuine multiplicity.
    interim_items = [
        {"prop": "sport", "subj": "St. Louis Blues", "obj": "ice hockey"},
        {"prop": "sport", "subj": "St. Louis Blues", "obj": "Ice Hockey"},
    ]
    index = build_relation_subject_object_index(interim_items)
    result = check_subject_multiplicity("sport", "St. Louis Blues", index)
    assert result.eligible


def test_subject_with_no_index_entry_is_treated_as_single_object():
    index = build_relation_subject_object_index([])
    result = check_subject_multiplicity("country", "Unseen Subject", index)
    assert result.eligible


def test_index_is_built_from_full_pool_not_just_candidates():
    # The index must reflect every interim item passed in, regardless of
    # whether that item was also selected as a screening candidate —
    # callers are responsible for passing the FULL interim pool.
    full_pool = [
        {"prop": "country", "subj": "Brown University", "obj": "United States of America"},
        {"prop": "country", "subj": "Brown University", "obj": "Tunisia"},
    ]
    index = build_relation_subject_object_index(full_pool)
    assert len(index[("country", "Brown University")]) == 2


# --- combined entry point -------------------------------------------------


def test_combined_check_primary_relation_single_object_is_eligible():
    interim_items = [{"prop": "sport", "subj": "St. Louis Blues", "obj": "ice hockey"}]
    index = build_relation_subject_object_index(interim_items)
    result = classify_primary_conflict_eligibility("sport", "St. Louis Blues", index)
    assert result.eligible


def test_combined_check_primary_relation_multi_object_is_ineligible():
    interim_items = [
        {"prop": "country", "subj": "Brown University", "obj": "United States of America"},
        {"prop": "country", "subj": "Brown University", "obj": "Tunisia"},
    ]
    index = build_relation_subject_object_index(interim_items)
    result = classify_primary_conflict_eligibility("country", "Brown University", index)
    assert not result.eligible
    assert result.reason == "relation_multi_object"


def test_combined_check_excluded_relation_short_circuits_before_subject_check():
    # A relation-level exclusion should be reported even if the subject
    # itself happens to be single-object — the relation-level reason
    # takes precedence, per classify_relation_policy running first.
    interim_items = [{"prop": "genre", "subj": "Some Album", "obj": "drama"}]
    index = build_relation_subject_object_index(interim_items)
    result = classify_primary_conflict_eligibility("genre", "Some Album", index)
    assert not result.eligible
    assert result.reason == "relation_not_primary_conflict"

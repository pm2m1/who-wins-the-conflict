"""Tests for the targeted primary-relation candidate pool construction
(conflict_eval.data.popqa.build_primary_relation_candidate_pool),
used for `dataset.candidate_pool: primary_conflict_relations`
(docs/decisions.md, "Support targeted primary conflict screening").

Pure fixture tests — no network, no dataset download, no model.
"""

from __future__ import annotations

import random
from unittest.mock import patch

from conflict_eval.data.popqa import build_primary_relation_candidate_pool


def _item(item_id, relation, subject, obj):
    return {"id": item_id, "prop": relation, "subj": subject, "obj": obj, "question": f"Q{item_id}?"}


BASE_ITEMS = [
    # PRIMARY relations, single-object subjects — all eligible.
    _item("1", "sport", "Athlete1", "hockey"),
    _item("2", "sport", "Athlete2", "basketball"),
    _item("3", "country", "Uni1", "USA"),
    _item("4", "mother", "Person1", "Mother1"),
    _item("5", "place of birth", "Person2", "CityA"),
    # PRIMARY relation, but multi-object subject — both rows excluded.
    _item("6", "country", "AmbiguousUni", "Germany"),
    _item("7", "country", "AmbiguousUni", "Austria"),
    # REVIEW relation — excluded from the primary pool.
    _item("8", "father", "Person3", "Father1"),
    # EXCLUDED relation — excluded from the primary pool.
    _item("9", "genre", "Album1", "drama"),
]


def test_only_primary_relations_survive():
    result = build_primary_relation_candidate_pool(BASE_ITEMS)
    relations = {item["prop"] for item in result.deduplicated_pool}
    assert relations == {"sport", "country", "mother", "place of birth"}


def test_review_and_excluded_relations_are_dropped():
    result = build_primary_relation_candidate_pool(BASE_ITEMS)
    ids = {item["id"] for item in result.deduplicated_pool}
    assert "8" not in ids  # father (REVIEW)
    assert "9" not in ids  # genre (EXCLUDED)


def test_subject_level_multi_object_records_are_excluded():
    result = build_primary_relation_candidate_pool(BASE_ITEMS)
    ids = {item["id"] for item in result.deduplicated_pool}
    assert "6" not in ids
    assert "7" not in ids
    # Also excluded from the pre-dedup eligible set, not merely deduped away.
    eligible_ids = {item["id"] for item in result.eligible_rows}
    assert "6" not in eligible_ids
    assert "7" not in eligible_ids


def test_eligible_single_object_primary_rows_survive():
    result = build_primary_relation_candidate_pool(BASE_ITEMS)
    ids = {item["id"] for item in result.deduplicated_pool}
    assert {"1", "2", "3", "4", "5"} <= ids


def test_deterministic_deduplication_prefers_smallest_string_id():
    items = [
        _item("42", "sport", "Athlete1", "Hockey"),  # same normalized object as "1"
        _item("1", "sport", "Athlete1", "hockey"),
        _item("7", "sport", "Athlete1", "HOCKEY"),
    ]
    result = build_primary_relation_candidate_pool(items)
    matching = [item for item in result.deduplicated_pool if item["subj"] == "Athlete1"]
    assert len(matching) == 1
    assert matching[0]["id"] == "1"
    # All three eligible individually (same normalized object => not
    # multi-object), even though only one survives deduplication.
    assert len(result.eligible_rows) == 3


def test_dedup_result_independent_of_input_order():
    items = [
        _item("3", "sport", "Athlete1", "hockey"),
        _item("1", "sport", "Athlete1", "hockey"),
        _item("2", "sport", "Athlete1", "hockey"),
    ]
    rng = random.Random(0)
    shuffled = list(items)
    rng.shuffle(shuffled)

    result_original = build_primary_relation_candidate_pool(items)
    result_shuffled = build_primary_relation_candidate_pool(shuffled)

    assert [i["id"] for i in result_original.deduplicated_pool] == [
        i["id"] for i in result_shuffled.deduplicated_pool
    ]
    assert result_original.deduplicated_pool[0]["id"] == "1"


def test_deduplicated_pool_is_sorted_by_id():
    items = [
        _item("30", "sport", "AthleteA", "hockey"),
        _item("2", "sport", "AthleteB", "basketball"),
        _item("100", "sport", "AthleteC", "soccer"),
    ]
    result = build_primary_relation_candidate_pool(items)
    ids = [item["id"] for item in result.deduplicated_pool]
    assert ids == sorted(ids)


def test_uses_the_shared_relation_policy_not_a_duplicate_hardcoded_list():
    # If a relation is added to PRIMARY_RELATIONS in the shared policy
    # module, build_primary_relation_candidate_pool must pick it up
    # automatically — proving it delegates rather than maintaining its
    # own separate relation list.
    items = [_item("1", "totally_new_relation", "SubjectX", "ValueX")]

    result_before = build_primary_relation_candidate_pool(items)
    assert result_before.deduplicated_pool == []

    with patch(
        "conflict_eval.data.conflict_eligibility.PRIMARY_RELATIONS",
        frozenset({"totally_new_relation"}),
    ):
        result_after = build_primary_relation_candidate_pool(items)
    assert [item["id"] for item in result_after.deduplicated_pool] == ["1"]


def test_empty_pool_returns_empty_result():
    result = build_primary_relation_candidate_pool([])
    assert result.eligible_rows == []
    assert result.deduplicated_pool == []

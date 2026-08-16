import pytest

from conflict_eval.source_preference.pairs import enumerate_unordered_pairs


def test_enumerates_all_unordered_pairs():
    pairs = enumerate_unordered_pairs(["A", "B", "C"])
    assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_pair_count_matches_combinatorics():
    labels = ["A", "B", "C", "D"]
    pairs = enumerate_unordered_pairs(labels)
    assert len(pairs) == 6  # n choose 2 = 4*3/2


def test_no_self_pairs():
    pairs = enumerate_unordered_pairs(["A", "B"])
    assert all(a != b for a, b in pairs)


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError):
        enumerate_unordered_pairs(["A", "A", "B"])


def test_two_labels_minimum():
    assert enumerate_unordered_pairs(["A", "B"]) == [("A", "B")]

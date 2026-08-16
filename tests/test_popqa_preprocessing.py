"""Tests for PopQA raw -> interim normalization (conflict_eval.data.popqa).
Uses small synthetic rows shaped like real PopQA fields — no download.
"""

from conflict_eval.data.popqa import _parse_alias_field, build_interim


def test_parses_string_encoded_list_aliases():
    assert _parse_alias_field("['Foo', 'Bar']") == ["Foo", "Bar"]


def test_treats_plain_string_as_single_alias():
    assert _parse_alias_field("Single Alias") == ["Single Alias"]


def test_none_alias_field_is_empty_list():
    assert _parse_alias_field(None) == []


def test_build_interim_keeps_well_formed_rows():
    raw_rows = [
        {
            "id": 1,
            "subj": "France",
            "prop": "capital_of",
            "obj": "Paris",
            "question": "What is the capital of France?",
            "o_aliases": "['City of Paris']",
            "possible_answers": "['Paris']",
        }
    ]
    interim, exclusions = build_interim(raw_rows)
    assert len(interim) == 1
    assert exclusions == []
    item = interim[0]
    assert item["id"] == "1"
    assert set(item["aliases"]) == {"City of Paris", "Paris"}
    assert item["gold_normalized"] == "paris"


def test_build_interim_excludes_empty_question():
    raw_rows = [{"id": 1, "obj": "Paris", "question": "  ", "prop": "capital_of"}]
    interim, exclusions = build_interim(raw_rows)
    assert interim == []
    assert exclusions[0]["reason"] == "empty_question"


def test_build_interim_excludes_empty_gold_answer():
    raw_rows = [{"id": 1, "obj": "", "question": "What is it?", "prop": "capital_of"}]
    interim, exclusions = build_interim(raw_rows)
    assert interim == []
    assert exclusions[0]["reason"] == "empty_gold_answer"

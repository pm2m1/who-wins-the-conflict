import random

from conflict_eval.data.foils import build_relation_index, sample_foil

CAPITAL_ITEMS = [
    {"id": "1", "prop": "capital_of", "obj": "Paris", "aliases": []},
    {"id": "2", "prop": "capital_of", "obj": "Berlin", "aliases": []},
    {"id": "3", "prop": "capital_of", "obj": "Madrid", "aliases": ["Spain's capital"]},
    {"id": "4", "prop": "founder_of", "obj": "Elon Musk", "aliases": []},
]


def test_foil_is_sampled_from_same_relation():
    index = build_relation_index(CAPITAL_ITEMS)
    item = CAPITAL_ITEMS[0]  # Paris, capital_of
    rng = random.Random(0)
    foil = sample_foil(item, index, rng)
    assert foil is not None
    source_item = next(i for i in CAPITAL_ITEMS if i["id"] == foil.source_item_id)
    assert source_item["prop"] == "capital_of"


def test_foil_never_equals_gold_or_alias():
    index = build_relation_index(CAPITAL_ITEMS)
    item = {"id": "5", "prop": "capital_of", "obj": "Madrid", "aliases": ["Spain's capital"]}
    index["capital_of"].append(item)
    rng = random.Random(0)
    for _ in range(20):
        foil = sample_foil(item, index, rng)
        assert foil is not None
        assert foil.foil_answer not in ("Madrid", "Spain's capital")


def test_foil_sampling_is_deterministic_given_seed():
    index = build_relation_index(CAPITAL_ITEMS)
    item = CAPITAL_ITEMS[0]
    foil_a = sample_foil(item, index, random.Random(42))
    foil_b = sample_foil(item, index, random.Random(42))
    assert foil_a == foil_b


def test_foil_sampling_can_differ_across_seeds():
    index = build_relation_index(CAPITAL_ITEMS)
    item = CAPITAL_ITEMS[0]
    results = {sample_foil(item, index, random.Random(s)).foil_answer for s in range(10)}
    # With two eligible same-relation candidates (Berlin, Madrid), varying
    # the seed should be able to produce more than one outcome.
    assert len(results) > 1


def test_no_defensible_foil_returns_none():
    lone_item = {"id": "99", "prop": "unique_relation", "obj": "Only Answer", "aliases": []}
    index = build_relation_index([lone_item])
    foil = sample_foil(lone_item, index, random.Random(0))
    assert foil is None


def test_foil_result_records_method_and_relation():
    index = build_relation_index(CAPITAL_ITEMS)
    item = CAPITAL_ITEMS[0]
    foil = sample_foil(item, index, random.Random(1))
    assert foil.relation == "capital_of"
    assert foil.generation_method == "same_relation_sample"

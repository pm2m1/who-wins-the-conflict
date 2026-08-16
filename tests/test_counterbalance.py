from conflict_eval.source_preference.counterbalance import (
    expand_pair_to_presentations,
    expand_pairs_to_presentations,
)


def test_expands_to_both_orders():
    presentations = expand_pair_to_presentations("Wikipedia", "a blog")
    orders = {p.presentation_order for p in presentations}
    assert orders == {"AB", "BA"}


def test_ab_presentation_shows_source_a_first():
    presentations = expand_pair_to_presentations("Wikipedia", "a blog")
    ab = next(p for p in presentations if p.presentation_order == "AB")
    assert ab.displayed_source_1 == "Wikipedia"
    assert ab.displayed_source_2 == "a blog"


def test_ba_presentation_shows_source_b_first():
    presentations = expand_pair_to_presentations("Wikipedia", "a blog")
    ba = next(p for p in presentations if p.presentation_order == "BA")
    assert ba.displayed_source_1 == "a blog"
    assert ba.displayed_source_2 == "Wikipedia"


def test_both_presentations_preserve_underlying_pair_identity():
    # Regardless of display order, source_a/source_b (the pair identity
    # used to group trials for pairwise statistics) must stay fixed.
    presentations = expand_pair_to_presentations("Wikipedia", "a blog")
    assert all(p.source_a == "Wikipedia" and p.source_b == "a blog" for p in presentations)


def test_expand_pairs_to_presentations_covers_every_pair():
    pairs = [("A", "B"), ("C", "D")]
    presentations = expand_pairs_to_presentations(pairs)
    assert len(presentations) == 4
    seen_pairs = {(p.source_a, p.source_b) for p in presentations}
    assert seen_pairs == {("A", "B"), ("C", "D")}

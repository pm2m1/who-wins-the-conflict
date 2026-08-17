"""Tests for the generic paired preferred-vs-dispreferred source
comparison helper (conflict_eval.analysis.paired_comparison).

All fixtures here are SYNTHETIC. The discordance-count shapes mirror the
structure of two real, already-published values (docs/qwen_pilot_results.md)
purely to cross-check the exact-p-value arithmetic against a known-correct
answer — the records themselves, item ids, and relations are invented,
not the frozen pilot data.
"""

from __future__ import annotations

import pytest

from conflict_eval.analysis.paired_comparison import paired_source_comparison


def _record(item_id, condition, context_adopted, relation="sport"):
    return {
        "item_id": item_id,
        "knowledge_group": "KW",
        "condition": condition,
        "context_adopted": context_adopted,
        "relation": relation,
    }


def test_preferred_only_discordance():
    records = [
        _record("1", "C1", True),
        _record("1", "C2", False),
    ]
    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.preferred_only == 1
    assert result.dispreferred_only == 0
    assert result.both == 0
    assert result.neither == 0
    assert result.n_items == 1


def test_dispreferred_only_discordance():
    records = [
        _record("1", "C1", False),
        _record("1", "C2", True),
    ]
    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.dispreferred_only == 1
    assert result.preferred_only == 0


def test_both_and_neither_counted_correctly():
    records = [
        _record("1", "C1", True),
        _record("1", "C2", True),
        _record("2", "C1", False),
        _record("2", "C2", False),
    ]
    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.both == 1
    assert result.neither == 1
    assert result.preferred_only == 0
    assert result.dispreferred_only == 0
    assert result.n_items == 2
    assert result.preferred_rate == 0.5
    assert result.dispreferred_rate == 0.5
    assert result.delta == 0.0


def test_relation_filtering():
    records = [
        _record("1", "C1", True, relation="sport"),
        _record("1", "C2", False, relation="sport"),
        _record("2", "C1", True, relation="country"),
        _record("2", "C2", True, relation="country"),
    ]
    sport_only = paired_source_comparison(records, "KW", "C1", "C2", relation="sport")
    assert sport_only.n_items == 1
    assert sport_only.preferred_only == 1

    country_only = paired_source_comparison(records, "KW", "C1", "C2", relation="country")
    assert country_only.n_items == 1
    assert country_only.both == 1


def test_duplicate_item_condition_raises():
    records = [
        _record("1", "C1", True),
        _record("1", "C1", False),  # duplicate C1 for the same item
        _record("1", "C2", False),
    ]
    with pytest.raises(ValueError):
        paired_source_comparison(records, "KW", "C1", "C2")


def test_incomplete_pair_is_excluded_not_errored():
    records = [
        _record("1", "C1", True),  # no matching C2 record
        _record("2", "C1", True),
        _record("2", "C2", False),
    ]
    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.n_items == 1  # only item "2" has a complete pair


def test_no_items_returns_none_fields():
    result = paired_source_comparison([], "KW", "C1", "C2")
    assert result.n_items == 0
    assert result.preferred_rate is None
    assert result.dispreferred_rate is None
    assert result.p_value is None


def test_no_discordant_pairs_gives_p_value_one():
    records = [
        _record("1", "C1", True),
        _record("1", "C2", True),
        _record("2", "C1", False),
        _record("2", "C2", False),
    ]
    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.preferred_only == 0
    assert result.dispreferred_only == 0
    assert result.p_value == 1.0


# --- exact p-value cross-checks (arithmetic correctness only; synthetic
# discordance shapes mirroring two already-published pilot values) ---------


def test_exact_p_value_matches_known_binomial_result_3_vs_0():
    # 3 preferred-only discordant pairs, 0 dispreferred-only: exact
    # two-sided binomial test on n=3 successes out of 3 trials at p=0.5
    # has a known closed-form value of 0.25.
    records = []
    for i in range(3):
        records.append(_record(f"disc-{i}", "C1", True))
        records.append(_record(f"disc-{i}", "C2", False))
    for i in range(24):
        records.append(_record(f"neither-{i}", "C1", False))
        records.append(_record(f"neither-{i}", "C2", False))

    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.preferred_only == 3
    assert result.dispreferred_only == 0
    assert result.neither == 24
    assert result.n_items == 27
    assert result.p_value == pytest.approx(0.25, abs=1e-9)


def test_exact_p_value_matches_known_binomial_result_8_vs_0():
    # 8 preferred-only discordant pairs, 0 dispreferred-only: exact
    # two-sided binomial test on n=8 successes out of 8 trials at p=0.5
    # has a known closed-form value of 2 * 0.5**8 = 0.0078125.
    records = []
    for i in range(8):
        records.append(_record(f"disc-{i}", "C1", True))
        records.append(_record(f"disc-{i}", "C2", False))
    for i in range(17):
        records.append(_record(f"both-{i}", "C1", True))
        records.append(_record(f"both-{i}", "C2", True))
    for i in range(5):
        records.append(_record(f"neither-{i}", "C1", False))
        records.append(_record(f"neither-{i}", "C2", False))

    result = paired_source_comparison(records, "KW", "C1", "C2")
    assert result.preferred_only == 8
    assert result.dispreferred_only == 0
    assert result.both == 17
    assert result.neither == 5
    assert result.n_items == 30
    assert result.p_value == pytest.approx(0.0078125, abs=1e-9)

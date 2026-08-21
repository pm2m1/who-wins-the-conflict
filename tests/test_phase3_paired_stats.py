"""Tests for the frozen Phase 3 primary paired statistics.

Covers the procedures pre-specified in
`docs/phase3_scaled_study_design.md` §26 and the §30 saturation
diagnostics. The Tango interval is validated against its own defining
property (|Z| equals the critical value at the endpoints), against the
constrained-MLE score equation it is derived from, and against independent
checks (symmetry, containment, large-sample agreement with Wald) -- not
merely against its own output.
"""

from __future__ import annotations

import math

import pytest

from conflict_eval.phase3.paired_stats import (
    build_paired_table,
    exact_mcnemar_p,
    holm_adjusted,
    paired_source_result,
    saturation_diagnostics,
    tango_interval,
    tango_score,
)


def _outcomes(both: int, a_only: int, b_only: int, neither: int):
    return (
        [(True, True)] * both
        + [(True, False)] * a_only
        + [(False, True)] * b_only
        + [(False, False)] * neither
    )


# --- paired 2x2 counts and risk difference --------------------------------


def test_paired_table_counts_all_four_cells():
    table = build_paired_table(_outcomes(both=17, a_only=8, b_only=0, neither=5))
    assert (table.both, table.a_only, table.b_only, table.neither) == (17, 8, 0, 5)
    assert table.n == 30
    assert table.discordant == 8


def test_paired_risk_difference_matches_frozen_phase2_qwen_value():
    # Frozen Phase 2 Qwen corrective cell (docs/cross_model_pilot_results.md):
    # 17 both, 8 preferred-only, 0 dispreferred-only, 5 neither.
    table = build_paired_table(_outcomes(17, 8, 0, 5))
    assert table.rate_a == pytest.approx(25 / 30)  # 83.3%
    assert table.rate_b == pytest.approx(17 / 30)  # 56.7%
    assert table.risk_difference == pytest.approx(0.26666, abs=1e-4)  # +26.7 pp


def test_empty_table_returns_none_rather_than_dividing_by_zero():
    table = build_paired_table([])
    assert table.n == 0
    assert table.risk_difference is None
    assert tango_interval(table) is None


# --- exact McNemar / exact binomial ---------------------------------------


def test_exact_mcnemar_matches_frozen_phase2_qwen_p_value():
    table = build_paired_table(_outcomes(17, 8, 0, 5))
    assert exact_mcnemar_p(table) == pytest.approx(0.0078125)


def test_exact_mcnemar_returns_one_when_no_discordant_pairs():
    # Not evidence of absence -- the saturation diagnostic covers that.
    table = build_paired_table(_outcomes(both=20, a_only=0, b_only=0, neither=10))
    assert exact_mcnemar_p(table) == 1.0


def test_exact_mcnemar_is_symmetric_under_swapping_arms():
    forward = exact_mcnemar_p(build_paired_table(_outcomes(5, 7, 2, 6)))
    reversed_ = exact_mcnemar_p(build_paired_table(_outcomes(5, 2, 7, 6)))
    assert forward == pytest.approx(reversed_)


# --- Tango interval -------------------------------------------------------


@pytest.mark.parametrize(
    "both,a_only,b_only,neither",
    [(17, 8, 0, 5), (10, 12, 5, 33), (40, 20, 14, 22), (12, 3, 3, 12), (0, 0, 0, 30)],
)
def test_tango_endpoints_satisfy_the_defining_score_equation(
    both, a_only, b_only, neither
):
    """At the returned endpoints the score statistic must equal +/- z.

    This is the interval's definition, checked independently of how the
    root was found.
    """
    from scipy.stats import norm

    table = build_paired_table(_outcomes(both, a_only, b_only, neither))
    lower, upper = tango_interval(table)
    z = norm.ppf(0.975)
    assert tango_score(table.a_only, table.b_only, table.n, lower) == pytest.approx(
        z, abs=1e-6
    )
    assert tango_score(table.a_only, table.b_only, table.n, upper) == pytest.approx(
        -z, abs=1e-6
    )


def test_tango_constrained_mle_solves_the_score_equation():
    """The quadratic root used internally must satisfy the constrained
    log-likelihood derivative it was derived from (interior solutions)."""
    from conflict_eval.phase3.paired_stats import _constrained_p21

    for b, c, n, delta in [(8, 2, 30, 0.10), (12, 5, 60, -0.05), (20, 14, 96, 0.05)]:
        x = _constrained_p21(b, c, n, delta)
        derivative = (
            b / (x + delta) + c / x - 2 * (n - b - c) / (1 - 2 * x - delta)
        )
        assert derivative == pytest.approx(0.0, abs=1e-8)


def test_tango_interval_contains_the_point_estimate():
    for args in [(17, 8, 0, 5), (10, 12, 5, 33), (12, 3, 3, 12), (0, 0, 0, 30)]:
        table = build_paired_table(_outcomes(*args))
        lower, upper = tango_interval(table)
        assert lower <= table.risk_difference <= upper


def test_tango_interval_is_symmetric_under_swapping_arms():
    forward = tango_interval(build_paired_table(_outcomes(17, 8, 0, 5)))
    reversed_ = tango_interval(build_paired_table(_outcomes(17, 0, 8, 5)))
    assert forward[0] == pytest.approx(-reversed_[1], abs=1e-9)
    assert forward[1] == pytest.approx(-reversed_[0], abs=1e-9)


def test_tango_interval_approaches_wald_in_a_large_well_behaved_sample():
    """Independent sanity check: far from the boundary and at large n, a
    score interval and a Wald interval must nearly coincide."""
    from scipy.stats import norm

    both, a_only, b_only, neither = 800, 200, 150, 850
    table = build_paired_table(_outcomes(both, a_only, b_only, neither))
    lower, upper = tango_interval(table)
    n = table.n
    estimate = table.risk_difference
    variance = (a_only + b_only - (a_only - b_only) ** 2 / n) / n
    half = norm.ppf(0.975) * math.sqrt(variance / n)
    assert lower == pytest.approx(estimate - half, abs=5e-4)
    assert upper == pytest.approx(estimate + half, abs=5e-4)


def test_tango_interval_is_finite_at_the_boundary_where_wald_degenerates():
    """Phase 2 produced a 30/30 cell with zero discordance one way; the
    frozen design chose Tango precisely because Wald is degenerate there."""
    table = build_paired_table(_outcomes(both=29, a_only=0, b_only=1, neither=0))
    lower, upper = tango_interval(table)
    assert -1.0 < lower < upper < 1.0
    assert upper > lower


def test_tango_score_is_monotone_decreasing_in_delta():
    previous = None
    for step in range(-95, 96):
        value = tango_score(8, 2, 30, step / 100)
        if previous is not None:
            assert value <= previous + 1e-9
        previous = value


def test_tango_interval_rejects_an_invalid_confidence_level():
    table = build_paired_table(_outcomes(10, 5, 5, 10))
    with pytest.raises(ValueError):
        tango_interval(table, confidence=1.5)


# --- saturation / low-information diagnostics (§30) -----------------------


def test_saturation_flag_requires_both_boundary_and_low_discordance():
    # Phase 2 Llama corrective shape: 29 both, 0/1 discordant, ceiling.
    table = build_paired_table(_outcomes(both=29, a_only=0, b_only=1, neither=0))
    diagnostics = saturation_diagnostics(table, tango_interval(table))
    assert diagnostics.near_boundary is True
    assert diagnostics.low_information is True
    assert diagnostics.saturated_uninformative is True


def test_mid_range_null_is_not_flagged_saturated():
    """Two arms at ~50% is a different observation from two arms at ~100%
    and must not be labeled the same way (§30)."""
    table = build_paired_table(_outcomes(both=15, a_only=8, b_only=8, neither=29))
    diagnostics = saturation_diagnostics(table, tango_interval(table))
    assert diagnostics.near_boundary is False
    assert diagnostics.saturated_uninformative is False


def test_boundary_with_ample_discordance_is_not_saturated():
    table = build_paired_table(_outcomes(both=90, a_only=6, b_only=4, neither=0))
    diagnostics = saturation_diagnostics(table, tango_interval(table))
    assert diagnostics.near_boundary is True
    assert diagnostics.low_information is False
    assert diagnostics.saturated_uninformative is False


def test_low_information_threshold_is_five_discordant_pairs():
    four = build_paired_table(_outcomes(10, 2, 2, 10))
    five = build_paired_table(_outcomes(10, 3, 2, 10))
    assert saturation_diagnostics(four).low_information is True
    assert saturation_diagnostics(five).low_information is False


# --- the full mandatory report (§26.1) -----------------------------------


def test_paired_source_result_reports_every_mandatory_field():
    result = paired_source_result(_outcomes(17, 8, 0, 5))
    for field in (
        "n", "both", "a_only", "b_only", "neither", "discordant",
        "risk_difference", "ci_lower", "ci_upper", "exact_p",
    ):
        assert getattr(result, field) is not None, field
    assert result.n == 30
    assert result.discordant == 8
    assert result.exact_p == pytest.approx(0.0078125)
    assert result.risk_difference == pytest.approx(0.26666, abs=1e-4)


# --- Holm (secondary family only, §28) ------------------------------------


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm_adjusted({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert adjusted["a"] == pytest.approx(0.004)
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())
    ordered = [adjusted[k] for k in ("a", "b", "c", "d")]
    assert ordered == sorted(ordered)


def test_holm_on_empty_family_returns_empty():
    assert holm_adjusted({}) == {}


def test_holm_single_test_is_unchanged():
    """A one-test family needs no correction; Holm must be a no-op there,
    matching the frozen treatment of the single primary test (§28)."""
    assert holm_adjusted({"only": 0.031}) == {"only": pytest.approx(0.031)}


# --- audit-gap: isolated strict-boundary saturation cases (§30) -----------
#
# Each case puts exactly ONE arm near a bound, so the other arm cannot
# independently trigger `near_boundary` and confound the test.


@pytest.mark.parametrize(
    "label,counts,expected_near_boundary",
    [
        # rate_a is EXACTLY 0.95 -> not > 0.95 -> not triggered
        ("rate_a == 0.95", (15, 4, 0, 1), False),
        # rate_a is 0.96 -> > 0.95 -> triggered
        ("rate_a == 0.96", (23, 1, 0, 1), True),
        # rate_a is EXACTLY 0.05 -> not < 0.05 -> not triggered
        ("rate_a == 0.05", (0, 1, 3, 16), False),
        # rate_a is 0.04 -> < 0.05 -> triggered
        ("rate_a == 0.04", (0, 1, 3, 21), True),
    ],
)
def test_saturation_uses_strict_inequalities(label, counts, expected_near_boundary):
    table = build_paired_table(_outcomes(*counts))
    diagnostics = saturation_diagnostics(table)
    assert diagnostics.near_boundary is expected_near_boundary, label


def test_saturation_requires_boundary_and_low_discordance_together():
    """0.96 with plenty of discordance is near-boundary but NOT saturated."""
    table = build_paired_table(_outcomes(both=90, a_only=6, b_only=4, neither=0))
    diagnostics = saturation_diagnostics(table)
    assert diagnostics.near_boundary is True
    assert diagnostics.low_information is False
    assert diagnostics.saturated_uninformative is False


@pytest.mark.parametrize(
    "a_only,b_only,expected_low_information", [(2, 2, True), (3, 2, False)]
)
def test_low_information_threshold_is_exactly_five(a_only, b_only, expected_low_information):
    table = build_paired_table(_outcomes(10, a_only, b_only, 10))
    assert table.discordant == a_only + b_only
    assert saturation_diagnostics(table).low_information is expected_low_information


def test_low_information_is_reported_independently_of_saturation():
    """§37 treats discordance < 5 as insufficient information even when the
    contrast is nowhere near a boundary."""
    table = build_paired_table(_outcomes(both=15, a_only=2, b_only=2, neither=15))
    diagnostics = saturation_diagnostics(table)
    assert diagnostics.near_boundary is False
    assert diagnostics.saturated_uninformative is False
    assert diagnostics.low_information is True


# --- audit-gap: exact McNemar with zero discordance ----------------------


def test_exact_mcnemar_with_zero_discordance_is_exactly_one():
    """n10 = n01 = 0 -> p = 1.0, and this is NOT evidence of absence."""
    table = build_paired_table(_outcomes(both=12, a_only=0, b_only=0, neither=18))
    assert table.discordant == 0
    assert exact_mcnemar_p(table) == 1.0
    assert saturation_diagnostics(table).low_information is True


def test_exact_mcnemar_never_exceeds_one():
    for a_only in range(6):
        for b_only in range(6):
            table = build_paired_table(_outcomes(5, a_only, b_only, 5))
            assert 0.0 <= exact_mcnemar_p(table) <= 1.0

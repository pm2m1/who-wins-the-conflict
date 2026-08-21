"""Phase 3 primary paired source-comparison statistics.

Implements exactly the procedures frozen in
`docs/phase3_scaled_study_design.md` §26.1/§26.2 and the saturation
diagnostics in §30. The frozen design pre-specifies **one** procedure for
each role and forbids substituting another for confirmatory inference:

- effect estimate: the paired risk difference;
- interval: the **95% Tango score interval** for the difference of paired
  proportions (the only confirmatory interval);
- test: the **exact two-sided McNemar test**, equivalently the exact
  binomial test on the discordant pairs against p = 0.5.

Bootstrap intervals are sensitivity-only (§26.2) and are deliberately not
implemented here as an alternative confirmatory path.

This module contains no Phase 3 scientific result. It computes statistics
from whatever paired outcomes it is given; in Phase 3B those are synthetic.
"""

from __future__ import annotations

import dataclasses
import math

from conflict_eval.phase3.constants import (
    MIN_INFORMATIVE_DISCORDANT_PAIRS,
    SATURATION_LOWER_BOUND,
    SATURATION_UPPER_BOUND,
)


@dataclasses.dataclass(frozen=True)
class PairedTable:
    """The paired 2x2 table for one source contrast.

    `a_only`/`b_only` are the discordant cells; `n` is the number of
    complete pairs. Field names use A/B rather than
    preferred/dispreferred because the same machinery serves the common
    fixed-source arm, where the labels are identities and not roles (§19).
    """

    both: int
    a_only: int
    b_only: int
    neither: int

    @property
    def n(self) -> int:
        return self.both + self.a_only + self.b_only + self.neither

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only

    @property
    def rate_a(self) -> float | None:
        return (self.both + self.a_only) / self.n if self.n else None

    @property
    def rate_b(self) -> float | None:
        return (self.both + self.b_only) / self.n if self.n else None

    @property
    def risk_difference(self) -> float | None:
        """Paired risk difference `Delta = P(adopt|A) - P(adopt|B)`."""
        if not self.n:
            return None
        return (self.a_only - self.b_only) / self.n


def build_paired_table(
    outcomes: list[tuple[bool, bool]],
) -> PairedTable:
    """Build the paired 2x2 table from `(outcome_under_A, outcome_under_B)`
    pairs. Callers are responsible for supplying exactly one complete pair
    per item; incomplete pairs cannot contribute to a paired comparison and
    must be excluded upstream (Phase 2 precedent,
    `analysis/paired_comparison.py`).
    """
    both = sum(1 for a, b in outcomes if a and b)
    a_only = sum(1 for a, b in outcomes if a and not b)
    b_only = sum(1 for a, b in outcomes if not a and b)
    neither = sum(1 for a, b in outcomes if not a and not b)
    return PairedTable(both=both, a_only=a_only, b_only=b_only, neither=neither)


# ---------------------------------------------------------------------------
# Tango (1998) score interval for the difference of paired proportions
# ---------------------------------------------------------------------------
#
# Reference: Tango, T. (1998), "Equivalence test and confidence interval
# for the difference in proportions for the paired-sample design",
# Statistics in Medicine 17:891-908; and Tango (1999), Biometrics 55:
# 1300-1303. Chosen by the frozen design because it is valid at and near
# the boundary, where a Wald interval is degenerate -- Phase 2 produced
# exactly such a cell (30/30 adoption, zero dispreferred-only discordance)
# (docs/phase3_scaled_study_design.md, §26.2).
#
# Notation for the paired 2x2 table:
#     b = a_only, c = b_only, n = total pairs
#     p12 = P(A=1, B=0), p21 = P(A=0, B=1), delta = p12 - p21
#
# Under the constraint p12 - p21 = delta, writing x = p21, the multinomial
# log-likelihood in the three free cells is proportional to
#
#     b*log(x + delta) + c*log(x) + (n - b - c)*log(1 - 2x - delta)
#
# Differentiating with respect to x and clearing denominators gives the
# quadratic whose admissible root is the constrained MLE x~:
#
#     2n*x^2 - [(b + c) - delta*(2n - b + c)]*x - c*delta*(1 - delta) = 0
#
# The variance of (b - c) under the constraint is
#
#     Var(b - c) = n*(p12 + p21 - (p12 - p21)^2) = n*(2x + delta*(1 - delta))
#
# giving the score statistic
#
#     Z(delta) = (b - c - n*delta) / sqrt(n*(2*x~ + delta*(1 - delta)))
#
# Z is monotone decreasing in delta, so the 100(1-alpha)% interval
# {delta : |Z(delta)| < z} is found by bisection on each side of the point
# estimate. Both the constrained-MLE quadratic and the variance expression
# above are verified numerically in tests/test_phase3_paired_stats.py:
# the root is checked against the score equation directly, and the
# resulting endpoints are checked to satisfy |Z| = z exactly.


def _constrained_p21(b: int, c: int, n: int, delta: float) -> float:
    """Constrained MLE of `p21` given `delta`, per the quadratic above.

    The larger root is taken because `p21` is a probability and must be
    non-negative; when `c == 0` the admissible solution can legitimately
    sit on the boundary at exactly 0, which this expression returns.
    """
    a_coef = 2.0 * n
    b_coef = delta * (2.0 * n - b + c) - (b + c)
    c_coef = -c * delta * (1.0 - delta)
    discriminant = max(b_coef * b_coef - 4.0 * a_coef * c_coef, 0.0)
    return (-b_coef + math.sqrt(discriminant)) / (2.0 * a_coef)


def tango_score(b: int, c: int, n: int, delta: float) -> float:
    """The Tango score statistic `Z(delta)`. Exposed for testing the
    interval's defining property (`|Z|` equals the critical value at the
    returned endpoints) rather than only its numeric output.
    """
    if n <= 0:
        raise ValueError("tango_score requires at least one pair")
    numerator = b - c - n * delta
    x = _constrained_p21(b, c, n, delta)
    variance = n * (2.0 * x + delta * (1.0 - delta))
    if variance <= 0.0:
        # Degenerate only at |delta| = 1 with no mass; preserve the sign so
        # bisection still brackets correctly.
        if numerator > 0:
            return math.inf
        if numerator < 0:
            return -math.inf
        return 0.0
    return numerator / math.sqrt(variance)


def tango_interval(
    table: PairedTable, confidence: float = 0.95, iterations: int = 200
) -> tuple[float, float] | None:
    """95% (by default) Tango score interval for the paired risk difference.

    This is the single pre-specified confirmatory interval
    (docs/phase3_scaled_study_design.md, §26.2). Returns `None` only when
    there are no pairs at all.
    """
    from scipy.stats import norm

    if table.n == 0:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    b, c, n = table.a_only, table.b_only, table.n
    estimate = table.risk_difference
    assert estimate is not None  # n > 0 checked above

    # Z is monotone decreasing in delta, so the lower endpoint is where
    # Z = +z and the upper endpoint is where Z = -z.
    low_a, low_b = -1.0, estimate
    for _ in range(iterations):
        mid = (low_a + low_b) / 2.0
        if tango_score(b, c, n, mid) > z:
            low_a = mid
        else:
            low_b = mid
    lower = (low_a + low_b) / 2.0

    high_a, high_b = estimate, 1.0
    for _ in range(iterations):
        mid = (high_a + high_b) / 2.0
        if tango_score(b, c, n, mid) > -z:
            high_a = mid
        else:
            high_b = mid
    upper = (high_a + high_b) / 2.0

    return lower, upper


def exact_mcnemar_p(table: PairedTable) -> float:
    """Exact two-sided McNemar test: the exact binomial test on the
    discordant pairs against p = 0.5 (§26.2).

    Returns 1.0 when there are no discordant pairs. Per the frozen design
    and the Phase 2 implementation this is **not** evidence of absence --
    zero discordance is a ceiling/floor signature (§30), which
    `saturation_diagnostics` flags separately.
    """
    from scipy.stats import binomtest

    if table.discordant == 0:
        return 1.0
    result = binomtest(table.a_only, table.discordant, 0.5, alternative="two-sided")
    return float(result.pvalue)


@dataclasses.dataclass(frozen=True)
class SaturationDiagnostics:
    """The §30 ceiling/floor diagnostics, computed for every contrast
    before it is interpreted."""

    discordant_pairs: int
    discordant_rate: float | None
    rate_a: float | None
    rate_b: float | None
    near_boundary: bool
    ci_width: float | None
    both_fraction: float | None
    neither_fraction: float | None
    low_information: bool
    saturated_uninformative: bool


def saturation_diagnostics(
    table: PairedTable, interval: tuple[float, float] | None = None
) -> SaturationDiagnostics:
    """Compute the §30 diagnostics.

    `saturated_uninformative` implements the frozen rule verbatim: a
    contrast is SATURATED / UNINFORMATIVE when *either arm* exceeds 0.95 or
    falls below 0.05 adoption **and** discordant pairs are fewer than 5. A
    contrast so flagged may never be reported as "no effect" (§30, §37).
    """
    n = table.n
    rate_a, rate_b = table.rate_a, table.rate_b
    near_boundary = any(
        rate is not None
        and (rate > SATURATION_UPPER_BOUND or rate < SATURATION_LOWER_BOUND)
        for rate in (rate_a, rate_b)
    )
    low_information = table.discordant < MIN_INFORMATIVE_DISCORDANT_PAIRS
    ci_width = (interval[1] - interval[0]) if interval is not None else None
    return SaturationDiagnostics(
        discordant_pairs=table.discordant,
        discordant_rate=(table.discordant / n) if n else None,
        rate_a=rate_a,
        rate_b=rate_b,
        near_boundary=near_boundary,
        ci_width=ci_width,
        both_fraction=(table.both / n) if n else None,
        neither_fraction=(table.neither / n) if n else None,
        low_information=low_information,
        saturated_uninformative=near_boundary and low_information,
    )


@dataclasses.dataclass(frozen=True)
class PairedSourceResult:
    """The complete mandatory report for one source contrast (§26.1).

    Every field in the frozen "mandatory primary report" is present so a
    caller cannot omit one: total n, all four paired cells, the discordant
    count, the risk difference, its interval, and the exact p-value.
    """

    n: int
    both: int
    a_only: int
    b_only: int
    neither: int
    discordant: int
    rate_a: float | None
    rate_b: float | None
    risk_difference: float | None
    ci_lower: float | None
    ci_upper: float | None
    exact_p: float
    diagnostics: SaturationDiagnostics


def paired_source_result(
    outcomes: list[tuple[bool, bool]], confidence: float = 0.95
) -> PairedSourceResult:
    """Run the full frozen paired procedure over `(A, B)` outcome pairs."""
    table = build_paired_table(outcomes)
    interval = tango_interval(table, confidence=confidence)
    diagnostics = saturation_diagnostics(table, interval)
    return PairedSourceResult(
        n=table.n,
        both=table.both,
        a_only=table.a_only,
        b_only=table.b_only,
        neither=table.neither,
        discordant=table.discordant,
        rate_a=table.rate_a,
        rate_b=table.rate_b,
        risk_difference=table.risk_difference,
        ci_lower=interval[0] if interval else None,
        ci_upper=interval[1] if interval else None,
        exact_p=exact_mcnemar_p(table),
        diagnostics=diagnostics,
    )


def holm_adjusted(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment within a family (§28).

    Used for the SECONDARY CONFIRMATORY family only. The primary family
    contains exactly one test and requires no correction; `analysis_status`
    enforces that separation, and this function is deliberately unaware of
    which family it is given so it cannot silently pool the two.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p) in enumerate(ordered):
        value = (m - index) * p
        running = max(running, min(value, 1.0))
        adjusted[name] = running
    return adjusted

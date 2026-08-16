from conflict_eval.scoring.parametric_margin import compute_parametric_margin


def test_positive_margin_when_memory_scores_higher():
    margin = compute_parametric_margin(memory_score=-1.0, conflicting_score=-4.0)
    assert margin == 3.0


def test_negative_margin_when_conflicting_scores_higher():
    margin = compute_parametric_margin(memory_score=-5.0, conflicting_score=-1.0)
    assert margin == -4.0


def test_zero_margin_when_scores_equal():
    assert compute_parametric_margin(memory_score=-2.0, conflicting_score=-2.0) == 0.0


def test_margin_is_memory_minus_conflicting_not_absolute_value():
    # Order matters: this is not a symmetric distance.
    a = compute_parametric_margin(-1.0, -3.0)
    b = compute_parametric_margin(-3.0, -1.0)
    assert a == -b

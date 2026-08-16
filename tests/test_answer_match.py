from conflict_eval.evaluation.answer_match import is_match


def test_exact_match_after_normalization():
    assert is_match("Washington, D.C.", "Washington DC")


def test_alias_match():
    assert is_match("Fab Four", "The Beatles", ["Fab Four", "The Fabs"])


def test_no_match_for_unrelated_answer():
    assert not is_match("Paris", "London", ["Greater London"])


def test_no_match_for_empty_candidate():
    assert not is_match("", "London", [])
    assert not is_match(None, "London", [])


def test_match_is_case_and_article_insensitive():
    assert is_match("the london eye", "London Eye", [])

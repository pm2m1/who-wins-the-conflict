from conflict_eval.data.normalize import normalize_answer, normalized_alias_set, token_f1


def test_lowercases_and_strips_punctuation():
    assert normalize_answer("Barack Obama!") == "barack obama"


def test_collapses_whitespace():
    assert normalize_answer("  New   York  ") == "new york"


def test_drops_single_leading_article():
    assert normalize_answer("The Beatles") == "beatles"
    assert normalize_answer("An Apple") == "apple"
    assert normalize_answer("A Study In Scarlet") == "study in scarlet"


def test_does_not_drop_mid_phrase_articles():
    # Only a single LEADING article is dropped; articles later in the
    # phrase are left alone since they can be semantically load-bearing.
    assert normalize_answer("Winnie the Pooh") == "winnie the pooh"


def test_empty_and_none_input():
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""


def test_normalized_alias_set_includes_gold_and_aliases():
    aliases = normalized_alias_set("The Beatles", ["Beatles", "Fab Four"])
    assert "beatles" in aliases
    assert "fab four" in aliases


def test_normalized_alias_set_drops_empty_entries():
    aliases = normalized_alias_set("Gold Answer", ["", "  ", None])
    assert aliases == {"gold answer"}


def test_token_f1_identical_strings():
    assert token_f1("New York City", "new york city") == 1.0


def test_token_f1_partial_overlap():
    score = token_f1("New York", "New York City")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    assert token_f1("Paris", "Tokyo") == 0.0

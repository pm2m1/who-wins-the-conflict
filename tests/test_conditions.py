import pytest

from conflict_eval.experiment.conditions import build_conditions
from conflict_eval.experiment.evidence import render_evidence

EVIDENCE_TEMPLATE = (
    'Source: {source}\n\nStatement:\nThe answer to the question "{question}" is "{asserted_answer}".\n'
)


def test_c0_has_no_evidence_and_no_conflict_status():
    specs = build_conditions("KC", "Paris", "Paris", "Berlin", "Wikipedia", "a blog")
    c0 = next(s for s in specs if s.condition == "C0")
    assert c0.evidence_truth == "none"
    assert c0.source_role == "none"
    assert c0.conflict_status == "none"
    assert c0.asserted_answer is None


def test_kc_agreement_and_conflict_mapping():
    specs = {s.condition: s for s in build_conditions("KC", "Paris", "Paris", "Berlin", "Wikipedia", "a blog")}
    # KC: correct evidence (C1/C2) agrees with the parametric answer.
    assert specs["C1"].conflict_status == "agreement"
    assert specs["C2"].conflict_status == "agreement"
    assert specs["C1"].asserted_answer == "Paris"
    assert specs["C2"].asserted_answer == "Paris"
    # KC: false evidence (C3/C4) is the primary conflict (harmful override).
    assert specs["C3"].conflict_status == "conflict"
    assert specs["C4"].conflict_status == "conflict"
    assert specs["C3"].asserted_answer == "Berlin"
    assert specs["C4"].asserted_answer == "Berlin"


def test_kw_agreement_and_conflict_mapping():
    specs = {
        s.condition: s
        for s in build_conditions("KW", "Paris", "Lyon", None, "Wikipedia", "a blog")
    }
    # KW: correct evidence (C1/C2) is the primary conflict (corrective override).
    assert specs["C1"].conflict_status == "conflict"
    assert specs["C2"].conflict_status == "conflict"
    assert specs["C1"].asserted_answer == "Paris"
    # KW: false evidence (C3/C4, using the baseline wrong answer) agrees with memory.
    assert specs["C3"].conflict_status == "agreement"
    assert specs["C4"].conflict_status == "agreement"
    assert specs["C3"].asserted_answer == "Lyon"


def test_source_roles_assigned_correctly():
    specs = {s.condition: s for s in build_conditions("KC", "Paris", "Paris", "Berlin", "Wikipedia", "a blog")}
    assert specs["C1"].source_role == "preferred"
    assert specs["C1"].source_label == "Wikipedia"
    assert specs["C2"].source_role == "dispreferred"
    assert specs["C2"].source_label == "a blog"
    assert specs["C3"].source_role == "preferred"
    assert specs["C4"].source_role == "dispreferred"


def test_all_five_conditions_present():
    specs = build_conditions("KC", "Paris", "Paris", "Berlin", "Wikipedia", "a blog")
    assert {s.condition for s in specs} == {"C0", "C1", "C2", "C3", "C4"}


def test_kc_requires_a_foil():
    with pytest.raises(ValueError):
        build_conditions("KC", "Paris", "Paris", None, "Wikipedia", "a blog")


def test_invalid_knowledge_group_rejected():
    with pytest.raises(ValueError):
        build_conditions("unknown", "Paris", "Paris", "Berlin", "Wikipedia", "a blog")


def test_evidence_content_unchanged_by_source_swap():
    # Changing only the source label must not alter the asserted-answer
    # content of the rendered evidence text (docs/phase2_research_design.md).
    evidence_wikipedia = render_evidence(EVIDENCE_TEMPLATE, "Wikipedia", "capital of France?", "Paris")
    evidence_blog = render_evidence(EVIDENCE_TEMPLATE, "a blog", "capital of France?", "Paris")

    assert 'is "Paris"' in evidence_wikipedia
    assert 'is "Paris"' in evidence_blog
    assert "Wikipedia" in evidence_wikipedia and "Wikipedia" not in evidence_blog
    assert "a blog" in evidence_blog and "a blog" not in evidence_wikipedia

    # The only substring difference between the two renders should be the
    # source label itself.
    diff_wikipedia = evidence_wikipedia.replace("Wikipedia", "<SOURCE>")
    diff_blog = evidence_blog.replace("a blog", "<SOURCE>")
    assert diff_wikipedia == diff_blog

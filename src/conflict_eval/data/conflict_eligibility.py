"""Relation-level and subject-level policy for PRIMARY context-memory
conflict trial eligibility.

The first real Qwen2.5-7B-Instruct 20-item PopQA baseline screen exposed a
critical methodological point: **different answer != validated semantic
conflict**. A PopQA relation can be genuinely multi-valued (e.g.
"screenwriter", "genre") or hierarchical/compatible (e.g. "religion":
Christianity vs. Baptist) without the model's baseline answer and the
gold/foil answer being in actual semantic contradiction. This module
centralizes which relations are safe for AUTOMATIC primary conflict trial
construction, and additionally checks each specific (relation, subject)
pair against the full interim PopQA pool for known multi-object cases
even within an otherwise-safe relation.

This is a deliberately conservative PILOT policy, not a claim about which
PopQA relations are objectively single-valued in the world — see
docs/decisions.md, "Restrict primary conflict trials to defensible
conflicts". "occupation", for instance, is conceptually multi-label even
though its observed per-subject duplicate rate in the pinned PopQA
snapshot happened to be low; it is excluded on semantic grounds, not
because of that duplicate rate. The duplicate-rate census that motivated
this review is not itself the classification rule.

KC/KW (baseline knowledge state, evaluation/baseline_eligibility.py) and
primary-conflict eligibility (this module) are deliberately kept as
separate concerns: a record can be a perfectly valid KC or KW item while
still being ineligible for automatic primary conflict trial construction.
"""

from __future__ import annotations

import dataclasses

from conflict_eval.data.normalize import normalize_answer

# Relations judged, by inspection of their semantic type — not by the
# observed PopQA duplicate rate, which only motivated this review — safe
# for automatic PRIMARY conflict trial construction: a subject has (at
# most) one correct object for these relations in ordinary usage.
PRIMARY_RELATIONS = frozenset(
    {
        "place of birth",
        "sport",
        "country",
        "mother",
    }
)

# Plausible candidates that need explicit researcher review before being
# trusted as single-valued — not included in PRIMARY_RELATIONS by default.
REVIEW_RELATIONS = frozenset(
    {
        "father",
        "capital",
        "color",
    }
)

# Relations excluded from primary conflict trial construction because
# they are commonly multi-valued, hierarchical/compatible, or otherwise
# not a validated semantic conflict merely because the answers differ.
EXCLUDED_PRIMARY_RELATIONS = frozenset(
    {
        "genre",
        "religion",
        "screenwriter",
        "director",
        "producer",
        "composer",
        "author",
        "occupation",
        "capital of",
    }
)


@dataclasses.dataclass(frozen=True)
class ConflictEligibility:
    eligible: bool
    reason: str | None  # populated when not eligible; None when eligible


def classify_relation_policy(relation: str | None) -> ConflictEligibility:
    """Classify a relation string into PRIMARY / REVIEW / EXCLUDED,
    independent of any specific subject. Does not consider subject-level
    multiplicity — see `check_subject_multiplicity` for that.

    A relation outside all three configured sets (unexpected for the
    pinned PopQA snapshot's 16 known relations, but not impossible for a
    different snapshot) is treated conservatively as needing review, not
    silently assumed eligible.
    """
    if relation in PRIMARY_RELATIONS:
        return ConflictEligibility(True, None)
    if relation in REVIEW_RELATIONS:
        return ConflictEligibility(False, "relation_requires_review")
    if relation in EXCLUDED_PRIMARY_RELATIONS:
        return ConflictEligibility(False, "relation_not_primary_conflict")
    return ConflictEligibility(False, "relation_unrecognized")


def build_relation_subject_object_index(
    interim_items: list[dict],
) -> dict[tuple[str, str], set[str]]:
    """Map (relation, subject) -> set of normalized known objects, built
    from the FULL interim PopQA pool — not just the sampled screening
    candidates — so multi-object detection is not limited by chance
    inclusion in a particular candidate subsample.
    """
    index: dict[tuple[str, str], set[str]] = {}
    for item in interim_items:
        key = (item["prop"], item["subj"])
        index.setdefault(key, set()).add(normalize_answer(item["obj"]))
    return index


def check_subject_multiplicity(
    relation: str, subject: str, relation_subject_index: dict[tuple[str, str], set[str]]
) -> ConflictEligibility:
    """A primary-conflict candidate is automatically eligible only if the
    full interim pool records exactly one distinct known object for this
    (relation, subject) pair. More than one known object means the
    relation is multi-valued FOR THIS SPECIFIC SUBJECT, regardless of the
    relation's general policy category.
    """
    objects = relation_subject_index.get((relation, subject), set())
    if len(objects) > 1:
        return ConflictEligibility(False, "relation_multi_object")
    return ConflictEligibility(True, None)


def classify_primary_conflict_eligibility(
    relation: str | None,
    subject: str | None,
    relation_subject_index: dict[tuple[str, str], set[str]],
) -> ConflictEligibility:
    """Combined relation-level + subject-level policy check — the single
    entry point callers should use.
    """
    relation_check = classify_relation_policy(relation)
    if not relation_check.eligible:
        return relation_check
    if relation is None or subject is None:
        return ConflictEligibility(False, "relation_requires_review")
    return check_subject_multiplicity(relation, subject, relation_subject_index)

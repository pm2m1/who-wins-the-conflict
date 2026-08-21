"""Tests for prompt-identical deduplication and cross-cohort reuse.

Covers `docs/phase3_scaled_study_design.md` §22 (deterministic
deduplication, aliasing, "one observation referenced by two planned
contrasts") and §23 (nominal condition slots != unique model generations).

The two failure modes guarded against are opposite: **missing** a genuine
duplicate (which would double-count one observation as independent
evidence) and **inventing** one (which would merge two genuinely different
prompts). Both are tested.
"""

from __future__ import annotations

import dataclasses

import pytest

from conflict_eval.phase3.dedup import (
    ConditionRequest,
    GenerationIdentity,
    collect_paired_outcomes,
    contrast_is_degenerate,
    deduplicate_requests,
    prompt_hash,
)

IDENTITY = GenerationIdentity(
    model_key="qwen",
    model_revision="synthetic-rev-1",
    prompt_version="v1",
    do_sample=False,
    num_beams=1,
    max_new_tokens=32,
    seed=42,
)
IDENTITY_V2 = dataclasses.replace(IDENTITY, prompt_version="v2")


def _request(item_id, condition, prompt, cohorts=("A",), model="qwen", **kwargs):
    defaults = {
        "arm": "common",
        "source_role": "identity_a",
        "source_label": "a government website",
        "evidence_truth": "true",
        "conflict_status": "conflict",
        "knowledge_group": "KW",
    }
    defaults.update(kwargs)
    return ConditionRequest(
        model_key=model,
        item_id=item_id,
        condition=condition,
        rendered_prompt=prompt,
        cohorts=tuple(cohorts),
        **defaults,
    )


# --- prompt identity ------------------------------------------------------


def test_prompt_hash_is_stable_and_distinguishes_any_difference():
    assert prompt_hash("abc") == prompt_hash("abc")
    assert prompt_hash("abc") != prompt_hash("abd")
    # Strict: whitespace is meaningful because it changes what the model sees.
    assert prompt_hash("a b") != prompt_hash("a  b")
    assert prompt_hash("Abc") != prompt_hash("abc")


# --- genuine duplicates collapse -----------------------------------------


def test_exact_duplicate_prompts_collapse_to_one_observation():
    prompt = "identical prompt text"
    result = deduplicate_requests(
        [_request("i1", "K1", prompt), _request("i1", "M1", prompt)],
        seed=1,
        identity=IDENTITY,
    )
    assert result.nominal_slots == 2
    assert result.unique_observations == 1
    assert result.collapsed_slots == 1
    observation = result.observations[0]
    assert observation.is_aliased
    assert {a["condition"] for a in observation.aliased_conditions} == {"K1", "M1"}


def test_aliased_conditions_record_full_condition_arm_role_labels():
    """§22 rule 2: the record carries the full set of (condition, arm, role)
    labels that resolve to it."""
    prompt = "same"
    result = deduplicate_requests(
        [
            _request("i1", "K1", prompt, arm="common", source_role="identity_a"),
            _request("i1", "M1", prompt, arm="model_specific", source_role="preferred"),
        ],
        seed=1,
        identity=IDENTITY,
    )
    aliases = result.observations[0].aliased_conditions
    assert {a["arm"] for a in aliases} == {"common", "model_specific"}
    assert {a["source_role"] for a in aliases} == {"identity_a", "preferred"}


def test_qwen_full_coincidence_collapses_both_model_specific_conditions():
    """Qwen's frozen pair IS the common pair, so both M1 and M2 collapse
    into common-arm conditions (§19, §22)."""
    requests = [
        _request("i1", "K1", "gold-from-gov"),
        _request("i1", "K2", "gold-from-forum"),
        _request("i1", "M1", "gold-from-gov"),
        _request("i1", "M2", "gold-from-forum"),
    ]
    result = deduplicate_requests(requests, seed=1, identity=IDENTITY)
    assert result.nominal_slots == 4
    assert result.unique_observations == 2
    assert result.collapsed_slots == 2
    assert result.alias_map[("qwen", "i1", "M1")] == result.alias_map[("qwen", "i1", "K1")]
    assert result.alias_map[("qwen", "i1", "M2")] == result.alias_map[("qwen", "i1", "K2")]


def test_llama_partial_coincidence_collapses_only_the_matching_condition():
    """§22 rule 5: partial overlap deduplicates only the matching side."""
    requests = [
        _request("i1", "K1", "gold-from-gov", model="llama"),
        _request("i1", "K2", "gold-from-forum", model="llama"),
        _request("i1", "M1", "gold-from-gov", model="llama"),
        _request("i1", "M2", "gold-from-social-media", model="llama"),
    ]
    result = deduplicate_requests(requests, seed=1, identity=IDENTITY)
    assert result.nominal_slots == 4
    assert result.unique_observations == 3
    assert result.collapsed_slots == 1
    assert result.alias_map[("llama", "i1", "M1")] == result.alias_map[("llama", "i1", "K1")]
    assert result.alias_map[("llama", "i1", "M2")] != result.alias_map[("llama", "i1", "K2")]


# --- false duplicates must NOT collapse ----------------------------------


def test_same_source_but_different_asserted_answer_does_not_collapse():
    result = deduplicate_requests(
        [
            _request("i1", "K1", 'from gov: answer is "gold"'),
            _request("i1", "K3", 'from gov: answer is "foil"'),
        ],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 2


def test_same_answer_but_different_question_does_not_collapse():
    result = deduplicate_requests(
        [
            _request("i1", "K1", 'Q1 ... answer is "gold"'),
            _request("i2", "K1", 'Q2 ... answer is "gold"'),
        ],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 2


def test_identical_metadata_with_different_prompt_does_not_collapse():
    """Deduplication is keyed on the exact rendered prompt, never on
    metadata resemblance (§22)."""
    result = deduplicate_requests(
        [
            _request("i1", "K1", "prompt A"),
            _request("i1", "K2", "prompt B"),
        ],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 2


def test_same_prompt_for_different_models_does_not_collapse():
    """Different models must each generate; only same-model duplicates
    collapse."""
    result = deduplicate_requests(
        [
            _request("i1", "K1", "same text", model="qwen"),
            _request("i1", "K1", "same text", model="llama"),
        ],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 2


# --- cross-cohort reuse ---------------------------------------------------


def test_cross_cohort_membership_merges_onto_one_observation():
    """An item in A, B and C is generated once and referenced from each
    (§15.2, §16, §23) -- compute is not multiplied by cohort membership."""
    prompt = "shared"
    result = deduplicate_requests(
        [
            _request("i1", "M1", prompt, cohorts=("A",)),
            _request("i1", "M1", prompt, cohorts=("B",)),
            _request("i1", "M1", prompt, cohorts=("C",)),
        ],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 1
    assert result.observations[0].cohorts == {"A", "B", "C"}
    assert result.cohort_membership_map()[result.observations[0].observation_id] == [
        "A", "B", "C",
    ]


def test_nominal_slots_differ_from_unique_observations_when_duplicates_exist():
    prompt = "dup"
    result = deduplicate_requests(
        [_request("i1", "K1", prompt), _request("i1", "M1", prompt)],
        seed=1,
        identity=IDENTITY,
    )
    assert result.nominal_slots != result.unique_observations
    assert result.nominal_slots == result.unique_observations + result.collapsed_slots


def test_no_collapse_leaves_the_two_counts_equal():
    result = deduplicate_requests(
        [_request("i1", "K1", "a"), _request("i1", "K2", "b")],
        seed=1,
        identity=IDENTITY,
    )
    assert result.nominal_slots == result.unique_observations == 2
    assert result.collapsed_slots == 0


# --- paired assembly through the alias map -------------------------------


def test_paired_outcomes_read_through_the_alias_map():
    requests = [
        _request("i1", "M1", "p-a"),
        _request("i1", "M2", "p-b"),
        _request("i2", "M1", "q-a"),
        _request("i2", "M2", "q-b"),
    ]
    result = deduplicate_requests(requests, seed=1, identity=IDENTITY)
    outcomes = {}
    for observation in result.observations:
        outcomes[observation.observation_id] = observation.rendered_prompt.endswith("-a")
    pairs = collect_paired_outcomes(result, outcomes, "qwen", ["i1", "i2"], "M1", "M2")
    assert pairs == [(True, False), (True, False)]


def test_deduplicated_observation_contributes_once_per_item():
    """A collapsed observation is counted once in n, never twice (§22 rule 3)."""
    requests = [
        _request("i1", "K1", "shared"),
        _request("i1", "M1", "shared"),
        _request("i1", "M2", "other"),
    ]
    result = deduplicate_requests(requests, seed=1, identity=IDENTITY)
    outcomes = {o.observation_id: True for o in result.observations}
    pairs = collect_paired_outcomes(result, outcomes, "qwen", ["i1"], "M1", "M2")
    assert len(pairs) == 1


def test_items_missing_a_condition_are_skipped_from_pairs():
    result = deduplicate_requests(
        [_request("i1", "M1", "only-one-side")], seed=1, identity=IDENTITY
    )
    outcomes = {o.observation_id: True for o in result.observations}
    assert collect_paired_outcomes(result, outcomes, "qwen", ["i1"], "M1", "M2") == []


def test_degenerate_contrast_is_detected():
    """Both sides resolving to the SAME observation is not a contrast; it
    would trivially show zero effect and must fail loudly."""
    result = deduplicate_requests(
        [_request("i1", "M1", "identical"), _request("i1", "M2", "identical")],
        seed=1,
        identity=IDENTITY,
    )
    assert contrast_is_degenerate(result, "qwen", ["i1"], "M1", "M2") is True


def test_normal_contrast_is_not_degenerate():
    result = deduplicate_requests(
        [_request("i1", "M1", "a"), _request("i1", "M2", "b")],
        seed=1,
        identity=IDENTITY,
    )
    assert contrast_is_degenerate(result, "qwen", ["i1"], "M1", "M2") is False


# --- determinism ----------------------------------------------------------


def test_deduplication_is_deterministic_across_runs():
    requests = [_request(f"i{i}", "K1", f"prompt-{i}") for i in range(10)]
    first = deduplicate_requests(requests, seed=5, identity=IDENTITY)
    second = deduplicate_requests(list(reversed(requests)), seed=5, identity=IDENTITY)
    assert [o.observation_id for o in first.observations] == [
        o.observation_id for o in second.observations
    ]


def test_observation_ids_change_with_prompt_version():
    """A prompt-version change must not silently reuse old generations."""
    requests = [_request("i1", "K1", "text")]
    first = deduplicate_requests(requests, seed=1, identity=IDENTITY)
    second = deduplicate_requests(requests, seed=1, identity=IDENTITY_V2)
    assert first.observations[0].observation_id != second.observations[0].observation_id


# --- generation-identity scoping (repair: canonical key hardening) --------


def test_same_identity_and_item_and_prompt_aliases():
    result = deduplicate_requests(
        [_request("i1", "K1", "p"), _request("i1", "M1", "p")],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 1


def test_different_model_revision_is_not_the_same_generation():
    """The same model label at a different revision is a different model
    artifact, so its output identity differs."""
    other = dataclasses.replace(IDENTITY, model_revision="synthetic-rev-2")
    first = deduplicate_requests([_request("i1", "K1", "p")], seed=1, identity=IDENTITY)
    second = deduplicate_requests([_request("i1", "K1", "p")], seed=1, identity=other)
    assert (
        first.observations[0].observation_id != second.observations[0].observation_id
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("do_sample", True),
        ("num_beams", 4),
        ("max_new_tokens", 64),
        ("temperature", 0.7),
        ("top_p", 0.9),
    ],
)
def test_output_affecting_generation_settings_change_the_identity(field, value):
    other = dataclasses.replace(IDENTITY, **{field: value})
    first = deduplicate_requests([_request("i1", "K1", "p")], seed=1, identity=IDENTITY)
    second = deduplicate_requests([_request("i1", "K1", "p")], seed=1, identity=other)
    assert (
        first.observations[0].observation_id != second.observations[0].observation_id
    )


def test_cohort_membership_alone_does_not_create_a_new_generation():
    result = deduplicate_requests(
        [
            _request("i1", "M1", "p", cohorts=("A",)),
            _request("i1", "M1", "p", cohorts=("B", "C")),
        ],
        seed=1,
        identity=IDENTITY,
    )
    assert result.unique_observations == 1
    assert result.observations[0].cohorts == {"A", "B", "C"}


def test_observation_records_its_generation_determinants():
    result = deduplicate_requests([_request("i1", "K1", "p")], seed=1, identity=IDENTITY)
    observation = result.observations[0]
    assert observation.model_revision == "synthetic-rev-1"
    assert observation.prompt_version == "v1"
    assert observation.generation_settings_fingerprint


def test_unresolved_revision_is_marked_not_silently_treated_as_resolved():
    unresolved = dataclasses.replace(IDENTITY, model_revision=None)
    result = deduplicate_requests([_request("i1", "K1", "p")], seed=1, identity=unresolved)
    assert result.observations[0].model_revision is None

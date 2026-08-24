"""Tests for Phase 3D evidence-condition generation.

Phase 3D has exactly one job: run what Phase 3C sealed. So the properties
worth pinning are all forms of *refusal* — the ways a run that would have
quietly departed from the freeze is stopped instead — plus the one
substantive definition Phase 3D applies, `context_adopted` (§24).

Nothing here loads a real model. The adapter is a stub whose reply is
scripted per test, because what is under test is the plumbing around the
generation, never the generation itself.
"""

from __future__ import annotations

import json

import pytest

from conflict_eval.models.base import GenerationConfig
from conflict_eval.phase3 import evidence_run as er
from conflict_eval.phase3.runtime_capture import sha256_text

BASELINE_TEMPLATE = (
    "Evidence:\n{evidence_or_none}\n\nQuestion:\n{question}\n\n"
    "Answer: <short answer>\nDecision: <answer or uncertain>\n"
)
EVIDENCE_TEMPLATE = (
    'Source: {source}\n\nStatement:\nThe answer to the question "{question}" '
    'is "{asserted_answer}".'
)
REV = "a" * 40


class _Model:
    """Stub adapter. Returns a scripted reply; never touches a real model."""

    def __init__(self, reply="Answer: Paris\nDecision: answer\nConfidence: 90",
                 model_id="Fake/Model", revision=REV):
        self.model_id = model_id
        self.model_revision = revision
        self.requested_revision = revision
        self.resolved_revision = revision
        self._reply = reply
        self.calls = 0

    def generate(self, messages, generation_config):
        self.calls += 1
        return self._reply


def _baseline(item_id="1", group="KC"):
    return {
        "item_id": str(item_id),
        "question": f"Where is {item_id}?",
        "gold_answer": "Paris",
        "gold_aliases": ["paris"],
        "memory_answer": "Paris" if group == "KC" else "Berlin",
        "foil_answer": "Rome" if group == "KC" else None,
        "knowledge_group": group,
    }


def _trial(item_id="1", condition="K1", *, model="qwen", asserted="Paris",
           source="a government website", group="KC", status="agreement",
           truth="true"):
    return {
        "model_key": model,
        "model_id": "Fake/Model",
        "model_revision": REV,
        "item_id": str(item_id),
        "relation": "country",
        "knowledge_group": group,
        "margin_stratum": "low",
        "cohorts": ["A"],
        "condition": condition,
        "arm": "common",
        "source_role": "identity_a",
        "source_label": source,
        "asserted_answer": asserted,
        "evidence_truth": truth,
        "conflict_status": status,
        "prompt_version": "v1",
        "rendered_prompt_sha256": "",
    }


def _render(trial, baseline):
    return er.render_planned_prompt(
        trial, baseline,
        baseline_template=BASELINE_TEMPLATE, evidence_template=EVIDENCE_TEMPLATE,
    )


def _fixture(conditions=("K1",), *, model="qwen", group="KC", asserted="Paris",
             source="a government website"):
    """A tiny sealed world: trials, observation, alias map and manifest."""
    baseline = _baseline("1", group)
    trials = [
        _trial("1", c, model=model, asserted=asserted, source=source, group=group)
        for c in conditions
    ]
    digest = sha256_text(_render(trials[0], baseline))
    for t in trials:
        t["rendered_prompt_sha256"] = digest
    observation = {
        "observation_id": "obs1",
        "model_key": model,
        "model_revision": REV,
        "item_id": "1",
        "prompt_hash": digest,
        "prompt_version": "v1",
        "generation_settings_fingerprint": "fp",
        "aliased_conditions": [
            {"condition": c, "arm": "common", "source_role": "identity_a",
             "source_label": source, "evidence_truth": "true",
             "conflict_status": "agreement"}
            for c in conditions
        ],
        "is_aliased": len(conditions) > 1,
        "cohorts": ["A"],
    }
    manifest = {
        "deduplication_alias_map": {f"{model}|1|{c}": "obs1" for c in conditions},
        "compute": {"unique_observations": 1, "nominal_condition_slots": len(conditions)},
        "deduplication_provenance": {
            "per_model": {model: {"unique_planned_generations": 1}}
        },
        "models": {model: {"model_specific_arm_enabled": True}},
    }
    return manifest, trials, [observation], {model: {"1": baseline}}


def _build(manifest, trials, observations, baselines, **kw):
    return er.build_run_plan(
        manifest=manifest, trial_rows=trials, observations=observations,
        baseline_by_model=baselines,
        baseline_template=kw.get("baseline_template", BASELINE_TEMPLATE),
        evidence_template=kw.get("evidence_template", EVIDENCE_TEMPLATE),
    )


# ---------------------------------------------------------------------------
# Freeze fidelity
# ---------------------------------------------------------------------------


def test_a_config_changed_since_the_freeze_stops_the_run(tmp_path):
    """The check the gate cannot do: the config's bytes still match.

    A drifted config could change model identity or seed while every
    manifest field still looked internally consistent.
    """
    class _Cfg:
        pass

    path = tmp_path / "phase3_study.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    manifest = {"artifact_hashes": {"phase3_config": "b" * 64}}

    import conflict_eval.phase3.real_run_gate as gate

    original = gate.check_readiness
    gate.check_readiness = lambda *a, **k: type(
        "R", (), {"ready": True, "describe": lambda self: ""}
    )()
    try:
        with pytest.raises(er.Phase3DError, match="RUNTIME_REPRODUCIBILITY_FAILURE"):
            er.assert_freeze_intact(_Cfg(), manifest, config_path=path)
    finally:
        gate.check_readiness = original


def test_a_closed_gate_stops_the_run(tmp_path):
    class _Cfg:
        pass

    path = tmp_path / "c.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    import conflict_eval.phase3.real_run_gate as gate

    original = gate.check_readiness
    gate.check_readiness = lambda *a, **k: type(
        "R", (), {"ready": False, "describe": lambda self: "  - blocked"}
    )()
    try:
        with pytest.raises(er.Phase3DError, match="VALIDATION_FAILURE"):
            er.assert_freeze_intact(_Cfg(), {"artifact_hashes": {}}, config_path=path)
    finally:
        gate.check_readiness = original


# ---------------------------------------------------------------------------
# The plan must be exactly the sealed plan
# ---------------------------------------------------------------------------


def test_the_run_plan_reproduces_the_sealed_prompt_digests():
    plan = _build(*_fixture())
    assert len(plan) == 1
    assert plan[0]["prompt_sha256"] == sha256_text(plan[0]["prompt"])
    assert plan[0]["conditions"] == ["K1"]


def test_a_changed_prompt_template_is_refused_before_any_generation():
    """The property that makes the freeze mean something.

    An edited template renders a different experiment. It must fail at plan
    time, not silently produce outcomes for prompts nobody pre-registered.
    """
    manifest, trials, obs, base = _fixture()
    with pytest.raises(er.Phase3DError, match="RUNTIME_REPRODUCIBILITY_FAILURE"):
        _build(manifest, trials, obs, base,
               baseline_template=BASELINE_TEMPLATE + "\nBe concise.\n")


def test_a_changed_source_label_is_refused():
    manifest, trials, obs, base = _fixture()
    trials[0]["source_label"] = "a personal blog"
    with pytest.raises(er.Phase3DError, match="RUNTIME_REPRODUCIBILITY_FAILURE"):
        _build(manifest, trials, obs, base)


def test_aliased_conditions_asserting_different_answers_are_refused():
    """§22 collapses only prompt-identical conditions. If two aliases assert
    different content, one generation cannot lawfully stand for both."""
    manifest, trials, obs, base = _fixture(("K1", "M1"))
    trials[1]["asserted_answer"] = "Lyon"
    with pytest.raises(er.Phase3DError, match="asserting different answers"):
        _build(manifest, trials, obs, base)


def test_aliased_conditions_disagreeing_on_conflict_status_are_refused():
    manifest, trials, obs, base = _fixture(("K1", "M1"))
    trials[1]["conflict_status"] = "conflict"
    with pytest.raises(er.Phase3DError, match="disagreeing on 'conflict_status'"):
        _build(manifest, trials, obs, base)


def test_an_m_condition_for_a_disabled_arm_is_refused():
    """Mistral and Gemma generate no M1/M2 at all (§34)."""
    manifest, trials, obs, base = _fixture(("K1",), model="mistral")
    manifest["models"]["mistral"]["model_specific_arm_enabled"] = False
    trials.append(_trial("1", "M1", model="mistral"))
    with pytest.raises(er.Phase3DError, match="disabled under the frozen"):
        _build(manifest, trials, obs, base)


def test_a_trial_absent_from_the_sealed_alias_map_is_refused():
    manifest, trials, obs, base = _fixture()
    manifest["deduplication_alias_map"] = {}
    with pytest.raises(er.Phase3DError, match="no entry in the sealed"):
        _build(manifest, trials, obs, base)


def test_a_selected_item_with_no_screening_record_is_refused():
    manifest, trials, obs, base = _fixture()
    base["qwen"] = {}
    with pytest.raises(er.Phase3DError, match="no screening record"):
        _build(manifest, trials, obs, base)


@pytest.mark.parametrize("total,per_model", [(2, 1), (1, 2)])
def test_a_plan_that_does_not_match_the_manifest_count_is_refused(total, per_model):
    """Checked in total AND per model, so over-planning one model cannot be
    hidden by under-planning another."""
    manifest, trials, obs, base = _fixture()
    manifest["compute"]["unique_observations"] = total
    manifest["deduplication_provenance"]["per_model"]["qwen"][
        "unique_planned_generations"
    ] = per_model
    with pytest.raises(er.Phase3DError, match="EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE"):
        _build(manifest, trials, obs, base)


def test_a_plan_containing_an_unplanned_model_is_refused():
    plan = _build(*_fixture())
    manifest = {
        "compute": {"unique_observations": 2},
        "deduplication_provenance": {
            "per_model": {"qwen": {"unique_planned_generations": 1}}
        },
    }
    rogue = dict(plan[0], model_key="llama", observation_id="obs2")
    with pytest.raises(er.Phase3DError, match="never planned"):
        er.assert_run_plan_matches_manifest([plan[0], rogue], manifest)


# ---------------------------------------------------------------------------
# The one definition Phase 3D applies: context_adopted (§24)
# ---------------------------------------------------------------------------


def test_context_adopted_is_true_only_on_a_committed_matching_answer():
    plan = _build(*_fixture(asserted="Rome"))
    record = er.build_evidence_record(
        plan[0],
        model=_Model("Answer: Rome\nDecision: answer\nConfidence: 90"),
        gen_config=GenerationConfig(),
    )
    assert record["context_adopted"] is True
    assert record["answer_class"] == "context"


def test_matching_text_under_decision_uncertain_never_counts_as_adoption():
    """§24, verbatim: 'Textual answer content under Decision: uncertain never
    counts.' The text is still recorded, for the separate §25 mechanistic
    analysis — but it must not reach the primary outcome."""
    plan = _build(*_fixture(asserted="Rome"))
    record = er.build_evidence_record(
        plan[0],
        model=_Model("Answer: Rome\nDecision: uncertain\nConfidence: 10"),
        gen_config=GenerationConfig(),
    )
    assert record["context_adopted"] is False
    assert record["answer_class"] == "uncertain"
    assert record["parsed_answer"] == "Rome"


def test_a_memory_answer_is_not_adoption():
    plan = _build(*_fixture(asserted="Rome"))
    record = er.build_evidence_record(
        plan[0],
        model=_Model("Answer: Paris\nDecision: answer\nConfidence: 90"),
        gen_config=GenerationConfig(),
    )
    assert record["context_adopted"] is False
    assert record["answer_class"] in ("memory", "gold")
    assert record["final_correct"] is True


def test_the_record_carries_every_alias_so_one_outcome_serves_both_contrasts():
    plan = _build(*_fixture(("K1", "M1")))
    record = er.build_evidence_record(
        plan[0], model=_Model(), gen_config=GenerationConfig()
    )
    assert record["conditions"] == ["K1", "M1"]
    assert record["is_aliased"] is True
    assert record["observation_id"] == "obs1"


def test_a_tampered_prompt_is_refused_at_generation_time():
    plan = _build(*_fixture())
    plan[0]["prompt"] = plan[0]["prompt"] + " (edited)"
    model = _Model()
    with pytest.raises(er.Phase3DError, match="EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE"):
        er.build_evidence_record(plan[0], model=model, gen_config=GenerationConfig())
    assert model.calls == 0, "nothing may be generated from a tampered plan row"


def test_an_incomplete_plan_row_is_refused():
    plan = _build(*_fixture())
    del plan[0]["memory_answer"]
    with pytest.raises(er.Phase3DError, match="missing"):
        er.build_evidence_record(plan[0], model=_Model(), gen_config=GenerationConfig())


# ---------------------------------------------------------------------------
# Execution, checkpointing and resume
# ---------------------------------------------------------------------------


def _plan_rows(n, model_key="qwen"):
    """`n` distinct planned generations for one model, in plan order."""
    baselines = {str(i): _baseline(str(i)) for i in range(n)}
    trials, observations = [], []
    for i in range(n):
        item_id = str(i)
        trial = _trial(item_id, "K1", model=model_key)
        digest = sha256_text(_render(trial, baselines[item_id]))
        trial["rendered_prompt_sha256"] = digest
        trials.append(trial)
        observations.append(
            {
                "observation_id": f"obs{i}",
                "model_key": model_key,
                "model_revision": REV,
                "item_id": item_id,
                "prompt_hash": digest,
                "prompt_version": "v1",
                "generation_settings_fingerprint": "fp",
                "aliased_conditions": [
                    {"condition": "K1", "arm": "common", "source_role": "identity_a",
                     "source_label": "a government website",
                     "evidence_truth": "true", "conflict_status": "agreement"}
                ],
                "is_aliased": False,
                "cohorts": ["A"],
            }
        )
    manifest = {
        "deduplication_alias_map": {
            f"{model_key}|{i}|K1": f"obs{i}" for i in range(n)
        },
        "compute": {"unique_observations": n, "nominal_condition_slots": n},
        "deduplication_provenance": {
            "per_model": {model_key: {"unique_planned_generations": n}}
        },
        "models": {model_key: {"model_specific_arm_enabled": True}},
    }
    return _build(manifest, trials, observations, {model_key: baselines})


def _run(tmp_path, rows, model=None, **kw):
    return er.run_evidence_generations(
        model_key="qwen",
        plan_rows=rows,
        results_dir=tmp_path,
        adapter_factory=lambda: model or _Model(),
        gen_config=GenerationConfig(),
        manifest_sha256="m" * 64,
        run_plan_sha256=kw.pop("run_plan_sha256", "p" * 64),
        expected_model_id="Fake/Model",
        expected_revision=REV,
        require_cuda=False,
        block_size=kw.pop("block_size", 2),
        **kw,
    )


def test_generations_are_checkpointed_and_digest_sealed(tmp_path):
    rows = _plan_rows(5)
    result = _run(tmp_path, rows)
    assert result.generated_total == 5
    assert result.blocks_completed == 3  # 2 + 2 + 1
    completed, records = er.load_completed_blocks(tmp_path, "qwen")
    assert len(records) == 5
    for block in completed:
        meta = json.loads(block.meta_path.read_text(encoding="utf-8"))
        assert meta["sha256"] == block.sha256
        assert meta["phase"] == "3D"
        assert meta["run_plan_sha256"] == "p" * 64


def test_a_resumed_run_regenerates_nothing_already_completed(tmp_path):
    rows = _plan_rows(5)
    _run(tmp_path, rows)
    model = _Model()
    result = _run(tmp_path, rows, model=model)
    assert result.resumed_total == 5
    assert result.generated_total == 0
    assert model.calls == 0, "a fully resumed run must not even load work"


def test_a_corrupted_completed_block_is_refused_not_silently_redone(tmp_path):
    rows = _plan_rows(3)
    _run(tmp_path, rows)
    path, _ = er.block_paths(tmp_path, "qwen", 0)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(er.Phase3DError, match="fails its recorded SHA256"):
        er.load_completed_blocks(tmp_path, "qwen")


def test_resuming_against_a_different_plan_is_refused(tmp_path):
    """Two plans must never be mixed into one artifact."""
    rows = _plan_rows(4)
    _run(tmp_path, rows)
    shuffled = list(reversed(rows))
    with pytest.raises(er.Phase3DError, match="do not match the current run plan"):
        _run(tmp_path, shuffled)


def test_resuming_under_a_different_run_plan_digest_is_refused(tmp_path):
    rows = _plan_rows(4)
    _run(tmp_path, rows)
    with pytest.raises(er.Phase3DError, match="produced from run plan"):
        _run(tmp_path, rows, run_plan_sha256="q" * 64)


def test_the_wrong_model_or_revision_is_refused(tmp_path):
    rows = _plan_rows(2)
    with pytest.raises(er.Phase3DError, match="frozen manifest requires"):
        _run(tmp_path, rows, model=_Model(model_id="Other/Model"))
    with pytest.raises(er.Phase3DError, match="frozen manifest requires"):
        _run(tmp_path, rows, model=_Model(revision="b" * 40))


def test_a_model_with_no_planned_generations_is_refused(tmp_path):
    with pytest.raises(er.Phase3DError, match="no generations for"):
        _run(tmp_path, _plan_rows(2, model_key="llama"))


def test_the_summary_reports_completeness_honestly(tmp_path):
    rows = _plan_rows(4)
    result = _run(tmp_path, rows)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["planned_generations"] == 4
    assert summary["complete"] is True
    assert summary["phase"] == "3D"

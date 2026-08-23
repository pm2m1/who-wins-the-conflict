"""Adversarial tests for the Phase 3C outcome-blind baseline screen.

The screening runner is the only Phase 3 code that executes a model, so
the properties that matter are the ones that keep it inside the §11/§41
boundary and make a GPU run mechanical and resumable:

- it can never build an evidence condition;
- blocks are the frozen size, capped at the frozen ceiling;
- a completed block is immutable and is verified by digest on resume;
- artifacts from different models or revisions can never be mixed;
- a corrupted or missing block is detected rather than silently reused.

No real model is loaded anywhere here: a deterministic fake adapter stands
in, which is exactly what makes these properties testable offline.
"""

from __future__ import annotations

import json

import pytest

from conflict_eval.models.base import GenerationConfig
from conflict_eval.phase3.artifact_verification import (
    EVIDENCE_LEAK_FIELDS,
    combine_blocks,
    verify_model_artifacts,
)
from conflict_eval.phase3.baseline_runner import (
    FORBIDDEN_EVIDENCE_MODULES,
    BaselineRunnerError,
    assert_no_evidence_machinery_imported,
    block_paths,
    load_completed_blocks,
    order_candidates,
    plan_blocks,
    run_baseline_screen,
)
from conflict_eval.phase3.constants import (
    PHASE3_DATASET_REVISION,
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
)
from conflict_eval.phase3.runtime_capture import (
    RuntimeRequirementError,
    assert_runtime_matches,
    sha256_file,
)

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
DATASET = {
    "hf_dataset_id": "akariasai/PopQA",
    "split": "test",
    "revision": PHASE3_DATASET_REVISION,
}
TEMPLATE = "Question: {question}\n\nAnswer with:\nAnswer: <answer>\nDecision: answer\n"


class _Score:
    def __init__(self, value):
        self.logprob_normalized = value


class FakeAdapter:
    """Deterministic stand-in. Answers correctly for even-numbered items."""

    def __init__(self, model_id=QWEN_ID, revision=QWEN_REV):
        self.model_id = model_id
        self.model_revision = revision
        self.requested_revision = revision
        self.loads = 0

    def generate(self, messages, gen_config):
        prompt = messages[0]["content"]
        index = int("".join(c for c in prompt if c.isdigit()) or 0)
        answer = f"obj-{index}" if index % 2 == 0 else f"wrong-{index}"
        return f"Answer: {answer}\nDecision: answer\nConfidence: high"

    def score_candidate(self, messages, candidate, answer_prefix=""):
        return _Score(-float(len(candidate)) / 10.0)


def _items(n, start=0):
    return [
        {
            "id": f"item-{i:05d}",
            "question": f"Q {i} ?",
            "obj": f"obj-{i}",
            "subj": f"subj-{i}",
            "prop": ["country", "sport", "place of birth", "mother"][i % 4],
            "aliases": [],
        }
        for i in range(start, start + n)
    ]


def _run(tmp_path, items, model_key="qwen", adapter=None, **kw):
    adapter = adapter or FakeAdapter()

    def factory():
        adapter.loads += 1
        return adapter

    return run_baseline_screen(
        model_key=model_key,
        model_id=kw.pop("model_id", QWEN_ID),
        requested_revision=kw.pop("requested_revision", QWEN_REV),
        candidates=items,
        interim_items=items,
        results_dir=tmp_path,
        dataset=dict(DATASET),
        prompt_template=TEMPLATE,
        prompt_version="v1",
        adapter_factory=factory,
        gen_config=GenerationConfig(do_sample=False, max_new_tokens=32, num_beams=1),
        seed=42,
        require_cuda=False,
        **kw,
    )


# --- the §41 boundary ------------------------------------------------------


def test_runner_holds_no_evidence_condition_machinery():
    """Structural, not aspirational: the runner's namespace is inspected."""
    assert_no_evidence_machinery_imported()


def test_forbidden_module_list_names_the_real_condition_builders():
    for module in FORBIDDEN_EVIDENCE_MODULES:
        __import__(module)  # they exist; the point is the runner must not hold them


def test_runner_executes_no_evidence_condition_call():
    """Check executable code, not prose: the docstring legitimately says
    the runner never produces a context_adopted value, so a naive text
    search would match its own safety statement."""
    import ast
    import inspect

    import conflict_eval.phase3.baseline_runner as runner

    tree = ast.parse(inspect.getsource(runner))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for forbidden in ("build_conditions", "render_evidence", "run_trials",
                      "build_phase3_conditions", "is_context_adopted"):
        assert forbidden not in called, forbidden
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported.intersection(FORBIDDEN_EVIDENCE_MODULES), imported


def test_screening_records_carry_no_evidence_fields(tmp_path):
    _run(tmp_path, _items(10))
    for record in combine_blocks(tmp_path, "qwen"):
        assert not EVIDENCE_LEAK_FIELDS.intersection(record)


# --- frozen runtime --------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "int8"])
def test_non_float16_runtime_is_refused(dtype):
    with pytest.raises(RuntimeRequirementError, match="float16"):
        assert_runtime_matches(dtype, "none")


@pytest.mark.parametrize("quant", ["int8", "int4", "gptq", "awq"])
def test_quantized_runtime_is_refused(quant):
    with pytest.raises(RuntimeRequirementError, match="UNQUANTIZED"):
        assert_runtime_matches("float16", quant)


def test_float16_unquantized_is_accepted():
    assert_runtime_matches("float16", "none")


# --- frozen block policy ---------------------------------------------------


def test_block_size_is_exactly_250():
    blocks = plan_blocks(_items(600))
    assert SCREENING_BLOCK_SIZE == 250
    assert [len(b) for b in blocks] == [250, 250, 100]


def test_ceiling_is_exactly_2000():
    assert SCREENING_CEILING_PER_MODEL == 2000
    blocks = plan_blocks(_items(3000))
    assert sum(len(b) for b in blocks) == 2000
    assert len(blocks) == 8


@pytest.mark.parametrize("bad", [100, 249, 251, 500])
def test_a_non_frozen_block_size_is_refused(bad):
    with pytest.raises(BaselineRunnerError, match="frozen"):
        plan_blocks(_items(10), block_size=bad)


@pytest.mark.parametrize("bad", [1000, 2001, 5000])
def test_a_non_frozen_ceiling_is_refused(bad):
    with pytest.raises(BaselineRunnerError, match="frozen"):
        plan_blocks(_items(10), ceiling=bad)


def test_candidate_order_is_deterministic_and_frame_only():
    items = _items(50)
    shuffled = list(reversed(items))
    assert order_candidates(items) == order_candidates(shuffled)


# --- resume ----------------------------------------------------------------


def test_completed_blocks_are_written_with_digests(tmp_path):
    _run(tmp_path, _items(300))
    completed, records = load_completed_blocks(tmp_path, "qwen")
    assert len(completed) == 2
    for block in completed:
        assert block.sha256 == sha256_file(block.path)
        meta = json.loads(block.meta_path.read_text(encoding="utf-8"))
        assert meta["sha256"] == block.sha256
        assert meta["block_size_policy"] == SCREENING_BLOCK_SIZE
        assert meta["ceiling_policy"] == SCREENING_CEILING_PER_MODEL
        assert meta["dtype"] == "float16"
        assert meta["quantization"] == "none"
    assert records


def test_resume_skips_completed_blocks_and_does_not_reload_the_model(tmp_path):
    items = _items(300)
    _run(tmp_path, items)
    before = {p: sha256_file(p) for p in (tmp_path / "qwen" / "blocks").glob("*.jsonl")}

    adapter = FakeAdapter()
    _run(tmp_path, items, adapter=adapter)
    after = {p: sha256_file(p) for p in (tmp_path / "qwen" / "blocks").glob("*.jsonl")}

    assert before == after, "a completed block was rewritten"
    assert adapter.loads == 0, "a fully-resumed run must not load the model at all"


def test_resume_continues_from_the_last_complete_block(tmp_path):
    items = _items(500)
    _run(tmp_path, items[:250])
    assert len(list((tmp_path / "qwen" / "blocks").glob("*.jsonl"))) == 1
    result = _run(tmp_path, items)
    assert result.blocks_completed == 2


def test_a_block_without_its_sidecar_is_treated_as_incomplete(tmp_path):
    _run(tmp_path, _items(500))
    _, meta_path = block_paths(tmp_path, "qwen", 1)
    meta_path.unlink()
    completed, _ = load_completed_blocks(tmp_path, "qwen")
    assert len(completed) == 1


def test_a_corrupted_completed_block_is_refused_not_silently_rescreened(tmp_path):
    _run(tmp_path, _items(250))
    path, _ = block_paths(tmp_path, "qwen", 0)
    path.write_text(path.read_text(encoding="utf-8") + '{"item_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(BaselineRunnerError, match="fails its recorded SHA256"):
        load_completed_blocks(tmp_path, "qwen")


def test_a_gap_in_the_block_sequence_stops_the_resume_prefix(tmp_path):
    _run(tmp_path, _items(750))
    for p in block_paths(tmp_path, "qwen", 1):
        p.unlink()
    completed, _ = load_completed_blocks(tmp_path, "qwen")
    assert len(completed) == 1


def test_blocks_from_another_model_cannot_be_reused(tmp_path):
    _run(tmp_path, _items(250))
    _, meta_path = block_paths(tmp_path, "qwen", 0)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["model_key"] = "llama"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BaselineRunnerError, match="different models"):
        load_completed_blocks(tmp_path, "qwen")


def test_a_changed_candidate_frame_cannot_be_mixed_into_a_resume(tmp_path):
    _run(tmp_path, _items(750))
    with pytest.raises(BaselineRunnerError, match="frame changed"):
        _run(tmp_path, _items(250))


# --- verification of returned artifacts ------------------------------------


def _verify(tmp_path, **kw):
    """Verify with require_cuda relaxed by default.

    These fixtures are produced on a CPU test host, so the CUDA check would
    mask every other assertion. It defaults to True in production and is
    exercised explicitly by
    `test_verification_requires_cuda_for_real_artifacts`.
    """
    return verify_model_artifacts(
        tmp_path,
        kw.pop("model_key", "qwen"),
        expected_model_id=kw.pop("expected_model_id", QWEN_ID),
        expected_revision=kw.pop("expected_revision", QWEN_REV),
        require_cuda=kw.pop("require_cuda", False),
        **kw,
    )


def test_verification_accepts_a_clean_run(tmp_path):
    _run(tmp_path, _items(300))
    report = _verify(tmp_path)
    assert report.ok, report.describe()
    assert report.blocks == 2
    assert report.resolved_revision == QWEN_REV


def test_verification_detects_a_tampered_block(tmp_path):
    _run(tmp_path, _items(250))
    path, _ = block_paths(tmp_path, "qwen", 0)
    path.write_text('{"item_id":"x"}\n', encoding="utf-8")
    report = _verify(tmp_path)
    assert not report.ok
    assert any("EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE" in f for f in report.failures)


def test_verification_detects_a_wrong_revision(tmp_path):
    _run(tmp_path, _items(250))
    report = _verify(tmp_path, expected_revision="0" * 40)
    assert not report.ok
    assert any("RUNTIME_REPRODUCIBILITY_FAILURE" in f for f in report.failures)


def test_verification_detects_a_wrong_model_id(tmp_path):
    _run(tmp_path, _items(250))
    report = _verify(tmp_path, expected_model_id="meta-llama/Llama-3.1-8B-Instruct")
    assert not report.ok
    assert any("model_id" in f for f in report.failures)


def test_verification_detects_a_missing_block(tmp_path):
    _run(tmp_path, _items(750))
    for p in block_paths(tmp_path, "qwen", 1):
        p.unlink()
    report = _verify(tmp_path)
    assert not report.ok
    assert any("not contiguous" in f for f in report.failures)


def test_verification_detects_a_duplicate_block_index(tmp_path):
    _run(tmp_path, _items(250))
    _, meta_path = block_paths(tmp_path, "qwen", 0)
    dup_data, dup_meta = block_paths(tmp_path, "qwen", 1)
    dup_meta.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    dup_data.write_text("", encoding="utf-8")
    report = _verify(tmp_path)
    assert not report.ok
    assert any("duplicate block index" in f for f in report.failures)


def test_verification_requires_cuda_for_real_artifacts(tmp_path):
    """The production default is strict: an artifact not produced under
    CUDA is a reproducibility failure, because the frozen runtime is GPU
    float16 (§7, §35)."""
    _run(tmp_path, _items(250))
    report = _verify(tmp_path, require_cuda=True)
    assert not report.ok
    assert any("not produced under CUDA" in f for f in report.failures)


def test_verify_model_artifacts_defaults_to_requiring_cuda():
    import inspect

    sig = inspect.signature(verify_model_artifacts)
    assert sig.parameters["require_cuda"].default is True


def test_verification_detects_an_evidence_leak(tmp_path):
    _run(tmp_path, _items(250))
    path, meta_path = block_paths(tmp_path, "qwen", 0)
    lines = path.read_text(encoding="utf-8").splitlines()
    leaked = json.loads(lines[0])
    leaked["context_adopted"] = True
    lines[0] = json.dumps(leaked, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sha256"] = sha256_file(path)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    report = _verify(tmp_path)
    assert not report.ok
    assert any("outcome-blind" in f for f in report.failures)


def test_verification_detects_a_wrong_dataset_revision(tmp_path):
    _run(tmp_path, _items(250))
    report = _verify(tmp_path, expected_dataset_revision="0" * 40)
    assert not report.ok
    assert any("dataset revision" in f for f in report.failures)


# --- schema compatibility with the frozen consumer -------------------------


def test_records_match_the_schema_screeningstate_consumes(tmp_path):
    """The runner's output must be directly consumable by the frozen
    Phase 3 screening state, or the two would drift apart silently."""
    from conflict_eval.phase3.screening import ScreeningState, item_id_of

    _run(tmp_path, _items(250))
    records = combine_blocks(tmp_path, "qwen")
    state = ScreeningState("qwen")
    state.add_block(records)
    finalized = state.finalize()
    assert finalized.screened_total == len(records)
    for record in records:
        assert item_id_of(record)
        assert "knowledge_group" in record
    assert any(r.get("knowledge_group") in ("KC", "KW") for r in records)

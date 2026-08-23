"""Phase 3C outcome-blind baseline screening runner (§11, §41).

This is the ONLY Phase 3 module that executes a real model, and it
executes exactly one thing: the no-context baseline prompt plus candidate
scoring, per item. That is the outcome-blind measurement §11 needs to
build the frozen cohorts, and §41 places it in Phase 3C because its output
is a design input, not an outcome.

**It cannot generate an evidence condition.** It imports no condition
builder, no evidence renderer and no trial runner; `assert_no_evidence_
machinery_imported()` asserts that structurally, and a test pins it. There
is no code path here that produces a `context_adopted` value.

Scoring, parsing, eligibility and foil selection are all imported from the
frozen Phase 2 modules rather than reimplemented, so a Phase 3 baseline
record is produced by the same definitions that produced the Phase 2 ones
(§10, §12, §14).

Blockwise and resumable (§11):

- candidates are ordered deterministically and cut into blocks of
  `SCREENING_BLOCK_SIZE`;
- each completed block is written atomically as one JSONL plus a sidecar
  `.meta.json` carrying its SHA256, model identity, resolved revision and
  runtime;
- a completed block is never rewritten -- a resumed run skips it after
  verifying its digest, so an interrupted GPU session costs at most the
  block in flight;
- screening stops at the frozen ceiling, or when supply criteria are met.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conflict_eval.data.conflict_eligibility import (
    build_relation_subject_object_index,
    classify_primary_conflict_eligibility,
)
from conflict_eval.data.foils import build_relation_index, sample_foil
from conflict_eval.data.normalize import is_match, normalize_answer
from conflict_eval.evaluation.baseline_eligibility import (
    classify_baseline_eligibility,
    is_clean_factual_candidate,
)
from conflict_eval.evaluation.parse import parse_response
from conflict_eval.experiment.prompts import ANSWER_FIELD_PREFIX, render_experiment_prompt
from conflict_eval.phase3.constants import (
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
)
from conflict_eval.phase3.runtime_capture import (
    assert_cuda_available,
    assert_runtime_matches,
    runtime_provenance,
    sha256_file,
    sha256_text,
)
from conflict_eval.phase3.screening import ScreeningState
from conflict_eval.scoring.parametric_margin import compute_parametric_margin

#: Modules that would let this runner build or run an evidence condition.
#: Importing any of them here would break the §41 boundary, so their
#: absence is asserted rather than merely intended.
FORBIDDEN_EVIDENCE_MODULES: tuple[str, ...] = (
    "conflict_eval.experiment.conditions",
    "conflict_eval.experiment.evidence",
    "conflict_eval.experiment.runner",
    "conflict_eval.phase3.conditions",
)


class BaselineRunnerError(RuntimeError):
    """Raised when screening input, sequencing or provenance is invalid."""


def assert_no_evidence_machinery_imported() -> None:
    """Assert this module's own imports cannot reach evidence generation.

    Checks this module's namespace, not `sys.modules`: another part of the
    process may legitimately have imported the condition builder, but THIS
    runner must not hold a reference to it.
    """
    namespace = vars(sys.modules[__name__])
    leaked = sorted(
        name
        for name, value in namespace.items()
        if getattr(value, "__module__", None) in FORBIDDEN_EVIDENCE_MODULES
        or (getattr(value, "__name__", None) in FORBIDDEN_EVIDENCE_MODULES)
    )
    if leaked:
        raise BaselineRunnerError(
            f"Baseline screening runner has imported evidence-condition "
            f"machinery {leaked}. Phase 3C screening is outcome-blind and may "
            "never construct C0/K/M conditions (§11, §41)."
        )


# --- deterministic block planning ----------------------------------------


def order_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic screening order.

    Sorted by item id, so the order is a property of the candidate frame
    alone -- independent of dict ordering, file order, or the machine. Two
    hosts screening the same frame produce identical blocks.
    """
    return sorted(candidates, key=lambda item: str(item["id"]))


def plan_blocks(
    candidates: list[dict[str, Any]],
    block_size: int = SCREENING_BLOCK_SIZE,
    ceiling: int = SCREENING_CEILING_PER_MODEL,
) -> list[list[dict[str, Any]]]:
    """Cut the ordered frame into frozen-size blocks, capped at the ceiling."""
    if block_size != SCREENING_BLOCK_SIZE:
        raise BaselineRunnerError(
            f"block_size must be the frozen {SCREENING_BLOCK_SIZE}, got {block_size} "
            "(§11); the block size is not a tuning parameter."
        )
    if ceiling != SCREENING_CEILING_PER_MODEL:
        raise BaselineRunnerError(
            f"ceiling must be the frozen {SCREENING_CEILING_PER_MODEL}, got "
            f"{ceiling} (§11)."
        )
    ordered = order_candidates(candidates)[:ceiling]
    return [ordered[i : i + block_size] for i in range(0, len(ordered), block_size)]


# --- per-block artifacts --------------------------------------------------


def block_paths(results_dir: str | Path, model_key: str, index: int) -> tuple[Path, Path]:
    directory = Path(results_dir) / model_key / "blocks"
    return (
        directory / f"block_{index:04d}.jsonl",
        directory / f"block_{index:04d}.meta.json",
    )


@dataclasses.dataclass(frozen=True)
class CompletedBlock:
    index: int
    path: Path
    meta_path: Path
    sha256: str
    record_count: int


def _write_block(
    records: list[dict[str, Any]],
    path: Path,
    meta_path: Path,
    *,
    index: int,
    model_key: str,
    model_id: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    dataset: dict[str, Any],
    prompt_version: str,
    runtime: dict[str, Any],
) -> CompletedBlock:
    """Write one completed block atomically, then its digest sidecar.

    The JSONL lands first and is hashed from disk; the sidecar is written
    last. A crash between the two leaves a block with no sidecar, which
    `load_completed_blocks` treats as incomplete and re-screens -- rather
    than a sidecar claiming a digest for a file that was never finished.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    tmp.replace(path)

    digest = sha256_file(path)
    meta = {
        "block_index": index,
        "model_key": model_key,
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "record_count": len(records),
        "block_size_policy": SCREENING_BLOCK_SIZE,
        "ceiling_policy": SCREENING_CEILING_PER_MODEL,
        "dataset": dataset,
        "prompt_version": prompt_version,
        "sha256": digest,
        "payload_sha256": sha256_text(payload),
        **runtime,
    }
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".partial")
    meta_tmp.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    meta_tmp.replace(meta_path)
    return CompletedBlock(
        index=index, path=path, meta_path=meta_path, sha256=digest, record_count=len(records)
    )


def load_completed_blocks(
    results_dir: str | Path, model_key: str
) -> tuple[list[CompletedBlock], list[dict[str, Any]]]:
    """Return verified completed blocks and their records, in block order.

    A block counts as complete only when its sidecar exists AND the file's
    recomputed digest matches it. A digest mismatch raises rather than
    being silently re-screened: a corrupted completed block is an integrity
    problem the researcher must see, not something to paper over.

    Stops at the first gap, so resume is always from a contiguous prefix.
    """
    completed: list[CompletedBlock] = []
    records: list[dict[str, Any]] = []
    index = 0
    while True:
        path, meta_path = block_paths(results_dir, model_key, index)
        if not path.exists() or not meta_path.exists():
            break
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual = sha256_file(path)
        if actual != meta.get("sha256"):
            raise BaselineRunnerError(
                f"Completed block {index} for {model_key!r} fails its recorded "
                f"SHA256 (recorded {meta.get('sha256')!r}, actual {actual!r}). "
                "Refusing to reuse a corrupted empirical artifact."
            )
        if meta.get("model_key") != model_key:
            raise BaselineRunnerError(
                f"Block {index} in {model_key!r}'s directory records model_key "
                f"{meta.get('model_key')!r}; artifacts from different models must "
                "never be mixed."
            )
        block_records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        completed.append(
            CompletedBlock(
                index=index,
                path=path,
                meta_path=meta_path,
                sha256=actual,
                record_count=len(block_records),
            )
        )
        records.extend(block_records)
        index += 1
    return completed, records


# --- the baseline measurement --------------------------------------------


def build_baseline_record(
    item: dict[str, Any],
    *,
    model: Any,
    gen_config: Any,
    prompt: str,
    prompt_version: str,
    relation_index: Any,
    relation_subject_index: Any,
    rng: random.Random,
) -> dict[str, Any] | None:
    """One outcome-blind baseline record, or None if malformed.

    Deliberately mirrors the frozen Phase 2 `cmd_screen` schema field for
    field, and calls the same imported helpers, so Phase 3 screening
    records are produced by the Phase 2 definitions rather than by a second
    implementation that could drift. Phase 2 code is historical and is not
    modified to share this path.
    """
    messages = [{"role": "user", "content": prompt}]
    raw_generation = model.generate(messages, gen_config)
    parsed = parse_response(raw_generation)
    if parsed.malformed:
        return None

    gold = item["obj"]
    aliases = item.get("aliases", [])
    baseline_correct = is_match(parsed.answer, gold, aliases)
    record: dict[str, Any] = {
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "requested_revision": getattr(model, "requested_revision", None),
        "item_id": item["id"],
        "subject": item.get("subj"),
        "relation": item.get("prop"),
        "question": item["question"],
        "gold_answer": gold,
        "gold_aliases": aliases,
        "raw_generation": raw_generation,
        "parsed_answer": parsed.answer,
        "parsed_decision": parsed.decision,
        "parsed_confidence": parsed.confidence,
        "normalized_answer": normalize_answer(parsed.answer),
        "baseline_correct": baseline_correct,
        "prompt_version": prompt_version,
        "prompt": prompt,
        "generation_config": gen_config.as_dict(),
        "manual_review": False,
    }

    eligibility = classify_baseline_eligibility(
        parsed.answer, parsed.decision, parsed.malformed
    )
    if not eligibility.eligible:
        record["knowledge_group"] = "excluded"
        record["exclusion_reason"] = eligibility.reason
        return record

    if baseline_correct:
        foil = sample_foil(item, relation_index, rng)
        if foil is None:
            record["knowledge_group"] = "excluded"
            record["exclusion_reason"] = "no_defensible_foil"
            return record
        record["knowledge_group"] = "KC"
        record["foil_answer"] = foil.foil_answer
        record["foil_source_item_id"] = foil.source_item_id
        record["foil_generation_method"] = foil.generation_method
        memory_answer, conflicting_answer = gold, foil.foil_answer
    else:
        if not is_clean_factual_candidate(parsed.answer):
            record["knowledge_group"] = "manual_review"
            record["manual_review"] = True
            return record
        record["knowledge_group"] = "KW"
        memory_answer, conflicting_answer = parsed.answer, gold

    conflict_eligibility = classify_primary_conflict_eligibility(
        item.get("prop"), item.get("subj"), relation_subject_index
    )
    record["primary_conflict_eligible"] = conflict_eligibility.eligible
    record["conflict_eligibility_reason"] = conflict_eligibility.reason
    if conflict_eligibility.reason in (
        "relation_multi_object",
        "relation_requires_review",
        "relation_unrecognized",
    ):
        record["manual_review"] = True

    memory_score = model.score_candidate(
        messages, memory_answer, answer_prefix=ANSWER_FIELD_PREFIX
    )
    conflicting_score = model.score_candidate(
        messages, conflicting_answer, answer_prefix=ANSWER_FIELD_PREFIX
    )
    record["memory_answer"] = memory_answer
    record["conflicting_context_answer"] = conflicting_answer
    record["memory_logprob_normalized"] = memory_score.logprob_normalized
    record["conflicting_answer_logprob_normalized"] = conflicting_score.logprob_normalized
    record["parametric_margin"] = compute_parametric_margin(
        memory_score.logprob_normalized, conflicting_score.logprob_normalized
    )
    return record


@dataclasses.dataclass(frozen=True)
class ScreenResult:
    model_key: str
    blocks_completed: int
    screened_total: int
    stopped_reason: str
    summary_path: Path
    summary_sha256: str


def run_baseline_screen(
    *,
    model_key: str,
    model_id: str,
    requested_revision: str,
    candidates: list[dict[str, Any]],
    interim_items: list[dict[str, Any]],
    results_dir: str | Path,
    dataset: dict[str, Any],
    prompt_template: str,
    prompt_version: str,
    adapter_factory: Callable[[], Any],
    gen_config: Any,
    seed: int,
    phase2_excluded_ids: frozenset[str] | None = None,
    dtype: str = "float16",
    quantization: str = "none",
    require_cuda: bool = True,
    block_size: int = SCREENING_BLOCK_SIZE,
    ceiling: int = SCREENING_CEILING_PER_MODEL,
) -> ScreenResult:
    """Screen one model blockwise, resumably, outcome-blind.

    `adapter_factory` is called only when at least one block still needs
    screening, so a fully-resumed run never loads the model at all -- which
    is what keeps a single model resident on the GPU at a time.
    """
    assert_no_evidence_machinery_imported()
    assert_runtime_matches(dtype, quantization)
    if require_cuda:
        assert_cuda_available()

    blocks = plan_blocks(candidates, block_size=block_size, ceiling=ceiling)
    completed, prior_records = load_completed_blocks(results_dir, model_key)
    if len(completed) > len(blocks):
        raise BaselineRunnerError(
            f"{len(completed)} completed blocks exist for {model_key!r} but the "
            f"candidate frame plans only {len(blocks)}. The frame changed under a "
            "resumed run; refusing to mix incompatible screening artifacts."
        )

    relation_index = build_relation_index(interim_items)
    relation_subject_index = build_relation_subject_object_index(interim_items)

    state = ScreeningState(
        model_key,
        phase2_excluded_ids=phase2_excluded_ids,
        block_size=block_size,
        ceiling=ceiling,
    )
    for start in range(0, len(prior_records), block_size):
        state.add_block(prior_records[start : start + block_size])

    model = None
    runtime = runtime_provenance(dtype=dtype, quantization=quantization)

    for index in range(len(completed), len(blocks)):
        if state.should_stop():
            break
        if model is None:
            model = adapter_factory()
        block_items = blocks[index]
        # A fresh RNG per block, seeded from the frozen seed and the block
        # index: foil sampling then depends only on (seed, block), never on
        # how many blocks a resumed process happened to run in one session.
        rng = random.Random((seed, index).__hash__())
        records = []
        for item in block_items:
            prompt = render_experiment_prompt(
                prompt_template, item["question"], evidence_text=None
            )
            record = build_baseline_record(
                item,
                model=model,
                gen_config=gen_config,
                prompt=prompt,
                prompt_version=prompt_version,
                relation_index=relation_index,
                relation_subject_index=relation_subject_index,
                rng=rng,
            )
            if record is not None:
                records.append(record)
        path, meta_path = block_paths(results_dir, model_key, index)
        _write_block(
            records,
            path,
            meta_path,
            index=index,
            model_key=model_key,
            model_id=model_id,
            requested_revision=requested_revision,
            resolved_revision=getattr(model, "model_revision", None),
            dataset=dataset,
            prompt_version=prompt_version,
            runtime=runtime,
        )
        state.add_block(records)
        completed.append(
            CompletedBlock(index, path, meta_path, sha256_file(path), len(records))
        )

    finalized = state.finalize()
    summary = {
        "model_key": model_key,
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": getattr(model, "model_revision", None)
        if model is not None
        else (
            json.loads(completed[0].meta_path.read_text(encoding="utf-8")).get(
                "resolved_revision"
            )
            if completed
            else None
        ),
        "dataset": dataset,
        "prompt_version": prompt_version,
        "blocks_completed": len(completed),
        "screened_total": finalized.screened_total,
        "stopped_reason": finalized.stopped_reason,
        "cohort_a_supply_met": finalized.cohort_a_supply_met,
        "cohort_b_supply_met": finalized.cohort_b_supply_met,
        "cohort_a_per_stratum": finalized.final_supply.cohort_a_per_stratum,
        "block_sha256": {str(b.index): b.sha256 for b in completed},
        "block_record_counts": {str(b.index): b.record_count for b in completed},
        "block_size_policy": block_size,
        "ceiling_policy": ceiling,
        **runtime,
    }
    summary_path = Path(results_dir) / model_key / "screening_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ScreenResult(
        model_key=model_key,
        blocks_completed=len(completed),
        screened_total=finalized.screened_total,
        stopped_reason=finalized.stopped_reason,
        summary_path=summary_path,
        summary_sha256=sha256_file(summary_path),
    )

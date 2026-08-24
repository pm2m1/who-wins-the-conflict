"""Phase 3D evidence-condition generation (§22, §24, §41).

Phase 3C sealed *what will be run*. This module executes exactly that and
nothing else. It is the only Phase 3D module that loads a model.

The whole design here is subordination to the freeze: every quantity that
determines a generation -- which items, which conditions, which source
label, which asserted answer, which model revision, which prompt -- is read
out of the sealed manifest and its referenced artifacts. Nothing is
recomputed from the screening data, nothing is re-selected, and there is no
parameter a caller can pass that would change any of them.

Three checks make that structural rather than aspirational:

- **The freeze must still be intact.** `assert_freeze_intact` refuses to
  run unless the real-run gate opens on the committed config *and* the
  manifest's recorded config digest still describes the config on disk. A
  freeze that has drifted is a `RUNTIME_REPRODUCIBILITY_FAILURE`, not
  something to run anyway.
- **Prompts are re-rendered and verified, never re-invented.** Each planned
  prompt is rebuilt with the frozen renderers and checked against the
  `rendered_prompt_sha256` Phase 3C recorded. Any drift in a template, an
  answer field or a source label fails loudly before a single token is
  generated.
- **Aliased conditions must genuinely agree.** §22 collapses conditions
  only when they are prompt-identical; this module additionally requires
  that every alias of an observation asserts the same answer from the same
  source with the same conflict status, so one generation can lawfully
  stand for all of them.

Scoring reuses the frozen Phase 2 definitions (`classify_answer`,
`is_context_adopted`, `is_final_correct`) rather than restating them, so a
Phase 3 outcome is produced by the same code that produced the Phase 2
ones (§24).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conflict_eval.data.normalize import normalize_answer
from conflict_eval.evaluation.classify import (
    classify_answer,
    is_context_adopted,
    is_final_correct,
)
from conflict_eval.evaluation.parse import parse_response
from conflict_eval.experiment.evidence import render_evidence
from conflict_eval.experiment.prompts import render_experiment_prompt
from conflict_eval.phase3.constants import (
    CONDITION_C0,
    MODEL_SPECIFIC_ARM_CONDITIONS,
)
from conflict_eval.phase3.runtime_capture import (
    assert_cuda_available,
    assert_runtime_matches,
    runtime_provenance,
    sha256_file,
    sha256_text,
)

#: Terminal blocker labels. A Phase 3D failure is reported under one of
#: these, never worked around.
INTEGRITY_FAILURE = "EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE"
REPRODUCIBILITY_FAILURE = "RUNTIME_REPRODUCIBILITY_FAILURE"
VALIDATION_FAILURE = "VALIDATION_FAILURE"

#: How many generations are written per checkpoint file.
#:
#: This is an I/O checkpoint interval and carries NO scientific meaning. It
#: is deliberately not one of the frozen §11 screening constants: the
#: screening block size is a design parameter that governs an adaptive
#: stopping rule, whereas this only bounds how much GPU time an interrupted
#: session can lose. Changing it cannot change which generations happen or
#: what any of them contain.
CHECKPOINT_BLOCK_SIZE = 100

#: Fields a run-plan row must carry before anything may be generated from it.
REQUIRED_PLAN_FIELDS = (
    "observation_id",
    "model_key",
    "model_id",
    "model_revision",
    "item_id",
    "knowledge_group",
    "prompt",
    "prompt_sha256",
    "prompt_version",
    "aliased_conditions",
    "gold_answer",
    "memory_answer",
)

#: The alias attributes that must be identical across every condition that
#: shares one canonical observation (§22).
_ALIAS_INVARIANT_FIELDS = ("source_label", "evidence_truth", "conflict_status")


class Phase3DError(RuntimeError):
    """Raised when Phase 3D cannot proceed under the sealed freeze."""


# ---------------------------------------------------------------------------
# Freeze fidelity
# ---------------------------------------------------------------------------


def assert_freeze_intact(config, manifest: dict[str, Any], *, config_path: str | Path) -> None:
    """Refuse to run unless the Phase 3C freeze is present and unchanged.

    Delegates the manifest rules to `real_run_gate.check_readiness`, which
    itself delegates to `validate_manifest`, so "Phase 3D started" can never
    mean something weaker than "the freeze is valid". The config-digest
    check is the part the gate cannot do: it proves the config on disk is
    still the byte-for-byte file whose SHA256 the manifest recorded.
    """
    from conflict_eval.phase3.real_run_gate import check_readiness

    report = check_readiness(config, manifest=manifest)
    if not report.ready:
        raise Phase3DError(
            f"{VALIDATION_FAILURE}: the Phase 3C freeze does not authorize a real "
            "run.\n" + report.describe()
        )
    recorded = (manifest.get("artifact_hashes") or {}).get("phase3_config")
    actual = sha256_file(config_path)
    if recorded != actual:
        raise Phase3DError(
            f"{REPRODUCIBILITY_FAILURE}: the Phase 3 config has changed since the "
            f"freeze was sealed (manifest records {recorded}, {config_path} hashes "
            f"to {actual}). Phase 3D runs the frozen study state or it does not run."
        )


def assert_run_plan_matches_manifest(
    plan_rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    """The plan must be exactly the planned generations the freeze counted.

    Checked per model as well as in total, so a plan that happened to reach
    the right grand total by over-planning one model and under-planning
    another is still refused.
    """
    expected_total = (manifest.get("compute") or {}).get("unique_observations")
    if len(plan_rows) != expected_total:
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: run plan has {len(plan_rows)} generations but the "
            f"sealed manifest planned {expected_total} (§23, §36)."
        )
    per_model = (manifest.get("deduplication_provenance") or {}).get("per_model") or {}
    actual: dict[str, int] = {}
    for row in plan_rows:
        actual[row["model_key"]] = actual.get(row["model_key"], 0) + 1
    for model_key, provenance in sorted(per_model.items()):
        expected = provenance["unique_planned_generations"]
        if actual.get(model_key, 0) != expected:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: run plan has {actual.get(model_key, 0)} "
                f"generations for {model_key!r}, sealed manifest planned {expected}."
            )
    unplanned = sorted(set(actual) - set(per_model))
    if unplanned:
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: run plan contains model(s) {unplanned} that the "
            "sealed manifest never planned."
        )


# ---------------------------------------------------------------------------
# Run-plan construction
# ---------------------------------------------------------------------------


def _shared_alias_attributes(
    observation_id: str, trial_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """The asserted content every alias of one observation must share.

    If two conditions collapsed into one generation but assert different
    answers or cite different sources, the collapse was unsound and the
    single stored outcome could not lawfully stand for both (§22 rule 5).
    That is an integrity failure, not something to reconcile by picking one.
    """
    asserted = {row.get("asserted_answer") for row in trial_rows}
    if len(asserted) != 1:
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: observation {observation_id!r} aliases conditions "
            f"asserting different answers {sorted(map(str, asserted))}; prompt-"
            "identical conditions must assert identical content (§22)."
        )
    shared: dict[str, Any] = {"asserted_answer": trial_rows[0].get("asserted_answer")}
    for field in _ALIAS_INVARIANT_FIELDS:
        values = {row.get(field) for row in trial_rows}
        if len(values) != 1:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: observation {observation_id!r} aliases "
                f"conditions disagreeing on {field!r} ({sorted(map(str, values))}); "
                "one generation cannot stand for both (§22)."
            )
        shared[field] = trial_rows[0].get(field)
    return shared


def render_planned_prompt(
    trial_row: dict[str, Any],
    baseline_record: dict[str, Any],
    *,
    baseline_template: str,
    evidence_template: str,
) -> str:
    """Rebuild one planned prompt with the frozen renderers.

    Identical to what Phase 3C rendered when it computed the trial-spec
    digest -- that identity is verified by the caller, which is the point:
    a template edit becomes a loud failure instead of a silently different
    experiment.
    """
    if trial_row["condition"] == CONDITION_C0:
        evidence_text = None
    else:
        evidence_text = render_evidence(
            evidence_template,
            trial_row["source_label"],
            baseline_record["question"],
            trial_row["asserted_answer"],
        )
    return render_experiment_prompt(
        baseline_template, baseline_record["question"], evidence_text
    )


def build_run_plan(
    *,
    manifest: dict[str, Any],
    trial_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    baseline_by_model: dict[str, dict[str, dict[str, Any]]],
    baseline_template: str,
    evidence_template: str,
) -> list[dict[str, Any]]:
    """One executable record per unique canonical observation.

    Joins the sealed trial specification to the sealed deduplication map and
    carries every field the generation and its scoring need, so the GPU host
    needs no access to the screening artifacts and no opportunity to
    re-derive anything.
    """
    alias_map = manifest.get("deduplication_alias_map") or {}
    disabled = {
        key
        for key, entry in (manifest.get("models") or {}).items()
        if entry.get("model_specific_arm_enabled") is False
    }
    model_specific = set(MODEL_SPECIFIC_ARM_CONDITIONS)

    trials_by_observation: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        key = f"{row['model_key']}|{row['item_id']}|{row['condition']}"
        if row["model_key"] in disabled and row["condition"] in model_specific:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: the trial specification contains "
                f"{row['condition']} for {row['model_key']!r}, whose model-specific "
                "arm was disabled under the frozen §34 rule; those generations must "
                "not exist."
            )
        observation_id = alias_map.get(key)
        if observation_id is None:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: planned trial {key!r} has no entry in the "
                "sealed deduplication alias map (§22, §36)."
            )
        trials_by_observation.setdefault(observation_id, []).append(row)

    if len(trials_by_observation) != len(observations):
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: the trial specification resolves to "
            f"{len(trials_by_observation)} observations but the sealed "
            f"deduplication map records {len(observations)}."
        )

    plan: list[dict[str, Any]] = []
    for observation in observations:
        observation_id = observation["observation_id"]
        model_key = observation["model_key"]
        item_id = str(observation["item_id"])
        rows = trials_by_observation.get(observation_id)
        if not rows:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: sealed observation {observation_id!r} has no "
                "planned trial referencing it."
            )
        baseline = (baseline_by_model.get(model_key) or {}).get(item_id)
        if baseline is None:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: no screening record for {model_key!r} item "
                f"{item_id!r}, which the freeze selected for generation."
            )
        shared = _shared_alias_attributes(observation_id, rows)

        prompt = render_planned_prompt(
            rows[0],
            baseline,
            baseline_template=baseline_template,
            evidence_template=evidence_template,
        )
        digest = sha256_text(prompt)
        for row in rows:
            if row["rendered_prompt_sha256"] != digest:
                raise Phase3DError(
                    f"{REPRODUCIBILITY_FAILURE}: prompt for {model_key!r} item "
                    f"{item_id!r} condition {row['condition']!r} re-renders to "
                    f"{digest} but the sealed trial specification recorded "
                    f"{row['rendered_prompt_sha256']}. A template, source label or "
                    "asserted answer has changed since the freeze."
                )
        if observation.get("prompt_hash") != digest:
            raise Phase3DError(
                f"{REPRODUCIBILITY_FAILURE}: observation {observation_id!r} records "
                f"prompt_hash {observation.get('prompt_hash')} but the prompt "
                f"re-renders to {digest}."
            )

        plan.append(
            {
                "observation_id": observation_id,
                "model_key": model_key,
                "model_id": rows[0]["model_id"],
                "model_revision": rows[0]["model_revision"],
                "item_id": item_id,
                "relation": rows[0].get("relation"),
                "knowledge_group": rows[0]["knowledge_group"],
                "margin_stratum": rows[0].get("margin_stratum"),
                "cohorts": sorted(observation.get("cohorts") or []),
                "aliased_conditions": sorted(
                    (dict(a) for a in observation["aliased_conditions"]),
                    key=lambda a: a["condition"],
                ),
                "conditions": sorted(row["condition"] for row in rows),
                "is_aliased": len(rows) > 1,
                **shared,
                "gold_answer": baseline["gold_answer"],
                "gold_aliases": baseline.get("gold_aliases", []),
                "memory_answer": baseline["memory_answer"],
                "foil_answer": baseline.get("foil_answer"),
                "prompt": prompt,
                "prompt_sha256": digest,
                "prompt_version": rows[0]["prompt_version"],
                "generation_settings_fingerprint": observation[
                    "generation_settings_fingerprint"
                ],
            }
        )

    plan.sort(key=lambda row: (row["model_key"], row["item_id"], row["observation_id"]))
    assert_run_plan_matches_manifest(plan, manifest)
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def build_evidence_record(
    plan_row: dict[str, Any],
    *,
    model: Any,
    gen_config: Any,
) -> dict[str, Any]:
    """Generate one planned observation and score it by the frozen rules.

    `context_adopted` is computed from `classify_answer`, which is the
    Phase 2 definition unchanged: true only when the model committed
    (`Decision: answer`) AND its answer matches the asserted context answer
    (§24). Answer text under `Decision: uncertain` is preserved in the
    record for the separate SECONDARY mechanistic analysis (§25) and never
    folded into the primary outcome here.
    """
    missing = [field for field in REQUIRED_PLAN_FIELDS if field not in plan_row]
    if missing:
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: run-plan row is missing {missing}; refusing to "
            "generate from an incomplete plan."
        )
    actual = sha256_text(plan_row["prompt"])
    if actual != plan_row["prompt_sha256"]:
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: run-plan row {plan_row['observation_id']!r} "
            f"carries a prompt hashing to {actual}, not its recorded "
            f"{plan_row['prompt_sha256']}."
        )

    messages = [{"role": "user", "content": plan_row["prompt"]}]
    raw_generation = model.generate(messages, gen_config)
    parsed = parse_response(raw_generation)

    answer_class = classify_answer(
        parsed_answer=parsed.answer,
        decision=parsed.decision,
        malformed=parsed.malformed,
        gold_answer=plan_row["gold_answer"],
        gold_aliases=plan_row.get("gold_aliases", []),
        memory_answer=plan_row["memory_answer"],
        context_answer=plan_row.get("asserted_answer"),
    )
    return {
        "observation_id": plan_row["observation_id"],
        "model_key": plan_row["model_key"],
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "requested_revision": getattr(model, "requested_revision", None),
        "item_id": plan_row["item_id"],
        "relation": plan_row.get("relation"),
        "knowledge_group": plan_row["knowledge_group"],
        "margin_stratum": plan_row.get("margin_stratum"),
        "cohorts": plan_row.get("cohorts", []),
        "conditions": plan_row["conditions"],
        "aliased_conditions": plan_row["aliased_conditions"],
        "is_aliased": plan_row.get("is_aliased", False),
        "source_label": plan_row.get("source_label"),
        "asserted_answer": plan_row.get("asserted_answer"),
        "evidence_truth": plan_row.get("evidence_truth"),
        "conflict_status": plan_row.get("conflict_status"),
        "gold_answer": plan_row["gold_answer"],
        "gold_aliases": plan_row.get("gold_aliases", []),
        "memory_answer": plan_row["memory_answer"],
        "prompt": plan_row["prompt"],
        "prompt_sha256": plan_row["prompt_sha256"],
        "prompt_version": plan_row["prompt_version"],
        "raw_generation": raw_generation,
        "parsed_answer": parsed.answer,
        "normalized_answer": normalize_answer(parsed.answer) if parsed.answer else None,
        "decision": parsed.decision,
        "confidence": parsed.confidence,
        "answer_class": answer_class,
        "context_adopted": is_context_adopted(answer_class),
        "final_correct": is_final_correct(
            parsed.answer, plan_row["gold_answer"], plan_row.get("gold_aliases", [])
        ),
        "manual_review": parsed.malformed,
        "generation_config": gen_config.as_dict(),
    }


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
    prompt_version: str,
    manifest_sha256: str,
    run_plan_sha256: str,
    runtime: dict[str, Any],
) -> CompletedBlock:
    """Write one checkpoint atomically, then its digest sidecar.

    Same ordering discipline as the Phase 3C screening runner: the JSONL
    lands and is hashed from disk before the sidecar is written, so a crash
    between the two leaves a block with no sidecar -- treated as incomplete
    and regenerated -- rather than a sidecar vouching for a truncated file.
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
        "phase": "3D",
        "block_index": index,
        "model_key": model_key,
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "record_count": len(records),
        "checkpoint_block_size": CHECKPOINT_BLOCK_SIZE,
        "prompt_version": prompt_version,
        "freeze_manifest_sha256": manifest_sha256,
        "run_plan_sha256": run_plan_sha256,
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
    """Verified completed checkpoints and their records, in block order.

    A digest mismatch raises rather than being silently regenerated: a
    corrupted completed block is an integrity problem the researcher must
    see. Stops at the first gap, so resume is always from a contiguous
    prefix.
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
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: completed block {index} for {model_key!r} "
                f"fails its recorded SHA256 (recorded {meta.get('sha256')!r}, actual "
                f"{actual!r}). Refusing to reuse a corrupted empirical artifact."
            )
        if meta.get("model_key") != model_key:
            raise Phase3DError(
                f"{INTEGRITY_FAILURE}: block {index} in {model_key!r}'s directory "
                f"records model_key {meta.get('model_key')!r}; artifacts from "
                "different models must never be mixed."
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


@dataclasses.dataclass(frozen=True)
class EvidenceRunResult:
    model_key: str
    blocks_completed: int
    generated_total: int
    resumed_total: int
    summary_path: Path
    summary_sha256: str


def run_evidence_generations(
    *,
    model_key: str,
    plan_rows: list[dict[str, Any]],
    results_dir: str | Path,
    adapter_factory: Callable[[], Any],
    gen_config: Any,
    manifest_sha256: str,
    run_plan_sha256: str,
    expected_model_id: str,
    expected_revision: str,
    dtype: str = "float16",
    quantization: str = "none",
    require_cuda: bool = True,
    block_size: int = CHECKPOINT_BLOCK_SIZE,
) -> EvidenceRunResult:
    """Execute one model's planned generations, resumably.

    The plan is consumed in its frozen order; already-completed checkpoints
    are verified and skipped, and their observation ids are checked against
    the plan so a resumed run can never silently continue from a *different*
    plan than the one it started with.
    """
    ordered = [row for row in plan_rows if row["model_key"] == model_key]
    if not ordered:
        raise Phase3DError(
            f"{INTEGRITY_FAILURE}: the run plan contains no generations for "
            f"{model_key!r}."
        )
    if require_cuda:
        assert_cuda_available()
    assert_runtime_matches(dtype, quantization)

    completed, done_records = load_completed_blocks(results_dir, model_key)
    resumed = len(done_records)
    if resumed:
        planned_prefix = [row["observation_id"] for row in ordered[:resumed]]
        if [r["observation_id"] for r in done_records] != planned_prefix:
            raise Phase3DError(
                f"{REPRODUCIBILITY_FAILURE}: the completed blocks for {model_key!r} "
                "do not match the current run plan's leading observations. The plan "
                "changed between runs; refusing to mix two plans in one artifact."
            )
        for meta_source in completed:
            meta = json.loads(meta_source.meta_path.read_text(encoding="utf-8"))
            if meta.get("run_plan_sha256") not in (None, run_plan_sha256):
                raise Phase3DError(
                    f"{REPRODUCIBILITY_FAILURE}: block {meta_source.index} for "
                    f"{model_key!r} was produced from run plan "
                    f"{meta.get('run_plan_sha256')}, not {run_plan_sha256}."
                )

    remaining = ordered[resumed:]
    runtime = runtime_provenance(dtype=dtype, quantization=quantization)
    generated = 0
    index = len(completed)
    model = None

    if remaining:
        model = adapter_factory()
        if model.model_id != expected_model_id:
            raise Phase3DError(
                f"{REPRODUCIBILITY_FAILURE}: loaded {model.model_id!r}, frozen "
                f"manifest requires {expected_model_id!r}."
            )
        resolved = getattr(model, "resolved_revision", None) or model.model_revision
        if resolved != expected_revision:
            raise Phase3DError(
                f"{REPRODUCIBILITY_FAILURE}: loaded revision {resolved!r}, frozen "
                f"manifest requires {expected_revision!r}."
            )

    for start in range(0, len(remaining), block_size):
        chunk = remaining[start : start + block_size]
        records = [
            build_evidence_record(row, model=model, gen_config=gen_config) for row in chunk
        ]
        path, meta_path = block_paths(results_dir, model_key, index)
        _write_block(
            records,
            path,
            meta_path,
            index=index,
            model_key=model_key,
            model_id=model.model_id,
            requested_revision=getattr(model, "requested_revision", None),
            resolved_revision=getattr(model, "resolved_revision", None)
            or model.model_revision,
            prompt_version=chunk[0]["prompt_version"],
            manifest_sha256=manifest_sha256,
            run_plan_sha256=run_plan_sha256,
            runtime=runtime,
        )
        generated += len(records)
        index += 1

    summary = {
        "phase": "3D",
        "model_key": model_key,
        "planned_generations": len(ordered),
        "generated_this_run": generated,
        "resumed_from_previous_run": resumed,
        "blocks_completed": index,
        "checkpoint_block_size": block_size,
        "freeze_manifest_sha256": manifest_sha256,
        "run_plan_sha256": run_plan_sha256,
        "model_id": expected_model_id,
        "requested_revision": expected_revision,
        "resolved_revision": expected_revision,
        "complete": (resumed + generated) == len(ordered),
        **runtime,
    }
    summary_path = Path(results_dir) / model_key / "evidence_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return EvidenceRunResult(
        model_key=model_key,
        blocks_completed=index,
        generated_total=generated,
        resumed_total=resumed,
        summary_path=summary_path,
        summary_sha256=sha256_file(summary_path),
    )

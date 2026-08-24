"""Verification of returned Phase 3C cloud screening artifacts.

The baseline screen runs on a GPU host; its artifacts come back as files.
Before any cohort is derived from them, this module checks that what came
back is what the frozen design asked for. Nothing here loads a model, and
nothing here modifies a returned file.

The checks are deliberately blunt, and each maps to a terminal blocker the
researcher must see rather than a value this code could quietly repair:

- a digest mismatch is `EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE`;
- a model identity, revision, dtype or quantization that differs from the
  frozen config is `RUNTIME_REPRODUCIBILITY_FAILURE`;
- a missing, duplicated or non-contiguous block is a sequencing failure.

None of these is recoverable by adjusting Phase 3 code, because doing so
after empirical results exist is exactly the outcome-driven change §36
forbids.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from conflict_eval.phase3.constants import (
    PHASE3_DATASET_REVISION,
    SCREENING_BLOCK_SIZE,
    SCREENING_CEILING_PER_MODEL,
)
from conflict_eval.phase3.runtime_capture import sha256_file, sha256_text

INTEGRITY_FAILURE = "EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE"
REPRODUCIBILITY_FAILURE = "RUNTIME_REPRODUCIBILITY_FAILURE"

#: Fields that would prove an evidence condition was generated. Their
#: presence anywhere in a returned screening artifact means the outcome-
#: blind boundary was crossed (§11, §41).
EVIDENCE_LEAK_FIELDS = frozenset(
    {
        "context_adopted",
        "condition",
        "evidence_truth",
        "source_role",
        "source_label",
        "conflict_status",
        "final_correct",
    }
)


@dataclasses.dataclass(frozen=True)
class VerificationReport:
    model_key: str
    ok: bool
    failures: tuple[str, ...]
    blocks: int
    records: int
    resolved_revision: str | None

    def describe(self) -> str:
        if self.ok:
            return f"{self.model_key}: OK ({self.blocks} blocks, {self.records} records)"
        return f"{self.model_key}: FAILED\n" + "\n".join(f"  - {f}" for f in self.failures)


def verify_checksums_file(root: str | Path) -> list[str]:
    """Verify every entry of a `CHECKSUMS.sha256` manifest under `root`.

    Accepts the standard ``<digest>  <relative path>`` format produced by
    `sha256sum`, tolerating the ``*`` binary marker.
    """
    root = Path(root)
    checksums = root / "CHECKSUMS.sha256"
    if not checksums.exists():
        return [f"{INTEGRITY_FAILURE}: {checksums} is missing"]
    failures: list[str] = []
    for line in checksums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            failures.append(f"{INTEGRITY_FAILURE}: unparseable CHECKSUMS line {line!r}")
            continue
        digest, name = parts[0], parts[1].lstrip("*").strip()
        target = root / name
        if not target.exists():
            failures.append(f"{INTEGRITY_FAILURE}: {name} listed in CHECKSUMS is missing")
            continue
        actual = sha256_file(target)
        if actual != digest:
            failures.append(
                f"{INTEGRITY_FAILURE}: {name} digest {actual} != recorded {digest}"
            )
    return failures


def verify_model_artifacts(
    root: str | Path,
    model_key: str,
    *,
    expected_model_id: str,
    expected_revision: str,
    expected_dataset_revision: str = PHASE3_DATASET_REVISION,
    expected_dtype: str = "float16",
    require_cuda: bool = True,
) -> VerificationReport:
    """Verify one model's returned screening artifacts.

    `require_cuda` defaults to True and must stay True for real returned
    artifacts -- the frozen runtime is GPU float16 (§7, §35). It exists so
    offline tests can exercise the structural checks on a CPU host without
    weakening what a real verification demands.
    """
    failures: list[str] = []
    model_dir = Path(root) / model_key
    block_dir = model_dir / "blocks"
    if not block_dir.is_dir():
        return VerificationReport(
            model_key, False, (f"{INTEGRITY_FAILURE}: {block_dir} missing",), 0, 0, None
        )

    metas = sorted(block_dir.glob("block_*.meta.json"))
    if not metas:
        failures.append(f"{INTEGRITY_FAILURE}: no completed blocks for {model_key!r}")

    seen: dict[int, dict[str, Any]] = {}
    records_total = 0
    resolved: str | None = None

    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = meta.get("block_index")
        if index in seen:
            failures.append(f"{INTEGRITY_FAILURE}: duplicate block index {index}")
            continue
        seen[index] = meta

        data_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".jsonl"))
        if not data_path.exists():
            failures.append(f"{INTEGRITY_FAILURE}: block {index} data file missing")
            continue
        actual = sha256_file(data_path)
        if actual != meta.get("sha256"):
            failures.append(
                f"{INTEGRITY_FAILURE}: block {index} digest {actual} != "
                f"recorded {meta.get('sha256')}"
            )

        if meta.get("model_key") != model_key:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} records model_key "
                f"{meta.get('model_key')!r}, expected {model_key!r}"
            )
        if meta.get("model_id") != expected_model_id:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} model_id "
                f"{meta.get('model_id')!r} != frozen {expected_model_id!r}"
            )
        for field in ("requested_revision", "resolved_revision"):
            value = meta.get(field)
            if value != expected_revision:
                failures.append(
                    f"{REPRODUCIBILITY_FAILURE}: block {index} {field} {value!r} != "
                    f"frozen {expected_revision!r}"
                )
        if resolved is None:
            resolved = meta.get("resolved_revision")
        elif meta.get("resolved_revision") != resolved:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} resolved_revision differs "
                "from earlier blocks; artifacts from different loads must not mix"
            )

        if str(meta.get("dtype", "")).lower() not in ("float16", "fp16"):
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} dtype {meta.get('dtype')!r} "
                f"!= {expected_dtype!r}"
            )
        if str(meta.get("quantization", "")).lower() not in ("none", "false"):
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} quantization "
                f"{meta.get('quantization')!r}; Phase 3 runs unquantized"
            )
        hardware = meta.get("hardware") or {}
        if require_cuda and not hardware.get("cuda_available"):
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} was not produced under CUDA"
            )
        dataset = meta.get("dataset") or {}
        if dataset.get("revision") != expected_dataset_revision:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} dataset revision "
                f"{dataset.get('revision')!r} != frozen {expected_dataset_revision!r}"
            )
        if meta.get("block_size_policy") != SCREENING_BLOCK_SIZE:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} block_size_policy "
                f"{meta.get('block_size_policy')!r} != {SCREENING_BLOCK_SIZE}"
            )
        if meta.get("ceiling_policy") != SCREENING_CEILING_PER_MODEL:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} ceiling_policy "
                f"{meta.get('ceiling_policy')!r} != {SCREENING_CEILING_PER_MODEL}"
            )

        lines = [
            line for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        records_total += len(lines)
        for raw in lines:
            leaked = EVIDENCE_LEAK_FIELDS.intersection(json.loads(raw).keys())
            if leaked:
                failures.append(
                    f"{INTEGRITY_FAILURE}: block {index} carries evidence-condition "
                    f"field(s) {sorted(leaked)}; screening must be outcome-blind "
                    "(§11, §41)"
                )
                break

    if seen:
        expected_indices = set(range(max(seen) + 1))
        missing = sorted(expected_indices - set(seen))
        if missing:
            failures.append(
                f"{INTEGRITY_FAILURE}: block sequence is not contiguous; missing {missing}"
            )
        if len(seen) * SCREENING_BLOCK_SIZE > SCREENING_CEILING_PER_MODEL:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: {len(seen)} blocks exceed the frozen "
                f"ceiling of {SCREENING_CEILING_PER_MODEL}"
            )

    return VerificationReport(
        model_key=model_key,
        ok=not failures,
        failures=tuple(failures),
        blocks=len(seen),
        records=records_total,
        resolved_revision=resolved,
    )


def verify_evidence_artifacts(
    root: str | Path,
    model_key: str,
    *,
    expected_model_id: str,
    expected_revision: str,
    expected_manifest_sha256: str,
    expected_run_plan_sha256: str,
    expected_count: int,
    expected_observation_ids: list[str] | None = None,
    expected_dtype: str = "float16",
    require_cuda: bool = True,
) -> VerificationReport:
    """Verify one model's returned **Phase 3D** evidence artifacts.

    Deliberately separate from `verify_model_artifacts`. That function
    refuses any artifact carrying an evidence-condition field, because a
    Phase 3C screening record must be outcome-blind. A Phase 3D record is
    the opposite: `context_adopted` is the primary outcome and its absence
    would be the defect. Keeping the two checks apart means neither can be
    pointed at the wrong stage and quietly pass.

    What is verified here is that the returned outcomes were produced under
    the sealed freeze: the right model at the right revision, from the right
    manifest and the right run plan, covering exactly the planned
    observations with no gap, duplicate or extra.
    """
    failures: list[str] = []
    block_dir = Path(root) / model_key / "blocks"
    if not block_dir.is_dir():
        return VerificationReport(
            model_key, False, (f"{INTEGRITY_FAILURE}: {block_dir} missing",), 0, 0, None
        )

    metas = sorted(block_dir.glob("block_*.meta.json"))
    if not metas:
        failures.append(f"{INTEGRITY_FAILURE}: no completed blocks for {model_key!r}")

    seen: dict[int, dict[str, Any]] = {}
    observation_ids: list[str] = []
    resolved: str | None = None

    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = meta.get("block_index")
        if index in seen:
            failures.append(f"{INTEGRITY_FAILURE}: duplicate block index {index}")
            continue
        seen[index] = meta

        data_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".jsonl"))
        if not data_path.exists():
            failures.append(f"{INTEGRITY_FAILURE}: block {index} data file missing")
            continue
        actual = sha256_file(data_path)
        if actual != meta.get("sha256"):
            failures.append(
                f"{INTEGRITY_FAILURE}: block {index} digest {actual} != recorded "
                f"{meta.get('sha256')}"
            )
        if meta.get("phase") != "3D":
            failures.append(
                f"{INTEGRITY_FAILURE}: block {index} is not marked phase 3D "
                f"(got {meta.get('phase')!r})"
            )
        if meta.get("freeze_manifest_sha256") != expected_manifest_sha256:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} was produced under freeze "
                f"manifest {meta.get('freeze_manifest_sha256')}, expected "
                f"{expected_manifest_sha256}"
            )
        if meta.get("run_plan_sha256") != expected_run_plan_sha256:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} was produced from run plan "
                f"{meta.get('run_plan_sha256')}, expected {expected_run_plan_sha256}"
            )
        if meta.get("model_key") != model_key:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} records model_key "
                f"{meta.get('model_key')!r}, expected {model_key!r}"
            )
        if meta.get("model_id") != expected_model_id:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} model_id "
                f"{meta.get('model_id')!r} != frozen {expected_model_id!r}"
            )
        for field in ("requested_revision", "resolved_revision"):
            if meta.get(field) != expected_revision:
                failures.append(
                    f"{REPRODUCIBILITY_FAILURE}: block {index} {field} "
                    f"{meta.get(field)!r} != frozen {expected_revision!r}"
                )
        if resolved is None:
            resolved = meta.get("resolved_revision")
        elif meta.get("resolved_revision") != resolved:
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} resolved_revision differs "
                "from earlier blocks; artifacts from different loads must not mix"
            )
        if str(meta.get("dtype", "")).lower() not in ("float16", "fp16"):
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} dtype {meta.get('dtype')!r} "
                f"!= {expected_dtype!r}"
            )
        if str(meta.get("quantization", "")).lower() not in ("none", "false"):
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} quantization "
                f"{meta.get('quantization')!r}; Phase 3 runs unquantized"
            )
        if require_cuda and not (meta.get("hardware") or {}).get("cuda_available"):
            failures.append(
                f"{REPRODUCIBILITY_FAILURE}: block {index} was not produced under CUDA"
            )

        for raw in data_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            observation_ids.append(record.get("observation_id"))
            # The primary outcome must be present and boolean. A missing or
            # non-boolean value is not "no effect" -- it is an unscored
            # generation, and pooling it with real zeros would be a fiction.
            if not isinstance(record.get("context_adopted"), bool):
                failures.append(
                    f"{INTEGRITY_FAILURE}: block {index} record "
                    f"{record.get('observation_id')!r} has non-boolean "
                    f"context_adopted {record.get('context_adopted')!r} (§24)"
                )
                break
            if record.get("prompt_sha256") != sha256_text(record.get("prompt") or ""):
                failures.append(
                    f"{INTEGRITY_FAILURE}: block {index} record "
                    f"{record.get('observation_id')!r} carries a prompt that does not "
                    "match its recorded digest"
                )
                break

    if seen:
        missing = sorted(set(range(max(seen) + 1)) - set(seen))
        if missing:
            failures.append(
                f"{INTEGRITY_FAILURE}: block sequence is not contiguous; missing {missing}"
            )

    if len(observation_ids) != expected_count:
        failures.append(
            f"{INTEGRITY_FAILURE}: {len(observation_ids)} generations returned for "
            f"{model_key!r}, the sealed freeze planned {expected_count}"
        )
    duplicates = len(observation_ids) - len(set(observation_ids))
    if duplicates:
        failures.append(
            f"{INTEGRITY_FAILURE}: {duplicates} duplicate observation id(s) returned "
            f"for {model_key!r}; each canonical observation is generated once (§22)"
        )
    if expected_observation_ids is not None:
        expected_set = set(expected_observation_ids)
        returned_set = set(observation_ids)
        unplanned = sorted(returned_set - expected_set)
        absent = sorted(expected_set - returned_set)
        if unplanned:
            failures.append(
                f"{INTEGRITY_FAILURE}: {len(unplanned)} returned observation(s) are "
                f"not in the sealed run plan, e.g. {unplanned[:3]}"
            )
        if absent:
            failures.append(
                f"{INTEGRITY_FAILURE}: {len(absent)} planned observation(s) were not "
                f"returned, e.g. {absent[:3]}"
            )

    return VerificationReport(
        model_key=model_key,
        ok=not failures,
        failures=tuple(failures),
        blocks=len(seen),
        records=len(observation_ids),
        resolved_revision=resolved,
    )


def combine_blocks(root: str | Path, model_key: str) -> list[dict[str, Any]]:
    """Deterministically combine verified blocks into one record list.

    Block order, then within-block file order. Callers must verify first;
    this function does not re-check digests, so that verification failures
    surface as a report rather than an exception mid-derivation.
    """
    block_dir = Path(root) / model_key / "blocks"
    records: list[dict[str, Any]] = []
    for path in sorted(block_dir.glob("block_*.jsonl")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records

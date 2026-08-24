"""Phase 3C command-line entrypoints.

Separate from `conflict_eval.cli` on purpose: that module is frozen Phase 2
historical code and is not modified. These commands drive Phase 3C only.

    python -m conflict_eval.phase3 prepare-data
    python -m conflict_eval.phase3 screen --model qwen
    python -m conflict_eval.phase3 verify-return --root ../phase3c-cloud-return
    python -m conflict_eval.phase3 extract-exclusions --artifact <phase2.jsonl>
    python -m conflict_eval.phase3 gate

`screen` is the ONLY command that loads a model, and it runs the
outcome-blind baseline measurement only (§11, §41). No command here can
generate a C0/K/M evidence condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from conflict_eval.config import load_models_config, load_prompts_config
from conflict_eval.data.popqa import (
    build_interim,
    build_primary_relation_candidate_pool,
    download_raw,
    load_raw_jsonl,
    write_jsonl,
)
from conflict_eval.io.results import read_jsonl
from conflict_eval.models.base import GenerationConfig
from conflict_eval.phase3.artifact_verification import verify_checksums_file, verify_model_artifacts
from conflict_eval.phase3.baseline_runner import run_baseline_screen
from conflict_eval.phase3.config import load_phase3_config
from conflict_eval.phase3.constants import SCREENING_BLOCK_SIZE
from conflict_eval.phase3.manifest import validate_manifest
from conflict_eval.phase3.phase2_exclusions import (
    extract_phase2_qwen_kw_exclusions,
    load_exclusion_file,
    write_exclusion_file,
)
from conflict_eval.phase3.real_run_gate import check_readiness
from conflict_eval.phase3.runtime_capture import sha256_file

DEFAULT_CONFIG = "configs/phase3/phase3_study.yaml"
INTERIM_NAME = "popqa_interim.jsonl"
CANDIDATES_NAME = "popqa_phase3_candidates.jsonl"


def _data_dir(config) -> Path:
    return Path(config.paths["results_dir"]).parent / "data"


def cmd_prepare_data(config_path: str) -> int:
    """Build the Phase 3 candidate frame from the pinned PopQA revision.

    Uses the frozen §9 primary-relation + subject-multiplicity pool, which
    is what the Phase 3 config selects. No model is loaded.
    """
    config = load_phase3_config(config_path)
    dataset = config.dataset
    out_dir = _data_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = download_raw(
        dataset["hf_dataset_id"],
        dataset["split"],
        str(out_dir / "raw"),
        revision=dataset["revision"],
    )
    interim_items, _ = build_interim(load_raw_jsonl(raw_path))
    write_jsonl(interim_items, out_dir / INTERIM_NAME)

    if dataset.get("candidate_pool") != "primary_conflict_relations":
        raise SystemExit(
            f"Phase 3 requires candidate_pool='primary_conflict_relations', got "
            f"{dataset.get('candidate_pool')!r} (§9)."
        )
    pool = build_primary_relation_candidate_pool(interim_items)
    write_jsonl(pool.deduplicated_pool, out_dir / CANDIDATES_NAME)
    print(
        f"prepare-data: {len(interim_items)} interim rows -> "
        f"{len(pool.deduplicated_pool)} primary-relation candidates "
        f"(dataset revision {dataset['revision']})"
    )
    return 0


def _adapter_factory(entry, models_config):
    """Build the HF adapter pinned to the Phase 3 identity and runtime.

    The revision, dtype and quantization come from the Phase 3 config, not
    from `configs/models.yaml` -- that file is frozen Phase 2 material and
    carries `revision: null` and bfloat16, neither of which Phase 3 permits.
    """

    def factory():
        from conflict_eval.models.hf_causal import HFCausalAdapter

        return HFCausalAdapter(
            entry.hf_model_id,
            revision=entry.revision,
            dtype="float16",
            device_map="auto",
            max_memory=None,
        )

    return factory


def cmd_screen(
    model_key: str, config_path: str, exclusions_path: str | None, allow_cpu: bool
) -> int:
    config = load_phase3_config(config_path)
    entry = config.model(model_key)
    if entry.hf_model_id is None or entry.revision is None:
        raise SystemExit(f"model {model_key!r} has no resolved identity/revision (§7).")

    data_dir = _data_dir(config)
    interim_path = data_dir / INTERIM_NAME
    candidates_path = data_dir / CANDIDATES_NAME
    if not candidates_path.exists():
        raise SystemExit(f"{candidates_path} not found -- run prepare-data first.")

    interim_items = read_jsonl(interim_path)
    candidates = read_jsonl(candidates_path)
    prompts_config = load_prompts_config(config.prompts_config)
    models_config = load_models_config("configs/models.yaml")

    excluded = None
    if exclusions_path:
        excluded, digest = load_exclusion_file(exclusions_path)
        print(f"Phase 2 exclusions: {len(excluded)} ids (sha256 {digest})")
    elif model_key == "qwen":
        print(
            "NOTE: no --exclusions supplied. Cohort A supply reporting will not "
            "reflect the §15.1 Phase 2 exclusion; screening itself is unaffected."
        )

    template = Path(prompts_config["baseline"]["template"]).read_text(encoding="utf-8")
    result = run_baseline_screen(
        model_key=model_key,
        model_id=entry.hf_model_id,
        requested_revision=entry.revision,
        candidates=candidates,
        interim_items=interim_items,
        results_dir=config.paths["results_dir"],
        dataset=dict(config.dataset),
        prompt_template=template,
        prompt_version=prompts_config["baseline"]["version"],
        adapter_factory=_adapter_factory(entry, models_config),
        gen_config=GenerationConfig(**models_config.generation),
        seed=config.seed,
        phase2_excluded_ids=excluded,
        require_cuda=not allow_cpu,
    )
    print(
        f"screen[{model_key}]: {result.blocks_completed} blocks, "
        f"{result.screened_total} records, stopped={result.stopped_reason}\n"
        f"  summary: {result.summary_path} (sha256 {result.summary_sha256})"
    )
    return 0


def cmd_verify_return(root: str, config_path: str) -> int:
    config = load_phase3_config(config_path)
    failures = verify_checksums_file(root)
    for line in failures:
        print(line)
    ok = not failures
    for key in sorted(config.models):
        entry = config.model(key)
        report = verify_model_artifacts(
            root,
            key,
            expected_model_id=entry.hf_model_id,
            expected_revision=entry.revision,
        )
        print(report.describe())
        ok = ok and report.ok
    print("\nVERIFIED" if ok else "\nVERIFICATION FAILED")
    return 0 if ok else 1


def cmd_extract_exclusions(artifacts: list[str], out: str) -> int:
    exclusions = extract_phase2_qwen_kw_exclusions(artifacts)
    path, digest = write_exclusion_file(exclusions, out)
    print(
        f"extract-exclusions: {len(exclusions.item_ids)} Phase 2 Qwen KW item ids\n"
        f"  written: {path}\n  sha256:  {digest}"
    )
    return 0


def cmd_build_freeze(
    config_path: str,
    return_root: str,
    derived_dir: str,
    freeze_dir: str,
    seal: bool,
) -> int:
    """Derive the Phase 3C freeze from verified cloud screening artifacts.

    Loads nothing from the network and constructs no model adapter. Every
    output is a deterministic function of the returned blocks, the Phase 2
    exclusion artifact, and the frozen constants.
    """
    from conflict_eval.phase3 import freeze_build as fb

    config = load_phase3_config(config_path)
    root = Path(return_root)
    derived = Path(derived_dir)
    freeze = Path(freeze_dir)
    model_keys = sorted(config.models)

    # --- §15.1 exclusion, re-derived locally from the raw Phase 2 artifact
    pilot = root / "phase2" / "qwen_pilot_trials.jsonl"
    exclusions = extract_phase2_qwen_kw_exclusions([pilot])
    _exclusion_path, exclusion_sha = write_exclusion_file(
        exclusions, derived / "phase2_qwen_kw_exclusions.json"
    )
    excluded_ids = frozenset(exclusions.item_ids)
    returned_ids, _ = load_exclusion_file(root / "qwen" / "qwen_phase2_kw_exclusions.json")
    if returned_ids != excluded_ids:
        raise SystemExit(
            "EMPIRICAL_ARTIFACT_INTEGRITY_FAILURE: the locally re-derived §15.1 "
            "exclusion list does not match the one returned from the GPU host."
        )
    print(
        f"exclusions: {len(excluded_ids)} Qwen KW ids re-derived locally and "
        f"matched against the returned artifact (sha256 {exclusion_sha})"
    )

    # --- Parts XVII/XIX: replay screening, derive artifacts, freeze strata
    finalized_by_model: dict[str, object] = {}
    derived_by_model: dict[str, dict] = {}
    screened_ids: dict[str, list[str]] = {}
    for key in model_keys:
        finalized = fb.replay_screening(
            root, key, phase2_excluded_ids=excluded_ids if key == "qwen" else None
        )
        finalized_by_model[key] = finalized
        raw = fb.combine_blocks(root, key)
        derived_by_model[key] = fb.derive_model_artifacts(
            finalized, derived, raw_records=raw
        )
        screened_ids[key] = sorted(str(r["item_id"]) for r in finalized.records)
        summary = derived_by_model[key]["summary"]
        print(
            f"screen[{key}]: {summary['screened_total']} records, "
            f"stopped={summary['stopped_reason']}, "
            f"KC={summary['knowledge_counts']['KC']} "
            f"KW={summary['knowledge_counts']['KW']}"
        )

    # Every model must have screened the SAME candidate frame prefix (§8,
    # §11). A model emits no record for a candidate whose generation failed
    # to parse (`build_baseline_record` returns None), so the per-model
    # record sets legitimately differ in SIZE while the underlying frame
    # must not differ in CONTENT: each is a subset of one common prefix
    # whose size is exactly blocks x block_size.
    frames = {key: set(ids) for key, ids in screened_ids.items()}
    candidate_ids = sorted(set().union(*frames.values()))
    expected_frame_size = (
        derived_by_model[model_keys[0]]["summary"]["blocks_screened"]
        * SCREENING_BLOCK_SIZE
    )
    if len(candidate_ids) != expected_frame_size:
        raise SystemExit(
            "RUNTIME_REPRODUCIBILITY_FAILURE: the union of screened item ids is "
            f"{len(candidate_ids)}, not the {expected_frame_size} candidates the "
            "frozen block plan covers; the models did not screen one common "
            "frame prefix (§8, §11)."
        )
    malformed_drops: dict[str, int] = {}
    for key in model_keys:
        extra = frames[key] - set(candidate_ids)
        if extra:
            raise SystemExit(
                f"RUNTIME_REPRODUCIBILITY_FAILURE: {key!r} screened "
                f"{len(extra)} item(s) outside the common candidate frame "
                "prefix (§8, §11)."
            )
        if screened_ids[key] != sorted(screened_ids[key]):
            raise SystemExit(
                f"RUNTIME_REPRODUCIBILITY_FAILURE: {key!r} emitted records out "
                "of the frozen ascending item-id order (§11)."
            )
        malformed_drops[key] = expected_frame_size - len(frames[key])
    print(
        f"candidate frame prefix: {len(candidate_ids)} items; per-model records "
        f"dropped as malformed: {malformed_drops}"
    )

    # --- Parts XX-XXIII: cohorts and membership
    bundle = fb.build_cohorts(
        finalized_by_model,
        config.seed,
        phase2_excluded_ids=excluded_ids,
        cohort_c_target=int(config.cohorts["c"]["target_size"]),
    )
    membership = fb.cross_cohort_membership(bundle)

    # --- Part XXIV: trial specification (no execution)
    prompts_config = load_prompts_config(config.prompts_config)
    baseline_template = Path(prompts_config["baseline"]["template"]).read_text(
        encoding="utf-8"
    )
    evidence_template = Path(prompts_config["evidence"]["template"]).read_text(
        encoding="utf-8"
    )
    prompt_version = prompts_config["baseline"]["version"]
    trial_rows, requests_by_model = fb.build_trial_specification(
        finalized_by_model,
        membership,
        config,
        baseline_template=baseline_template,
        evidence_template=evidence_template,
        prompt_version=prompt_version,
    )
    trial_path, trial_sha = fb.write_jsonl(derived / "trial_specification.jsonl", trial_rows)

    # --- Part XXV: realized dedup map
    models_config = load_models_config("configs/models.yaml")
    dedup = fb.build_dedup_map(
        requests_by_model,
        config,
        prompt_version=prompt_version,
        gen_config=GenerationConfig(**models_config.generation),
    )

    # The per-observation records carry every rendered condition and run to
    # tens of MB; under the repository's artifact policy
    # (configs/frozen/README.md) bulk runtime output stays outside Git and is
    # referenced by immutable digest. The alias map -- what §36 actually
    # requires -- goes into the manifest and the committed freeze record.
    dedup_obs_path, dedup_obs_sha = fb.write_json(
        derived / "dedup_observations.json", dedup["observations"]
    )
    dedup["observations_file"] = str(dedup_obs_path)
    dedup["observations_file_sha256"] = dedup_obs_sha

    # --- Part XXVI: analysis-status realization
    analysis = fb.realize_analysis_status(config, bundle)

    # --- Part XXVII: the §36 manifest
    runtime_root = root / "runtime"
    frame_line = (runtime_root / "candidate-frame.sha256").read_text(
        encoding="utf-8"
    ).split()
    reference_meta = json.loads(
        (root / model_keys[0] / "blocks" / "block_0000.meta.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = fb.assemble_manifest(
        config=config,
        config_sha256=sha256_file(config_path),
        repository_commit=(runtime_root / "git-head.txt").read_text(encoding="utf-8").strip(),
        candidate_ids=candidate_ids,
        candidate_file_sha256=frame_line[0],
        trial_file_sha256=trial_sha,
        prompt_version=prompt_version,
        finalized_by_model=finalized_by_model,
        derived_by_model=derived_by_model,
        bundle=bundle,
        membership=membership,
        dedup=dedup,
        analysis=analysis,
        phase2_excluded_ids=excluded_ids,
        phase2_exclusion_sha256=exclusion_sha,
        environment=reference_meta["environment"],
        hardware=reference_meta["hardware"],
        device_map="auto",
        max_memory="none (unconstrained; single 24GiB RTX 3090)",
        screening_extras={
            "stopped_reason": _combined_stop_reason(derived_by_model),
            "candidate_frame_sha256": frame_line[0],
            "candidate_frame_path": frame_line[1] if len(frame_line) > 1 else None,
            "candidate_item_ids_scope": (
                "the screened prefix of the frozen §9 primary-relation candidate "
                "frame; all four models screened exactly this set of candidates"
            ),
            "malformed_generation_drops": malformed_drops,
        },
    )
    problems = validate_manifest(manifest, expected_model_keys=model_keys)

    # --- write the freeze record
    fb.write_json(freeze / "cohort_a.json", manifest.data["cohorts"]["A"])
    fb.write_json(freeze / "cohort_b.json", manifest.data["cohorts"]["B"])
    fb.write_json(freeze / "cohort_c.json", manifest.data["cohorts"]["C"])
    fb.write_json(freeze / "cohort_membership_map.json", membership)
    fb.write_json(freeze / "final_margin_strata.json", manifest.data["final_margin_strata"])
    fb.write_json(freeze / "analysis_status.json", {
        "entries": manifest.data["analysis_status"],
        "realization": analysis["provenance"],
    })
    fb.write_json(
        freeze / "deduplication_map.json",
        {
            "alias_map": dedup["alias_map"],
            "per_model": dedup["per_model"],
            "totals": dedup["totals"],
            "observations_file": str(dedup_obs_path),
            "observations_file_sha256": dedup_obs_sha,
            "observation_count": len(dedup["observations"]),
        },
    )

    if problems:
        for problem in problems:
            print(f"  - {problem}")
        print(f"\nVALIDATION_FAILURE: {len(problems)} unmet §36 requirement(s)")
        return 1

    if seal:
        manifest = fb.freeze_manifest(manifest)
    manifest_path, manifest_sha = fb.write_json(
        freeze / "phase3c_pre_run_manifest.json", manifest.data
    )
    print(
        f"\ntrial specification: {trial_path} (sha256 {trial_sha})\n"
        f"dedup observations:  {dedup_obs_path}\n"
        f"manifest:            {manifest_path} (sha256 {manifest_sha})\n"
        f"frozen={manifest.data['frozen']} "
        f"ready_for_real_run={manifest.data['ready_for_real_run']}\n"
        f"validate_manifest -> [] ({len(problems)} problems)"
    )
    return 0


def _combined_stop_reason(derived_by_model: dict[str, dict]) -> str:
    reasons = {d["summary"]["stopped_reason"] for d in derived_by_model.values()}
    return reasons.pop() if len(reasons) == 1 else "mixed:" + ",".join(sorted(reasons))


def cmd_gate(config_path: str, manifest_path: str | None) -> int:
    config = load_phase3_config(config_path)
    manifest = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    report = check_readiness(config, manifest=manifest)
    print(report.describe())
    print(f"\nready_for_real_run = {config.ready_for_real_run}")
    print(f"READY = {report.ready}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m conflict_eval.phase3",
        description="Phase 3C outcome-blind screening and freeze tooling.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Build the pinned Phase 3 candidate frame.")

    p = sub.add_parser("screen", help="Outcome-blind baseline screen (loads a model).")
    p.add_argument("--model", required=True, choices=["qwen", "llama", "mistral", "gemma"])
    p.add_argument("--exclusions", default=None, help="Phase 2 exclusion artifact (Qwen).")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Debug only; the frozen runtime is unquantized float16 on GPU.",
    )

    p = sub.add_parser("verify-return", help="Verify returned cloud screening artifacts.")
    p.add_argument("--root", required=True)

    p = sub.add_parser("extract-exclusions", help="Derive the §15.1 Qwen KW exclusion list.")
    p.add_argument("--artifact", action="append", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "build-freeze", help="Derive the Phase 3C freeze from verified cloud artifacts."
    )
    p.add_argument("--return-root", required=True)
    p.add_argument("--derived-dir", default="runs/phase3/derived")
    p.add_argument("--freeze-dir", default="configs/phase3/freeze")
    p.add_argument(
        "--seal",
        action="store_true",
        help="Mark the manifest frozen; only at the Phase 3C freeze point (§36).",
    )

    p = sub.add_parser("gate", help="Report Phase 3C real-run readiness.")
    p.add_argument("--manifest", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        return cmd_prepare_data(args.config)
    if args.command == "screen":
        return cmd_screen(args.model, args.config, args.exclusions, args.allow_cpu)
    if args.command == "verify-return":
        return cmd_verify_return(args.root, args.config)
    if args.command == "extract-exclusions":
        return cmd_extract_exclusions(args.artifact, args.out)
    if args.command == "build-freeze":
        return cmd_build_freeze(
            args.config,
            args.return_root,
            args.derived_dir,
            args.freeze_dir,
            args.seal,
        )
    if args.command == "gate":
        return cmd_gate(args.config, args.manifest)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())

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
from conflict_eval.phase3.phase2_exclusions import (
    extract_phase2_qwen_kw_exclusions,
    load_exclusion_file,
    write_exclusion_file,
)
from conflict_eval.phase3.real_run_gate import check_readiness

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
    if args.command == "gate":
        return cmd_gate(args.config, args.manifest)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())

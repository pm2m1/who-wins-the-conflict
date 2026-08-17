"""Command-line interface.

Implements the command functions used by both `python -m conflict_eval
<command>` and the standalone scripts under scripts/ (which are thin
argparse wrappers around the `cmd_*` functions here — see
docs/decisions.md for why the logic lives in one place).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from conflict_eval.config import (
    ConfigError,
    ModelSpec,
    load_models_config,
    load_pilot_config,
    load_prompts_config,
    load_sources_config,
)
from conflict_eval.data.conflict_eligibility import (
    build_relation_subject_object_index,
    classify_primary_conflict_eligibility,
)
from conflict_eval.data.foils import build_relation_index, sample_foil
from conflict_eval.data.normalize import is_match, normalize_answer
from conflict_eval.data.popqa import (
    build_interim,
    build_primary_relation_candidate_pool,
    download_raw,
    load_raw_jsonl,
    screen_candidates,
    write_jsonl,
)
from conflict_eval.data.sampling import (
    assign_margin_bin,
    compute_margin_bin_edges,
    sample_balanced_across_bins,
)
from conflict_eval.evaluation.baseline_eligibility import (
    classify_baseline_eligibility,
    is_clean_factual_candidate,
)
from conflict_eval.evaluation.classify import classify_answer, is_context_adopted, is_final_correct
from conflict_eval.evaluation.parse import parse_response
from conflict_eval.experiment.conditions import build_conditions
from conflict_eval.experiment.evidence import render_evidence
from conflict_eval.experiment.prompts import ANSWER_FIELD_PREFIX, render_experiment_prompt
from conflict_eval.experiment.runner import run_trials
from conflict_eval.io.results import ResultWriter, read_jsonl
from conflict_eval.models.base import GenerationConfig
from conflict_eval.models.dummy import DummyModelAdapter
from conflict_eval.scoring.parametric_margin import compute_parametric_margin

DRY_RUN_BANNER = (
    "*** SYNTHETIC/DEBUG OUTPUT (dummy model adapter) — "
    "never use these files as real pilot results ***"
)


def _output_prefix(model_key: str, spec: ModelSpec) -> str:
    # Dummy-adapter output is always written under a "dryrun_" prefix so
    # it can never be mistaken for, or accidentally merged with, real
    # model results (docs/decisions.md).
    return "dryrun" if spec.adapter == "dummy" else model_key


def _build_model_adapter(spec: ModelSpec):
    if spec.adapter == "dummy":
        print(DRY_RUN_BANNER)
        return DummyModelAdapter(model_id="dummy")
    if spec.adapter == "hf_causal":
        from conflict_eval.models.hf_causal import HFCausalAdapter

        if spec.requires_gated_access:
            print(
                f"Note: {spec.hf_model_id} requires gated Hugging Face access. "
                "If access has not been granted, loading will fail below rather "
                "than silently substituting a different model."
            )
        return HFCausalAdapter(
            spec.hf_model_id,
            revision=spec.revision,
            dtype=spec.dtype,
            device_map=spec.device_map,
            max_memory=spec.max_memory,
        )
    raise ConfigError(f"Unknown adapter type: {spec.adapter!r} for model spec {spec.key!r}")


# ---------------------------------------------------------------------------
# prepare-data
# ---------------------------------------------------------------------------


def cmd_prepare_data(config_path: str) -> None:
    config = load_pilot_config(config_path)

    raw_path = download_raw(
        config.dataset["hf_dataset_id"], config.dataset["split"], config.paths["raw_dir"]
    )
    raw_items = load_raw_jsonl(raw_path)
    interim_items, exclusions = build_interim(raw_items)

    interim_dir = Path(config.paths["interim_dir"])
    write_jsonl(interim_items, interim_dir / "popqa_interim.jsonl")
    write_jsonl(exclusions, interim_dir / "popqa_exclusions.jsonl")

    # dataset.candidate_pool selects the SCREENING FRAME that
    # screening_candidates/seed subsample from (docs/decisions.md,
    # "Support targeted primary conflict screening"). "all" (default)
    # preserves prior behavior exactly: the frame is the full interim
    # pool. "primary_conflict_relations" restricts the frame first, using
    # the already-committed PRIMARY relation + subject-multiplicity
    # policy — this changes the SAMPLING FRAME, not the eligibility rule
    # itself, and results from it must not be read as prevalence
    # estimates over all of PopQA.
    candidate_pool = config.dataset["candidate_pool"]
    pool_result = None
    if candidate_pool == "primary_conflict_relations":
        pool_result = build_primary_relation_candidate_pool(interim_items)
        screening_frame = pool_result.deduplicated_pool
    else:
        screening_frame = interim_items

    candidates = screen_candidates(
        screening_frame, config.dataset["screening_candidates"], config.seed
    )
    processed_dir = Path(config.paths["processed_dir"])
    write_jsonl(candidates, processed_dir / "popqa_candidates.jsonl")

    if pool_result is not None:
        print(
            f"candidate pool = {candidate_pool}\n"
            f"  interim rows = {len(interim_items)}\n"
            f"  eligible primary rows = {len(pool_result.eligible_rows)}\n"
            f"  unique primary relation/subject facts = {len(pool_result.deduplicated_pool)}\n"
            f"  screened candidates = {len(candidates)}"
        )
    else:
        print(f"candidate pool = {candidate_pool}")

    print(
        f"prepare-data: {len(raw_items)} raw rows -> {len(interim_items)} interim "
        f"({len(exclusions)} excluded) -> {len(candidates)} screened candidates."
    )


# ---------------------------------------------------------------------------
# screen (baseline screening + KC/KW classification + parametric margin)
# ---------------------------------------------------------------------------


def _baseline_prompt(templates: dict[str, Any], question: str) -> str:
    template_text = Path(templates["baseline"]["template"]).read_text(encoding="utf-8")
    return render_experiment_prompt(template_text, question, evidence_text=None)


def cmd_screen(model_key: str, config_path: str) -> None:
    config = load_pilot_config(config_path)
    models_config = load_models_config(config.models_config)
    prompts_config = load_prompts_config(config.prompts_config)
    spec = models_config.get(model_key)

    interim_dir = Path(config.paths["interim_dir"])
    processed_dir = Path(config.paths["processed_dir"])
    interim_items = read_jsonl(interim_dir / "popqa_interim.jsonl")
    candidates = read_jsonl(processed_dir / "popqa_candidates.jsonl")
    if not candidates:
        raise RuntimeError("No screened candidates found — run prepare-data first.")

    relation_index = build_relation_index(interim_items)
    relation_subject_index = build_relation_subject_object_index(interim_items)
    rng = random.Random(config.seed)

    model = _build_model_adapter(spec)
    gen_config = GenerationConfig(**models_config.generation)
    prompt_version = prompts_config["baseline"]["version"]

    output_prefix = _output_prefix(model_key, spec)
    results_dir = Path(config.paths["results_dir"])
    baseline_records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for item in candidates:
        question = item["question"]
        gold = item["obj"]
        aliases = item.get("aliases", [])
        prompt = _baseline_prompt(prompts_config, question)
        messages = [{"role": "user", "content": prompt}]

        raw_generation = model.generate(messages, gen_config)
        parsed = parse_response(raw_generation)

        if parsed.malformed:
            exclusions.append({"item_id": item["id"], "reason": "malformed_baseline_response"})
            continue

        baseline_correct = is_match(parsed.answer, gold, aliases)
        record: dict[str, Any] = {
            "model_id": model.model_id,
            "model_revision": model.model_revision,
            "requested_revision": getattr(model, "requested_revision", None),
            "item_id": item["id"],
            "subject": item.get("subj"),
            "relation": item.get("prop"),
            "question": question,
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

        # Eligibility (Decision == "answer", not an explicit abstention)
        # is checked BEFORE KC/KW assignment, not just before the KW
        # cleanliness check — a gold-matching answer accompanied by
        # "Decision: uncertain" must not become KC either
        # (docs/decisions.md, "Baseline abstentions must not become KC/KW
        # memory candidates").
        eligibility = classify_baseline_eligibility(parsed.answer, parsed.decision, parsed.malformed)
        if not eligibility.eligible:
            record["knowledge_group"] = "excluded"
            record["exclusion_reason"] = eligibility.reason
            baseline_records.append(record)
            continue

        if baseline_correct:
            foil = sample_foil(item, relation_index, rng)
            if foil is None:
                record["knowledge_group"] = "excluded"
                record["exclusion_reason"] = "no_defensible_foil"
                baseline_records.append(record)
                continue
            record["knowledge_group"] = "KC"
            record["foil_answer"] = foil.foil_answer
            record["foil_source_item_id"] = foil.source_item_id
            record["foil_generation_method"] = foil.generation_method
            memory_answer, conflicting_answer = gold, foil.foil_answer
        else:
            # A clean KW candidate must be a short, unambiguous factual
            # guess — not a hedge, list, or malformed fragment. Borderline
            # cases are routed to manual_review rather than forced into KW.
            if not is_clean_factual_candidate(parsed.answer):
                record["knowledge_group"] = "manual_review"
                record["manual_review"] = True
                baseline_records.append(record)
                continue
            record["knowledge_group"] = "KW"
            memory_answer, conflicting_answer = parsed.answer, gold

        # Relation-level + subject-level policy for PRIMARY conflict trial
        # eligibility — orthogonal to KC/KW (baseline knowledge state),
        # which is preserved above regardless: different answer != validated
        # semantic conflict (docs/decisions.md, "Restrict primary trials to
        # defensible conflicts"). A record can be a perfectly valid KC/KW
        # item while still being ineligible for automatic primary conflict
        # trial construction; build-pilot filters on this field, not on
        # knowledge_group alone.
        conflict_eligibility = classify_primary_conflict_eligibility(
            item.get("prop"), item.get("subj"), relation_subject_index
        )
        record["primary_conflict_eligible"] = conflict_eligibility.eligible
        record["conflict_eligibility_reason"] = conflict_eligibility.reason
        if conflict_eligibility.reason in ("relation_multi_object", "relation_requires_review", "relation_unrecognized"):
            # A genuinely ambiguous case for a researcher to inspect, as
            # opposed to a settled policy exclusion (relation_not_primary_
            # conflict), which is not flagged for manual review.
            record["manual_review"] = True

        # answer_prefix="Answer: " matches the field label the model is
        # instructed to produce before its answer (docs/decisions.md,
        # "Scoring prefix must include the Answer: field label") — without
        # it, memory_answer/conflicting_answer would be scored as if they
        # were the literal first tokens of the assistant's turn instead of
        # the value of the Answer field.
        memory_score = model.score_candidate(messages, memory_answer, answer_prefix=ANSWER_FIELD_PREFIX)
        conflicting_score = model.score_candidate(
            messages, conflicting_answer, answer_prefix=ANSWER_FIELD_PREFIX
        )
        margin = compute_parametric_margin(
            memory_score.logprob_normalized, conflicting_score.logprob_normalized
        )
        record["memory_answer"] = memory_answer
        record["conflicting_context_answer"] = conflicting_answer
        record["memory_logprob_normalized"] = memory_score.logprob_normalized
        record["conflicting_answer_logprob_normalized"] = conflicting_score.logprob_normalized
        record["parametric_margin"] = margin
        baseline_records.append(record)

    # Margin bins are computed within each model-specific KC/KW pool
    # separately (docs/phase2_research_design.md, "Sample across
    # parametric strength").
    for group in ("KC", "KW"):
        eligible = [r for r in baseline_records if r.get("knowledge_group") == group]
        margins = [r["parametric_margin"] for r in eligible]
        if not margins:
            continue
        edges = compute_margin_bin_edges(margins, n_bins=len(config.sampling["margin_bins"]))
        for r in eligible:
            r["margin_bin"] = assign_margin_bin(
                r["parametric_margin"], edges, tuple(config.sampling["margin_bins"])
            )

    write_jsonl(baseline_records, results_dir / f"{output_prefix}_baseline.jsonl")
    write_jsonl(exclusions, results_dir / f"{output_prefix}_baseline_exclusions.jsonl")

    n_kc = sum(1 for r in baseline_records if r.get("knowledge_group") == "KC")
    n_kw = sum(1 for r in baseline_records if r.get("knowledge_group") == "KW")
    # Wording only, not a semantics change: a record can retain
    # knowledge_group == "KC"/"KW" while manual_review == True (e.g. a
    # conflict-eligibility review flag), so "manual_review=N" alone read
    # as if it meant knowledge_group == "manual_review" specifically.
    # "manual_review_flagged" makes clear this counts the boolean flag
    # across all records, not a third knowledge group value.
    n_manual_review_flagged = sum(1 for r in baseline_records if r.get("manual_review"))
    print(
        f"screen[{model_key}]: {len(baseline_records)} baseline records "
        f"(KC={n_kc}, KW={n_kw}, manual_review_flagged={n_manual_review_flagged}), "
        f"{len(exclusions)} malformed exclusions."
    )


# ---------------------------------------------------------------------------
# diagnose-score (real-model infrastructure validation, not an experiment)
# ---------------------------------------------------------------------------


def cmd_diagnose_score(
    model_key: str, config_path: str, question: str, candidate_a: str, candidate_b: str
) -> None:
    """Score two explicit candidate answers to one question under the
    identical no-evidence prompt prefix, and print a token-level
    breakdown of each score plus their margin.

    This is a diagnostic for validating the model adapter and
    sequence-logprob implementation (see the real-model validation task in
    docs/decisions.md) — its output is never written to results/ and is
    not a research finding.
    """
    config = load_pilot_config(config_path)
    models_config = load_models_config(config.models_config)
    prompts_config = load_prompts_config(config.prompts_config)
    spec = models_config.get(model_key)

    model = _build_model_adapter(spec)
    prompt = _baseline_prompt(prompts_config, question)
    messages = [{"role": "user", "content": prompt}]

    print(
        f"model_id={model.model_id} model_revision={model.model_revision} "
        f"(requested_revision={getattr(model, 'requested_revision', None)!r}, "
        f"resolved_revision={getattr(model, 'resolved_revision', None)!r})"
    )
    print(f"question={question!r}")
    print("messages (safe readable representation, not raw tensors):")
    print(f"  [{{'role': 'user', 'content': <{len(prompt)} chars, see prompts/baseline.txt>}}]")
    print(f"answer_prefix appended after the chat-template generation marker: {ANSWER_FIELD_PREFIX!r}")

    scores = {}
    for label, candidate in (("A", candidate_a), ("B", candidate_b)):
        detailed = model.score_candidate_detailed(
            messages, candidate, answer_prefix=ANSWER_FIELD_PREFIX
        )
        scores[label] = detailed.scored
        print(f"\ncandidate_{label.lower()}={candidate!r}")
        print(f"  tokens={detailed.answer_tokens}")
        print(f"  token_logprobs={[round(x, 4) for x in detailed.token_logprobs]}")
        print(f"  raw_summed_logprob={detailed.scored.logprob_sum:.4f}")
        print(f"  token_count={detailed.scored.token_count}")
        print(f"  normalized_logprob={detailed.scored.logprob_normalized:.4f}")

    margin = compute_parametric_margin(
        scores["A"].logprob_normalized, scores["B"].logprob_normalized
    )
    print(f"\nmargin (A - B) = {margin:.4f}")
    if margin > 0:
        print(f"  -> model prefers candidate A ({candidate_a!r}) over candidate B ({candidate_b!r})")
    elif margin < 0:
        print(f"  -> model prefers candidate B ({candidate_b!r}) over candidate A ({candidate_a!r})")
    else:
        print("  -> no preference measured between A and B")


# ---------------------------------------------------------------------------
# calibrate-sources
# ---------------------------------------------------------------------------


def cmd_calibrate_sources(model_key: str, config_path: str) -> None:
    from conflict_eval.source_preference.calibration import run_calibration_trial
    from conflict_eval.source_preference.counterbalance import expand_pairs_to_presentations
    from conflict_eval.source_preference.pairs import enumerate_unordered_pairs
    from conflict_eval.source_preference.ranking import (
        build_preference_matrix,
        compute_pairwise_stats,
        rank_sources_pilot_heuristic,
    )

    config = load_pilot_config(config_path)
    models_config = load_models_config(config.models_config)
    sources_config = load_sources_config(config.sources_config)
    spec = models_config.get(model_key)

    model = _build_model_adapter(spec)
    gen_config = GenerationConfig(**models_config.generation)
    template = Path("prompts/source_calibration.txt").read_text(encoding="utf-8")
    prompt_version = sources_config["calibration_prompt_version"]

    pairs = enumerate_unordered_pairs(sources_config["source_labels"])
    presentations = expand_pairs_to_presentations(pairs)

    output_prefix = _output_prefix(model_key, spec)
    results_dir = Path(config.paths["results_dir"])
    trials = []
    for i, presentation in enumerate(presentations):
        trial = run_calibration_trial(
            model, template, presentation, prompt_version, config.seed, run_id=f"{output_prefix}-{i}",
            generation_config=gen_config,
        )
        trials.append(dataclasses_asdict(trial))

    write_jsonl(trials, results_dir / f"{output_prefix}_source_calibration.jsonl")

    stats = compute_pairwise_stats(trials)
    matrix = build_preference_matrix(stats)
    ranking = rank_sources_pilot_heuristic(stats)

    summary = {
        "model_id": model.model_id,
        "pairwise_stats": [dataclasses_asdict(s) for s in stats],
        "preference_matrix": matrix,
        "pilot_heuristic_ranking": ranking,
    }
    import json

    with open(results_dir / f"{output_prefix}_source_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"calibrate-sources[{model_key}]: {len(trials)} trials over {len(pairs)} source pairs.")
    print("Pilot heuristic ranking (most- to least-preferred, NOT a statistical threshold):")
    for r in ranking:
        print(
            f"  {r['source']!r}: mean_pairwise_preference_rate="
            f"{r['mean_pairwise_preference_rate']:.2f} ({r['wins']}/{r['total_valid_trials']})"
        )
    print(
        "Set source_roles.<model>.preferred_source / dispreferred_source in "
        f"{config_path} before running build-pilot."
    )


def dataclasses_asdict(obj: Any) -> dict[str, Any]:
    import dataclasses

    return dataclasses.asdict(obj)


# ---------------------------------------------------------------------------
# build-pilot
# ---------------------------------------------------------------------------


def cmd_build_pilot(model_key: str, config_path: str) -> None:
    config = load_pilot_config(config_path)
    models_config = load_models_config(config.models_config)
    prompts_config = load_prompts_config(config.prompts_config)
    spec = models_config.get(model_key)
    output_prefix = _output_prefix(model_key, spec)

    role_config = config.source_roles_for(model_key)
    if not role_config.is_set():
        raise RuntimeError(
            f"build-pilot refuses to run for model '{model_key}': preferred_source/"
            f"dispreferred_source are not set in {config_path}. Run calibrate-sources, "
            "inspect the pilot-heuristic ranking, and set both fields under "
            f"source_roles.{model_key} before building the pilot sample."
        )

    results_dir = Path(config.paths["results_dir"])
    baseline_records = read_jsonl(results_dir / f"{output_prefix}_baseline.jsonl")
    if not baseline_records:
        raise RuntimeError(f"No baseline records found for '{model_key}' — run screen first.")

    # Only relation/subject pairs judged to be a defensible primary
    # conflict are sampled into the pilot — KC/KW alone is not sufficient
    # (docs/decisions.md, "Restrict primary trials to defensible
    # conflicts"): different answer != validated semantic conflict.
    kc_items = [
        r
        for r in baseline_records
        if r.get("knowledge_group") == "KC" and r.get("primary_conflict_eligible") is True
    ]
    kw_items = [
        r
        for r in baseline_records
        if r.get("knowledge_group") == "KW" and r.get("primary_conflict_eligible") is True
    ]

    sampled_kc = sample_balanced_across_bins(
        kc_items, config.sampling["target_kc_items"], config.seed, id_key="item_id"
    )
    sampled_kw = sample_balanced_across_bins(
        kw_items, config.sampling["target_kw_items"], config.seed + 1, id_key="item_id"
    )

    evidence_template = Path(prompts_config["evidence"]["template"]).read_text(encoding="utf-8")
    baseline_template = Path(prompts_config["baseline"]["template"]).read_text(encoding="utf-8")
    evidence_version = prompts_config["evidence_template_version"]
    prompt_version = prompts_config["baseline"]["version"]

    trials: list[dict[str, Any]] = []
    for item in sampled_kc + sampled_kw:
        specs = build_conditions(
            knowledge_group=item["knowledge_group"],
            gold_answer=item["gold_answer"],
            baseline_answer=item["parsed_answer"],
            foil_answer=item.get("foil_answer"),
            preferred_source=role_config.preferred_source,
            dispreferred_source=role_config.dispreferred_source,
        )
        for trial_spec in specs:
            if trial_spec.asserted_answer is None:
                evidence_text = None
            else:
                evidence_text = render_evidence(
                    evidence_template, trial_spec.source_label, item["question"], trial_spec.asserted_answer
                )
            prompt = render_experiment_prompt(baseline_template, item["question"], evidence_text)
            trials.append(
                {
                    "model_id": item["model_id"],
                    "model_revision": item.get("model_revision"),
                    "item_id": item["item_id"],
                    "subject": item.get("subject"),
                    "relation": item.get("relation"),
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "gold_aliases": item.get("gold_aliases", []),
                    "memory_answer": item["memory_answer"],
                    "foil_answer": item.get("foil_answer"),
                    "baseline_answer": item["parsed_answer"],
                    "baseline_correct": item["baseline_correct"],
                    "knowledge_group": item["knowledge_group"],
                    "parametric_margin": item.get("parametric_margin"),
                    "margin_bin": item.get("margin_bin"),
                    "condition": trial_spec.condition,
                    "conflict_status": trial_spec.conflict_status,
                    "evidence_truth": trial_spec.evidence_truth,
                    "source_role": trial_spec.source_role,
                    "source_label": trial_spec.source_label,
                    "evidence_template_version": evidence_version,
                    "evidence_text": evidence_text,
                    "prompt_version": prompt_version,
                    "prompt": prompt,
                }
            )

    write_jsonl(trials, results_dir / f"{output_prefix}_pilot_trials.jsonl")
    print(
        f"build-pilot[{model_key}]: {len(sampled_kc)} KC + {len(sampled_kw)} KW items "
        f"x 5 conditions = {len(trials)} trials written to "
        f"{results_dir / f'{output_prefix}_pilot_trials.jsonl'}."
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(model_key: str, config_path: str) -> None:
    config = load_pilot_config(config_path)
    models_config = load_models_config(config.models_config)
    spec = models_config.get(model_key)
    output_prefix = _output_prefix(model_key, spec)

    results_dir = Path(config.paths["results_dir"])
    trials = read_jsonl(results_dir / f"{output_prefix}_pilot_trials.jsonl")
    if not trials:
        raise RuntimeError(f"No pilot trials found for '{model_key}' — run build-pilot first.")

    model = _build_model_adapter(spec)
    gen_config = GenerationConfig(**models_config.generation)
    writer = ResultWriter(results_dir / f"{output_prefix}_pilot_results.jsonl")

    def generate_fn(trial: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": trial["prompt"]}]
        raw_generation = model.generate(messages, gen_config)
        parsed = parse_response(raw_generation)

        answer_class = classify_answer(
            parsed_answer=parsed.answer,
            decision=parsed.decision,
            malformed=parsed.malformed,
            gold_answer=trial["gold_answer"],
            gold_aliases=trial.get("gold_aliases", []),
            memory_answer=trial["memory_answer"],
            context_answer=_asserted_answer_for_trial(trial),
        )

        record = dict(trial)
        record.update(
            {
                "raw_generation": raw_generation,
                "parsed_answer": parsed.answer,
                "normalized_answer": normalize_answer(parsed.answer) if parsed.answer else None,
                "decision": parsed.decision,
                "confidence": parsed.confidence,
                "answer_class": answer_class,
                "context_adopted": is_context_adopted(answer_class),
                "final_correct": is_final_correct(parsed.answer, trial["gold_answer"], trial.get("gold_aliases", [])),
                "manual_review": parsed.malformed,
                "generation_config": gen_config.as_dict(),
            }
        )
        return record

    stats = run_trials(trials, generate_fn, writer, experiment_type="pilot", seed=config.seed)
    print(f"run[{model_key}]: {stats['run']} generated, {stats['skipped']} already completed (resumed).")


def _asserted_answer_for_trial(trial: dict[str, Any]) -> str | None:
    if trial["condition"] == "C0":
        return None
    if trial["condition"] in ("C1", "C2"):
        return trial["gold_answer"]
    return trial.get("foil_answer") if trial["knowledge_group"] == "KC" else trial["baseline_answer"]


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def cmd_analyze(config_path: str) -> None:
    from conflict_eval.analysis.plots import (
        plot_abstention,
        plot_condition_summary,
        plot_corrective_vs_harmful,
        plot_signature_interaction,
    )
    from conflict_eval.analysis.regression import fit_primary_model
    from conflict_eval.analysis.summaries import condition_summary, override_summary
    from conflict_eval.evaluation.metrics import (
        abstention_rate,
        context_adoption_rate,
        source_effect_on_corrective_override,
        source_effect_on_harmful_override,
    )

    config = load_pilot_config(config_path)
    results_dir = Path(config.paths["results_dir"])
    figures_dir = Path(config.paths["figures_dir"])

    all_records: list[dict[str, Any]] = []
    for model_key in config.models:
        models_config = load_models_config(config.models_config)
        spec = models_config.get(model_key)
        prefix = _output_prefix(model_key, spec)
        records = read_jsonl(results_dir / f"{prefix}_pilot_results.jsonl")
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No pilot results found. Run screen / build-pilot / run for at least one "
            "model before analyze. (Dummy/dry-run output is intentionally not picked "
            "up here unless 'dummy' is listed under models: in the config.)"
        )

    conflict_records = [r for r in all_records if r["conflict_status"] == "conflict"]
    print(f"analyze: {len(all_records)} total records, {len(conflict_records)} conflict trials.")

    if conflict_records:
        car = context_adoption_rate(conflict_records)
        print(f"  CAR (context adoption rate, conflict trials): {car.rate} (n={car.n})")
    harm = source_effect_on_harmful_override(all_records)
    corr = source_effect_on_corrective_override(all_records)
    print(f"  Delta_harm (source effect on harmful override): {harm['delta_harm']}")
    print(f"  Delta_correct (source effect on corrective override): {corr['delta_correct']}")
    ar = abstention_rate(conflict_records) if conflict_records else None
    if ar:
        print(f"  Abstention rate under conflict: {ar.rate} (n={ar.n})")

    print(condition_summary(all_records).to_string(index=False))
    print(override_summary(all_records).to_string(index=False))

    try:
        model_fit = fit_primary_model(all_records)
        print(model_fit.summary())
    except ValueError as e:
        print(f"  Primary regression skipped: {e}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    for plot_fn, filename in [
        (plot_signature_interaction, "plot1_signature_interaction.png"),
        (plot_corrective_vs_harmful, "plot2_corrective_vs_harmful.png"),
        (plot_condition_summary, "plot3_condition_summary.png"),
        (plot_abstention, "plot4_abstention.png"),
    ]:
        try:
            path = plot_fn(all_records, figures_dir / filename)
            print(f"  Wrote {path}")
        except ValueError as e:
            print(f"  Skipped {filename}: {e}")


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conflict_eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="Download and preprocess PopQA.")
    p.add_argument("--config", required=True)

    p = sub.add_parser("screen", help="Baseline screening + KC/KW classification.")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)

    p = sub.add_parser(
        "diagnose-score",
        help="Score two candidate answers to one question; infra validation, not an experiment.",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--candidate-a", required=True)
    p.add_argument("--candidate-b", required=True)

    p = sub.add_parser("calibrate-sources", help="Direct pairwise source-preference calibration.")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)

    p = sub.add_parser("build-pilot", help="Build the C0-C4 pilot sample.")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)

    p = sub.add_parser("run", help="Run the pilot experiment (resumable).")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)

    p = sub.add_parser("analyze", help="Evaluate, summarize, and plot pilot results.")
    p.add_argument("--config", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare-data":
            cmd_prepare_data(args.config)
        elif args.command == "screen":
            cmd_screen(args.model, args.config)
        elif args.command == "diagnose-score":
            cmd_diagnose_score(args.model, args.config, args.question, args.candidate_a, args.candidate_b)
        elif args.command == "calibrate-sources":
            cmd_calibrate_sources(args.model, args.config)
        elif args.command == "build-pilot":
            cmd_build_pilot(args.model, args.config)
        elif args.command == "run":
            cmd_run(args.model, args.config)
        elif args.command == "analyze":
            cmd_analyze(args.config)
    except (ConfigError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Pilot Protocol

Step-by-step execution sequence for running the real pilot. Each step names
the CLI command that performs it; see `README.md` for setup. This protocol
has not yet been executed for a real model as of this writing — see
`README.md` for current pilot status.

1. **Prepare PopQA.**

       python -m conflict_eval prepare-data --config configs/pilot.yaml

   Downloads `akariasai/PopQA` (test split) into `data/raw/`, runs
   deterministic normalization into `data/interim/`, and writes the
   screened candidate pool plus exclusion log into `data/processed/`.
   `dataset.candidate_pool` in the config controls the screening frame
   (docs/decisions.md, "Support targeted primary conflict screening"):
   `all` (default) screens from the full interim pool, unchanged from
   prior behavior; `primary_conflict_relations` restricts the frame,
   before deterministic sampling, to items eligible under the
   already-committed PRIMARY relation and subject-multiplicity policy —
   an efficiency-oriented option for constructing causal conflict trials,
   not a change to which relations are policy-eligible. A targeted-frame
   screen is a different sampling frame from an unrestricted screen, so
   its output must not be read as a prevalence estimate over all of
   PopQA.

2. **Baseline screening, model A.**

       python -m conflict_eval screen --model llama --config configs/pilot.yaml

   Generates clean baseline (no-evidence) answers, classifies KC/KW,
   computes conflict-specific parametric margins and margin bins.

3. **Baseline screening, model B.**

       python -m conflict_eval screen --model qwen --config configs/pilot.yaml

4. **Inspect exclusions.** Manually review
   `data/processed/<model>_baseline_exclusions.jsonl` and
   `manual_review` items before proceeding. This is a real research step,
   not a formality — malformed or ambiguous baseline answers should not
   silently enter the KC/KW pools.

5. **Calibrate sources, separately per model.**

       python -m conflict_eval calibrate-sources --model llama --config configs/pilot.yaml
       python -m conflict_eval calibrate-sources --model qwen --config configs/pilot.yaml

   Writes pairwise calibration trials and a preference-matrix summary per
   model to `results/`.

6. **Researcher confirms source pair(s).** Inspect the calibration summary
   and set `preferred_source` / `dispreferred_source` per model in
   `configs/pilot.yaml`. This is an explicit researcher decision, not an
   automatic selection (see `docs/phase2_research_design.md`).

7. **Build pilot sample.**

       python -m conflict_eval build-pilot --model llama --config configs/pilot.yaml
       python -m conflict_eval build-pilot --model qwen --config configs/pilot.yaml

   Selects ~30 KC + ~30 KW items per model (sampled across margin bins),
   attaches foils, and constructs the C0-C4 trial specifications. Refuses
   to run if `preferred_source`/`dispreferred_source` are unset for that
   model.

8. **Inspect prompts.** Spot-check a sample of rendered prompts written to
   `results/<model>_pilot_trials.jsonl` (pre-generation) for template
   correctness before spending generation compute.

9. **Run pilot.**

       python -m conflict_eval run --model llama --config configs/pilot.yaml
       python -m conflict_eval run --model qwen --config configs/pilot.yaml

   Resumable; safe to interrupt and re-run.

10. **Evaluate.**

        python -m conflict_eval analyze --config configs/pilot.yaml

    (Evaluation/classification runs as part of `analyze`, before the
    regression and plotting steps; see `src/conflict_eval/evaluation/`.)

11. **Analyze.** The same `analyze` command produces descriptive summaries,
    the exploratory logistic regression, and the four figures under
    `figures/`.

12. **Manually review ambiguous outputs.** Inspect every record with
    `manual_review = true` or `answer_class = other` before drawing
    conclusions from the metrics.

13. **Make go/no-go decision.** Using the Signal 1/2/3 report from
    `analyze`, the researcher (not the pipeline) decides whether the pilot
    justifies scaling up. Record the decision and reasoning in
    `docs/decisions.md`.

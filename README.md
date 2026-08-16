# Who Wins the Conflict?

**Parametric Preference Strength and Source Preference in LLM Evidence Use**

*(alternative title: When Sources Override Memory: How Source Preference
Mediates Context-Memory Conflict in LLMs)*

## Research motivation

Four strands of prior work — WildHallucinations, HALoGEN, factuality
finetuning via internal signals, and source-preference in agentic
selection — jointly suggest that faithful LLM use of external evidence
depends on more than whether correct information is available in context.
It plausibly also depends on how strongly the model already prefers its
own parametric answer, and on who the evidence is attributed to. See
`docs/phase1_literature_synthesis.md` for the full literature synthesis
and `docs/research_proposal.md` for the resulting research gap and
proposal.

## Research questions

- **RQ1 (parametric strength):** How does the strength of an LLM's
  pre-existing parametric preference affect its willingness to adopt
  conflicting external evidence?
- **RQ2 (source effect):** Holding evidence content exactly constant, does
  changing only its attributed source change the probability that the
  model follows that evidence instead of its baseline parametric answer?
- **RQ3 (truth asymmetry):** Does source preference amplify both
  beneficial correction of incorrect parametric answers and harmful
  override of correct parametric answers?

Full hypotheses, experimental design, and metrics are specified in
`docs/phase2_research_design.md`.

## Experimental idea (brief)

A controlled pilot on PopQA short-answer factual questions. Evidence is
placed directly in the prompt with a fixed template (no retrieval system —
this isolates evidence integration from retrieval quality). Each eligible
item is run through five conditions (C0: no evidence; C1/C2: correct
evidence from a preferred/dispreferred source; C3/C4: false evidence from
a preferred/dispreferred source), separately for items the model already
answers correctly (KC) and items it already answers incorrectly (KW).
Interpretation is restricted to genuine conflict trials — see
`docs/phase2_research_design.md`, "Conflict vs. agreement."

## Project status

- Literature synthesis, research design, and methodology documents:
  **written** (`docs/`).
- Implementation (data pipeline, model adapters, scoring, source
  calibration, experiment runner, evaluation, analysis): **written and
  unit-tested**.
- Real PopQA download: **validated** during implementation
  (`akariasai/PopQA`, test split) — see `docs/reference_implementations.md`
  and `data/README.md` for recreation instructions. Not committed to this
  repository.
- Real-model adapter/scoring validation: **partially validated**. The
  real Qwen2.5-7B-Instruct tokenizer and chat template were downloaded
  (tokenizer/config files only, ~11MB — no model weights) and used to
  confirm chat-template rendering, answer-token boundary/masking, and
  determinism against real BPE tokenization
  (`tests/test_hf_tokenizer_integration.py`, gated behind
  `CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1`). Actual log-probability scoring
  on real model weights was not run: this environment has no CUDA GPU and
  insufficient free RAM for safe CPU inference of a ~15GB bf16 model —
  see `docs/decisions.md` ("Real-model validation blocked by insufficient
  RAM"). Use `diagnose-score` (above) to complete this validation once
  adequate compute is available.
- Real 8B/7B model runs (baseline screening, source calibration, the
  C0-C4 pilot itself): **not yet run**. This requires an explicit,
  separately-initiated command with the intended model, per
  `docs/pilot_protocol.md`, and — per the point above — a machine with a
  GPU or substantially more free RAM than is available here.
  Llama-3.1-8B-Instruct additionally requires gated Hugging Face access
  that has not been provisioned in this environment.
- **Pilot status: not yet run.**

## Repository layout

    docs/            research documents (read these first)
    configs/          YAML configuration (models, pilot, sources, prompts)
    prompts/          versioned prompt templates
    src/conflict_eval/ implementation package
    scripts/          thin CLI entry-point wrappers
    tests/            unit tests (no model downloads)
    data/             raw/interim/processed PopQA (not committed; see data/README.md)
    results/          pipeline-generated JSONL (not committed; see results/README.md)
    figures/          pipeline-generated figures (not committed; see figures/README.md)

See `docs/decisions.md` for why `scripts/` are thin wrappers around
`src/conflict_eval/cli.py` rather than a second implementation.

## Environment setup

    python -m venv .venv
    .venv\Scripts\pip install -e ".[dev]"   # Windows
    # or: .venv/bin/pip install -e ".[dev]"  # macOS/Linux

Torch is installed CPU-only by default via the plain PyPI index; install a
CUDA build separately if running the 7B/8B models on GPU.

## Commands

    python -m conflict_eval prepare-data --config configs/pilot.yaml
    python -m conflict_eval screen --model llama --config configs/pilot.yaml
    python -m conflict_eval diagnose-score --model qwen --config configs/pilot.yaml \
        --question "What is the capital of France?" --candidate-a "Paris" --candidate-b "London"
    python -m conflict_eval calibrate-sources --model llama --config configs/pilot.yaml
    python -m conflict_eval build-pilot --model llama --config configs/pilot.yaml
    python -m conflict_eval run --model llama --config configs/pilot.yaml
    python -m conflict_eval analyze --config configs/pilot.yaml

`diagnose-score` is an infrastructure-validation diagnostic, not part of
the experiment: it scores two explicit candidate answers to one question
under the identical no-evidence prompt prefix and prints a token-level
breakdown of each score plus their margin, so the chat-template
rendering, answer-token boundary/masking, and margin sign convention can
be inspected directly against a real model before trusting the pipeline's
aggregate output. Its output is never written to `results/`.

Each command has an equivalent standalone script under `scripts/` (e.g.
`python scripts/screen_baseline.py --model llama --config configs/pilot.yaml`).
Full execution order and researcher checkpoints (inspecting exclusions,
confirming source pairs before `build-pilot`, manual review before the
go/no-go decision) are in `docs/pilot_protocol.md`.

### Dry run

Every command above also accepts `--model dummy`, which uses
`DummyModelAdapter` (see `configs/models.yaml`) instead of a real model —
no download, no network call beyond `prepare-data`'s dataset fetch, and no
GPU required. Output from the dummy adapter is always written under a
`dryrun_` filename prefix and preceded by a `*** SYNTHETIC/DEBUG OUTPUT
***` banner, so it can never be mistaken for real pilot data; `analyze`
only aggregates models listed under `models:` in the config, and `dummy`
is deliberately not listed in `configs/pilot.yaml`. This dry run was
exercised end to end during implementation (real PopQA download -> dummy
screening -> dummy source calibration -> dummy build-pilot -> dummy run,
including a verified resumability check -> dummy analyze, including all
four figures) using a scratch config outside this repository, so its
output never touched `data/`, `results/`, or `figures/` here. The
equivalent pipeline is also exercised automatically by
`tests/test_dry_run_pipeline.py` on every test run.

## Reproducibility notes

- All stochastic steps (candidate subsampling, foil sampling, decoding)
  are seeded via `configs/pilot.yaml`.
- Every generation record stores model id, model revision, the exact
  rendered prompt, prompt version, and generation config. `model_revision`
  is the exact resolved Hugging Face commit SHA whenever one could be
  determined (from local cache metadata, no extra download), not just the
  requested `revision` from `configs/models.yaml` — see
  `docs/decisions.md`, "Exact model revision recording".
- Raw PopQA data is not committed (license on the dataset card is
  unstated); `data/README.md` documents exact recreation steps.
  `data/raw/manifest.json` records the exact resolved dataset commit SHA
  (`resolved_revision`) alongside the dataset id/split, when it can be
  determined — see `docs/decisions.md`, "Exact PopQA dataset revision
  recording".
- Experiments are resumable: interrupting `run` and re-invoking it skips
  already-completed generations rather than duplicating them.
- Secrets: this project currently has no `.env` file and reads no
  credentials from the environment; `.gitignore` excludes `.env`/`.env.*`
  (keeping any future `.env.example` trackable) so this stays true if that
  changes later.

## Limitations

This is a pilot (~60 items/model x 5 conditions x 2 models ≈ 600
generations), not a publication-scale study. Evidence is prompt-injected
and templated, not retrieved from real documents. Source preference is
calibrated via direct pairwise elicitation only. Self-reported confidence
is exploratory and not validated as calibrated. See
`docs/research_proposal.md` ("Limitations") and `docs/decisions.md` for
the full list of scope decisions and their rationale.

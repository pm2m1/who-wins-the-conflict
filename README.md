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

The controlled pilot has now been completed on **two instruction-tuned
models**:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

Each model was screened independently on the same pinned PopQA candidate
frame and then evaluated on a model-specific 60-item sample
(30 knowledge-correct / KC and 30 knowledge-wrong / KW items), with each
item run through all five C0-C4 conditions.

That gives:

- **300 generations per model**
- **600 total pilot generations**
- **120 genuine conflict trials per model**
- **240 total primary conflict trials**
- deterministic C0 baseline reproduction of **60/60** selected items for
  each model

The first Qwen pilot is frozen in
[`docs/qwen_pilot_results.md`](docs/qwen_pilot_results.md).

The completed two-model synthesis is frozen in
[`docs/cross_model_pilot_results.md`](docs/cross_model_pilot_results.md).

### Results at a glance

The primary outcome is **committed context adoption**: whether the model
explicitly commits to the answer asserted by conflicting evidence.

| Conflict type | Qwen preferred | Qwen dispreferred | Qwen delta | Llama preferred | Llama dispreferred | Llama delta |
|---|---:|---:|---:|---:|---:|---:|
| Harmful override (KC) | 20.0% | 10.0% | +10.0 pp | 33.3% | 36.7% | -3.3 pp |
| Corrective override (KW) | 83.3% | 56.7% | **+26.7 pp** | 96.7% | 100.0% | -3.3 pp |

For Qwen corrective conflicts, the paired source comparison was
17 both, 8 preferred-only, 0 dispreferred-only, and 5 neither
(exact two-sided `p = 0.0078125`).

The corresponding Llama corrective comparison was
29 both, 0 preferred-only, 1 dispreferred-only, and 0 neither
(exact two-sided `p = 1.0`).

The strongest Qwen result therefore **did not replicate in Llama**.

The current pilot evidence supports a **model-dependent boundary
condition**, not the universal claim that preferred source attribution
causes LLMs to follow conflicting evidence more often.

A secondary mechanistic analysis also distinguishes **tentative answer
content** from **committed adoption**. Both models can place the contextual
candidate in the required `Answer:` field without necessarily committing
to it, although this behavior is much more frequent in Qwen than in
Llama. The Qwen decomposition was post-hoc; the Llama follow-up was
pre-specified before the replication outcomes were observed.

### Reproducibility

Both research-model runs used:

- the exact PopQA revision
  `098765c79ea10a2cb19c828324e33281b8336ec0`;
- deterministic generation;
- exact pinned Hugging Face model revisions;
- unquantized float16 execution;
- an NVIDIA RTX 3090;
- recorded prompt/model/dataset provenance;
- independently SHA256-verified off-host final archives.

Model revisions:

- Qwen2.5-7B-Instruct:
  `a09a35458c702b33eeacc393d103063234e8bc28`
- Llama-3.1-8B-Instruct:
  `0e9e39f249a16976918f6564b8830bc894c89659`

The implementation includes the data pipeline, baseline screening,
candidate scoring, source calibration, pilot construction, deterministic
generation, evaluation, plotting, paired checks, sensitivity analyses,
and exploratory regression infrastructure.

The original experimental design and methodological decisions remain in
`docs/`; model outputs and large runtime artifacts are intentionally not
committed.

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

## Continuous integration

`.github/workflows/ci.yml` runs `pytest -q` and `ruff check .` on every
push and pull request to `main` (Python 3.10, matching this project's
supported baseline), and can also be triggered manually. It never
downloads model weights, runs the gated real-tokenizer tests, or uses
repository secrets — it verifies the existing software test suite only,
not the research pipeline itself. A separate, manually-triggered workflow
(`.github/workflows/tokenizer-integration.yml`) runs the gated real
Qwen2.5 tokenizer/config tests on demand; see `docs/decisions.md`, "CI:
separate tokenizer integration job." No status badge is included yet —
one will be added once an actual workflow run has succeeded.

## Reproducibility notes

- All stochastic steps (candidate subsampling, foil sampling, decoding)
  are seeded via `configs/pilot.yaml`.
- Every generation record stores model id, model revision, the exact
  rendered prompt, prompt version, and generation config. A real model
  run resolves the exact Hugging Face commit SHA for the configured
  `revision` (or "main") *before* loading, via one small Hub metadata
  request, and loads both the tokenizer and the model from that same
  pinned SHA — `model_revision` equals it whenever resolution succeeded.
  By default, if the SHA cannot be resolved (e.g. offline, or no
  credentials for a gated repo), model construction raises rather than
  silently proceeding with an unpinned snapshot — see `docs/decisions.md`,
  "Resolve, pin, load, record".
- Raw PopQA data is not committed (license on the dataset card is
  unstated); `data/README.md` documents exact recreation steps.
  `prepare-data` resolves the exact PopQA commit SHA before calling
  `datasets.load_dataset(..., revision=...)`, and `data/raw/manifest.json`
  records that same SHA as `resolved_revision`. If the SHA cannot be
  resolved, `prepare-data` fails clearly rather than downloading data it
  cannot exactly attribute — see `docs/decisions.md`, "Resolve, pin, load,
  record".
- Experiments are resumable: interrupting `run` and re-invoking it skips
  already-completed generations rather than duplicating them.
- Secrets: this project currently has no `.env` file and reads no
  credentials from the environment; `.gitignore` excludes `.env`/`.env.*`
  (keeping any future `.env.example` trackable) so this stays true if that
  changes later.

## Limitations

This remains a **pilot-scale two-model study**, not a publication-scale
benchmark.

Important limitations include:

- only two instruction-tuned models were tested;
- each model contributes only 60 selected factual items and 120 primary
  conflict trials;
- KC/KW membership and parametric margins are model-specific, so the final
  selected item sets differ between Qwen and Llama;
- the selected relation distributions are not balanced;
- source calibration was performed independently for each model, so the
  dispreferred labels differ (`anonymous online forum post` for Qwen,
  `social media post` for Llama);
- source preference is based on direct elicitation rather than a validated
  latent trust or credibility measure;
- evidence is standardized and prompt-injected rather than retrieved from
  naturally occurring documents;
- Llama corrective adoption is near ceiling under both source conditions,
  which limits sensitivity to source effects;
- the Qwen tentative-answer decomposition was post-hoc, while the Llama
  version was pre-specified as a secondary follow-up;
- the exploratory Llama ordinary logistic regression did not converge due
  to quasi-separation;
- the pilot does not use an item-aware mixed-effects model;
- results should not be generalized to LLMs as a class.

The main next step is a larger, pre-specified study with more model
families, a shared cross-model item set, balanced relations, a common
validated source hierarchy alongside model-specific calibration, and
item-aware statistical modeling.

See
[`docs/cross_model_pilot_results.md`](docs/cross_model_pilot_results.md)
for the complete result interpretation, hypothesis-by-hypothesis synthesis,
provenance, sensitivity analyses, mechanistic findings, and follow-up
design.

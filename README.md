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

**Phase 3 is complete and frozen.** A preregistered, four-model
confirmatory study has been run end to end: design frozen before any data,
cohorts built from an outcome-blind baseline screen, a sealed §36 pre-run
manifest, 4 197 evidence-condition generations, and only the analyses
declared in advance.

| Phase | What it was | Status |
| --- | --- | --- |
| 1 | literature synthesis | complete |
| 2 | two-model **pilot** (Qwen, Llama), 30 items/model | complete, frozen |
| 3 | four-model **confirmatory** study, preregistered | **complete, frozen** |

Phase 2 and Phase 3 are different studies and are never pooled. Phase 2 is
a pilot whose estimates are treated as very likely inflated; Phase 3 is the
confirmatory test on fresh items.

Models in Phase 3: `Qwen/Qwen2.5-7B-Instruct`,
`meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
`google/gemma-2-9b-it` — all at pinned immutable revisions, unquantized
float16.

### Headline result (Phase 3, confirmatory)

**Cohort A · Qwen · 96 fresh KW items · corrective conflict · frozen source
pair**, holding evidence content exactly constant and changing only the
attributed source:

| | |
| --- | ---: |
| adoption under `a government website` | **0.833** |
| adoption under `an anonymous online forum post` | **0.583** |
| paired cells (both / preferred-only / dispreferred-only / neither) | 56 / 24 / 0 / 16 |
| discordant pairs | 24 |
| **paired risk difference** | **+25.00 pp** |
| **95% Tango matched-pair CI** | **[+17.41, +34.51] pp** |
| **exact two-sided p** | **1.19 × 10⁻⁷** |

**Classification: FULL CONFIRMATORY REPLICATION** of the Phase 2 pilot
finding, on items the pilot never touched.

A common-source effect also appeared for Llama (+10.9 pp) and Mistral
(+17.5 pp) after Holm correction. Eight of the fifteen Holm-corrected
secondary tests survive correction.

### What the study does *not* claim

- **No architecture effect.** The four models differ in family, size,
  training data and instruction tuning simultaneously; this design cannot
  attribute anything to architecture.
- **No universal trust hierarchy.** Two of the four models produced no
  usable source-preference calibration at all.
- **Llama's Phase 3 corrective contrast is INCONCLUSIVE, not a null.**
  The chronology matters here, and the two phases say different things.

  *Phase 2 (pilot, 2026-08)* suggested **no replicated Llama source
  effect**: its corrective cells were 96.7% / 100.0%, and the pilot
  reported the Qwen finding as not replicating in Llama. Those documents
  are frozen historical record and keep that wording.

  *Phase 3 (confirmatory)* does not overturn that observation — it
  **classifies it**. Under the preregistered §30 rule, written before any
  Phase 3 data existed, a contrast is SATURATED / UNINFORMATIVE when
  either arm passes 0.95 (or falls below 0.05) adoption **and** there are
  fewer than five discordant pairs. Llama's Phase 3 corrective contrast is
  53/54 adopting under **both** sources — 0.981 each — with **zero**
  discordant pairs, so both conditions hold and the frozen classification
  is **INCONCLUSIVE due to saturation**.

  The distinction is not cosmetic: the design had almost no room for a
  source manipulation to act, so this is an inability to observe an
  effect, not evidence that none exists. It may not be counted as evidence
  against a source effect, and may not be pooled with non-saturated nulls.
- **Mistral and Gemma have no model-specific arm at all.** Their frozen
  calibration yielded no valid unique preferred/dispreferred pair, so under
  the frozen §34 rule they run the common arm only and their
  model-specific contrasts are NOT APPLICABLE — never measured, never
  pooled with nulls.

Full results: [`docs/phase3_final_report.md`](docs/phase3_final_report.md).
Reproduction: [`docs/phase3_reproducibility.md`](docs/phase3_reproducibility.md).

### Phase 2 pilot results (historical)

The pilot ran 5 conditions over 60 items per model — 600 generations, 240
primary conflict trials. Its Qwen corrective contrast was 25/30 vs 17/30,
Δ = +26.7 pp, exact p = 0.0078. Frozen in
[`docs/qwen_pilot_results.md`](docs/qwen_pilot_results.md) and
[`docs/cross_model_pilot_results.md`](docs/cross_model_pilot_results.md).

Phase 3 uses that +26.7 pp only as a **compatibility diagnostic** — it can
distinguish a full from an attenuated replication, and can never overturn
a Phase 3 estimate or test.

## Repository layout

    docs/                       research documents (read these first)
      phase3_scaled_study_design.md   the FROZEN preregistration
      phase3_final_report.md          Phase 3 confirmatory results
      phase3_reproducibility.md       end-to-end runbook + artifact digests
      decisions.md                    chronological methodology log
    configs/
      phase3/phase3_study.yaml        Phase 3 study config (sealed)
      phase3/freeze/                  the sealed §36 pre-run freeze record
      frozen/                         Phase 2 post-run provenance copies
      models.yaml, pilot.yaml, sources.yaml, prompts.yaml
    prompts/                    versioned prompt templates
    src/conflict_eval/          implementation package
      phase3/                   Phase 3 design, freeze, execution, analysis
    scripts/                    thin CLI entry-point wrappers
    tests/                      unit tests (no model downloads, no network)
    data/    results/   figures/    runtime artifacts, not committed
    runs/                       Phase 3 runtime output, not committed

See `docs/decisions.md` for why `scripts/` are thin wrappers around
`src/conflict_eval/cli.py` rather than a second implementation.

## Environment setup

    python -m venv .venv
    .venv\Scripts\pip install -e ".[dev]"   # Windows
    # or: .venv/bin/pip install -e ".[dev]"  # macOS/Linux

Torch is installed CPU-only by default via the plain PyPI index; install a
CUDA build separately if running the 7B/8B models on GPU.

## Commands

### Phase 3 (the confirmatory study)

Reproduce the analysis from the verified Phase 3D return — no GPU, no
network:

    python -m conflict_eval.phase3 verify-evidence-return --root <phase3d-return>
    python -m conflict_eval.phase3 gate --manifest configs/phase3/freeze/phase3c_pre_run_manifest.json
    python -m conflict_eval.phase3 analyze-3e --root <phase3d-return>

Regenerating the observations needs a 24 GB CUDA GPU:

    python -m conflict_eval.phase3 build-run-plan
    python -m conflict_eval.phase3 run-evidence --model qwen   # then llama, mistral, gemma

`run-evidence` refuses to start unless the real-run gate opens and the
config still hashes to what the sealed manifest recorded, and it verifies
every planned prompt against its frozen digest before generating anything.
Full runbook and artifact digests:
[`docs/phase3_reproducibility.md`](docs/phase3_reproducibility.md).

### Phase 2 (the frozen pilot pipeline)

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

Phase 3 is a preregistered four-model confirmatory study, but it is still
narrow in scope.

Important limitations include:

- only four instruction-tuned models, one dataset (PopQA at one pinned
  revision), and four relations (`country`, `sport`, `place of birth`,
  `mother`);
- **two of the four models contribute no model-specific evidence at all** —
  Mistral and Gemma are common-arm only, so that family rests on Qwen and
  Llama;
- **ceiling effects limit several cells**: Llama adopts corrective evidence
  near-universally, leaving the source manipulation almost no room to act,
  and five contrasts are INCONCLUSIVE by the frozen saturation or
  discordance rules;
- Cohort C is eligibility-limited (81 of 96) and three of eight Cohort B
  model × group cells fall out of the confirmatory families under the
  frozen §32 ladder;
- KC/KW membership and parametric margins are model-specific, so selected
  item sets differ between models;
- source labels are **textual attributions, not real provenance**: evidence
  is synthetic and template-rendered, with no retrieval and no real
  documents;
- source preference rests on direct elicitation, not a validated latent
  trust or credibility measure;
- a single decoding configuration and one prompt template version;
  sensitivity to phrasing is untested;
- no human baseline, and no measure of whether these source preferences are
  normatively appropriate;
- the Phase 2 comparator is a 30-item pilot estimate, used only as a
  diagnostic, and is very likely inflated;
- results should not be generalized to LLMs as a class.

The main next steps are broader model coverage with usable
source calibration for every model, designs that avoid ceiling regimes for
corrective conflict, real retrieved documents rather than templated
evidence, and item-aware mixed-effects modeling.

See [`docs/phase3_final_report.md`](docs/phase3_final_report.md) for the
complete Phase 3 result interpretation, and
[`docs/cross_model_pilot_results.md`](docs/cross_model_pilot_results.md)
for the frozen Phase 2 pilot synthesis.

## License and third-party materials

This repository's **own** code and documentation are released under the
**MIT License** — see [`LICENSE`](LICENSE).

That covers what was written here: the `conflict_eval` package, the tests,
the configuration and prompt templates, the research documents under
`docs/`, and the frozen Phase 3 provenance records under
`configs/phase3/freeze/`.

**It does not cover anything external, and nothing here relicenses any
third-party material.** In particular:

- **Model weights.** No weights are contained in or distributed by this
  repository. `Qwen/Qwen2.5-7B-Instruct`,
  `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`
  and `google/gemma-2-9b-it` are each downloaded by the user from Hugging
  Face and remain governed by their own model-card licenses and
  acceptable-use terms. The Llama and Gemma repositories are **gated**: you
  must accept their terms before the pipeline can load them.
- **PopQA.** No dataset rows are committed. The dataset card for
  `akariasai/PopQA` states no explicit license, which is precisely why the
  raw and interim data are gitignored rather than redistributed here;
  `data/README.md` documents exact recreation from the pinned revision
  instead. Cite the original PopQA authors, not this repository, for the
  data.
- **Dependencies.** PyTorch, Transformers, Datasets, scikit-learn,
  statsmodels, SciPy, pandas, NumPy, Matplotlib and the rest retain their
  own licenses.
- **Experimental outputs.** The generations analysed here are model
  outputs, and their use may be constrained by the terms of the model that
  produced them.

Using this code to reproduce the study means obtaining those external
resources yourself, under their terms.

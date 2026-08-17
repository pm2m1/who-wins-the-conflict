# Decisions Log

Methodological decisions and their rationale. New entries are appended, not
inserted retroactively, so this file also serves as a chronological record
of how the design changed as implementation proceeded.

## Why no full RAG retriever

The pilot's target variable is evidence *integration* (does the model use
supplied evidence faithfully, and does that depend on parametric strength
and source identity), not evidence *retrieval*. A real retriever
(embeddings, FAISS, reranking) would confound integration with retrieval
quality, chunking, and recall. Evidence is instead placed directly and
deterministically in the prompt using a fixed template
(`prompts/evidence.txt`), so the only manipulated variables are asserted
content and attributed source. Real retrieval is recorded as future
ecological validation (see `docs/reference_implementations.md`,
WildHallucinations entry).

## Why PopQA

PopQA provides short, closed-form factual questions with an explicit
subject/relation/object structure and alias lists, which is what a
deterministic exact/alias matcher needs. Its `prop` (relation) field also
supports a defensible same-relation foil-sampling control (see foils
section, `docs/phase2_research_design.md`). Long-form datasets
(WildHallucinations) are unsuitable for the pilot's deterministic
evaluation approach and are deferred to future ecological validation.

## Why deterministic evaluation, not LLM-as-judge

An LLM judge introduces its own confounds (potential source-preference or
parametric-preference effects in the judge itself) and its own hallucination
risk, which is exactly what this project studies. Deterministic
normalization plus exact/alias matching is simpler, fully reproducible, and
adequate for PopQA's short-answer format. Token F1 and judge-based
evaluation are retained as possible future diagnostics, not primary
metrics. See WildBench in `docs/reference_implementations.md` for why its
LLM-judge approach is future-only.

## Why no chain-of-thought

Requesting rationale or explanation would let the model narrate a
justification for its answer that may not reflect the actual mechanism
producing it, and would substantially complicate deterministic parsing of
the answer/decision/confidence fields. The prompt requests a fixed
three-field format only.

## Why source calibration is independent of the test set

Declaring a source "preferred" or "dispreferred" without measurement would
be an assumption dressed as a finding. Direct pairwise calibration
(`configs/sources.yaml`, `prompts/source_calibration.txt`) measures actual
per-model source preference before the conflict experiment uses it, with
A/B and B/A counterbalancing to separate position preference from source
preference.

## Why knowledge groups are model-specific

Different models have different parametric knowledge. Assuming a KC/KW
label transfers across models would silently inject one model's knowledge
state into another model's trial construction. Groups, margins, and margin
bins are computed and stored per `(model_id, item_id)`.

## Why conflict trials are primary

The research questions concern what happens when evidence and memory
disagree. Agreement trials (C1/C2 for KC, C3/C4 for KW) cannot identify
whether context or memory caused the observed answer, since both point to
the same answer. They are retained as controls (e.g., to check baseline
condition-following behavior) but are not used to compute CAR, HOR, COR, or
the source-effect deltas.

## Why the parametric margin is conflict-specific

An earlier, simpler design considered one universal gold-vs-foil margin for
every trial. That would conflate two different comparisons: for KC items,
resistance of the gold answer against a false foil; for KW items,
resistance of a wrong baseline answer against the gold answer. These are
not the same quantity and averaging them would obscure both RQ1 and RQ3.
The margin is therefore defined per trial type against the actual
memory-vs-conflicting-context pair relevant to that trial (see
`docs/phase2_research_design.md`, H1).

## Why source effects must not be inferred from agreement trials

If context and memory agree, adopting the contextual answer is
indistinguishable from simply retaining the memory answer. Any apparent
"source effect" measured on an agreement trial would be an artifact of
this non-identifiability, not a real effect of source attribution. Source
effects (`Delta_harm`, `Delta_correct`) are therefore computed only from
conflict trials.

## Why hidden-state probes are postponed

The reference implementation for internal factuality signals
(`bnewm0609/epistemic-training`) documents a probe-training workflow
requiring substantially more compute (a custom vLLM fork, ~150GB RAM) than
this pilot has budget or need for. The pilot uses a behavioral proxy
(length-normalized sequence log-probability margin) instead, and records
hidden-state probing as future work.

## Repository license — pending researcher decision

No `LICENSE` file has been added to this repository. This is a deliberate
placeholder, not an oversight: the choice of license (if any) is a
researcher decision that depends on eventual publication plans and is left
pending. Do not assume MIT/Apache/other terms apply to this repository's
own code until a `LICENSE` file is added.

## scripts/ are thin wrappers around src/conflict_eval/cli.py

The repository layout calls for both a `python -m conflict_eval <command>`
CLI and standalone scripts under `scripts/`. Rather than implementing the
same orchestration logic twice (data loading, model construction, trial
generation) in two places where it could silently drift out of sync, each
`scripts/*.py` file is a short argparse wrapper that calls the
corresponding `cmd_*` function in `src/conflict_eval/cli.py`, which holds
the actual implementation. This is a material simplification relative to
treating `scripts/` as an independent implementation, recorded here per
the project's own instruction to log such simplifications.

## Scoring prefix must include the Answer: field label

Found during real-model-adapter validation (auditing `models/hf_causal.py`
and `scoring/sequence_logprob.py` before a real Qwen2.5-7B-Instruct run).

The experimental prompt (`prompts/baseline.txt`) instructs the model to
respond in a fixed three-field format: `Answer: <short answer>\nDecision:
...\nConfidence: ...`. The original `score_candidate` implementation
scored a candidate answer (e.g. `"Paris"`) as the continuation
immediately following the bare chat-template assistant-turn marker — i.e.
it measured the probability that the literal first tokens of the
assistant's entire turn are the candidate text, with no `"Answer: "`
label at all. That is a different, and for an instruction-following model
far less meaningful, quantity than what the parametric margin is supposed
to capture: how strongly the model prefers a given value *for the Answer
field*, given that it is instructed to (and, per `generate()`, actually
does) produce that field label first.

Fix: `BaseModelAdapter.score_candidate` and the new
`score_candidate_detailed` both gained an `answer_prefix: str = ""`
parameter. Callers that score candidates against the real experimental
prompt format pass `answer_prefix=ANSWER_FIELD_PREFIX` (`"Answer: "`,
defined in `experiment/prompts.py`), which is appended to the
chat-template prefix before the candidate is tokenized and scored
(`cli.py:cmd_screen`, the only current caller of `score_candidate`).
`DummyModelAdapter` also accepts the parameter, so the interface stays
uniform across adapters and the parameter is exercised by the dummy-based
test suite even without a real model.

This was confirmed against the real Qwen2.5-7B-Instruct tokenizer (not
the model weights — see the compute-constraints entry below): appending
`answer_prefix` measurably lengthens the real tokenized prefix, and
answer-token boundary detection still exactly recovers the candidate text
on both sides of the change (`tests/test_hf_tokenizer_integration.py`,
gated behind `CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1`). `docs/methodology.md`
section 3 has been corrected to describe the scoring prefix accurately.

No baseline screening had been run against a real model before this fix
(only the dummy-adapter dry run, which does not exercise `hf_causal.py`
at all), so no real result files needed to be invalidated or rerun.

## Real-model validation blocked by insufficient RAM (compute-constraints finding)

Attempted real-model validation of `HFCausalAdapter` against
`Qwen/Qwen2.5-7B-Instruct` (the preferred model per project
configuration). The environment has no CUDA GPU (`torch.cuda.is_available()
== False`; the installed `torch` build is CPU-only) and 16GB total system
RAM, of which only about 3GB was free at the time of inspection (checked
via `Get-CimInstance Win32_OperatingSystem`). Qwen2.5-7B-Instruct's
weights alone are approximately 15GB in bfloat16 (the dtype configured in
`configs/models.yaml`); safe CPU inference typically needs free RAM well
in excess of the raw weight size once activation memory and framework
overhead are included. Loading the full model here would very likely
either fail with an out-of-memory error or force heavy paging that could
make the machine unresponsive.

Per this project's own instruction to stop rather than proceed when
compute is inadequate, the model weights were not downloaded and the
model was not loaded or run. Only the model's tokenizer/config files
(~11MB — vocab, merges, tokenizer_config, chat_template; no `.safetensors`
weights) were downloaded, to validate chat-template rendering and
answer-token boundary/masking logic against the real tokenizer
independently of the compute-heavy forward pass — see
`tests/test_hf_tokenizer_integration.py`.

**Update:** this local environment still has no GPU and insufficient RAM
for CPU inference of a ~15GB bf16 model, and that has not changed. Real
logit/log-probability validation was instead performed on separate
GPU-equipped compute (Google Colab), using a smaller model as an
infrastructure-validation stand-in rather than waiting on this
environment's hardware — see "Real-model infrastructure validation on
Google Colab" below for what was actually run and confirmed.

## PopQA row count discrepancy

`docs/reference_implementations.md` records 14,300 test-split rows based
on the Hugging Face dataset card at inspection time. Actually downloading
`akariasai/PopQA` (test split) during implementation validation yielded
14,267 rows. The difference is small (33 rows) and most likely reflects a
minor dataset-card/artifact version difference rather than a wrong
identifier — the fields, split name, and general size all match. Recorded
here rather than silently editing the earlier figure, per this project's
own rule against silently altering recorded observations. `data/raw/manifest.json`
(written by `prepare-data`) captures the actual row count for whichever
revision is downloaded at pilot time, which is the authoritative figure
for any real run.

## Directory naming note

The working directory name contains a literal space
(`who-wins-the -conflict`) rather than the originally intended
`who-wins-the-conflict`. This was already the state of the directory
before implementation began and was not renamed, since renaming the
working directory is outside the scope of writing project files and could
disrupt any external references the researcher already has to this path.
The Python package name (`conflict_eval`) and repository documentation are
unaffected by this.

## Local secret-file protection

`.gitignore` did not previously mention `.env` files explicitly. Even
though the project currently reads no environment variables for
credentials (confirmed by inspection — no `os.environ`/`os.getenv` calls
exist in `src/`), a future need (e.g. `HF_TOKEN` for gated Llama access)
should never be one accidental `git add .` away from being committed.
Added `.env`, `.env.*`, and a `!.env.example` exception (so a future
template file documenting expected variables would remain trackable) to
`.gitignore`. No `.env` or `.env.example` file exists in the repository;
none was created, per the instruction not to invent one without an actual
documented need. Verified with `git check-ignore` (`tests/test_gitignore_secrets.py`),
not by asserting on `.gitignore`'s text content, so the test reflects
git's actual matching behavior rather than a specific line's wording.

## Exact model revision recording

`HFCausalAdapter` previously stored only the `revision` argument passed
in as `model_revision`, which stayed `None` whenever `configs/models.yaml`
left `revision: null` (the default for both Llama and Qwen) — meaning
real result records could not identify which concrete model snapshot was
actually loaded, undermining reproducibility for any run that did not
pin an explicit revision.

Fix: after loading, `HFCausalAdapter` now reads
`self.model.config._commit_hash`, which `transformers` populates with the
exact resolved commit SHA extracted from the local Hugging Face Hub cache
path (`.../snapshots/<sha>/...`) during the `from_pretrained` calls that
already happened — this adds no extra network call or download beyond the
load itself. Three attributes are now distinguished:

- `requested_revision` — the `revision` value from configuration, possibly
  `None`.
- `resolved_revision` — the concrete commit SHA if one could be
  determined, otherwise `None`. Never fabricated: if `config._commit_hash`
  is missing or empty (e.g. loading from a plain local directory, or an
  older/newer transformers version that does not expose the attribute),
  this stays `None`.
- `model_revision` — the field already used throughout result records and
  `docs/methodology.md`; unchanged in name and meaning ("the revision to
  cite for this run"), but now prefers `resolved_revision` and only falls
  back to `requested_revision` when no concrete SHA is available.

`DummyModelAdapter` gained matching `requested_revision`/`resolved_revision`
attributes, both always `None` (no real model is loaded, so there is
nothing to resolve), for interface consistency with `BaseModelAdapter`.
`cmd_screen`'s baseline records gained an additional `requested_revision`
field alongside the existing `model_revision` field, and `diagnose-score`'s
printed output now shows both values — both additive changes, not a
rename of any existing field. Validated against the real
`Qwen/Qwen2.5-7B-Instruct` config (config.json only, not the model
weights — `tests/test_hf_tokenizer_integration.py::test_config_exposes_resolved_commit_hash`,
opt-in via `CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1`) and against mocked
`transformers` internals for the fallback/unavailable paths
(`tests/test_hf_causal_revision.py`), so no 7B/8B weights were downloaded
to validate this.

## Exact PopQA dataset revision recording

`data/raw/manifest.json` previously recorded `hf_dataset_id`, `split`,
`num_rows`, and `fields`, but no exact dataset revision — its docstring
claimed otherwise, which was inaccurate and has been corrected.

Fix: `conflict_eval.data.popqa.resolve_dataset_revision` scans the local
Hugging Face Hub cache via `huggingface_hub.scan_cache_dir()` (a public,
documented API) for the dataset repo, and returns the commit SHA of
whichever cached revision's refs include `"main"`, falling back to the
most recently modified cached revision if no `"main"` ref is present. This
is purely local — it adds no network call beyond whatever
`datasets.load_dataset` already made — verified against this
environment's real, already-cached `akariasai/PopQA` snapshot during
implementation (interactively, matching the `refs/main` file's contents
byte for byte), and covered by mocked-cache unit tests
(`tests/test_dataset_revision.py`) for the CI-reproducible path. The
result is stored in the manifest as `resolved_revision`; it is `None`,
never guessed, when the local cache cannot be scanned or the repo is not
found in it. `huggingface_hub` is not a new direct dependency —
it is already installed transitively via `transformers`/`datasets`, so
`pyproject.toml` was not changed.

Manifest construction was factored out into a small pure function,
`build_manifest`, purely so this schema (in particular, that
`resolved_revision` is always present, even as `None`) is unit-testable
without a real dataset download; `download_raw`'s behavior is otherwise
unchanged.

## Resolve, pin, load, record (not load, then infer)

The previous two entries recorded a resolved revision, but both did so by
*inferring* it after the fact — from `transformers`' post-load
`config._commit_hash` (model side) or by scanning the local Hugging Face
cache for whatever happened to be there (dataset side). That is weaker
than it looks: the revision that ends up recorded is not necessarily the
revision that was *deliberately selected* before loading, and the
dataset-side cache scan in particular could, in principle, report an
unrelated cached revision as if it were proven to be the one the current
call used.

For real research runs, this project now enforces the stronger
invariant:

    resolve concrete Hub commit SHA
        -> load using that exact SHA
        -> record that exact same SHA

**Model side** (`models/hf_causal.py`): `resolve_model_revision(hf_model_id,
requested_revision)` calls `huggingface_hub.HfApi().model_info(...)` —
one small metadata request, no weight download — *before* constructing
the tokenizer or model, and returns the exact commit SHA (or `None` if
Hub access fails; never guessed). `HFCausalAdapter` passes that same
resolved SHA as `revision=` to *both* `AutoTokenizer.from_pretrained` and
`AutoModelForCausalLM.from_pretrained`, guaranteeing they load from the
identical snapshot. By default (`require_pinned_revision=True`),
construction raises `ModelRevisionResolutionError` if resolution fails,
rather than silently loading an unpinned snapshot — a real experimental
run must fail clearly here, not produce results attributable to an
unknown model version. `config._commit_hash` is still read after
loading, but now purely as a **consistency check**: if it disagrees with
the SHA that was deliberately requested, construction raises rather than
continuing with a snapshot that cannot be verified. An explicit
`require_pinned_revision=False` opt-out is available and documented for
non-strict use (e.g. offline exploratory work); it cannot claim exact
revision pinning, and falls back to the old post-load inference (then to
the bare requested revision string) precisely because it is not claiming
the stronger guarantee.

**Dataset side** (`data/popqa.py`): `resolve_dataset_revision(hf_dataset_id,
requested_revision="main")` now calls `huggingface_hub.HfApi().dataset_info(...)`
(one metadata request) instead of scanning the local cache post-download.
This replaces the earlier cache-scanning implementation entirely — it did
not distinguish "the revision this call actually used" from "whatever
happens to be cached," which the earlier entry's own "most recently
modified" fallback heuristic made concrete. `download_raw` now resolves
the SHA first and passes it explicitly to
`datasets.load_dataset(..., revision=resolved_sha)`; if resolution fails,
it raises `DatasetRevisionResolutionError` before attempting any load,
with no opt-out (unlike the model side, there is no non-strict variant of
a real PopQA download in this project). The manifest's `resolved_revision`
field is unchanged in name and meaning, but now reflects the SHA that was
actually used to load the data, not one inferred afterward.

Both resolution functions were validated against real Hugging Face Hub
metadata during implementation (`Qwen/Qwen2.5-7B-Instruct` and
`akariasai/PopQA`, `revision="main"` — a metadata-only request, no
weights or dataset files downloaded); the resolved SHAs matched the
values previously observed via post-load/cache-based inference exactly,
which is expected since "main" had not moved. Ordinary test coverage
uses mocked `huggingface_hub`/`transformers`/`datasets` throughout
(`tests/test_hf_causal_revision.py`, `tests/test_dataset_revision.py`) —
no test downloads model weights or the dataset.

## CI: separate tokenizer integration job

GitHub Actions CI (`.github/workflows/ci.yml`) runs the deterministic
unit test suite and `ruff check .` on every push/PR to `main`, plus
`workflow_dispatch`. It deliberately does not set
`CONFLICT_EVAL_RUN_TOKENIZER_TESTS`, so the gated real-tokenizer tests in
`tests/test_hf_tokenizer_integration.py` stay skipped there, matching how
they already behave in a plain local `pytest -q`.

A second, separate workflow (`.github/workflows/tokenizer-integration.yml`)
runs only those gated tests, with `CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1`
set explicitly. It is `workflow_dispatch`-only (never triggered by
push/PR), so it never adds an unnecessary Hugging Face Hub network call
to ordinary CI runs. Kept as an entirely separate workflow file (not an
extra job inside `ci.yml`) so it has its own distinct name and run history
in the Actions UI — a failure there reads unambiguously as "Hugging Face
Hub was unreachable, or its tokenizer/chat-template changed" rather than
being interleavable with, or mistaken for, a deterministic code
regression in the main CI run. It downloads only the
Qwen/Qwen2.5-7B-Instruct tokenizer/config files (~11MB), never model
weights, and both workflows request only `permissions: contents: read`
and use no repository secrets.

This is judged worth the extra file: it is fully isolated from the
required CI signal (a PR can merge on `ci.yml` alone), costs nothing
unless someone deliberately runs it, and gives a way to periodically
re-confirm the chat-template/tokenizer assumptions this project's scoring
code depends on still hold against the real Hugging Face repo, without
coupling that check to every ordinary push.

## Baseline abstentions must not become KC/KW memory candidates

The first real-model baseline screen — 20 PopQA items,
`Qwen/Qwen2.5-3B-Instruct`, run as infrastructure validation, not a
research result — surfaced a genuine methodological bug: of 20 baseline
records (KC=1, KW=19, manual_review=0), 18 of the 19 "KW" records had
`parsed_answer = "uncertain"` as the supposed memory answer (e.g.
`memory_answer = "uncertain"`, `conflicting_context_answer = "jazz"`).

The model was abstaining far more often than expected on this small
sample, and the prior screening logic had no check for that: it only
required (a) the baseline answer not match gold, and (b) the answer text
be short with no comma/"or" — "uncertain" trivially satisfies both. The
resulting "parametric margin," `score("uncertain") - score(gold)`, is not
the quantity `docs/phase2_research_design.md` (H1) defines: a KW margin
is supposed to measure resistance of the model's *wrong factual guess*
against the correct answer, and an abstention is not a wrong factual
guess — it is the model declining to guess at all. Treating it as one
would have silently fed a meaningless quantity into RQ1/H1's primary
analysis had this not been caught before the real pilot.

Corrective rule, implemented in the new
`evaluation/baseline_eligibility.py` (`classify_baseline_eligibility`,
`is_clean_factual_candidate`) and applied in `cli.py:cmd_screen` before
any KC/KW assignment: a baseline response is only eligible for KC or KW
if `parsed.decision == "answer"` **and** its answer text, after the same
normalization used for gold matching, is not an explicit uncertainty
marker (`"uncertain"`, `"unknown"`, `"i don't know"`, `"i do not know"`,
`"cannot determine"`, `"can't determine"`) — this second check catches a
model that emits `Decision: answer` inconsistently alongside an
abstention-shaped answer, which is exactly what made 18/19 records slip
through the old check. This eligibility check runs before the
gold-match/KC branch too, not only before the KW-cleanliness branch: a
response matching gold but carrying `Decision: uncertain` must not become
KC either, since KC and KW both represent usable parametric answers, not
abstentions.

Ineligible-but-syntactically-valid responses are recorded with
`knowledge_group = "excluded"` and `exclusion_reason =
"baseline_uncertain"` — kept in the baseline record stream (not the
separate malformed-response exclusions stream) for auditability, since
the response itself parsed successfully; they receive no
`memory_answer`/`conflicting_context_answer`/`parametric_margin`/
`margin_bin` at all, and margin-bin computation (which iterates only
`knowledge_group in {"KC", "KW"}`) was already structurally unable to
include them. `parsed_decision` and `parsed_confidence` are now also
stored on every baseline record (additive; `parsed_answer` and
`raw_generation` are unchanged), so the full parsed response — not just
the derived classification — is auditable after the fact.

This is a small, explicit marker list, not an attempt to catch every
possible hedge or refusal phrasing (e.g. "I'm not sure" is not on it) —
consistent with this project's general preference for narrow, defensible,
documented heuristics over broad automatic judgment. An answer that is
not one of these explicit markers but still fails the shape-based
clean-candidate check (too long, contains a comma, contains " or ") is
routed to `manual_review` rather than forced into KW.

No prior research results were invalidated by this fix: the 20-item run
that exposed it was itself explicitly infrastructure validation, run
before baseline screening had produced any data used in, or intended for,
the real pilot, source calibration, or C0-C4. Nothing needed to be rerun
for research purposes — only the screening code needed correcting before
the real pilot uses it.

## Real-model infrastructure validation on Google Colab

This is a record of **infrastructure validation**, not a pilot run, not
an experiment result, and not a research finding. `Qwen/Qwen2.5-3B-Instruct`
was used strictly as a stand-in to validate the model-adapter and
scoring code paths on real GPU hardware; it is not, and does not become,
a research/pilot model for this project. The intended research model
remains `Qwen/Qwen2.5-7B-Instruct` (per `configs/models.yaml`), which has
not been run, alongside `meta-llama/Llama-3.1-8B-Instruct`.

**Environment:** Google Colab, NVIDIA Tesla T4 GPU. Model loaded in
`float16` on CUDA (this local repository's own `configs/models.yaml`
still specifies `bfloat16`; the Colab session used `float16`, a deliberate
adaptation to the T4, which lacks native bfloat16 tensor-core support —
not a change to the checked-in config).

**Model revision pinning (`docs/decisions.md`, "Resolve, pin, load,
record"):** exact resolution succeeded before loading. Observed pinned
SHA: `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Real model weights loaded
successfully under that pinned revision.

**Teacher-forced scoring, empirically checked against real model logits:**

- deterministic generation
- `Answer: ` scoring-prefix alignment (docs/decisions.md, "Scoring prefix
  must include the Answer: field label")
- candidate token boundary handling
- A/B versus B/A score-order invariance (scoring one candidate does not
  affect the independently-computed score of the other)
- repeated-score determinism (scoring the same candidate twice under the
  same prompt gives identical results)
- multi-token candidate scoring
- length normalization
- punctuation/token-boundary reconstruction

Example diagnostic (via `diagnose-score`): for "What is the capital of
France?", "Paris" received a much higher normalized sequence
log-probability than "London", and the model's own greedy generation
also produced "Paris" — the scoring and generation pathways agree, on a
real model, for at least this case.

**PopQA preparation, real and pinned (`docs/decisions.md`, "Resolve, pin,
load, record"):** resolved dataset SHA
`098765c79ea10a2cb19c828324e33281b8336ec0` (the same commit observed
earlier during local `HfApi().dataset_info()` metadata-only validation —
consistent, since "main" had not moved). The pinned test snapshot
contained 14,267 rows, matching what `docs/methodology.md` §1 already
documents as the observed count for this pinned revision (distinct from
the Hugging Face dataset card's stated 14,300).

**20-item baseline smoke screen, run twice, before and after the
abstention fix:** deterministic, `Qwen/Qwen2.5-3B-Instruct`, same 20 item
IDs, same model revision, same raw generations in both runs (confirmed,
not assumed) — only the screening *code* differed:

| | KC | KW | excluded | 
|---|---|---|---|
| before commit `72928b7` | 1 | 19 | 0 |
| after commit `72928b7` | 1 | 0 | 19 (all `exclusion_reason = baseline_uncertain`) |

No excluded record contained `memory_answer`, `conflicting_context_answer`,
`memory_logprob_normalized`, `conflicting_answer_logprob_normalized`,
`parametric_margin`, or `margin_bin` — confirming, on a real model and
real generations (not mocked/scripted, unlike `tests/test_cmd_screen_eligibility.py`),
that the fix in commit `72928b7fd624394526e7b4ba3c1cd22439d30f2a` behaves
as intended: the 19 abstentions that previously became meaningless "KW"
memory candidates are now correctly excluded and margin-free.

**What remains unrun:** `Qwen/Qwen2.5-7B-Instruct` (the intended research
model), `Llama-3.1-8B-Instruct`, source calibration, and the C0-C4
experimental conditions. No pilot results or scientific conclusions exist
yet — this entry validates infrastructure only.

## Decision output format made strict

The actual intended research model, `Qwen/Qwen2.5-7B-Instruct`, was run
unquantized in FP16 on a free Colab T4 (GPU+CPU offload) — still
infrastructure validation, not a pilot run. A real-model diagnostic
exposed a prompt/parser ambiguity that had to be fixed before any 7B
screening or research data could be produced.

The committed prompt (`prompts/baseline.txt`) previously requested:

    Decision: answer | uncertain

with the pipe intended to mean "choose one." The real Qwen2.5-7B-Instruct
model instead partially reproduced the pipe syntax literally, generating:

    Answer: Paris
    Decision: answer | certain
    Confidence: 100

The old Decision parser, `Decision:\s*(answer|uncertain)` with no line
anchoring, matches anywhere in the text and accepts a valid prefix even
if trailing junk follows — so `"Decision: answer | certain"`,
`"Decision: answer | uncertain"`, and `"Decision: answer blah"` were all
silently parsed as `decision = "answer"`. This matters directly:
`parsed.decision` controls baseline KC/KW eligibility
(`evaluation/baseline_eligibility.py`), so a response the model did not
actually commit to unambiguously could still have become a KC/KW memory
candidate.

Fix, in two parts:

1. The prompt now reads:

       Answer: <short answer>
       Decision: <answer or uncertain>
       Confidence: <integer from 0 to 100>

       For Decision, write exactly one word: answer or uncertain.
       Do not write both choices and do not use the | symbol.

   (`prompts/baseline.txt`; the canonical prompt quoted in
   `docs/phase2_research_design.md`, "Prompt design," was updated to
   match exactly, and the same wording was mirrored into synthetic
   prompt fixtures in `tests/test_dry_run_pipeline.py` and
   `tests/test_hf_tokenizer_integration.py` for consistency — neither of
   those tests parses its own fixture text, so this is a documentation
   consistency fix, not a behavior fix, for those two files.)

2. `evaluation/parse.py`'s Decision regex is now line-anchored and exact:
   `^Decision:\s*(answer|uncertain)\s*$`, case-insensitive and
   multiline, so `^`/`$` bind to individual lines within the full
   response rather than the whole string. A Decision line is accepted
   only if, after "Decision:" and surrounding whitespace, the entire
   rest of that line is exactly "answer" or "uncertain" (any case).
   `"Decision: answer | certain"`, `"Decision: answer | uncertain"`,
   `"Decision: answer blah"`, `"Decision: uncertain because ..."`, and
   `"Decision: answer/uncertain"` are all now rejected: `parsed.decision
   = None` and `parsed.malformed = True`, which — via the existing
   malformed-response handling in `cli.py:cmd_screen` — routes the item
   to the syntactic-malformed exclusions stream, never KC/KW.

A Colab-only strict-prompt test on the same loaded, already-pinned 7B
model (revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, from "Real-model
infrastructure validation on Google Colab" above), using the wording now
committed here, produced exactly:

    Answer: Paris
    Decision: answer
    Confidence: 100

which the current parser accepts as `answer='Paris'`, `decision='answer'`,
`confidence=100`, `malformed=False` — confirming the new prompt reliably
elicits the strict format from the real research model, not just from the
regex's own test suite.

This was discovered before any 7B PopQA screening or research results
were produced, so no research results were invalidated. It compounds with
the earlier baseline-abstention fix (`72928b7fd624394526e7b4ba3c1cd22439d30f2a`):
that fix ensures an *unambiguous* abstention cannot become KC/KW; this fix
ensures the model cannot produce an *ambiguous* Decision value that the
parser would have silently resolved one way or the other.

## Real Qwen2.5-7B-Instruct feasibility validation on Google Colab

This is a record of **real-model infrastructure/feasibility validation**,
not a pilot result and not a scientific finding. The France diagnostic
below is a sanity check on the scoring pipeline, not evidence bearing on
RQ1/RQ2/RQ3.

The actual intended research model, `Qwen/Qwen2.5-7B-Instruct`, was run
**unquantized**, in float16, on a free Google Colab NVIDIA Tesla T4, using
`device_map="auto"` with an explicit Accelerate memory cap
(`max_memory={0: "12.0GiB", "cpu": "5GiB"}`) to force GPU+CPU offload —
the free T4's ~15GB nominal VRAM is not reliably fully available in
practice, so an unconstrained `device_map="auto"` load is not guaranteed
to succeed. Exact revision resolution (`docs/decisions.md`, "Resolve,
pin, load, record") succeeded before loading. Observed pinned SHA:
`a09a35458c702b33eeacc393d103063234e8bc28`.

**Observed placement:** 23 modules on GPU, 9 on CPU, 0 on disk.
CPU-offloaded modules: `model.layers.22` through `model.layers.27`,
`model.norm`, `model.rotary_emb`, `lm_head`.

**Real diagnostic** ("What is the capital of France?"): generation
succeeded —

    Answer: Paris
    Decision: answer
    Confidence: 100

— and teacher-forced scoring gave Paris a normalized log-probability of
`-0.00019059749320149422` against London's `-21.82831573486328` (margin
`+21.82812513737008`): generation time ~20.68s, each candidate score
~1.03s/~1.05s, total ~22.75s. Post-inference: ~11.78GiB GPU used,
~2.78GiB GPU free, ~3.69GiB CPU RAM available. This demonstrates the
feasibility of small unquantized 7B runs on a free T4 through CPU
offload — it does not, by itself, establish anything about the research
questions.

The strict Decision prompt/parser (`ffa67c517738b8ccb1521b9ae6fc39e3300f8e82`,
"Decision output format made strict") was subsequently re-validated
against this same real loaded Qwen2.5-7B-Instruct model, under the
committed prompt: the generation above (`Decision: answer`, no pipe
syntax) parses as `parsed.decision == "answer"`, `malformed == False`.

**What this validates:** that the committed model-loading, revision-pinning,
teacher-forced-scoring, and strict-parsing code paths work end to end
against the real 7B research model on realistically constrained free-tier
hardware. **What this does not validate:** no 7B PopQA baseline screening
has been run, no source calibration has been run, no C0-C4 conditions have
been run, and no scientific conclusion should be drawn from the France
diagnostic or any other single-question sanity check above.

**Reproducibility gap this closed:** the Colab session used a temporary,
uncommitted, direct `transformers` load with `max_memory` hardcoded
inline — the committed `HFCausalAdapter`/`ModelSpec`/CLI path had no way
to express that placement. `ModelSpec` gained an optional `max_memory`
field (validated, passed through unchanged — never reinterpreted — to
`AutoModelForCausalLM.from_pretrained`), and `configs/models.yaml` keeps
`max_memory: null` for both `llama` and `qwen`: the free T4's specific
12GiB/5GiB split is a property of that runtime environment, not a
research default, and belongs in a separate, machine-specific scratch
config that overrides these values for an actual Colab run, not in the
primary committed configuration. No result-record schema changed in this
task; whether a run manifest should persist hardware placement metadata
is left as a separate, later decision.

## Restrict primary trials to defensible conflicts (2026-08-17)

The first real Qwen/Qwen2.5-7B-Instruct 20-item PopQA baseline screen
completed successfully through the committed CLI and exposed a
methodological issue in conflict construction, distinct from the earlier
abstention (`72928b7`) and Decision-format (`ffa67c5`) fixes: **different
answer != validated semantic conflict**. KC/KW eligibility alone does not
establish that a KC item's foil, or a KW item's gold answer, is in actual
semantic contradiction with the model's memory answer.

Six examples from that screen (full detail in
`docs/phase2_research_design.md`, "Primary conflict trial eligibility"):
`sport`/St. Louis Blues (ice hockey vs. handball) and `country`/Brown
University (USA vs. Tunisia) are defensible conflicts; `genre` (drama vs.
erotica) is ambiguous, since genres are not mutually exclusive;
`religion` (Christianity vs. Baptists) is hierarchical/compatible, not a
contradiction; two `screenwriter` examples showed the relation can
legitimately have multiple correct values, including a multi-name answer
("Eric Paul Friedmann and Christophe Beck") that the old KW-cleanliness
check did not catch.

A full census of the pinned PopQA snapshot (14,267 rows, 16 relations)
was inspected for per-subject duplicate-object rates across all 16
relations. That census **motivated** this review but is deliberately
**not** the classification rule itself — a relation's semantic type
decides its category, not its observed duplicate rate (e.g. `occupation`
has a low observed duplicate rate in this snapshot, ~0.19%, but is
excluded anyway because it is conceptually multi-label in general, and
`capital of` has a comparatively high rate, ~20.4%, which is corroborating
evidence for excluding it rather than the reason on its own).

**Fix:** a new, centralized, independently-testable relation policy
(`src/conflict_eval/data/conflict_eligibility.py`) plus a subject-level
multi-object check against the full interim PopQA pool. Three researcher-
defined categories — `PRIMARY_RELATIONS` (`place of birth`, `sport`,
`country`, `mother`), `REVIEW_RELATIONS` (`father`, `capital`, `color`),
`EXCLUDED_PRIMARY_RELATIONS` (`genre`, `religion`, `screenwriter`,
`director`, `producer`, `composer`, `author`, `occupation`, `capital of`)
— cover all 16 observed relations; any relation outside all three
(unexpected for this snapshot) defaults to requiring review, never to
silent eligibility. This is a deliberately conservative **pilot policy**,
not a claim about which relations are objectively single-valued in the
world — a researcher may revisit `REVIEW_RELATIONS`, or expand
`PRIMARY_RELATIONS`, after direct inspection.

KC/KW semantics were deliberately **not** redefined: `knowledge_group`
still means only "the model's answer matches gold, or not, and is
usable." Primary-conflict eligibility is a new, orthogonal pair of
fields — `primary_conflict_eligible` (bool) and
`conflict_eligibility_reason` — added to every KC/KW record, computed
independently of KC/KW assignment, and checked in `cli.py:cmd_screen`
right after `knowledge_group` is set. `manual_review` (the existing
general-purpose flag) is set `True` for the two reasons that represent a
genuinely ambiguous case worth a researcher's attention
(`relation_multi_object`, `relation_requires_review`, and the defensive
`relation_unrecognized` fallback), but left unset for
`relation_not_primary_conflict`, since that is a settled policy exclusion
rather than an open question. Parametric margins are still computed for
every KC/KW record regardless of conflict eligibility — margins measure
the model's own confidence, not the relation's semantic properties, and
this project does not discard usable baseline information unnecessarily.
`cmd_build_pilot` now filters to `primary_conflict_eligible == True` in
addition to the existing `knowledge_group` filter, so only defensible
conflict items are ever sampled into a pilot trial.

`is_clean_factual_candidate` (`evaluation/baseline_eligibility.py`)
additionally rejects a word-level " and " conjunction, alongside the
existing comma/" or " checks, so a multi-name or multi-item answer like
the screenwriter example above is routed to `manual_review` instead of
becoming an automatic KW memory candidate. This is a small, explicit
addition, not a broad speculative blacklist.

`foils.py` itself is unchanged: same-relation sampling remains the
minimum defensible type-compatibility control, and a foil is never
claimed to be logically impossible merely because it came from another
same-relation item — whether the resulting KC item is usable as a
primary conflict trial is decided separately, by the policy above, not
by the foil-sampling mechanism.

This was discovered from a 20-item infrastructure smoke screen, before
scaling to the 100/500-item screening pool and before source calibration
or C0-C4, so no scientific result was invalidated.

## Support targeted primary conflict screening (2026-08-17)

After the real Qwen2.5-7B-Instruct validation gates passed, real-model
PRIMARY-relation candidate screening at scale (the run leading into the
500-candidate screen) required constructing the eligible, deduplicated
PRIMARY-relation candidate frame independently on the GPU with an ad-hoc
script, since the committed `prepare-data` pipeline only ever screened
from the full interim pool. That is a reproducibility gap: the exact
frame a real run screens from should be reconstructible from the
committed CLI and config alone, not from an undocumented one-off script.

**Fix:** `dataset.candidate_pool` in `configs/pilot.yaml`, validated in
`config.load_pilot_config` (`ConfigError` on anything other than `all` or
`primary_conflict_relations`; defaults to `all` when omitted, so every
config written before this option existed keeps its exact prior
behavior). `all` is unchanged: `cmd_prepare_data` screens from the full
interim pool exactly as before. `primary_conflict_relations` calls the
new `data/popqa.build_primary_relation_candidate_pool`, which:

1. classifies every interim row via the already-committed
   `data/conflict_eligibility.classify_primary_conflict_eligibility` —
   the same relation policy (`PRIMARY_RELATIONS`) and subject-level
   multiplicity check `cmd_screen` already uses for
   `primary_conflict_eligible`, not a second, parallel relation list
   (verified directly by a test that patches `PRIMARY_RELATIONS` and
   confirms the pool construction responds to it);
2. deterministically deduplicates the eligible rows by `(relation,
   subject)`, keeping the row with the lexicographically smallest string
   `id` when more than one eligible row shares a subject/relation pair
   (e.g. two re-scraped rows recording the same fact);
3. only *after* that frame is built does the existing seeded
   `screen_candidates`/`sample_candidates` logic apply, unmodified.

The eligibility check depends only on the full interim pool's content
(via `build_relation_subject_object_index`), and the dedup choice and
final ordering are both determined by comparing/sorting item ids — so the
resulting frame, and therefore the candidate file built from it, is
reproducible independent of interim row ordering. `cmd_prepare_data` logs
each stage (`interim rows`, `eligible primary rows`,
`unique primary relation/subject facts`, `screened candidates`) so a
targeted run's provenance is visible without re-deriving it.

**PRIMARY_RELATIONS predates this change.** The relation policy
(`place of birth`, `sport`, `country`, `mother`) was committed in
`d4e732a` ("Restrict primary trials to defensible conflicts"), motivated
by a semantic review of the relation types plus a duplicate-rate census —
*not* by which relations happened to give favorable outcomes with
Qwen2.5-7B-Instruct. This screening option reuses that already-fixed
policy verbatim; it does not introduce a new relation list, and it is not
a post-hoc choice of relations based on real-model results.

**What this is not:** a targeted-frame screen is an *efficiency*
mechanism for constructing causal conflict trials — it changes which
subset of PopQA is screened, not the eligibility rule, KC/KW semantics,
foils, prompts, scoring, or generation settings. Because it changes the
sampling frame, relation/answer statistics from a targeted-frame screen
must not be interpreted as prevalence estimates over all of PopQA. Final
candidate sampling within the (possibly restricted) frame remains
deterministic and seed-pinned, exactly as before.

**Screen summary wording.** The human-readable `screen` command summary
previously printed `manual_review=N`, which could be misread as counting
`knowledge_group == "manual_review"` specifically — but a KC/KW record
can independently carry `manual_review = True` (e.g. flagged for
conflict-eligibility review) while keeping its `knowledge_group`. The
summary now prints `manual_review_flagged=N` instead. This is a wording
change to the printed summary only; no record field, value, or semantics
changed.

No real model, PopQA download, or experiment was run to implement this
option — the algorithm was validated with synthetic fixtures only (see
`tests/test_primary_relation_pool.py`,
`tests/test_prepare_data_candidate_pool.py`); the real 2,156 / 2,143 /
2,136 counts and the reported candidate-file SHA256 were observed
independently on the GPU and are not re-derived or re-verified here.

## Make source calibration output strict

Fixed before the first real Qwen2.5-7B-Instruct source-preference
calibration was run, analogous to "Make decision output format strict"
(`ffa67c5`) for the baseline `Answer`/`Decision`/`Confidence` format:
that fix's real-model finding — a model can partially reproduce
pipe-alternatives wording literally (`"Decision: answer | certain"`),
and an unanchored prefix-matching parser silently accepts it — applies
identically to the source-calibration `Choice` field, which used the
same `"Choice: 1 | 2"` wording and the same
`Choice:\s*([12])` unanchored regex. This was caught by inspection before
any real calibration data existed, so **no source-preference result is
invalidated**; there is nothing to invalidate.

**Prompt** (`prompts/source_calibration.txt`): `"Choice: 1 | 2"` replaced
with `"Choice: <1 or 2>"`, plus explicit instructions ("write exactly one
digit... do not use the | symbol... do not provide an explanation"). The
substantive source-preference question itself
("For answering general factual questions, which of these two sources
would you prefer to rely on...") is unchanged — this remains **direct
stated preference**, not latent/behavioral preference (see
`docs/reference_implementations.md` for that terminology distinction).

**Parser** (`source_preference/calibration.py:parse_choice`): now
`^Choice:\s*([12])\s*$` with `re.MULTILINE`, line-anchored and exact,
mirroring the Decision-field fix exactly. `"Choice: 1 | 2"`,
`"Choice: 1 because..."`, `"Choice: 12"`, `"Choice: 1/2"`, and a missing
Choice line all now return `None` rather than a coerced guess — malformed
choices continue to be excluded from pairwise statistics, not coerced,
which was already the documented behavior for a `None` result.

**Prompt version**: `configs/sources.yaml`'s `calibration_prompt_version`
bumped from `v1` to `v2`. `v1` is retired, not reinterpreted — no trial
can claim `v1` semantics for the new prompt text, and no `v1` calibration
data exists to reinterpret.

**Model provenance**: `CalibrationTrial` gained `model_revision`,
`requested_revision`, and `resolved_revision`, populated from the
adapter's own attributes (never a hard-coded SHA) — the same convention
already used for baseline result records
(`docs/decisions.md`, "Exact model revision recording"). Source
preference is model-specific, so a calibration record that only names
the model family (`model_id`) without the exact checkpoint is not fully
reproducible. The calibration summary JSON gains the same three fields
at the top level. This is purely additive: `PairwiseStat`,
`compute_pairwise_stats`, `build_preference_matrix`, and
`rank_sources_pilot_heuristic` are all unchanged, since they only read
`source_a`/`source_b`/`selected_source` from trial records — no pairwise
preference statistic is affected by this addition.

**What did not change**: the six labels in `configs/sources.yaml`, pair
enumeration (all 15 unordered pairs from 6 labels), AB/BA counterbalancing
(30 total presentations), the direct-stated-preference framing,
`rank_sources_pilot_heuristic`'s status as an explicitly-labeled pilot
heuristic (not a statistical threshold), and the requirement that a
researcher — not the pipeline — sets `preferred_source`/
`dispreferred_source` in `configs/pilot.yaml` after inspecting real
calibration output. No repetitions, thresholds, or new ranking method
were introduced.

No real model, source calibration, or experiment was run to implement
this fix — all tests use synthetic fixtures and fake model adapters
(`tests/test_source_calibration.py`).

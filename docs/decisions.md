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

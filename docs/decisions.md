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
`tests/test_hf_tokenizer_integration.py`. Full logit/log-probability
validation on real model outputs remains open until adequate compute
(a GPU, or a machine with substantially more free RAM) is available.

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

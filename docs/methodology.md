# Methodology

This document is meant to be detailed enough that another researcher could
reproduce the pilot without reading every implementation file. It describes
what the implementation actually does; `docs/phase2_research_design.md` is
the specification it implements, and `docs/decisions.md` records why
methodological choices were made.

## 1. Dataset and preprocessing

**Source:** `akariasai/PopQA`, Hugging Face `datasets` identifier, test
split. Fields used: `id`, `subj`, `prop`, `obj`, `s_aliases`, `o_aliases`,
`question`, `possible_answers`. The Hugging Face dataset card states
14,300 test rows; the pinned `revision="main"` snapshot actually resolved
and downloaded during implementation (`resolve -> pin -> load -> record`,
see `docs/decisions.md`) contained 14,267 rows. These are not claimed to
be the same figure — the dataset-card count is a description of the
repository at some point in time, not a guarantee about any specific
pinned commit; `data/raw/manifest.json`'s `num_rows` is the authoritative
count for any given real run.

**Pipeline stages** (`data/raw/` -> `data/interim/` -> `data/processed/`,
implemented in `src/conflict_eval/data/popqa.py`):

1. `data/raw/` holds the untouched downloaded dataset (not committed; see
   `data/README.md` for recreation).
2. `data/interim/` holds the dataset after deterministic normalization but
   before candidate screening: lowercasing (for comparison only — original
   casing is preserved in a separate field), punctuation normalization,
   whitespace normalization, defensible article normalization (dropping a
   leading "a"/"an"/"the" for comparison purposes only), and alias set
   construction from `possible_answers` plus any `o_aliases`.
3. `data/processed/` holds the sampled candidate pool actually used to
   build baseline-screening inputs, plus an exclusion log recording every
   item dropped and why (e.g., empty gold answer, empty alias set, parse
   failure).

A fixed random seed (`configs/pilot.yaml: seed`) controls the initial
candidate subsample (default: screen ~500-1000 candidates, configurable).

`dataset.candidate_pool` selects the **screening frame** that this
subsample is drawn from — an efficiency-oriented option for constructing
causal conflict trials, not a redesign of the experiment
(docs/decisions.md, "Support targeted primary conflict screening"):

- `all` (default; every config predating this option behaves exactly as
  before) — the frame is the full interim pool.
- `primary_conflict_relations` — before sampling, the frame is restricted
  to interim rows whose relation is in `PRIMARY_RELATIONS` and whose
  `(relation, subject)` pair is subject-multiplicity-eligible, using the
  same `data/conflict_eligibility.classify_primary_conflict_eligibility`
  policy `cmd_screen` already applies per-model
  (`src/conflict_eval/data/popqa.py:build_primary_relation_candidate_pool`).
  Rows are then deterministically deduplicated by `(relation, subject)` —
  when more than one eligible row shares a subject/relation, the row with
  the lexicographically smallest string `id` is kept — before the same
  seeded `screen_candidates`/`sample_candidates` sampling applies
  unchanged. This is deliberately the same relation policy that already
  determines `primary_conflict_eligible` at screening time, not a second,
  parallel relation list, and it **predates** this screening option — see
  `docs/decisions.md` for why it is not a post-hoc choice based on
  favorable model outcomes. Because it changes the sampling frame, a
  targeted-frame screen's relation distribution must not be interpreted
  as a prevalence estimate over all of PopQA.

## 2. Model adapters

`src/conflict_eval/models/base.py` defines `BaseModelAdapter` with:

- `generate(messages, generation_config) -> str` — chat-template-formatted
  generation.
- `score_candidate(messages, candidate_text, answer_prefix="") -> ScoredSequence` —
  teacher-forced scoring of a candidate continuation. `answer_prefix` is
  literal text (e.g. `"Answer: "`) that the model is expected to produce,
  as part of its own turn, immediately before the candidate — see section
  3 below and `docs/decisions.md` ("Scoring prefix must include the
  Answer: field label").
- `score_candidate_detailed(messages, candidate_text, answer_prefix="") -> DetailedScore` —
  same computation, plus the decoded answer tokens and their individual
  log probabilities. Diagnostic only (used by the `diagnose-score` CLI
  command); never written to result records.

Two adapters implement it:

- `dummy.py` — `DummyModelAdapter`, a deterministic, seedable mock used in
  tests and dry runs. It never loads a real model.
- `hf_causal.py` — `HFCausalAdapter`, wraps a Hugging Face
  `AutoModelForCausalLM` + `AutoTokenizer`, applies the tokenizer's chat
  template, and supports CUDA if available. Generation config defaults to
  `do_sample=False` (greedy decoding), matching the library's documented
  deterministic mode rather than only setting `temperature=0`. The actual
  `generation_config` used is stored on every result record.

## 3. Sequence log-probability scoring

Implemented in `src/conflict_eval/scoring/sequence_logprob.py`.

For a candidate answer `a` under a chat-formatted prompt `prompt`:

1. Render the prompt through the tokenizer's chat template up to (and
   including) the assistant generation-start marker, then append
   `answer_prefix` — the literal field label the model is instructed to
   produce before its answer (`"Answer: "`, from
   `experiment/prompts.py:ANSWER_FIELD_PREFIX`). This combined string is
   the **scoring prefix**. Including `answer_prefix` is required: without
   it, the candidate would be scored as if it were the literal first
   tokens of the assistant's turn, not as the value of the Answer field —
   a substantially different and far less meaningful quantity for a model
   that is instructed to always start its response with `"Answer: "`
   (found during real-model-adapter validation; see `docs/decisions.md`).
2. Tokenize `prefix + a` as a single sequence, and locate the boundary
   between prefix tokens and answer tokens by finding the longest common
   token-id prefix between `tokenize(prefix)` and `tokenize(prefix + a)`,
   rather than assuming the boundary is exactly `len(tokenize(prefix))` —
   retokenizing a longer string can change how the last prefix token(s)
   are merged (`scoring/sequence_logprob.py:answer_token_boundary`).
3. Run a single forward pass (teacher forcing — the full sequence is fed at
   once, not autoregressively resampled) and take `log_softmax` over the
   vocabulary dimension at each position.
4. For each answer token at position `t` (t >= prefix_len), the model's
   prediction for that token comes from the logits at position `t-1`. Sum
   `log P(a_t | prefix, a_<t))` over answer-token positions only —
   prompt-token positions are masked out of the sum.
5. Report both the raw summed log probability and the length-normalized
   score `(1/N) * sum(...)`, where `N` is the number of answer tokens.
   Downstream comparisons (the parametric margin) always use the
   normalized score, per `docs/phase2_research_design.md`; the raw sum is
   retained for diagnostics only, never compared across differently sized
   answers.

Care points, each covered by a dedicated unit test in `tests/test_logprob.py`:

- A single leading whitespace token can attach to the first answer token
  depending on the tokenizer; the prefix/answer boundary is computed by
  tokenizing the prefix and prefix+answer separately and finding the
  longest common token-id prefix between them, not by assuming the
  boundary is at `len(tokenize(prefix))`.
- The scoring prefix includes `answer_prefix` (the "Answer: " field
  label) — real-tokenizer validation (`tests/test_hf_tokenizer_integration.py`,
  gated behind `CONFLICT_EVAL_RUN_TOKENIZER_TESTS=1`) confirms this adds
  real tokens to the prefix under the actual Qwen2.5-7B-Instruct
  tokenizer and that answer-token boundary detection still recovers the
  exact candidate text on both sides of that change.
- BOS/EOS and chat-template special tokens are part of the prefix, never
  scored as if they were answer content.
- Multi-token answers are summed correctly, not scored as a single token.

## 4. Parametric preference margin

Implemented in `src/conflict_eval/scoring/parametric_margin.py`. There is
no single universal gold-vs-foil margin used for every trial (this was
flagged as a design risk in the original prompt and is avoided). Instead,
for each **primary conflict trial**:

- KC conflict trials: `memory_answer = gold`,
  `conflicting_context_answer = false foil`.
- KW conflict trials: `memory_answer = baseline wrong answer`,
  `conflicting_context_answer = gold`.

    parametric_margin = normalized_score(memory_answer) - normalized_score(conflicting_context_answer)

Both scores are computed under the identical C0-style baseline prompt
prefix (no evidence), so the only thing that differs between the two scored
sequences is the candidate answer text itself.

## 5. Baseline screening and knowledge groups

`scripts/screen_baseline.py` (via `src/conflict_eval/experiment/*` and
`evaluation/*`) generates a baseline (no-evidence) answer per item per
model, deterministically, then:

1. Parses the response into `Answer` / `Decision` / `Confidence` fields
   (`evaluation/parse.py`); syntactically malformed responses (no
   locatable answer or decision field) are logged to a separate
   exclusions stream, not written as a baseline record at all.
2. Normalizes and matches the parsed answer against gold/aliases
   (`evaluation/answer_match.py`).
3. Checks baseline *eligibility* before KC/KW assignment
   (`evaluation/baseline_eligibility.py`): a syntactically well-formed
   response is only eligible to become KC or KW if `Decision == "answer"`
   and the answer text is not itself an explicit uncertainty/refusal
   marker (`"uncertain"`, `"unknown"`, `"i don't know"`, `"i do not
   know"`, `"cannot determine"`, `"can't determine"`, checked after the
   same normalization used for answer matching). This check runs before
   the gold-match comparison, so an abstention that happens to restate
   the gold answer text is still not eligible for KC, and an abstention
   never becomes a KW memory candidate — a real 20-item smoke screen with
   Qwen2.5-3B-Instruct found the model emitting `Answer:
   uncertain\nDecision: uncertain` (and, inconsistently, sometimes
   `Decision: answer`) far more often than expected; see
   `docs/decisions.md`, "Baseline abstentions must not become KC/KW
   memory candidates". Ineligible responses are recorded with
   `knowledge_group = "excluded"` and `exclusion_reason =
   "baseline_uncertain"` — kept as a baseline record (not the malformed
   exclusions stream) since the response itself was syntactically valid.
4. Classifies eligible items as `KC` (answer matches gold/alias), `KW`
   (answer does not match gold/alias and is additionally a clean,
   unambiguous candidate — short, no comma, no " or ", no " and " as a
   word-level conjunction), or `manual_review` (an eligible but not
   obviously clean non-matching answer), per model. A real Qwen2.5-7B-Instruct
   screen produced a screenwriter answer ("Eric Paul Friedmann and
   Christophe Beck") that the comma/" or " checks alone did not catch;
   see `docs/decisions.md`, "Restrict primary trials to defensible
   conflicts".
5. Checks primary conflict trial eligibility for every KC/KW item,
   independent of KC/KW assignment itself (`data/conflict_eligibility.py`):
   a relation-level policy (`PRIMARY_RELATIONS` / `REVIEW_RELATIONS` /
   `EXCLUDED_PRIMARY_RELATIONS`) plus a subject-level check against the
   full interim pool for relations/subjects with more than one distinct
   known object. Stored as `primary_conflict_eligible` (bool) and
   `conflict_eligibility_reason`; does not change `knowledge_group`. See
   `docs/phase2_research_design.md`, "Primary conflict trial eligibility:
   different answer != semantic conflict", for the full rationale and
   the exact relation lists.
6. Computes the conflict-specific parametric margin for every KC and KW
   item (using the relevant foil for KC, gold for KW), regardless of
   primary conflict eligibility, and assigns a within-pool quantile
   `margin_bin` (`low`/`medium`/`high`) for pilot sampling convenience
   only.

Knowledge groups and margins are stored per `(model_id, item_id)` and are
never assumed to transfer across models. `build-pilot` additionally
filters to `primary_conflict_eligible == true` before sampling.

## 6. Foil construction (KC items)

`src/conflict_eval/data/foils.py`. For each KC item, sample a foil answer
deterministically (seeded) from another PopQA item sharing the same
`prop` (relation), excluding any candidate equal to the gold answer or a
known alias. If no valid same-relation foil exists, the item is excluded
and logged, not forced with a mismatched-type foil. Foil metadata (foil
answer, source item id, relation, generation method = "same_relation_sample")
is stored on the item record. Same-relation sampling establishes
type-compatibility only, not semantic conflict — whether the resulting
KC item is usable as a primary conflict trial is decided separately by
the relation/subject policy in step 5 above.

## 7. Source-preference calibration

`scripts/calibrate_sources.py` runs `src/conflict_eval/source_preference/`:

1. `pairs.py` enumerates all unordered pairs from `configs/sources.yaml`.
2. `counterbalance.py` expands each pair into both presentation orders
   (`AB`, `BA`).
3. `calibration.py` renders `prompts/source_calibration.txt` for each
   presentation, calls the model, and parses a structured `Choice: <1 or
   2>` response into a `selected_source`. Parsing is line-anchored and
   exact (`^Choice:\s*([12])\s*$`, multiline): a Choice line with any
   trailing content — including the pipe-alternatives phrasing an earlier
   prompt version used — is treated as malformed rather than accepted as
   a prefix match (docs/decisions.md, "Make source calibration output
   strict"). Each trial record also carries `model_revision`,
   `requested_revision`, and `resolved_revision` from the adapter, since
   source preference is model-specific.
4. `ranking.py` aggregates trial-level records into pairwise
   `P(S_i preferred to S_j)` statistics and a preference matrix per model.

Calibration output is written to its own JSONL stream (see
`docs/pilot_protocol.md` and the result-record schema below), separate from
experimental generations. Calibration recommends candidate pairs; it does
not write `preferred_source`/`dispreferred_source` into
`configs/pilot.yaml` automatically. `build-pilot` reads those two fields
from configuration and refuses to proceed if either is unset, printing the
calibration summary so the researcher can decide.

## 8. Controlled evidence and conditions

`src/conflict_eval/experiment/evidence.py` renders the fixed evidence
template (`prompts/evidence.txt`) with only `{source}` and
`{asserted_answer}` substituted. `conditions.py` builds the five
per-item trial specifications (C0-C4) and assigns `conflict_status` and
`evidence_truth` according to the KC/KW mapping in
`docs/phase2_research_design.md`. `prompts.py` renders the full
experimental prompt (`prompts/baseline.txt` style, with evidence slotted
in) for each trial and stores the exact rendered text on the result record.

## 9. Resumable execution

`src/conflict_eval/experiment/runner.py` computes a deterministic record
key from `(experiment_type, model_id, item_id, condition, prompt_version,
seed)`. Before generating, it checks `src/conflict_eval/io/results.py`'s
index of already-written keys in the target JSONL file and skips completed
work. Writes are append-only and flushed per record, so an interrupted run
can be resumed without duplication or partial-record corruption.

## 10. Evaluation and metrics

`src/conflict_eval/evaluation/classify.py` assigns each generation one of
`gold`, `memory`, `context`, `other`, `uncertain`, `manual_review`, using
the deterministic matcher from `answer_match.py`. `metrics.py` computes
CAR, HOR, COR, `Delta_harm`, `Delta_correct`, and abstention rate exactly as
defined in `docs/phase2_research_design.md`, restricting each metric to the
correct trial subset (conflict-only for CAR/HOR/COR).

## 11. Analysis

`src/conflict_eval/analysis/` computes descriptive summaries
(`summaries.py`), an exploratory logistic regression on conflict trials
using `statsmodels` (`regression.py`), and the four documented figures
(`plots.py`) using matplotlib only. All analysis functions raise rather
than silently proceed when given empty or synthetic-only input, so
placeholder numbers cannot end up in a figure or summary table.

## 12. Reproducibility notes

- All stochastic steps (candidate subsampling, foil sampling, generation
  decoding) are seeded via `configs/*.yaml`.
- Every result record stores model id, model revision (if resolvable),
  generation config, prompt version, and the exact rendered prompt.
- Raw PopQA data is not committed; `data/README.md` documents exact
  recreation steps and the dataset identifier/revision used.

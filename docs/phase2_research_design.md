# Phase 2 — Research Design

This is the current experimental specification. If implementation reveals a
genuine methodological problem, it is not silently changed here — see
`docs/decisions.md` for the process (document the issue, implement the
least invasive correction, record what changed and why).

## Research questions and hypotheses

See `docs/research_proposal.md` for the motivating narrative. Restated
precisely for implementation:

**RQ1.** How does the strength of an LLM's pre-existing parametric
preference affect its willingness to adopt conflicting external evidence?

**RQ2.** Holding evidence content exactly constant, does changing only its
attributed source change the probability that the model follows that
evidence instead of its baseline parametric answer?

**RQ3.** Does source preference amplify both beneficial correction of
incorrect parametric answers and harmful override of correct parametric
answers?

**Exploratory.** Does the model become appropriately uncertain under
conflict, or does one source simply win?

### H1 — Parametric-strength hypothesis

For a conflict trial, define the conflict-specific parametric margin:

    B(q) = score(memory_answer | q) - score(conflicting_context_answer | q)

where `score` is a length-normalized sequence log probability (teacher
forced; see `docs/methodology.md`). Prediction: as B increases, P(context
adoption) decreases.

### H2 — Source-preference hypothesis

For identical contextual evidence content:

    P(follow context | preferred source) > P(follow context | dispreferred source)

### H3 — Interaction hypothesis

The source effect may be strongest at intermediate levels of parametric
preference strength: weak preference allows override regardless of source;
intermediate preference is where source identity may exert the most
influence; extremely strong preference may resist override even from a
preferred source.

### H4 — Truth-blind source-effect hypothesis

If source preference behaves like a general weighting heuristic rather than
truth discrimination, a preferred source may increase both corrective and
harmful overrides. This is a hypothesis to be tested, stated here so it is
not silently written into conclusions after the fact.

## Conflict vs. agreement (critical design point)

C0-C4 trials must not be pooled indiscriminately. The RQs concern
**conflict** between the model's memory answer and the supplied evidence.
Interpretation depends on the model-specific knowledge group.

For each model and item:

    memory_answer = model's clean baseline answer without external evidence

### Known-Correct (KC) items

Baseline: `memory_answer == gold`.

- Correct external evidence agrees with the parametric answer -> **agreement**.
- False external evidence (a plausible foil) conflicts with the parametric
  answer -> **conflict**.

Primary KC conflict trials: **C3** (false evidence, preferred source) and
**C4** (false evidence, dispreferred source). These measure harmful
override.

### Known-Wrong (KW) items

Baseline: `memory_answer != gold`, and the baseline wrong answer is clean
enough to use as a candidate.

- Correct external evidence (containing gold) conflicts with the parametric
  answer -> **conflict**.
- False external evidence containing the baseline wrong answer agrees with
  the parametric answer -> **agreement**.

Primary KW conflict trials: **C1** (correct evidence, preferred source) and
**C2** (correct evidence, dispreferred source). These measure corrective
override.

The remaining conditions in each group are agreement controls. Context
adoption in an agreement trial is never interpreted as evidence that the
source *caused* the answer, because context and memory point to the same
answer and the causal question is unidentified.

Every result record stores `conflict_status: conflict | agreement | none`
(C0 is `none`) and `context_truth: true | false | none` (renamed
`evidence_truth` on the record; see `docs/methodology.md` for the exact
schema).

## Model-specific knowledge groups

Groups are computed per model and never assumed to transfer. An item marked
KC for one model may be KC, KW, excluded, or manual_review for another.

- **KC** — the model gave an answer (not an abstention) that matches gold
  or a listed alias.
- **KW** — the model gave an answer (not an abstention) that does not
  match gold, and is a clean factual candidate (unambiguous, not
  malformed, not excessively long, not a list of alternatives). An
  explicit abstention (e.g. "uncertain") is never a clean factual
  candidate, even if it happens to coincide with the gold text or the
  model inconsistently reports `Decision: answer` alongside it — KC and
  KW both represent usable parametric answers, not abstentions.
- **excluded / manual_review** — baseline answer is unsuitable for either
  role, including an explicit abstention (see `docs/methodology.md` for
  exact exclusion criteria).

## Sampling across parametric strength

We do not select only extreme-margin items. H3 requires observing multiple
strength levels. Within each model x KC/KW pool, items retain a continuous
`parametric_margin`, and are additionally labeled with a `margin_bin`
(`low`/`medium`/`high`) computed from within-pool quantiles, purely as a
pilot sampling convenience. Pilot sampling draws approximately evenly across
bins when enough eligible items exist. The continuous margin, not the bin,
is the primary quantity for analysis; bins are not treated as fundamental
categories.

## Foils (KC false evidence)

Rules (implemented in `src/conflict_eval/data/foils.py`):

1. Prefer foils sampled from another PopQA item with the **same relation**
   (`prop`), which is the minimum defensible type-compatibility control.
2. A foil must not equal the gold answer or any known gold alias.
3. Sampling is deterministic given a configured seed.
4. Each foil record stores: foil answer, source item id, relation,
   generation method.
5. Foils are not hand-crafted to be absurd or implausible.
6. If no defensible foil can be constructed for an item, the item is
   excluded and the reason is logged — not forced.
7. Foils are not LLM-generated in the pilot.

## Source-preference calibration

Source preference is calibrated independently of the PopQA test set, using
**direct pairwise calibration** (not "latent preference" — that term is
reserved for indirect/behavioral measurement, which is future work; see
`docs/reference_implementations.md`).

- Candidate source labels live in `configs/sources.yaml`, not hardcoded.
- Every unordered pair `(S_A, S_B)` is presented in both orders
  (`AB` and `BA`) to separate position preference from source preference.
- The calibration prompt (`prompts/source_calibration.txt`) asks only for a
  structured choice (`Choice: 1` or `Choice: 2`), not an explanation.
- Each calibration trial records: model, source_a, source_b,
  displayed_order, selected_source, seed, exact prompt, prompt version.
- Pairwise statistics (`P(S_i preferred to S_j)`) and a preference matrix
  are computed per model.
- `build-pilot` does not auto-select a preferred/dispreferred pair. The
  calibration step recommends candidate pairs; the researcher sets
  `preferred_source` / `dispreferred_source` in configuration. If unset,
  `build-pilot` refuses to proceed and explains what is needed. Any ranking
  heuristic used to recommend pairs is labeled a pilot heuristic, exposed
  in configuration, and reported with its underlying counts.

## Controlled evidence

Evidence is prompt-injected, not retrieved, to isolate evidence integration
from retrieval quality, chunking, reranking, and context-window confounds
(see `docs/decisions.md`). Canonical template (version tracked as
`evidence_template_version`):

    Source: {source}

    Statement:
    The answer to the question "{question}" is "{asserted_answer}".

Changing source identity changes only `{source}`; changing factual content
changes only `{asserted_answer}`. Evidence is not LLM-generated in the
pilot.

## Five conditions

- **C0** — no evidence, no source (baseline-in-experiment-format).
- **C1** — correct evidence, preferred source.
- **C2** — correct evidence, dispreferred source.
- **C3** — false evidence, preferred source.
- **C4** — false evidence, dispreferred source.

Interpretation depends on knowledge group:

| Group | C0 | C1/C2 | C3/C4 |
|---|---|---|---|
| KC | baseline | agreement (correct evidence) | **primary conflict** — harmful override |
| KW | baseline | **primary conflict** — corrective override | agreement (false evidence = baseline wrong answer) |

Every trial record includes `condition`, `evidence_truth`, `source_role`,
`conflict_status`.

## Prompt design

The experimental prompt deliberately does not instruct the model to answer
only from context, which would force context adherence and defeat the
purpose of the experiment:

    You are answering a factual question.

    Use the information you consider most reliable.
    You may use the supplied information as evidence.

    If you cannot determine the answer reliably because the available
    information is insufficient or conflicting, indicate uncertainty.

    Evidence:
    {evidence_or_none}

    Question:
    {question}

    Respond using exactly this format:

    Answer: <short answer>
    Decision: answer | uncertain
    Confidence: <integer from 0 to 100>

No chain-of-thought, explanation, or rationale is requested. Confidence is
exploratory only and is not treated as a calibrated probability without
separate validation. The exact rendered prompt is stored per generation.
Prompt templates are versioned under `prompts/`.

## Pilot size

Target approximately 60 selected items per model (30 KC, 30 KW), 5
conditions each, 2 models -> approximately 600 experimental generations.
This is a pilot: it validates the setup, tests whether effects exist,
estimates effect sizes, surfaces parsing failures and confounds, validates
source calibration, and informs a go/no-go decision on scaling. It is not
sized for publication-level power. All counts are configurable in
`configs/pilot.yaml`.

## Primary outcome classification

Each generated response is classified deterministically into exactly one of:
`gold`, `memory`, `context`, `other`, `uncertain`, `manual_review`. On
agreement trials, `memory` and `context` labels may coincide with the same
normalized answer; in that case causal attribution to source is not
claimed, because the two hypotheses are not separable in that trial.

## Answer matching

Deterministic normalization (lowercasing, punctuation/whitespace handling,
defensible article normalization) plus exact/alias matching is the primary
classifier. No LLM judge is used in the pilot. Token F1 may be stored as a
diagnostic but does not replace exact/alias classification. Ambiguous
generations are marked `manual_review = true` rather than silently coerced.

## Primary metrics

- **Context Adoption Rate (CAR)**, conflict trials only:
  `CAR = P(final answer == contextual conflicting answer)`.
- **Harmful Override Rate (HOR)**, KC + false conflicting context only:
  `HOR = P(false contextual answer adopted | baseline answer correct)`.
  Lower is better.
- **Corrective Override Rate (COR)**, KW + correct conflicting context only:
  `COR = P(correct contextual answer adopted | baseline answer wrong)`.
  Higher is better.
- **Source effect on harmful override:** `Delta_harm = HOR_preferred - HOR_dispreferred`.
- **Source effect on corrective override:** `Delta_correct = COR_preferred - COR_dispreferred`.
- **Abstention Rate (AR)** (exploratory): `AR = P(Decision == uncertain)`.
- Also retained: final answer accuracy, other-answer rate, self-reported
  confidence, parametric margin, knowledge group, source, conflict status.

## Primary statistical analysis

Restricted to conflict trials. `Y = 1` if conflicting external context was
adopted, else `0`. Predictors: `B` (conflict-specific parametric margin),
`S` (source role, preferred vs. dispreferred), `T` (context truth: 1 =
corrective true context, 0 = harmful false context). Conceptual model:

    logit P(Y=1) = b0 + b1*B + b2*S + b3*T + b4*(B*S) + b5*(S*T) + b6*(B*T) + ...

The primary interaction of interest is `b4` (whether source preference
changes how parametric margin affects adoption). For the pilot, ordinary
logistic regression is exploratory, and its limitations (small N,
non-independence of repeated items/models, no pre-registration) are stated
explicitly rather than treating pilot p-values as confirmatory. A scaled
study should use a mixed-effects treatment with item/model random effects.
Effect sizes, uncertainty intervals, plots, and qualitative inspection are
prioritized over significance claims.

## Figures

Generated only from real pilot output; never from fabricated placeholder
numbers.

- **Plot 1 (signature interaction):** x = parametric margin, y = empirical
  context-adoption rate, curves split by source role, and by evidence truth
  (corrective vs. harmful), using sensible empirical bins or a documented
  fitted model — not manufactured smooth curves.
- **Plot 2 (corrective vs. harmful override):** 2x2 summary of override
  rate by (preferred/dispreferred) x (corrective/harmful).
- **Plot 3 (condition summary):** C0-C4 rates by model and knowledge group,
  distinguishing agreement vs. conflict conditions.
- **Plot 4 (abstention under conflict):** exploratory.

Matplotlib only, restrained defaults, no decorative themes. Confidence
intervals, if shown, are computed correctly and the method is documented.

## Go/No-Go criteria

After a real pilot, report on three signals with supporting empirical
values, each labeled `detected`, `weak`, or `inconclusive` — never claimed
automatically successful:

- **Signal 1:** Is context adoption associated with parametric preference
  strength?
- **Signal 2:** Does changing only source attribution produce a repeatable
  effect on conflict trials?
- **Signal 3:** Is source influence different between corrective override
  and harmful override?

The go/no-go decision itself is made by the researcher, not automatically
by the analysis code.

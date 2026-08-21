# Phase 3 — Scaled Confirmatory Study Design

## 1. Status and purpose

**Status: PHASE 3A (DESIGN ONLY). NOT RUN. DESIGN DECISIONS APPROVED;
EXECUTION NOT FROZEN.**

No Phase 3 model has been run. No Phase 3 data exists. No Phase 3 model
revision has been resolved for the *new* models. No Phase 3 source
calibration has been performed. Nothing in this document is a result, and
nothing in it may be cited as evidence about any model's behavior.

This document is a protocol/preregistration: its purpose is to fix, in
advance, the decisions that would otherwise become researcher degrees of
freedom once Phase 3 outcomes exist. As of this revision the researcher
has **approved** the cohort architecture, the primary test, the source
pairs, the common pair, the cell-size and screening rules, the statistical
plan, the replication criteria, the Qwen/Llama revision policy, the two
additional model families, and the compute budget (§42, "Resolved").
**One item remains open** — the exact releases/revisions of the two
approved new model families — and is listed in §42.1.

The execution freeze itself still happens in **Phase 3C** (§36); approval
of this design is not the pre-run freeze.

The completed Phase 2 pilot is frozen and is not modified by this
document. `docs/qwen_pilot_results.md`, `docs/cross_model_pilot_results.md`,
`docs/phase2_research_design.md`, `docs/llama_replication_protocol.md`, and
`configs/frozen/` remain exactly as they are.

## 2. Motivation from the completed Phase 2 pilot

Phase 2 ran a controlled context-memory conflict pilot on two
instruction-tuned models, 60 selected items each (30 KC + 30 KW), five
conditions per item, 300 generations per model, 120 genuine conflict
trials per model. The frozen results
(`docs/cross_model_pilot_results.md`) are:

| Model | Conflict type | Preferred | Dispreferred | Delta | Paired exact p |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | Corrective (KW) | 25/30 = 83.3% | 17/30 = 56.7% | +26.7 pp | 0.0078125 |
| Qwen2.5-7B-Instruct | Harmful (KC) | 6/30 = 20.0% | 3/30 = 10.0% | +10.0 pp | 0.25 |
| Llama-3.1-8B-Instruct | Corrective (KW) | 29/30 = 96.7% | 30/30 = 100.0% | -3.3 pp | 1.0 |
| Llama-3.1-8B-Instruct | Harmful (KC) | 10/30 = 33.3% | 11/30 = 36.7% | -3.3 pp | 1.0 |

The Qwen corrective source effect was the strongest signal in the pilot.
**It did not replicate in Llama.** The current evidence therefore supports
a *model-dependent boundary condition*, not a universal preferred-source
effect.

Phase 3 exists because Phase 2 cannot distinguish among several
explanations for that divergence. The following are all live and all
confounded in the Phase 2 design:

1. **Genuine model-specific behavior.** The two models really do weight
   source attribution differently.
2. **Source-pair non-equivalence.** The dispreferred label differed
   (`an anonymous online forum post` for Qwen, `a social media post` for
   Llama), because calibration was independently performed per model. The
   cross-model contrast is therefore not a same-stimulus contrast.
3. **Item non-equivalence.** KC/KW status and margins are model-specific,
   so the two models were evaluated on different selected factual items.
4. **Relation composition.** Neither model's selected sample was
   relation-balanced, and their compositions differed from each other
   (Qwen KC was entirely country/sport; Llama KC included place of birth
   and mother).
5. **Ceiling regime.** Llama corrective adoption was 96.7% / 100.0%.
   A source manipulation cannot express an increase from a near-saturated
   baseline. Llama's corrective "null" and a genuine mid-range null are
   not the same observation and must not be read the same way.
6. **Sampling noise.** 30 paired items per cell, with 8 discordant pairs
   driving the Qwen corrective result, is a small basis for a point
   estimate. The Phase 2 estimate is subject to winner's-curse inflation.

Additional Phase 2 constraints Phase 3 must respect: the Phase 2
regression evidence was **not confirmatory**; the Llama ordinary logistic
regression **did not converge** (quasi-separation) and its coefficients
and p-values must not be interpreted; the Qwen tentative-answer analysis
was **post-hoc**, the Llama one was **pre-specified secondary**; only two
models were tested. Nothing in Phase 2 licenses architecture-level
language, and Phase 3 must not introduce it.

## 3. Confirmatory research questions

**RQ-A (direct replication — the primary question).** Does the Phase 2
Qwen corrective preferred-source effect reproduce on a larger, genuinely
new set of Qwen KW items, under essentially the same manipulation (same
pinned model revision, same frozen source pair, same prompts, same
outcome)? Answered by **Cohort A** (§15.1). Note that this question is
deliberately *not* conditioned on relation balance — see RQ-F.

**RQ-B (model dependence).** Does the Phase 2 Llama near-zero corrective
source effect also reproduce, and is the Qwen/Llama divergence itself
reproducible?

**RQ-C (source identity vs. elicited preference).** When every model sees
the *same* two source labels (common contrast), does source identity
change committed context adoption? Is that the same effect as each model's
own model-specific preferred/dispreferred contrast?

**RQ-D (parametric strength).** Within genuine conflict trials, does
stronger parametric preference predict lower context adoption, and does
the source effect depend on parametric strength?

**RQ-E (truth asymmetry).** Does a source effect appear in both corrective
(KW) and harmful (KC) conflict, as truth-blind source weighting would
predict, or only in one?

**RQ-F (composition — generalization).** Do source effects survive
*enforced relation balance* (**Cohort B**, §15.2) and *shared-item*
controls (**Cohort C**, §16), or are they explained by item/relation
composition? This is a generalization question, distinct from RQ-A's
replication question, and a negative answer here does not overturn a
positive RQ-A result — it qualifies its scope.

Overall framing question: *when* does source attribution alter an LLM's
willingness to override parametric knowledge, and is that effect
reproducible across models, relations, and parametric-strength levels?

## 4. Confirmatory and exploratory hypotheses

Each hypothesis carries an explicit status. Status is fixed now and may
not be changed after Phase 3 outcomes exist.

**H1 — Parametric strength. (SECONDARY CONFIRMATORY.)**
Within genuine conflict trials, larger `B` (parametric margin favoring the
memory answer) predicts lower P(context adoption). Tested on the
continuous margin, not on bins.

**H2a — Common source identity. (SECONDARY CONFIRMATORY.)**
Holding evidence content exactly fixed, changing only the attributed
source between the two *common* labels changes P(context adoption).
This hypothesis is deliberately **directionless per model**: the common
labels are fixed source *identities*, and it is not assumed that either is
the preferred label for any given model. Direction is only predicted for a
model in which calibration independently establishes an ordering over
those same two labels, and that is a separate, secondary claim.

**H2b — Model-specific source preference.**
Within each model, evidence attributed to that model's preferred source is
adopted more often than identical evidence attributed to its dispreferred
source. This is the direct descendant of the Phase 2 H2. Its status is
**split by model**:

- **H2b-Qwen-A (PRIMARY CONFIRMATORY — the sole primary test).** The Qwen
  corrective (KW) contrast **on Cohort A** (§15.1: 96 fresh Qwen KW items,
  32/32/32 margin strata, no relation quota), using the frozen Phase 2
  source pair (`a government website` vs. `an anonymous online forum
  post`) at the frozen Phase 2 revision, is the single designated primary
  confirmatory test of Phase 3. It is the direct replication of the frozen
  Phase 2 result (25/30 vs. 17/30, Delta = +26.7 pp, paired exact
  p = 0.0078125).
- **H2b-other (SECONDARY CONFIRMATORY).** The corresponding contrasts on
  **Cohort B** (relation-balanced, all models including Qwen), for Llama,
  and for each new model belong to the secondary confirmatory family and
  are Holm-corrected within it (§28). The Cohort B Qwen contrast is a
  *generalization* test, **not** a second direct replication, and is never
  reported as corroborating Cohort A independently (they share items and
  observations, §15.2).

**H3 — Source × parametric strength. (EXPLORATORY.)**
The source effect may depend on parametric-preference strength. The
Phase 2 "strongest at intermediate strength" shape is retained only as an
**exploratory** hypothesis, not a confirmatory one, for two reasons: the
Phase 2 `B:S` interaction was unsupported (Qwen p = 0.563), and three
quantile bins cannot identify a nonlinear shape. If tested, nonlinearity
must be modeled explicitly on the continuous margin (e.g. a restricted
cubic spline or a quadratic term in `B` interacted with source), not
inferred from three bin means. Bin-level displays are DIAGNOSTIC only.

**H4 — Truth-blind source weighting. (SECONDARY CONFIRMATORY.)**
If source preference acts as a general weighting heuristic rather than
truth discrimination, a preferred-source effect appears in *both*
corrective (KW) and harmful (KC) conflict. Both directions must be
estimated and reported. A corrective-only effect does **not** support
truth-blind weighting and must not be described as such.

**H5 — Replication / model dependence.**
The Qwen corrective effect is subjected to direct confirmatory replication
under the criteria in §37-38; that test **is** H2b-Qwen, the sole primary
confirmatory test. Whether the Llama near-zero result also reproduces is
evaluated separately, in the secondary confirmatory family — a
"non-replication of Qwen" and a "replication of Llama's null" are
different claims and are reported as such. Success is never defined as
`p < .05` alone (§37).

## 5. Scope of inference

Phase 3 evaluates approximately four instruction-tuned open-weight causal
LMs of broadly similar scale, on English PopQA short-answer factual
questions, under prompt-injected templated evidence, with directly
elicited source preferences.

Phase 3 does **not** support, and its report must not claim:

- conclusions about "LLMs" as a class;
- architecture effects (four models cannot separate architecture from
  training data, instruction tuning, RLHF recipe, or tokenizer);
- conclusions about retrieval-augmented systems in deployment (evidence
  here is injected, not retrieved);
- claims about universal source credibility, trustworthiness, or latent
  epistemic preference (calibration is *directly elicited stated*
  preference only);
- conclusions about model scale (scale is deliberately held roughly
  constant and is therefore not manipulated).

Four models is a small, non-random sample of model families. Model-level
inference is descriptive across a fixed set, not generalization to a model
population (§27).

## 6. Model-selection principles

Additional models are selected on stated methodological criteria, applied
before any Phase 3 outcome exists:

1. **Instruction-tuned causal LM** with a chat template, so the existing
   prompt-rendering and `Answer:`-prefix scoring path applies unchanged.
2. **Comparable scale** (roughly 7B-9B), so scale is not a confound with
   family. Deliberately not a scale study.
3. **Feasible on the existing RTX 3090-class setup** in unquantized
   float16 with `device_map: auto`.
4. **Deterministic generation support** (`do_sample: false`,
   `num_beams: 1`).
5. **Hugging Face revision pinning** available, so `resolve -> pin ->
   load -> record` works unchanged.
6. **Usable token-level teacher-forced sequence scoring**, so the
   parametric margin is computable under the existing scoring method.
7. **Distinct family/training lineage** from Qwen and Llama and from each
   other, to improve generality per unit of compute.
8. **Licensing/access constraints documented** before selection is
   frozen; gated models require provisioned access.
9. **No quantization change** to fit a model in. If a candidate does not
   fit in unquantized float16 on the available hardware, it is dropped
   rather than quantized, because changing numeric precision for one model
   only would confound family with precision. (If quantization ever
   becomes scientifically necessary, it must be applied as a documented,
   pre-registered manipulation with its own control, not as a
   convenience.)

## 7. Proposed model set

Two models are fixed by the replication requirement:

- `Qwen/Qwen2.5-7B-Instruct` — direct Phase 2 replication target.
- `meta-llama/Llama-3.1-8B-Instruct` — direct Phase 2 replication target.

Two additional families are **RECOMMENDED, NOT FROZEN** (researcher
approval item, §42.1). Repository evidence alone cannot establish current
availability, licensing status, hardware fit, or tokenizer/scoring
compatibility for models this project has never loaded. The shortlist
below is a reasoned recommendation against §6, not a verified selection:

| Candidate family | Why it fits the criteria | Known caveats to verify in 3C |
|---|---|---|
| `mistralai/Mistral-7B-Instruct` (a specific released instruct version) | ~7B, distinct lineage from both Qwen and Llama, established chat template, open weights | Exact version must be chosen and pinned; instruct-version differences are substantive |
| `google/gemma-2-9b-it` | ~9B, distinct lineage/tokenizer, instruction-tuned | Gated access; 9B float16 memory headroom on a 24GB card must be verified |
| `microsoft/Phi-3-small-8k-instruct` or a comparable Phi release | Distinct training-data philosophy (heavily curated/synthetic), useful contrast | Scoring/chat-template compatibility must be verified; family behavior may differ enough to complicate comparability |
| `01-ai/Yi-1.5-9B-Chat` | ~9B, distinct lineage | Availability/licensing must be verified |

**APPROVED AT THE FAMILY LEVEL (researcher decision): the
Mistral-7B-Instruct family and the Gemma-2-9B-it family.** Together they
add two clearly distinct training lineages at comparable scale.

They remain **provisional at the exact-release level until Phase 3C**. For
each new model, Phase 3C must verify and freeze:

- the exact Hugging Face repository ID;
- the exact instruction-tuned release;
- the exact resolved commit SHA;
- **tokenizer/model revision equality** (both loaded from the same SHA);
- licensing and access requirements (Gemma is gated);
- **float16 feasibility** on the RTX-3090-class setup;
- **no-quantization feasibility** (§6.9);
- compatibility with the existing sequence-logprob scoring pipeline
  (chat template + `Answer: `-prefix teacher-forced scoring).

**If either family cannot satisfy these frozen runtime requirements, STOP
and return to researcher approval.** Another model is never substituted
silently, and a model that does not fit in float16 is dropped rather than
quantized.

### Model revision policy — FROZEN (researcher decision)

For the replicated models, Phase 3 uses **the same exact model artifacts
as Phase 2**, not re-resolved current versions:

| Model | Repository | Frozen revision |
|---|---|---|
| Qwen | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |

Rationale: a model update between phases is an uncontrolled variable that
could by itself produce an apparent non-replication of the primary test.
Pinning the Phase 2 artifacts makes Cohort A a test of the *effect* on new
items, not a test of a moving model.

- **Do not re-resolve to a newer revision for the replication analyses.**
- **Phase 3C must verify that these exact pinned revisions are still
  available and loadable.**
- **If an exact pinned artifact cannot be loaded, that is an
  infrastructure/version failure** (§34) — stop and report. **Never
  silently substitute a newer revision.**

**No exact revision SHA is invented for the two new models.** Theirs are
resolved only in Phase 3C via the existing metadata-only
`conflict_eval.models.hf_causal.resolve_model_revision` and recorded in
the freeze manifest.

## 8. Dataset and exact snapshot policy

- Dataset: `akariasai/PopQA`, split `test`, unchanged from Phase 2.
- Revision: pinned to the exact Phase 2 snapshot
  `098765c79ea10a2cb19c828324e33281b8336ec0`, via the existing generic
  `dataset.revision` config option (`docs/methodology.md`, §1). Using the
  same snapshot means Phase 3 draws from the same underlying factual
  population as Phase 2, which is a precondition for calling RQ-A a
  replication.
- The `resolve -> pin -> load -> record` behavior of `download_raw` is
  unchanged; a revision that cannot be resolved raises rather than
  silently falling back to `main`.
- The raw/interim/processed artifacts remain uncommitted runtime outputs.
- Normalization, alias construction, and the exclusion log are unchanged
  from `docs/methodology.md` §1.

## 9. Relation eligibility

Unchanged from the already-committed policy in
`src/conflict_eval/data/conflict_eligibility.py` and
`docs/phase2_research_design.md`:

- **PRIMARY** (automatically eligible): `place of birth`, `sport`,
  `country`, `mother`.
- **REVIEW** (excluded from automatic sampling): `father`, `capital`,
  `color`.
- **EXCLUDED**: `genre`, `religion`, `screenwriter`, `director`,
  `producer`, `composer`, `author`, `occupation`, `capital of`.
- Unrecognized relations are treated conservatively as requiring review.

Phase 3 does **not** expand PRIMARY_RELATIONS. Expanding the relation set
between Phase 2 and Phase 3 would change the sampling frame in the same
step that tests replication, confounding the two. If the four PRIMARY
relations cannot supply the target cell counts (§32), the response is to
reduce the per-cell target or declare cells exploratory — not to add
relations after the fact. Any future expansion of PRIMARY_RELATIONS must
be a separate, pre-outcome, documented decision with its own manual
review of the added relation's single-valuedness.

## 10. Subject-level multiplicity handling

Unchanged: a `(relation, subject)` pair with more than one distinct known
object in the **full interim pool** is automatically ineligible
(`relation_multi_object`) regardless of relation category, and is flagged
for manual review. The index is built from the entire interim pool, not
the sampled candidates, so detection is not limited by subsample luck.

Phase 3 addition (procedural, not a rule change): because Phase 3 screens
more candidates than Phase 2 (§22), the absolute number of
`relation_multi_object` exclusions will grow. That count must be reported
per relation in the Phase 3 screening summary, so relation-level
eligibility attrition is visible rather than buried.

## 11. Baseline screening

Unchanged mechanism (`docs/methodology.md` §5): deterministic no-evidence
generation per model per item; strict field parsing; baseline eligibility
check (an abstention is never KC or KW, checked *before* gold matching);
KC/KW/manual_review classification; primary-conflict-eligibility flagging;
conflict-specific parametric margin computed for every KC/KW record.

Phase 3 requirements:

- Baseline screening is run **per model**, independently. KC/KW labels are
  never transferred across models (§16).
- The full screening summary (KC, KW, excluded, manual_review, malformed,
  per relation, per margin stratum) is recorded before any evidence
  condition is generated.

### Adaptive pre-outcome screening procedure (frozen)

Screening acquires **baseline eligibility supply**. It is not optional
stopping on a scientific outcome, and the distinction is enforced by the
rule below.

- **Block size: 250 candidates per model.** Candidates are drawn
  deterministically from the seeded, pinned candidate frame, in order, so
  block *k* is always the same set of items for a given seed and frame.
- **Hard ceiling: 2,000 candidates per model** (8 blocks). Screening never
  exceeds this, regardless of whether quotas are met.
- **Early-stop condition.** Screening for a model stops after any block in
  which **all** of that model's applicable supply criteria below are met.
  The **reserve is 2 items per cell/stratum** throughout, absorbing losses
  from manual review (§33) and foil-construction failures (§21/§31) that
  are only discovered after a cell is counted.

  - **Cohort A (Qwen only) — the primary-test supply criterion.** At least
    **34 usable *fresh* Qwen KW items in each final margin stratum**
    (32 target + 2 reserve), i.e. **≥ 34 low, ≥ 34 medium, ≥ 34 high**,
    under the current pool's stratum definition. "Fresh" excludes every
    Qwen KW item used in the frozen Phase 2 pilot (§15.1); Phase 2 items
    do **not** count toward this supply. There is **no relation quota** in
    this criterion.
  - **Cohort B — the relation-balanced supply criterion.** Every required
    (relation × KC/KW × margin stratum) cell holds at least **target +
    reserve = 10** eligible, `primary_conflict_eligible` candidates
    (target 8, §15.2).
  - **Cohort C** draws from the same screened pools; it imposes no
    additional stopping requirement beyond the above.

  For Qwen, both criteria apply and screening continues until both are met
  or the ceiling is reached — but they **fail independently** (§34): the
  Cohort B criterion failing does not make Cohort A eligibility-limited,
  and vice versa.
- **Permitted stopping information:** baseline records, KC/KW labels,
  eligibility flags, relations, parametric margins, and margin strata —
  i.e. *supply*. **Prohibited stopping information:** any Phase 3
  evidence-condition generation, any `context_adopted` value, any source
  contrast, any Phase 3 outcome of any kind. Evidence conditions are not
  generated until after the freeze (§36), so at screening time no outcome
  physically exists to leak.
- The realized block count, per-block counts, and the reason screening
  stopped (supply criteria met vs. ceiling reached) are recorded per model
  in the freeze manifest.

### Margin-stratum stability under block-wise screening (frozen rule)

Quantile boundaries shift as candidates are added, so the rule is stated
explicitly to remove any ambiguity:

1. **After each 250-candidate block**, recompute the margin strata
   deterministically from the **complete currently screened eligible
   pool** for that (model × knowledge group) — not incrementally, not from
   the block alone.
2. **Evaluate the stopping criteria (above) against those newly
   recomputed strata**, so the supply check always refers to the strata
   that would actually be used if screening stopped at that block.
3. **Once a stopping criterion is satisfied and screening stops, those
   final boundaries and the resulting per-item stratum assignments are
   frozen** — recorded in the freeze manifest — before any evidence
   generation.
4. **Phase 3 sampling strata are never recomputed after observing any
   evidence outcome.** Post-outcome re-stratification is prohibited
   outright; it would let outcome data influence which items sit in which
   stratum.

Continuous margin remains scientifically primary for modeling (§13, §14);
strata remain sampling and diagnostic devices only, never treated as
latent categories.
- If the ceiling is reached with cells still short, §32's
  downsampling/eligibility-limited ladder applies. **Screening is never
  extended past the ceiling to rescue a cell.**

## 12. KC/KW definitions

Unchanged from Phase 2:

- **KC** — eligible response (`Decision: answer`, not an abstention
  marker) whose answer matches gold or a listed alias.
- **KW** — eligible response whose answer does not match gold/alias and is
  a clean, unambiguous factual candidate (short; no comma; no " or "; no
  word-level " and " conjunction).
- **excluded / manual_review** — everything else, including all
  abstentions (`exclusion_reason = baseline_uncertain`).

These definitions are frozen for Phase 3. Changing them would make RQ-A
not a replication.

## 13. Parametric-preference scoring

Unchanged conceptual quantity:

    B(q) = normalized_score(memory_answer | q)
         - normalized_score(conflicting_context_answer | q)

using length-normalized teacher-forced token log probability under the
exact `Answer: ` scoring prefix, both candidates scored under the
identical C0-style no-evidence prompt prefix, with the longest-common-
token-prefix boundary detection already implemented in
`src/conflict_eval/scoring/`. KC trials use `memory = gold`,
`context = foil`; KW trials use `memory = baseline wrong answer`,
`context = gold`.

**Continuous margin is scientifically primary.** Bins are for balanced
sampling, visualization, and diagnostics only, and are never treated as
latent categories.

## 14. Margin standardization/binning

Raw margin magnitudes are **not comparable across models**: different
tokenizers, vocabularies, and calibration produce different log-prob
scales. Naively pooling raw `B` across four models would confound scale
with behavior.

Phase 3 policy:

- **Within-model analyses** (including the primary within-model H2b
  analysis and H1 within each model) use the **raw continuous margin**,
  which is meaningful on its own scale.
- **Cross-model analyses** use a **within-model, within-knowledge-group
  standardized margin**. The pre-specified default is the within-(model ×
  KC/KW) **normalized rank** (empirical quantile in [0,1]), because rank
  is invariant to monotone rescaling of the log-prob scale and is robust
  to the heavy-tailed margin distributions Phase 2 observed (Qwen KC
  margins ranged -0.2001 to 25.8750 with median 13.9619). A z-score
  within (model × group) is recorded as a pre-specified robustness
  alternative; if rank and z-score analyses disagree materially, both are
  reported and neither is presented as the single answer.
- **Bins** (`low`/`medium`/`high`) are within-(model × group) tertiles of
  the raw margin, used for sampling quotas (§15) and diagnostic display.
  Bin edges are model-specific and are recorded in the freeze manifest.

## 15. Cohort architecture: three distinct cohorts

Phase 3 uses **three deliberately distinct cohorts**. They are related but
**not interchangeable**, and conflating them is the specific error this
architecture exists to prevent.

| Cohort | Question it answers | Status |
|---|---|---|
| **A — Direct Qwen replication** | *"Does the original Qwen corrective-source effect replicate on new items under essentially the same manipulation?"* | **SOLE PRIMARY CONFIRMATORY** |
| **B — Relation-balanced model-specific** | *"Does source-sensitive evidence use survive stronger control over relation composition and parametric-strength strata?"* | SECONDARY CONFIRMATORY / generalization |
| **C — Shared cross-model** | *"How do models differ when evaluated on the same underlying facts?"* | SECONDARY CONFIRMATORY / cross-model |

**Cohort B is not a direct replication of Phase 2 and must never be
described as one.** It imposes relation balance that Phase 2 did not have,
so it changes the sampling frame relative to the Phase 2 result. It tests
*generalization under stronger control*, which is a different — and also
valuable — claim.

**Critical independence property.** Cohort A's design deliberately does
**not** depend on Cohort B's relation-balanced grid succeeding. Phase 2
showed severe relation imbalance, especially for Qwen KW (14 country, 2
sport, 12 mother, 2 place of birth). Under a relation-quota requirement, a
single scarce relation could render the sole primary replication
untestable — an unacceptable design fragility. Cohort A therefore drops
relation quotas entirely (§15.1). A Cohort-B eligibility failure **never**
invalidates or terminates Cohort A (§34).

### 15.1 Cohort A — direct Qwen replication cohort (PRIMARY)

**Status: the sole PRIMARY CONFIRMATORY analysis of Phase 3.**

**Purpose.** Directly test whether the Phase 2 Qwen corrective source
effect replicates on a larger, genuinely new set of factual items.

**Composition:**

- **96 fresh Qwen KW items.**
- **Balanced by parametric-margin stratum only: 32 low / 32 medium /
  32 high.**
- Eligibility: existing `PRIMARY_RELATIONS` only (`country`, `sport`,
  `place of birth`, `mother`), `primary_conflict_eligible == true`, and
  genuinely KW for Qwen under the unchanged §12 definitions.
- **No equal-relation quota is imposed.** Relation composition is whatever
  the eligible KW supply yields within each margin stratum, sampled
  deterministically and seeded.

**Item freshness (required).** The 30 Qwen KW items used in the frozen
Phase 2 pilot are **excluded** from Cohort A, by item id, so the
replication is measured on a genuinely new item sample rather than partly
re-measuring the original items. The excluded id list is recorded in the
freeze manifest.

**Manipulation held identical to Phase 2:**

- Model: `Qwen/Qwen2.5-7B-Instruct`, **exact revision
  `a09a35458c702b33eeacc393d103063234e8bc28`** (§7, §35).
- Source contrast: the **frozen Phase 2 Qwen pair** — preferred
  `a government website`, dispreferred `an anonymous online forum post`.
- Same prompt logic, same controlled evidence template, same generation
  settings, same primary committed `context_adopted` outcome.

**Replication target (frozen Phase 2 result):** preferred 25/30 = 83.3%,
dispreferred 17/30 = 56.7%, Delta = +26.7 pp, paired exact
p = 0.0078125.

**Mandatory reporting.** The **relation distribution of the realized
96-item Cohort A** is always reported alongside the primary estimate, so
that a reader can see exactly what item mix produced it. This is a
transparency requirement, not a quota.

**Relation-dominance diagnostic (DIAGNOSTIC, non-gating).** The share of
Cohort A contributed by its most frequent relation is computed and
reported. If a single relation exceeds **60%** of the cohort, that is
flagged in the report and the leave-one-relation-out sensitivity (§29.1)
is given added prominence in the discussion. **This flag never blocks,
re-samples, or invalidates the primary analysis**, and no relation cap is
imposed on sampling — deliberately, because a quota complex enough to
force balance is exactly what would threaten feasibility and reintroduce
the fragility this cohort exists to remove.

**Eligibility-limited condition — narrow by construction.** Cohort A can
be declared ELIGIBILITY-LIMITED **only** if the §11 screening ceiling
(2,000 Qwen candidates) cannot produce 96 usable *fresh* Qwen KW items
meeting the 32/32/32 margin balance. It is **never** eligibility-limited
because a PRIMARY relation is scarce, because a relation × margin cell is
sparse, or because Cohort B failed.

### 15.2 Cohort B — relation-balanced model-specific cohort (SECONDARY)

**Status: SECONDARY CONFIRMATORY / generalization.** Not a direct
replication of Phase 2.

For each model independently, and for KC and KW separately:

- Items must be genuinely KC or KW **for that model**, and
  `primary_conflict_eligible == true`.
- Target grid per knowledge group: **4 relations × 3 margin strata × 8
  items = 96 items**, i.e. **96 KC + 96 KW = 192 selected items per
  model**.
- Sampling within each (relation × margin stratum) cell is deterministic
  and seeded.
- Margin strata are computed within (model × knowledge group) as tertiles
  (§11, §14) before quota sampling.
- **Target cell size: 8. Minimum acceptable balanced cell size: 6.**
- If all cells reach ≥ 6 but not all reach 8, **deterministically
  downsample every relevant cell to a common cell count**, before any
  evidence outcome exists (§32.3).
- If any required cell remains below 6 at the screening ceiling, the fully
  relation-balanced analysis for that model × group is classified
  **ELIGIBILITY-LIMITED** (§32.4). **No selective backfilling.**

**It is unknown whether 96 KC + 96 KW exist for each model.** That is an
empirical property of each model's baseline screen and cannot be asserted
in advance. §11 defines how screening supply is grown, §32 defines what
happens when a cell falls short, §23 explains why 8/cell was chosen.

**Overlap with Cohort A.** Qwen KW items can qualify for both cohorts.
Where an item-condition observation is genuinely identical across cohorts,
it is **generated once and referenced by both** (§22 deduplication /
aliasing), never regenerated. Cohort membership and each observation's
statistical role are recorded explicitly per record, so an observation is
counted once within any single estimate and its dual membership is
auditable.

**A Cohort-B eligibility failure does not affect Cohort A** (§34).

## 16. Cohort C — shared cross-model cohort

**Purpose.** Compare models on the same underlying factual questions,
removing item non-equivalence as an explanation for cross-model
divergence (Phase 2 limitation #3).

**Construction** (deterministic; occurs after *all* model baseline screens
are complete and before any Phase 3 evidence-condition generation):

1. Take the union candidate frame screened by all models.
2. Retain items that are `primary_conflict_eligible` and have a
   non-`excluded`, non-`manual_review` baseline record **for every model**
   in the study.
3. Retain items whose baseline record for every model is either KC or KW
   (i.e. every model has a usable parametric answer on that question).
4. Apply a deterministic, seeded relation-balanced quota to reach the
   shared-cohort target size.

**KC/KW labels are not required to agree across models, and are never
forced to.** A shared item may be KC for one model and KW for another.
Each model keeps its own baseline state, its own memory answer, its own
margin, and its own conflict construction (KC → false foil evidence,
KW → correct gold evidence). What is shared is the *question*, not the
knowledge label.

**Analysis separation.** Cohort C is analyzed **separately from the primary
Qwen replication (Cohort A)** and from Cohort B. Shared-cohort results are
SECONDARY CONFIRMATORY for cross-model comparison and never replace,
override, or corroborate the Cohort A primary estimate.

**Selection integrity.** Shared-cohort membership is determined using
baseline/eligibility information only. It may never use Phase 3
context-adoption outcomes, evidence-condition generations, or any
Phase 3 result. The shared cohort is fixed in the Phase 3C freeze manifest
before generation begins.

**Overlap with Cohorts A and B** is allowed and expected; an item may
appear in more than one cohort. Records are tagged with **all** their
cohort memberships so analyses can be run on any cohort without
re-deriving membership post hoc. Overlapping item-conditions are generated
**once** and referenced from each cohort (§22 deduplication; the runner's
deterministic record key makes this safe), never regenerated per cohort —
compute is not multiplied by cohort membership (§23).

## 17. Relation balancing

Phase 2's selected samples were substantially relation-imbalanced (Qwen KC
was entirely country/sport; Qwen KW was 14 country / 2 sport / 12 mother /
2 place of birth). **Cohort B** targets balance by construction: 8 items
per (relation × margin stratum × knowledge group) cell, across all four
PRIMARY relations.

**Cohort A deliberately does not impose relation balance** (§15.1). Its
relation distribution is reported and diagnosed (§15.1 dominance flag) but
never quota-controlled, precisely so that a scarce relation cannot make
the sole primary replication untestable. Relation balance is Cohort B's
scientific contribution, not a precondition for Cohort A.

Pre-specified relation analyses (Cohort B unless stated):

- **Relation-balanced analysis** — the Cohort B estimates are
  computed on the balanced cohort as constructed.
- **Relation-specific descriptive results** — reported for every relation,
  DIAGNOSTIC status, with explicit n per cell.
- **Leave-one-relation-out sensitivity** — recompute the primary contrast
  four times, each dropping one relation, to check that no single relation
  drives the result. SECONDARY CONFIRMATORY (§29).
- If balance cannot be achieved (§32), the realized relation composition
  is reported in the freeze manifest and every affected estimate is
  labeled with its actual composition.

A relation subgroup with small n is never interpreted as evidence of
absence. Relation-specific nulls are reported with confidence intervals
and explicitly labeled underpowered where they are.

## 18. Source manipulations

Phase 3 separates two questions that Phase 2 conflated:

- **Is the model sensitive to a fixed source identity?** (common contrast,
  §19) — comparable across models by construction, because every model
  sees the identical two labels.
- **Is the model sensitive to its own model-specific source contrast?**
  (model-specific arm, §20) — the direct descendant of Phase 2's design.
  For Qwen and Llama this arm uses their **frozen Phase 2 pairs**, making
  it a direct replication; for new models it uses a freshly calibrated
  pair.

Both arms use the unchanged evidence template, so only `{source}` and
`{asserted_answer}` vary. The source label set (`configs/sources.yaml`,
six labels) and `calibration_prompt_version: v2` are unchanged.

## 19. Common source contrast

**FROZEN (researcher decision). The common pair is:**

    common source A = "a government website"
    common source B = "an anonymous online forum post"

Exactly these two labels, **identical for every model**, fixed now, before
any Phase 3 outcome exists. This is no longer selected adaptively at 3C;
it is a fixed design constant recorded in the freeze manifest.

**These are fixed source identities, not preferred/dispreferred roles.**
It must not be assumed or written that either common label is the
preferred label for any model. `a government website` was the measured
preferred label for both Phase 2 models, but that is a Phase 2
measurement about those two models, not a property of the labels and not
a prediction for the two new models. Whether any model's calibration
orders these two labels in a particular direction is a separate, measured
fact reported per model (§25, calibration diagnostics). H2a is therefore
stated **directionlessly** (§4).

**Transparency note — deliberate coincidence with Qwen's replication
pair.** This common pair is *identical* to Qwen's frozen Phase 2
replication pair (`a government website` vs. `an anonymous online forum
post`, §20). This is stated explicitly rather than left for a reader to
discover, because it has three consequences that must be carried into
every Qwen result:

1. For **Qwen only**, the common arm and the model-specific arm use the
   same two labels. Qwen's `M1`/`M2` conditions are therefore
   prompt-identical to the corresponding common-arm conflict conditions,
   and the deterministic deduplication rule in §22 applies: **one
   generation, referenced by two planned contrasts, not two independent
   observations.**
2. Qwen's common-arm and model-specific-arm estimates are consequently
   **the same estimate**, not two corroborating results. They must never
   be presented as independent evidence, counted twice in any summary, or
   allowed to enter the same multiplicity family twice (§28).
3. For Llama and the two new models, the common pair and the
   model-specific pair differ (Llama's dispreferred label is `a social
   media post`; new models' pairs are calibrated fresh), so for those
   models the two arms remain genuinely distinct contrasts.

This collapse is a known, accepted cost of using Qwen's exact Phase 2 pair
as the common pair: it buys an exact, uncompromised direct replication of
the primary target at the price of Qwen contributing one contrast rather
than two. The freeze manifest records the collapse explicitly per model.

## 20. Model-specific source contrast

This arm supplies each model's `M1`/`M2` conditions. Its source pair is
determined differently for the replicated models than for the new models,
and both rules are frozen now.

### 20.1 Qwen and Llama — frozen Phase 2 pairs (replication)

**FROZEN (researcher decision).** The two Phase 2 models reuse their
**exact frozen Phase 2 measured pairs**, so that the Phase 3 contrast is
the *same manipulation* the Phase 2 result came from:

| Model | `M1` source (Phase 2 preferred) | `M2` source (Phase 2 dispreferred) |
|---|---|---|
| Qwen2.5-7B-Instruct | `a government website` | `an anonymous online forum post` |
| Llama-3.1-8B-Instruct | `a government website` | `a social media post` |

These are taken from the frozen Phase 2 record
(`docs/qwen_pilot_results.md`, `docs/cross_model_pilot_results.md`,
`configs/frozen/`). They are **not re-derived, re-measured, or
re-selected** for the confirmatory analysis.

Rationale: if Phase 3 re-calibrated Qwen and the pair shifted, a
"non-replication" could reflect a *different manipulation* rather than a
different result — the confound this design exists to remove. Holding the
pair fixed makes RQ-A a clean test of the Phase 2 effect.

**Fresh Phase 3 calibration for Qwen and Llama may be run, but only as a
DIAGNOSTIC stability analysis** (§25, §44). It measures whether each
model's elicited preference is stable across phases, which is itself
previously unmeasured and interesting. It **must not** redefine either
model's confirmatory replication pair, and it may not be substituted into
`M1`/`M2` for any confirmatory analysis, regardless of what it shows. If
Phase 3 calibration disagrees with the Phase 2 pair, that disagreement is
**reported as a diagnostic finding and as a limitation on interpretation**
— it does not change the frozen contrast.

### 20.2 New models — fresh independent calibration

Each newly added model undergoes the existing direct pairwise calibration,
unchanged: all 15 unordered pairs from the six labels in
`configs/sources.yaml`, both AB and BA presentation orders, strict
`^Choice:\s*([12])\s*$` parsing, per-model preference matrix,
`calibration_prompt_version: v2`.

The researcher then selects that model's `preferred_source` and
`dispreferred_source` after inspecting its own output. There is **no
automatic source-pair selection**, unchanged from Phase 2. **No source
roles are assigned in this document for either new model**, because
neither has been calibrated and inventing them is prohibited. They are
recorded in the Phase 3C freeze manifest, before any evidence generation.

If a new model's calibration is tied, unstable, or heavily malformed, §34's
rule applies: that model runs the **common arm only** and does not
contribute to the model-specific family.

### 20.3 Interpretation constraint (unchanged)

This is **direct stated source preference for this prompt and this
model**. It is never described as latent trust, credibility, epistemic
judgment, or a universal source hierarchy — for Phase 2's frozen pairs or
for any newly calibrated pair.

## 21. Evidence construction

Unchanged from Phase 2 (`prompts/evidence.txt`, versioned as
`evidence_template_version`):

    Source: {source}

    Statement:
    The answer to the question "{question}" is "{asserted_answer}".

Only `{source}` changes between source conditions; only
`{asserted_answer}` changes between truth conditions. Evidence is not
LLM-generated. Foils are sampled deterministically from another item with
the same relation, excluding gold and gold aliases; if no defensible foil
exists, the item is excluded and logged (`src/conflict_eval/data/foils.py`,
unchanged).

The experimental prompt (`prompts/baseline.txt`) is unchanged and still
deliberately does **not** instruct the model to answer only from context.

## 22. Experimental conditions

**Recommended structure: seven conditions per item.**

| Condition | Evidence | Source | Arm |
|---|---|---|---|
| `C0` | none | none | baseline |
| `K1` | correct | common label A | common |
| `K2` | correct | common label B | common |
| `K3` | false (foil) | common label A | common |
| `K4` | false (foil) | common label B | common |
| `M1` | conflicting | model-specific source A | model-specific |
| `M2` | conflicting | model-specific source B | model-specific |

The model-specific sources are fixed per §20: **Qwen and Llama use their
frozen Phase 2 pairs** (`M1` = that model's Phase 2 preferred label, `M2`
= its Phase 2 dispreferred label); **new models use their freshly
calibrated pair**.

For the model-specific arm, "conflicting" is resolved by knowledge group,
exactly as in Phase 2: **KC → false (foil) evidence** (harmful override),
**KW → correct (gold) evidence** (corrective override). This arm therefore
contributes only conflict trials, which is the point — it deliberately
omits model-specific-source *agreement* conditions, which cost generations
and answer no Phase 3 question.

The common arm retains all four truth × source cells because the common
contrast is the cross-model comparability arm, and dropping its agreement
cells would remove the within-model agreement control that makes a common
source effect interpretable (an apparent common-source effect that also
appears on agreement trials would indicate a general response-style
effect, not conflict-specific source weighting).

**Why seven and not five.** A five-condition design (Phase 2's C0-C4)
cannot test both contrasts: it has exactly one source pair, so it forces
the study to choose between cross-model comparability (common pair) and
replication of Phase 2's design (model-specific pair). Since RQ-A demands
the model-specific contrast and RQ-C demands the common contrast, five
conditions are insufficient.

**Why seven and not a full factorial.** A complete factorial (2 truth × 2
source-arms × 2 labels-within-arm = 8, plus C0 = 9) would add
model-specific agreement conditions. Those cells are causally
uninterpretable for the source question in exactly the way Phase 2 already
documented — when context and memory assert the same answer, "adopted the
context" and "kept its memory" are not separable — so they would cost 2
generations per item (≈29% more compute) to produce cells that cannot
enter the primary analysis. The common arm already supplies agreement
controls. Seven is therefore the minimum design that answers every Phase 3
confirmatory question.

### Deterministic deduplication of prompt-identical conditions

An `M` condition and a `K` condition can be **prompt-identical**: same
item, same asserted answer, same source label — therefore the same
rendered prompt, and under deterministic decoding the same generation.
This happens whenever a model's model-specific label equals a common
label for the matching truth condition. **For Qwen it happens for both
`M1` and `M2`**, because Qwen's frozen Phase 2 pair *is* the common pair
(§19).

Pre-specified rule:

1. Prompt-identical conditions are **generated exactly once**. The
   runner's existing deterministic record key (which already includes the
   rendered prompt's determinants) detects the duplicate *before*
   generation, so no identical prompt is ever sent twice.
2. The single stored generation is referenced by **both** planned
   contrasts via condition aliasing: the record carries the full set of
   `(condition, arm, role)` labels that resolve to it.
3. **The analysis layer must treat this as one observation referenced by
   two planned contrasts, not two independent generations.** Concretely:
   it is counted once in any n; it may not contribute twice to a pooled
   estimate; the common-arm and model-specific-arm estimates built on it
   are not independent and are never reported as mutual corroboration;
   and it enters the multiplicity families only once (§28).
4. Deduplication is deterministic and decided **before** generation, from
   the frozen source assignments, not discovered afterward. The exact set
   of aliased conditions per model is recorded in the freeze manifest.
5. Partial overlap (one label shared, the other differing — e.g. Llama,
   whose `M1` label `a government website` matches common A but whose
   `M2` label `a social media post` does not) deduplicates only the
   matching conditions; the rest generate normally.

**Nominal condition workload.** At full target: 4 models × 192 items × 7
conditions = **5,376 nominal condition slots**. This is the count *before*
deterministic duplicate collapse; the realized number of actual
generations is lower (materially so for Qwen, whose `M1`/`M2` collapse
entirely into common-arm conditions) and is knowable only once the frozen
source assignments and cohorts are fixed at 3C. Both the nominal and the
realized counts are recorded in the freeze manifest.

Genuine conflict trials per model at full target, counted as distinct
observations after collapse: the model-specific arm contributes 192
(96 KC harmful + 96 KW corrective) and the common arm contributes 384
(KC under K3/K4, KW under K1/K2) — for models with no collapse, 576; for
Qwen, the model-specific 192 are the same observations as 192 of the
common arm's, giving 384 distinct conflict observations.

Item-conditions shared between Cohorts A, B, and C are **generated once
and referenced from each** (§15.2, §16, §23); only non-overlapping items
add further generations. Compute is not multiplied by cohort membership.
All counts are recorded in the 3C manifest.

Every record carries `condition`, `arm`
(`baseline | common | model_specific`), `source_label`,
`model_specific_role` (`preferred | dispreferred | null`),
`aliased_conditions` (for deduplicated records), `evidence_truth`,
`conflict_status`, and cohort tags. No post-result decision determines
which conditions are primary — the mapping in §44 is fixed now.

## 23. Sample-size target

**DESIGN CALCULATION — not a power guarantee.** The following uses *only*
the already-frozen Phase 2 aggregate numbers. No Phase 3 data exists. It
is a sensitivity sketch for a paired sign test on discordant pairs, not a
formal preregistered power analysis, and it deliberately does not treat
the Phase 2 point estimate as ground truth.

Phase 2's Qwen corrective result rested on **8 discordant pairs out of 30
items (26.7% discordance), all 8 favoring the preferred source**. Taking
"all discordant pairs favor preferred" (p ≈ 0.95) as truth would be
precisely the winner's-curse error this design must avoid, so power is
evaluated across a range of assumed true discordant-favor-preferred
proportions:

| Discordant pairs | true p=0.65 | true p=0.75 | true p=0.85 | true p≈0.95 |
|---:|---:|---:|---:|---:|
| 8 | 0.03 | 0.10 | 0.27 | 0.66 |
| 16 | 0.13 | 0.41 | 0.79 | 0.99 |
| 24 | 0.21 | 0.61 | 0.94 | 1.00 |
| 32 | 0.27 | 0.74 | 0.98 | 1.00 |
| 48 | 0.47 | 0.93 | 1.00 | 1.00 |

(Exact two-sided binomial test, α = 0.05. Non-monotonicity across rows is
a real artifact of the discreteness of exact binomial critical regions,
not an error.)

If the Phase 2 discordance rate (~26.7%) held, 96 paired items per cell
would yield ≈26 discordant pairs — adequate power (>0.9) only if the true
effect is large (p ≥ 0.85), and clearly underpowered for a moderate effect
(p ≈ 0.65). **This is the honest conclusion: 96/group is well-powered only
against a large effect, and Phase 3 must be reported as such.** Detecting
a moderate effect reliably would require roughly 150+ discordant pairs per
cell, implying ~560+ paired items per cell under the Phase 2 discordance
rate — infeasible for four models on the available hardware.

Assumptions and their fragility, stated plainly:

- The discordance rate may not persist. It is a Phase 2 Qwen-specific
  observation; Llama's corrective discordance was 1/30, near ceiling.
- Under ceiling regimes (Llama corrective), discordant pairs approach
  zero and **no sample size within reach rescues power**. This is a
  structural limit, not a sample-size problem (§30).
- Relation balancing changes the item mix relative to Phase 2 and may
  change both adoption rates and discordance.
- The Phase 2 effect estimate is very likely inflated.

**Conclusion.** 96 KC + 96 KW per model is retained as a
**bounded, feasibility- and precision-driven confirmatory target**,
justified primarily by the 4 × 3 × 8 balanced grid (§15) and by giving
materially tighter interval estimates than Phase 2's 30/group, **not** by
a claim of adequate power against a moderate effect. Phase 3 is explicitly
framed as improving *precision, balance, and comparability*. **This is not
a publication-powered sample**, and no such claim is made anywhere in this
design or in the eventual report.

Emphasis for the eventual report: **interval estimates and effect
direction carry the interpretation; p-values are secondary** (§26, §37).

### Compute budget (bounded)

**APPROVED (researcher decision): the bounded budget below.**

| Workload | Bound |
|---|---|
| **Nominal condition slots**, full balanced target | 4 models × 192 items × 7 conditions = **5,376** (before deterministic duplicate collapse, §22) |
| **Baseline-screen ceiling** | 4 models × 2,000 candidates = **8,000 baseline generations** (hard cap, §11) |
| Source calibration | 15 unordered pairs × 2 presentation orders = 30 generations per calibrated model; required for the 2 new models, plus optional DIAGNOSTIC stability calibration for Qwen and Llama (§20.1) |

**NOMINAL CONDITION SLOTS ≠ UNIQUE MODEL GENERATIONS.** The two must be
reported separately and never conflated, because the realized number of
actual forward passes is strictly lower:

- **prompt-identical common/model-specific conditions are deduplicated**
  (§22) — for Qwen, both `M1` and `M2` collapse entirely into common-arm
  conditions;
- **Cohorts A, B, and C share item-condition observations.** An
  observation belonging to several cohorts is **generated once and
  referenced from each**. **Compute is never multiplied by cohort
  membership**, and cohort membership is metadata on a record, not a
  reason to regenerate it;
- **early baseline-screen stopping** (§11) usually ends screening well
  before the 2,000/model ceiling.

**Implementation requirement:** the runner must generate a given prompt
**once** and reference that same stored generation from every cohort,
arm, and contrast for which it is genuinely identical, via the
deterministic record key and the `aliased_conditions` / cohort-membership
metadata (§22, §43).

Both the **nominal** and the **realized unique-generation** counts are
recorded in the Phase 3C freeze manifest.

This remains a **bounded, feasibility- and precision-driven confirmatory
design**. It is **not** publication-powered (§23), and no such claim is
made.

## 24. Primary outcomes

**Primary outcome: `context_adopted`** — unchanged. `True` only if
`Decision == "answer"` **and** the parsed `Answer:` field matches the
conflicting context's asserted answer. Textual answer content under
`Decision: uncertain` never counts.

Primary causal analysis is restricted to **genuine conflict trials**.
Agreement trials are controls and are never interpreted as identifying a
source-caused adoption effect, because context and memory assert the same
answer and the causal question is unidentified there.

Derived, unchanged: CAR (conflict trials), HOR (KC + false conflicting
evidence), COR (KW + correct conflicting evidence), `Delta_harm`,
`Delta_correct`.

## 25. Secondary/mechanistic outcomes

- **Tentative answer content vs. commitment** — contextual answer text
  appearing in the `Answer:` field despite `Decision: uncertain`.
  **SECONDARY mechanistic outcome, pre-specified for all four models.** It
  never replaces or merges into `context_adopted`. (For Qwen this analysis
  was post-hoc in Phase 2 and pre-specified here; for Llama it was already
  pre-specified secondary; that provenance distinction is preserved in the
  report.)
- **Abstention rate** (`Decision == "uncertain"`) — DIAGNOSTIC, and a key
  input to ceiling/floor interpretation (§30).
- **`parsed_answer_accuracy`** (`final_correct` mean) — DIAGNOSTIC,
  reported to keep the content/commitment distinction visible.
- **Self-reported confidence** — EXPLORATORY only, still not validated as
  calibrated.
- **Calibration stability** — whether each model's Phase 3 calibrated pair
  matches its Phase 2 pair (Qwen/Llama only). SECONDARY.

## 26. Confirmatory statistical analysis

### 26.1 The sole primary test (Cohort A)

Fully specified so it cannot be reinterpreted after results exist:

| Element | Specification |
|---|---|
| Cohort | **A** (96 fresh Qwen KW items, 32/32/32 margin strata, no relation quota, Phase 2 pilot items excluded) |
| Model | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` |
| Trials | **KW corrective conflict** (correct gold evidence conflicting with the model's wrong baseline answer) |
| Source contrast | Frozen Phase 2 Qwen pair: `a government website` (preferred) vs. `an anonymous online forum post` (dispreferred) |
| Outcome | **committed context adoption** (`context_adopted`) |
| Effect estimate | paired **risk difference** |
| Interval | one pre-specified **95% Tango matched-pair score interval** |
| Test | **exact two-sided McNemar / exact binomial test on discordant pairs** |
| Multiplicity | **none required** — single-test primary family (§28) |

**Mandatory primary report.** The primary result is always reported with
all of: **total n**; the four paired cells **both / preferred-only /
dispreferred-only / neither**; the **number of discordant pairs**; the
**paired risk difference**; its **95% CI**; and the **exact p-value**.
The realized **relation distribution** of Cohort A and the §15.1
relation-dominance flag are reported alongside it.

**The Qwen common-arm contrast is prompt-identical to this same frozen
Qwen pair (§19, §22) and is therefore the same observations. It must never
be described as independent corroboration of the primary test**, counted
as a second supporting result, or added to the primary family.

### 26.2 Paired procedures (applied to every source contrast)

Every selected item appears under both source conditions of a contrast, so
the analysis is **paired and item-aware**, never a comparison of unpaired
proportions. Exactly **one** procedure is pre-specified for each role — no
interchangeable alternatives, no post-hoc choice among methods:

- **Effect estimate:** the paired risk difference
  `Delta = P(adopt | source A) - P(adopt | source B)`, where for the
  model-specific arm A = that model's preferred label and B = its
  dispreferred label.
- **Confidence interval (the single pre-specified method):** the **95%
  Tango score interval for the difference of paired proportions.** This is
  the only interval used for confirmatory inference. It is chosen because
  it is valid at and near the boundary — Phase 2 produced exactly such
  cells (30/30 adoption; zero dispreferred-only discordance) where a Wald
  interval is degenerate or undefined. No other interval may be
  substituted for a confirmatory estimate.
- **Test (the single pre-specified procedure):** the **exact two-sided
  McNemar test — equivalently, the exact binomial test on the discordant
  pairs against p = 0.5** — as already implemented in
  `src/conflict_eval/analysis/paired_comparison.py`.
- **Mandatory reporting.** Every source contrast reports, without
  exception: **both**, **source-A-only**, **source-B-only**, **neither**,
  and the **number of discordant pairs** (= A-only + B-only), alongside
  `Delta` and its interval. The discordance count is not optional
  supporting detail: near-zero discordance is a ceiling/floor signature,
  not evidence of no effect (§30). The existing implementation already
  returns `p = 1.0` at zero discordance and documents that this is not
  evidence of absence.
- **Bootstrap is sensitivity-only.** A percentile bootstrap over items may
  be reported as a robustness check (§29.7). It is never the confirmatory
  interval and never substitutes for the Tango interval.
- Run separately for corrective (KW) and harmful (KC) conflict, per model.

**Common-arm analysis (H2a).** Identical paired machinery and identical
single pre-specified CI/test procedures, contrasting common label A vs.
common label B, run per model, per knowledge group, and additionally on
agreement cells as a control. For Qwen, note the §19/§22 collapse: its
common-arm conflict estimate and its model-specific estimate are the same
observations and are reported once.

**Parametric-strength analysis (H1).** Continuous margin, within model:

- **Primary specification:** ordinary logistic regression of
  `context_adopted` on the continuous margin with **item-clustered robust
  standard errors**, restricted to conflict trials — used **only if it is
  estimable**, meaning it converges, reports no quasi-/complete
  separation, and yields finite standard errors.
- **If it is not estimable** — the exact Phase 2 Llama failure — the
  single pre-specified alternative is **Firth penalized (bias-reduced)
  logistic regression**, whose estimate is defined under separation. Its
  estimand is the same log-odds coefficient on the margin; it is reported
  as Firth-penalized, with penalized profile-likelihood intervals.
- **If Firth also fails to produce a finite estimate**, the analysis is
  reported as **NOT ESTIMABLE** for that model, accompanied only by the
  DIAGNOSTIC descriptive bin display (§44). A "not estimable" result is
  never converted into a null, and no further model variants are tried.
- There is **no rank-based fallback**. (An earlier draft named a vague
  Spearman fallback without specifying its estimand or role; it is removed
  rather than kept as an unspecified escape hatch. Rank correlations, if
  computed at all, are DIAGNOSTIC description only and carry no
  confirmatory status.)
- Unconverged ordinary-logit coefficients are **never interpreted**, and
  this two-step ladder is fixed now precisely so that a convergence
  failure cannot become post-hoc model shopping.

**H3 (exploratory).** If tested, nonlinearity in margin × source is
modeled on the continuous standardized margin (restricted cubic spline or
quadratic × source interaction), never inferred from three bin means.
Reported as EXPLORATORY regardless of outcome.

## 27. Cross-model statistical analysis

With approximately **four** models, "model" is **not** treated as a
random-effect population. A `(1 | model)` term on four levels does not
support inference to a model population, and Phase 3 does not pretend
otherwise.

Pre-specified structure:

- **Model as a fixed effect** (four levels), with **item random
  intercepts** to account for repeated observations within factual item.
  On Cohort C, items are crossed with models, so the structure is
  item random intercept + model fixed effects + the source contrast + the
  model × source interaction (the cross-model heterogeneity term).
- The **model × source interaction** is the formal cross-model
  heterogeneity test: does the source effect differ by model? This is the
  statistical expression of RQ-B.
- **Trade-off, stated explicitly:** fixed model effects give valid
  inference about *these four models* and no generalization beyond them;
  random model effects would nominally generalize but are not credibly
  estimable from four levels. Phase 3 chooses valid narrow inference over
  nominal broad inference.
- **Convergence fallback, fixed in advance (exactly three steps, in this
  order, no others):**
  1. If the mixed model fails to converge, **simplify** to item random
     intercept + model fixed effects + source + model × source, dropping
     any additional random slopes.
  2. If that fails, use **GEE** (binomial family, logit link) with
     **items as the clustering unit** and an exchangeable working
     correlation, with robust sandwich standard errors. Estimand: the
     population-averaged log-odds of the source contrast and its
     model interaction.
  3. If GEE also fails, report per-model paired estimates (§26) with a
     descriptive heterogeneity display (forest-style per-model `Delta`
     with Tango intervals) and declare the pooled cross-model model
     **NOT ESTIMABLE**.

  No further model variants are tried, and "not estimable" is never
  reported as a null or as absence of heterogeneity.
- Naive pooling of item-level observations across models as if independent
  is prohibited, unchanged from the Phase 2 synthesis warning.

## 28. Multiple-comparison policy

Tests are assigned to families now.

**Primary confirmatory family — exactly ONE test:**

> **Cohort A · Qwen · KW corrective conflict · frozen Phase 2 Qwen source
> pair (`a government website` vs. `an anonymous online forum post`) ·
> frozen Phase 2 Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`
> · committed context adoption (`context_adopted`).**
>
> The direct replication of the frozen Phase 2 result (25/30 vs. 17/30,
> Delta = +26.7 pp, paired exact p = 0.0078125), measured on 96 *fresh*
> Qwen KW items.

Because the primary family contains a single test, **no multiplicity
correction is required within it**, and it is evaluated at α = 0.05 with
the §26 procedures and the §37 classification. Confining the primary
family to one test is deliberate: it keeps the study's headline claim
unambiguous and prevents the primary α from being diluted across models
whose behavior is not the replication target.

**Secondary confirmatory family — Holm-Bonferroni within the family**,
controlled separately from (and never pooled with) the primary test:

- **Cohort B** relation-balanced model-specific source contrasts, per
  model and knowledge group (including Qwen — a generalization test, not a
  second direct replication);
- Llama corrective model-specific source contrast (frozen Phase 2 pair);
- each new model's corrective model-specific source contrast (freshly
  calibrated pair);
- harmful-conflict model-specific contrasts, per model (H4);
- common fixed-source (common-arm) contrasts, per model (H2a) — counted
  **once** for Qwen, whose common-arm conflict estimate rests on the same
  observations as its frozen-pair contrast (§19, §22) and therefore does
  not re-enter as an additional independent test;
- the cross-model model × source interaction (RQ-B);
- **Cohort C** shared-cohort cross-model contrasts;
- H1 (continuous margin), per model;
- leave-one-relation-out and other §29 sensitivity analyses that qualify
  as secondary.

**Cohort A never enters the secondary family**, and no secondary result
may be presented as independent corroboration of the primary test where it
shares observations with it.

Holm is chosen over Bonferroni (uniformly more powerful, same family-wise
error control) and over FDR (family-wise control is appropriate for a
small confirmatory family where each individual claim matters).

**Exploratory and diagnostic analyses are not multiplicity-corrected and
are never reported as confirmatory**, regardless of how small their
p-values are. This includes H3, relation-specific subgroups, margin-bin
displays, confidence analyses, and any analysis conceived after Phase 3
data exists (which must be labeled post-hoc).

Dozens of model × relation × bin cells are explicitly **not** treated as
independent primary hypotheses.

## 29. Sensitivity analyses

All pre-specified; all SECONDARY CONFIRMATORY unless marked:

1. **Leave-one-relation-out** — primary contrast recomputed dropping each
   relation in turn.
2. **Country-only** — the Phase 2 sensitivity, retained for continuity
   with the frozen reports.
3. **Shared-cohort restriction** — primary contrast recomputed on shared
   items only, testing whether item composition explains cross-model
   divergence.
4. **Model-specific cohort restriction** — the complement of (3).
5. **Margin-standardization robustness** — rank vs. z-score (§14).
6. **KC/KW-agreement subset** (EXPLORATORY) — shared items where all
   models share the same knowledge label, if enough exist.
7. **Bootstrap intervals** over items, as a robustness check on the
   paired analytic intervals.

A sensitivity analysis that disagrees with the primary analysis is
**reported, not suppressed**, and weakens the primary claim rather than
being explained away.

## 30. Ceiling/floor diagnostics

Phase 2's Llama corrective cells (96.7% / 100.0%) demonstrate that a
"no source difference" observation is uninterpretable without a saturation
check. Pre-specified diagnostics, computed and reported for **every**
source contrast before it is interpreted:

- **Discordant-pair count** and rate. Near-zero discordance means the
  manipulation had almost no opportunity to express an effect.
- **Proximity to boundary**: whether either arm's adoption rate exceeds
  0.95 or falls below 0.05.
- **Confidence-interval width** for the paired risk difference.
- **Marginal saturation**: the fraction of items adopting under *both*
  conditions (`both`) or *neither* (`neither`) — Phase 2's Llama
  corrective cell was 29 `both`, 0 `neither`.
- **Abstention rate**, since near-universal abstention produces a
  different kind of floor.

**Pre-specified interpretation rule.** A contrast is labeled
**SATURATED / UNINFORMATIVE** — not "no effect" — when either arm exceeds
0.95 or falls below 0.05 adoption **and** discordant pairs are fewer than
5. A saturated cell's null must be reported as "this design could not
detect a source effect in this regime," and it may **not** be counted as
evidence against H2b, and may **not** be aggregated with non-saturated
nulls as if equivalent. Two arms both at ~50% and two arms both at ~100%
are different observations and are reported differently.

## 31. Exclusion rules

Unchanged from Phase 2 and applied identically to all models:

- Malformed baseline responses (no locatable Answer/Decision field) → the
  exclusions stream, never a baseline record.
- Abstentions → `knowledge_group = excluded`,
  `exclusion_reason = baseline_uncertain`; never KC or KW.
- Non-clean non-matching answers → `manual_review`; never silently KW.
- Items without a defensible same-relation foil → excluded and logged.
- Relation-ineligible and subject-multi-object items → excluded from
  primary conflict construction.
- Generation-time malformed experimental responses → recorded, counted,
  and manually reviewed; never silently coerced into an outcome.

Exclusion rules are frozen before generation. No exclusion criterion may
be added, tightened, or loosened after Phase 3 outcomes exist.

## 32. Missing-cell / insufficient-eligibility rules

**These rules govern Cohort B only.** Cohort A has no relation-quota
cells and therefore no missing-cell rule; its single, narrow
eligibility-limited condition is defined in §15.1 and its failure mode in
§34. **A Cohort B failure never propagates to Cohort A.**

**FROZEN (researcher decision).** Cohort B target cell size: **8 items**
per (relation × margin stratum × knowledge group × model). **Minimum
acceptable balanced cell size: 6 items.**

Pre-specified handling, in strict order, all decided **before any
evidence-generation outcome exists**:

1. **If every required cell reaches the target**, sample **8** per cell.
2. **If some cells exceed the target**, they are deterministically
   downsampled to the common target — *balanced downsampling is always
   preferred over leaving cells unequal*.
3. **If the target 8 cannot be reached but every required cell reaches
   at least 6**, deterministically **downsample all corresponding cells
   to the same common count** (the realized minimum across cells, ≥ 6),
   preserving exact balance, and record the reduced grid. Balance is
   protected at the cost of n. This happens before generation, using
   eligibility information only.
4. **If any required cell remains below 6 after the §11 screening
   ceiling**, there is **no selective backfilling**. That model ×
   knowledge group's fully balanced confirmatory cohort is declared
   **ELIGIBILITY-LIMITED**, and the following reporting policy applies:
   - the affected relation(s) are identified explicitly, with their
     realized counts;
   - the balanced estimate is computed on the remaining relations that do
     meet the minimum, and every reported number carries its realized
     relation composition;
   - if fewer than three of the four PRIMARY relations meet the minimum
     for that model × group, the whole model × group is declared
     **ELIGIBILITY-LIMITED / EXPLORATORY**, is removed from the
     confirmatory families (§28), and is still run and reported
     descriptively;
   - **if this occurs for Qwen KW, the primary test is unaffected.** The
     Cohort B Qwen KW *generalization* analysis is eligibility-limited;
     Cohort A's primary replication proceeds normally, because it imposes
     no relation quota (§15.1). This decoupling is the central purpose of
     the three-cohort architecture.
5. **No post-result backfilling, ever.** Once generation begins, no items
   are added to any cell for any reason. Cells are never topped up after
   seeing outcomes, and short cells are never selectively filled while
   full cells are left untouched.
6. Any application of rules 3-4 is recorded in the freeze manifest
   **before** generation, so the realized design is fixed pre-outcome.

## 33. Manual-review policy

Every `manual_review` baseline record, every malformed exclusion, and
every malformed calibration response is inspected by the researcher
**before** the affected items can enter any pool — unchanged from
`docs/pilot_protocol.md` step 4 and the Phase 2 replication protocol.

Additions for scale: because Phase 3 screens substantially more candidates
(§22), the manual-review burden grows proportionally. The review is still
required, not sampled. If the volume makes full review infeasible, the
correct response is to **reduce the screening ceiling or the cohort
target** — never to switch to spot-checking, and never to auto-accept
ambiguous items.

Review decisions are recorded (item id, decision, reason) and archived in
the freeze manifest, so the review is auditable rather than a claim.

## 34. Failure/stopping rules

Fixed now. Each category is handled differently, and **a scientific null
result is never a failure condition.**

**Infrastructure failures (stop, fix, document, resume):**

| Situation | Rule |
|---|---|
| **A frozen Phase 2 pinned artifact (Qwen `a09a354…` / Llama `0e9e39f…`) cannot be loaded** | **Infrastructure/version failure. STOP.** Never silently substitute a newer revision — doing so would replace the replication target. Report which artifact is unavailable and return to researcher approval. |
| A new model cannot be loaded at its Phase 3C-resolved revision | STOP for that model. Do not substitute silently. Document; researcher decides whether to re-resolve (recording both SHAs) or drop the model. |
| A new model family cannot satisfy the frozen runtime requirements (fp16, no quantization, tokenizer/scoring compatibility, access) | STOP and return to researcher approval (§7). Never substitute another family silently. |
| Exact revision resolution fails | STOP. Never proceed unpinned. |
| Model does not fit the runtime regime (unquantized float16) | Drop the model from Phase 3 and document. Do **not** quantize to fit (§6.9). |
| Raw generation interrupted | Resume via the existing deterministic record-key resumability. No duplicates, no partial records. |
| Tokenizer/model revision mismatch | STOP. Both must load from the same resolved SHA. |
| **C0 baseline reproducibility fails** | STOP before any scientific interpretation for that model. C0 must exactly reproduce the baseline record (raw generation, parsed answer, decision, confidence). Report the mismatch; do not proceed past it. |
| Parsing failure rate exceeds **5%** of a model's generations | STOP for that model, inspect, and document. A parser fix (if a real bug) must be applied to *all* models and the affected stages re-run, or the model is dropped — never patched for one model only. |

**Eligibility failures (pre-outcome design reductions, per §32):**

| Situation | Rule |
|---|---|
| **Cohort A acquisition failure** — cannot obtain 96 fresh Qwen KW items with 32/32/32 final margin balance by the 2,000-candidate ceiling | The primary test is declared **ELIGIBILITY-LIMITED**, and §37 classifies it `INCONCLUSIVE DUE TO SATURATION OR INSUFFICIENT INFORMATION`. Report the realized fresh-KW supply per stratum and the reason. Do **not** relax freshness, add relations, loosen eligibility, or extend past the ceiling. This is the **only** way Cohort A becomes eligibility-limited. |
| **Cohort B eligibility failure** — one or more relation × margin cells remain below 6 | Apply §32 rules 3-4 (downsample to a common count ≥ 6, else label that model × group ELIGIBILITY-LIMITED). **This does not terminate, delay, or invalidate Cohort A** — the two failures are independent and are reported separately. Never add relations or loosen eligibility. |
| Screening ceiling (2,000/model) reached without meeting a supply criterion | Stop screening (§11), apply the relevant rule above for whichever criterion failed, record the realized design. Screening is never extended past the ceiling. |
| A scientific null result (any cohort) | **Never** a trigger for additional acquisition, additional screening, cohort redefinition, or any design change. Nulls are reported under §37-§39. |
| **New model's** calibration is tied, weak, or unstable (heavy malformed output, AB/BA reversals) | STOP before building that model's model-specific arm. Do not force a pair. The model proceeds with the **common arm only**, contributing to H2a but not to the model-specific secondary family — recorded as such before generation. |
| No clearly separated preferred/dispreferred pair exists for a **new** model | Same as above: common arm only, documented. This is an informative measurement about that model, not a failure. |
| **Qwen/Llama** Phase 3 diagnostic calibration disagrees with their frozen Phase 2 pair | **Not a failure and not a trigger to change anything.** The frozen Phase 2 pair still defines `M1`/`M2` (§20.1). The disagreement is reported as a DIAGNOSTIC finding and as a stated limitation on interpretation. |
| A model-specific label coincides with a common label (including Qwen's full coincidence) | Not a failure. Apply §22's deterministic deduplication; treat the shared generation as one observation referenced by two contrasts, and annotate the affected estimates as non-independent across arms. |

**Analysis-stage contingencies (fallback ladders fixed in advance):**

| Situation | Rule |
|---|---|
| Ordinary logistic regression not estimable (non-convergence / separation) | §26 ladder, exactly two steps: **Firth** penalized logit → **NOT ESTIMABLE** (with DIAGNOSTIC bin display only). Never interpret unconverged coefficients; never convert "not estimable" into a null. |
| Mixed-effects model fails to converge | §27 ladder: simplify → GEE → per-model descriptive heterogeneity. No further variants. |
| A model shows near-universal abstention | Report as a DIAGNOSTIC finding about that model. Its conflict cells are labeled SATURATED/UNINFORMATIVE per §30 if discordance collapses. Not a failure. |
| A contrast is saturated | §30 rule: labeled SATURATED/UNINFORMATIVE, not "no effect". |

**Explicitly not failures:** a null result; non-replication of the Qwen
effect; replication of the Llama null; heterogeneity across models; an
effect smaller than Phase 2's. All are reportable scientific outcomes.

## 35. Reproducibility and provenance requirements

Phase 3 preserves and extends the Phase 2 discipline. Required at every
stage:

- Exact Git commit SHA used for each stage.
- Exact dataset id, split, and resolved revision.
- Exact model revision **resolved before model use** (`resolve → pin →
  load → record`), recorded as both `requested_revision` and
  `resolved_revision` on every record.
- Tokenizer and model loaded from the **same** resolved SHA, verified.
- Deterministic seeds for every stochastic step.
- Exact rendered prompts stored per generation, with prompt versions.
- Exact source labels, arm, model-specific role, and any
  `aliased_conditions` stored per record (§22).
- Exact condition construction and cohort tags stored per record.
- Full environment capture (Python, torch, CUDA, transformers, datasets,
  accelerate versions) and GPU name/VRAM.
- SHA256 of every important artifact (candidates, baselines, exclusions,
  calibration, trials, results, analysis outputs).
- **Pre-run archive** (§36), **post-run raw archive**, **final analysis
  archive** — each hashed.
- **Off-host verification**: archives copied off the GPU host and
  independently SHA256-verified, as Phase 2 did.
- **Resume-safe generation**, append-only writes, no silent overwrites.
- **C0 exact reproduction check** per model (§34).
- A **frozen final interpretation document** at the end of Phase 3F.

**No secrets, tokens, or credentials** in archives, manifests, configs, or
Git. Gated-model authentication stays in the local environment only,
consistent with `.gitignore`'s `.env` rules.

## 36. Pre-run freeze procedure

Before **any** Phase 3 evidence-condition generation, a freeze manifest is
created, hashed, and archived. It must contain at minimum:

- Repository commit SHA; Phase 3 config file contents and SHA256.
- Every model: id, requested revision, **resolved revision**, precision,
  quantization status (none), `device_map`, `max_memory` actually used.
- Dataset id, split, resolved revision; candidate file SHA256 and IDs.
- Per model: baseline and exclusion file SHA256; KC/KW membership; margins;
  margin-bin edges; manual-review decisions.
- Per model: the **model-specific source pair actually used for `M1`/`M2`**
  and its provenance — for Qwen and Llama, the frozen Phase 2 pair (§20.1);
  for each new model, its calibration output SHA256, preference matrix, and
  the researcher's stated reason for the selected pair (§20.2).
- Any DIAGNOSTIC Phase 3 calibration run for Qwen/Llama, recorded as
  diagnostic and explicitly **not** used to define `M1`/`M2` (§20.1).
- The **frozen common source pair** (§19) and the recorded note of its
  coincidence with Qwen's replication pair.
- The **deduplication map** per model: which `M`/`K` conditions are
  prompt-identical and therefore stored as one aliased observation (§22).
- **Cohort A**: the 96 selected fresh Qwen KW item IDs, their margin
  strata (32/32/32), the **excluded Phase 2 Qwen KW item IDs**, the
  realized relation distribution, and the §15.1 dominance flag value.
- **Cohort B**: selected item IDs per (relation × stratum × group) per
  model, and any §32 reductions actually applied.
- **Cohort C**: membership, and each model's per-item knowledge label on
  those items.
- **Cohort-membership map**: for every item-condition observation, all
  cohorts and contrasts that reference it, so shared observations are
  auditable and never double-counted (§23, §43).
- Any **duplicate-condition collapse** (§22) in effect, per model.
- The full condition specification and trial file SHA256.
- Which analyses are PRIMARY/SECONDARY/EXPLORATORY/DIAGNOSTIC (§39 table).
- Environment and hardware capture.

Once archived and hashed, the manifest is frozen. Changes after that point
require a new dated `docs/decisions.md` entry explaining the change and
why it is not outcome-driven — and any change made *after* outcomes exist
invalidates the confirmatory status of the affected analyses.

## 37. What counts as replication

Fixed before results exist. **`p < .05` alone is explicitly rejected as a
replication criterion**, and **no minimum practically meaningful effect
threshold is invented** (researcher decision). The report therefore makes
no claim about *practical* significance; it reports direction, magnitude,
uncertainty, and information content.

Classification inputs, all pre-specified:

- **(a) effect direction** — the sign of the paired risk difference;
- **(b) paired risk-difference estimate** `Delta` (§26);
- **(c) the 95% Tango score interval** for `Delta` (§26 — the single
  pre-specified interval);
- **(d) the exact two-sided McNemar / exact binomial test** on discordant
  pairs (§26 — the single pre-specified test);
- **(e) the Phase 2 / Phase 3 compatibility diagnostic** — defined
  operationally below, and **explicitly SECONDARY**;
- **(f) the ceiling/floor diagnostic** (§30).

**Definition of (e), the compatibility diagnostic.** "Compatibility" is
not left vague. It is computed as: **does the 95% Tango interval for the
Phase 3 Cohort A `Delta` contain the Phase 2 point estimate of
+26.7 pp?** Secondarily, the interval for the Phase 3 − Phase 2 difference
is reported descriptively.

> **(e) is a SECONDARY DIAGNOSTIC and can never override the Phase 3
> estimate or the Phase 3 test.** The Phase 3 estimate is scientifically
> primary: Phase 2 was a 30-item pilot whose estimate is very likely
> inflated, so a Phase 3 result that is smaller than +26.7 pp is evidence
> about the effect's true size, not evidence that Phase 3 is wrong.
> Criterion (e) only distinguishes *full* from *attenuated* replication;
> it never converts a positive, non-saturated, interval-excluding-zero
> result into a NON-REPLICATION.

The **primary test (Cohort A · Qwen · KW corrective · frozen Phase 2 pair,
§26.1)** is classified into exactly one of four categories. Each condition
is evaluated mechanically from pre-specified quantities, so the category
cannot be chosen subjectively after seeing results:

**FULL CONFIRMATORY REPLICATION** — all of: (a) `Delta` > 0; (c) the 95%
Tango interval excludes 0; (d) exact test p < 0.05 (no correction needed,
single-test family, §28); (e) the interval contains +26.7 pp; (f) not
flagged SATURATED/UNINFORMATIVE and discordant pairs ≥ 5.

**DIRECTIONAL / ATTENUATED REPLICATION** — (a) `Delta` > 0, (c) interval
excludes 0, (d) p < 0.05, (f) not saturated and discordant pairs ≥ 5,
**but** (e) fails — the interval does not contain +26.7 pp because the
Phase 3 estimate is materially smaller. Reported as a genuine but smaller
effect, with explicit discussion of winner's-curse inflation in the
Phase 2 pilot estimate. This outcome should be considered *a priori*
likely under genuine replication, and it is a distinct, honestly-labeled
result — neither a clean replication nor a failure.

**NON-REPLICATION** — (f) not saturated **and** discordant pairs ≥ 5
(i.e. the test was genuinely informative), **and** either `Delta` ≤ 0, or
the 95% interval includes 0. The Phase 2 Qwen effect is then reported as
unstable / pilot-specific (§38).

**INCONCLUSIVE DUE TO SATURATION OR INSUFFICIENT INFORMATION** — any of:
(f) flags the contrast SATURATED/UNINFORMATIVE (§30); **or** discordant
pairs < 5; **or** the 95% interval simultaneously includes 0 **and**
+26.7 pp (compatible with both no effect and the full Phase 2 effect);
**or** Cohort A is eligibility-limited under the narrow §15.1 condition
(fewer than 96 fresh Qwen KW items at 32/32/32 by the screening ceiling).

Evaluation order is fixed: check (f) and the discordance floor **first**;
if either triggers, the result is INCONCLUSIVE and the other categories
are not considered. This ordering prevents a saturated cell from being
read as a null.

> **A ceiling or floor regime is classified INCONCLUSIVE. It is never
> automatically labeled a scientific null and never labeled a
> non-replication.** This applies to every model, not only Qwen, and is
> the single most important guard against misreading Phase 2's Llama
> corrective result pattern.

The same four categories are applied to every other model's corrective
contrast, and separately to harmful contrasts, with the difference that
those tests must additionally survive Holm correction within the secondary
family (§28) to be classified as FULL CONFIRMATORY REPLICATION.

## 38. What counts as non-replication

**Qwen NON-REPLICATION** (as defined in §37) requires that the contrast be
**non-saturated and adequately informative**. When it is declared, the
Phase 2 Qwen effect must be reported as **unstable / pilot-specific**; the
Phase 2 documents remain frozen as the historical record; the effect is
*not* rhetorically preserved as established; and Phase 3's report must say
plainly that the earlier headline result did not hold up.

**Qwen INCONCLUSIVE** is declared under any of §37's inconclusive
triggers. Inconclusive is reported as inconclusive — never as
non-replication, and never as support for the effect.

**Llama replication (of a null)** is declared when Phase 3's Llama
corrective `Delta` is again near zero with an interval excluding large
positive effects **and** the contrast is **not** saturated. **Critically:**
if Llama is again near ceiling, the classification is
**INCONCLUSIVE DUE TO SATURATION OR INSUFFICIENT INFORMATION**, not
"Llama has no source effect" and not a confirmed null. Phase 2's Llama
corrective result (96.7% vs. 100.0%, 1 discordant pair) is already
substantially attributable to ceiling, and Phase 3 must not launder that
into a scientific null.

**Llama non-replication** is declared when Phase 3 Llama shows a clear
positive corrective source effect, contradicting Phase 2.

**Cross-model heterogeneity** is declared when the model × source
interaction (§27) is supported after multiplicity control, or, if the
pooled model is not estimable, when per-model intervals are clearly
non-overlapping in a consistent direction. Heterogeneity supports
*model-dependent* source sensitivity — never architecture-level language.

## 39. Interpretation rules

Language is pre-specified for the main outcome patterns, so interpretation
cannot drift toward whatever is most flattering:

| Outcome pattern | Pre-specified interpretation |
|---|---|
| Cohort A Qwen replicates; other models near zero (non-saturated) | Evidence supports **model-dependent source sensitivity**. Not architecture. Not universal. |
| Cohort A Qwen replicates; other models also show the effect | Evidence for **broader source sensitivity** strengthens, still limited to four models and this stimulus set. |
| Cohort A Qwen does not replicate | The Phase 2 Qwen effect is treated as **unstable / pilot-specific**. Stated plainly; not preserved rhetorically. |
| Multiple models show the common-pair effect | Evidence for **source-identity sensitivity** across models strengthens. |
| Calibrated effects appear but common-pair effects do not | **Model-specific elicited preference** may matter more than fixed source identity. |
| Common-pair effects appear but calibrated effects do not | The specific labels, not elicited preference, may drive the effect; direct calibration may not predict behavior. |
| Effects in **both** corrective and harmful directions | More compatible with **truth-blind source weighting** (H4). |
| Effects in corrective only | **Do not claim truth-blind weighting.** Report the asymmetry as an asymmetry. |
| Cohort A replicates but **Cohort B** (relation-balanced) does not | **Relation composition** is an important explanation for the effect: it holds on a naturally-composed item sample but weakens under enforced relation balance. Cohort A remains the replication verdict; Cohort B qualifies its generality. |
| Cohort A replicates and Cohort B also shows the effect | The effect **survives stronger compositional control** — a stronger generality claim than Cohort A alone supports. |
| Effects vanish after shared-item (Cohort C) restriction | **Item composition** becomes an important explanation for Phase 2's cross-model pattern. |
| Contrast saturated | **SATURATED/UNINFORMATIVE**; not evidence of absence (§30). |
| Analysis not estimable after fallback ladder | Reported as **not estimable**; not converted into a null. |

Additional standing rules: no "proves"; no "demonstrates universal"; no
architecture-effect language; no p-value-only claims; no retrospective
hypothesis changes; "significant" only alongside the exact test and
number. Every interpretation above is falsifiable by a stated pattern.

## 40. Known limitations

Stated in advance, and to be carried into the final report regardless of
outcome:

- Four models is a small, non-random sample of families; no generalization
  to LLMs as a class.
- Scale is deliberately held roughly constant, so scale effects are
  untested.
- Evidence is prompt-injected and templated, not retrieved — ecological
  validity is limited by design.
- Source preference is **directly elicited stated** preference for one
  prompt, not validated latent trust or credibility.
- Source labels are abstractions ("a government website"), not real
  documents with content, formatting, or metadata.
- PopQA short-answer factual questions in English only; four relations.
- **Phase 3 is powered only against large effects** (§23); moderate
  effects may go undetected, and nulls must be read with the interval, not
  as absence.
- Ceiling regimes may again limit sensitivity for some models, and no
  feasible sample size fixes that.
- The Phase 2 estimate that motivates the replication is likely inflated;
  Phase 3's Qwen estimate should be expected to shrink even under genuine
  replication.
- Self-reported confidence remains unvalidated.
- Shared-cohort comparisons still cannot equate models' *knowledge states*
  on shared items, only the questions.
- Deterministic decoding fixes one decoding regime; sampling behavior is
  untested.

## 41. Phase 3 execution stages

| Stage | Content | Gate |
|---|---|---|
| **3A** | Design (this document) | **Design decisions approved**; one open item (§42.1: exact new-model releases) |
| **3B** | Implementation, tests, dry/synthetic validation only | No real model. Dummy adapter only. All new logic unit-tested. |
| **3C** | Exact new-model releases/revisions; verification that the frozen Qwen/Llama artifacts still load; new-model calibration; cohort construction (A, B, C); **pre-run freeze** (§36) | Freeze manifest archived and hashed |
| **3D** | Real baseline screens, source calibration, C0 + evidence-condition generations | Only after 3C is frozen |
| **3E** | Pre-specified analysis per §26-§30 | Only analyses declared in the freeze manifest |
| **3F** | Final report, archives, repository freeze | Frozen interpretation document |

> **NO REAL PHASE 3 MODEL MAY RUN BEFORE 3C IS FROZEN.**

Source calibration for the **new models** is deliberately placed in 3C
rather than 3D because its output is required to select their `M1`/`M2`
roles before the freeze. Calibration is not an outcome variable; it is a
design input. (Qwen and Llama need no calibration to proceed — their pairs
are already frozen from Phase 2, §20.1 — so their optional DIAGNOSTIC
stability calibration may run in 3C or 3D without affecting the frozen
design.) Calibration is still real model execution, so it may only occur
after 3B is
complete and after the researcher has approved §42, and its results must
be archived before the freeze manifest is sealed.

## 42. Researcher approval required before Phase 3B

### Resolved by researcher decision (no longer open)

The following were open in the Phase 3A draft and are now **frozen** in
this revision; they are listed so the record shows they were decided, not
defaulted:

| Decision | Resolution | Section |
|---|---|---|
| Primary confirmatory scope | Sole primary test = **Cohort A** Qwen corrective replication; all other source tests are secondary (Holm) | §4, §26.1, §28 |
| Cohort architecture | Three distinct cohorts (A direct replication / B relation-balanced / C shared), with Cohort A explicitly decoupled from relation balance | §15, §16 |
| Source pair for Qwen/Llama | Frozen Phase 2 pairs reused for the confirmatory contrast; fresh calibration is DIAGNOSTIC only | §20.1 |
| Common source pair | Frozen: `a government website` vs. `an anonymous online forum post`, with the Qwen coincidence documented | §19 |
| **Additional model families** | **APPROVED at family level: Mistral-7B-Instruct and Gemma-2-9B-it; exact releases verified/frozen in Phase 3C; stop-and-return if runtime requirements fail** | §7 |
| **Qwen/Llama revision policy** | **APPROVED: reuse the exact Phase 2 pinned revisions (`a09a354…`, `0e9e39f…`); never re-resolve for replication; unloadable artifact = infrastructure failure** | §7, §34 |
| **Compute budget** | **APPROVED: bounded ceiling — 5,376 nominal condition slots, 8,000 baseline-generation cap, plus calibration; nominal slots ≠ unique generations** | §23 |
| Cohort B cell sizes | Target 8, minimum balanced 6, downsample-to-common, eligibility-limited fallback, no backfilling | §32 |
| Screening procedure | Deterministic 250-candidate blocks, 2,000/model ceiling, reserve of 2, eligibility-only stopping, Cohort A supply criterion ≥ 34/stratum fresh Qwen KW | §11 |
| Margin-stratum stability | Recompute per block from the full eligible pool; freeze at stop; never recompute after outcomes | §11 |
| Practical-effect threshold | **None invented**; four-category replication classification with an operationalized compatibility diagnostic | §37 |
| Paired statistics | Single CI (Tango score), single test (exact McNemar/binomial), mandatory paired-cell + discordance reporting, bootstrap sensitivity-only | §26 |
| Margin-analysis fallback | Ordinary logit if estimable → Firth → NOT ESTIMABLE; vague rank fallback removed | §26 |
| Condition structure | Seven nominal conditions with deterministic deduplication of prompt-identical conditions | §22 |

### Still unresolved — approval required before Phase 3B

Only genuinely open items appear here. No decision is listed merely to
populate the section.

### 42.1 Exact releases and revisions for the two approved model families

The **families are approved** (§7). What remains genuinely open is the
release-level identity, which cannot be settled from repository evidence
and must be verified against the live Hub in Phase 3C:

- **Open:** which exact `mistralai/Mistral-7B-Instruct-v0.x` release and
  which exact `google/gemma-2-9b-it` repository/release to pin; their
  exact resolved commit SHAs; confirmation of gated access for Gemma.
- **Must be verified in 3C before freezing:** tokenizer/model revision
  equality, licensing/access, float16 feasibility on the RTX-3090-class
  setup, no-quantization feasibility, and compatibility with the existing
  sequence-logprob scoring pipeline.
- **Why it still matters:** instruct-release differences are substantive
  behavioral differences, so "the Mistral family" does not by itself
  determine the stimulus-response system being measured.
- **If a family fails the runtime requirements:** stop and return to
  researcher approval (§7, §34). Never substitute another model silently.
- **Frozen once resolved:** the exact repository IDs and SHAs in the 3C
  manifest.

*(No other decision remains open. Cohort design, source pairs, common
pair, cell sizes, screening rule, statistics, replication criteria,
revision policy, model families, and compute budget are all frozen above.)*

## 43. Implementation implications (planning only — nothing implemented)

Inspecting the current architecture, Phase 3B will likely need the
following. **None of this is implemented, and none of it should be
implemented before the remaining §42 item is resolved.**

- **New Phase 3 config namespace** — e.g. `configs/phase3/`, with per-model
  pilot/model configs and a `runs/phase3/<model>/` isolated path namespace
  (the existing `.gitignore` `runs/` rule already covers it).
- **Expanded condition builder** — `experiment/conditions.py` currently
  builds exactly C0-C4 from a single preferred/dispreferred pair. Phase 3
  needs the seven-condition structure with an `arm` concept, two source
  pairs, and the KC/KW-dependent model-specific conflict resolution.
- **Deterministic deduplication logic** (§22) — detect prompt-identical
  `M`/`K` conditions before generation (for Qwen, both `M1` and `M2`),
  store one aliased observation, and expose it to the analysis layer as
  *one observation referenced by two contrasts* so it is never
  double-counted.
- **Three-cohort sampling** — (a) **Cohort A**: margin-stratum-only quota
  (32/32/32) over *fresh* Qwen KW items with the Phase 2 item-id exclusion
  list, no relation quota; (b) **Cohort B**: two-dimensional relation ×
  margin quota with the §32 downsample-to-common-count /
  eligibility-limited ladder (target 8, minimum 6); (c) **Cohort C**:
  deterministic cross-model intersection. `data/sampling.py` currently
  spreads across margin bins only.
- **Cohort-membership and observation-reuse layer** — records must carry
  *all* cohort memberships and the contrasts that reference them, so one
  stored generation can serve several cohorts without being regenerated or
  double-counted in any single estimate (§23).
- **Block-wise screening driver** (§11) — deterministic 250-candidate
  blocks up to the 2,000/model ceiling, with the eligibility-only
  early-stop check and per-block quota accounting.
- **Cohort C construction** — a new deterministic module that
  intersects per-model baseline eligibility across models and applies a
  relation-balanced quota, with cohort tags on records.
- **Margin standardization utilities** (§14) — within-(model × group) rank
  and z-score transforms for cross-model analysis.
- **Paired-analysis extensions** — `analysis/paired_comparison.py` already
  computes discordance and the exact test; Phase 3 adds the paired
  risk-difference **Tango score confidence interval**, a bootstrap
  alternative, and the §30 saturation flags.
- **Robust/penalized regression** — Firth logistic fallback and the §26
  ladder; `analysis/regression.py` currently raises on non-convergence,
  which is correct but insufficient for Phase 3's pre-specified fallbacks.
- **Hierarchical/cross-model analysis** — model-fixed-effects + item
  random intercept, GEE fallback, and a heterogeneity display.
- **Multiplicity utilities** — Holm correction over declared families.
- **Extended provenance manifest** (§36) — richer than Phase 2's, covering
  cohorts, arms, both source pairs, per-cell realized counts, and the
  analysis status table.
- **Additional tests** — condition-structure invariants, duplicate-collapse
  behavior, quota-sampling determinism and downsampling ladder,
  shared-cohort determinism and outcome-independence, margin-standardization
  correctness, paired-CI correctness on boundary cases (including the
  Phase 2 30/30 and zero-discordance cases), Firth fallback triggering,
  and Phase 3 config invariants.

## 44. Analysis status table

Fixed before Phase 3 runs. A reader can determine every analysis's status
from this table without waiting for results.

| Analysis | Cohort | Outcome | Contrast | Status | Multiplicity family |
|---|---|---|---|---|---|
| **Qwen corrective frozen-pair replication (RQ-A)** | **A** (96 fresh Qwen KW items, 32/32/32 strata) | `context_adopted` | frozen Phase 2 Qwen pair (`government website` vs. `anonymous online forum post`), KW, @ `a09a354…` | **PRIMARY CONFIRMATORY — sole primary test** | **Primary (single test; no correction required)** |
| Relation-balanced Qwen source effect (generalization, **not** a direct replication) | **B**, Qwen | `context_adopted` | frozen Phase 2 Qwen pair, KW and KC | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Llama corrective source effect | **B**, Llama | `context_adopted` | frozen Phase 2 Llama pair (`government website` vs. `social media post`), KW | SECONDARY CONFIRMATORY | Secondary (Holm) |
| New-model corrective calibrated contrasts (Mistral, Gemma) | **B**, M3/M4 | `context_adopted` | that model's freshly calibrated pair, KW | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Harmful model-specific source effect, per model (H4) | **B** | `context_adopted` | that model's `M1` vs. `M2`, KC | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Common fixed-source contrast, per model (H2a) | **B** | `context_adopted` | common A vs. common B, conflict cells | SECONDARY CONFIRMATORY | Secondary (Holm); **Qwen counted once** — same observations as its frozen-pair contrast (§19, §22), never independent corroboration of Cohort A |
| Cross-model model × source interaction (RQ-B) | **B + C** | `context_adopted` | model × source | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Shared-cohort cross-model contrasts | **C** | `context_adopted` | model-specific and common | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Parametric strength (H1), per model | **B** (and **A** for Qwen, descriptively) | `context_adopted` | continuous margin | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Leave-one-relation-out | **B**; also run on **A** as a robustness check | `context_adopted` | contrast, per relation dropped | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Country-only sensitivity | **B** | `context_adopted` | contrast, country only | SECONDARY CONFIRMATORY | Secondary (Holm) |
| Tentative answer content vs. commitment | All cohorts | tentative content | by source and group | SECONDARY (mechanistic) | none |
| Common-arm agreement control | **B** | `context_adopted` | common A vs. B, agreement cells | DIAGNOSTIC | none |
| Source × parametric strength (H3), nonlinear/intermediate-strength | **B** | `context_adopted` | margin × source, nonlinear | EXPLORATORY | none |
| Cohort A relation distribution + dominance flag (§15.1) | **A** | composition | most-frequent-relation share | DIAGNOSTIC (non-gating) | none |
| Calibration stability (Phase 2 → 3) | Qwen, Llama | elicited pair | Phase 3 calibrated pair vs. frozen Phase 2 pair | **DIAGNOSTIC** — may never redefine the confirmatory contrast (§20.1) | none |
| Relation-specific descriptives | All cohorts | `context_adopted` | per relation | DIAGNOSTIC | none |
| Margin-bin displays | All cohorts | `context_adopted` | per stratum | DIAGNOSTIC | none |
| Parsing-failure rate check (§34) | All | malformed rate | per model | DIAGNOSTIC (gating) | none |
| Ceiling/floor diagnostics (§30) | All | discordance, boundary, CI width | per contrast | DIAGNOSTIC | none |
| Abstention rates | All | `Decision == uncertain` | per condition | DIAGNOSTIC | none |
| `parsed_answer_accuracy` | All | `final_correct` | per condition | DIAGNOSTIC | none |
| Self-reported confidence | All | confidence | per condition | EXPLORATORY | none |
| C0 reproducibility check | All | exact match | C0 vs. baseline | DIAGNOSTIC (gating, §34) | none |

Any analysis not in this table, conceived after Phase 3 data exists, is
**post-hoc** and must be labeled as such in the final report — it is
EXPLORATORY by definition and belongs to no multiplicity family.

**Multiplicity families are explicit and disjoint:** the **Primary**
family contains exactly one test (Cohort A, no correction required); the
**Secondary** family is Holm-corrected within itself; EXPLORATORY and
DIAGNOSTIC analyses are uncorrected and can never be reported as
confirmatory, however small their p-values.

---

**Phase 3A status: design approved except for the exact releases/revisions
of the two approved new model families (§42.1). Not implemented, not
frozen for execution, not run. The pre-run freeze happens in Phase 3C. No
Phase 3 model has been executed and no Phase 3 result exists.**

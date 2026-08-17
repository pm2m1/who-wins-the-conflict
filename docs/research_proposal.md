# Research Proposal

## Title

**Who Wins the Conflict? Parametric Preference Strength and Source Preference
in LLM Evidence Use**

Alternative title: *When Sources Override Memory: How Source Preference
Mediates Context-Memory Conflict in LLMs*

**Status:** the first pilot (`Qwen/Qwen2.5-7B-Instruct`) has been executed
and frozen; see `docs/qwen_pilot_results.md` for results. This proposal is
retained as originally written.

## Motivation

Four strands of prior work, synthesized in
`docs/phase1_literature_synthesis.md`, jointly suggest that faithful use of
external evidence by LLMs depends on more than whether correct information is
available in context. It plausibly depends on (a) how strongly the model
already prefers its own parametric answer, and (b) who the evidence is
attributed to. No single prior study measures both factors jointly, on the
same conflicting-evidence trials, holding evidence content fixed.

## Related-work boundary and research gap

Generic context-versus-parametric-memory conflict has already received
significant research attention (e.g., Tug-of-War between Knowledge,
FaithfulRAG, CARE, and other task-dependent context-memory conflict or
multi-source/user/document assertion conflict studies). We therefore do not
claim novelty for the general observations that:

- LLMs experience knowledge conflicts, or
- LLMs sometimes follow context instead of memory, or
- source credibility can affect decisions.

Those phenomena already have prior work. Our narrower working gap is:

> Does source preference alter the threshold at which an LLM overrides its
> own parametric preference?

More formally, we investigate whether

    P(follow conflicting context) = f(parametric preference strength, source preference, evidence truthfulness)

can be characterized empirically, and in particular whether source
preference and parametric preference strength **interact** rather than
acting as independent, additive effects. This is a working research gap,
not a claim of proven novelty. We use language such as "we investigate the
interaction between..." and "the pilot focuses on..." rather than "this is
the first work to..." throughout this project's documentation.

## Research questions

**RQ1 — Parametric preference strength.** How does the strength of an LLM's
pre-existing parametric preference affect its willingness to adopt
conflicting external evidence?

**RQ2 — Source effect.** Holding evidence content exactly constant, does
changing only its attributed source change the probability that the model
follows that evidence instead of its baseline parametric answer?

**RQ3 — Truth asymmetry.** Does source preference amplify both (a)
beneficial correction of incorrect parametric answers, and (b) harmful
override of correct parametric answers?

**Exploratory question.** When baseline knowledge and supplied evidence
disagree, does the model become appropriately uncertain, or does one
information source simply win?

## Hypotheses

Full statements are in `docs/phase2_research_design.md`. Summary:

- **H1 (parametric-strength):** larger parametric margin in favor of the
  memory answer predicts lower probability of context adoption.
- **H2 (source-preference):** for identical evidence content, a preferred
  source is adopted more often than a dispreferred source.
- **H3 (interaction):** the source effect may be strongest at intermediate
  parametric preference strength, weaker at both extremes.
- **H4 (truth-blind source effect):** if source preference behaves like a
  general weighting heuristic rather than truth discrimination, a preferred
  source may increase both corrective and harmful overrides. This is a
  hypothesis to be tested, not an assumed result.

## Independent variables

- Parametric preference strength: conflict-specific length-normalized
  sequence log-probability margin between the memory answer and the
  conflicting context answer (`parametric_margin`).
- Source role: preferred vs. dispreferred, from independent direct
  pairwise calibration (`source_role`).
- Evidence truth status: true vs. false relative to the gold answer
  (`evidence_truth`).
- Knowledge group: known-correct (KC) vs. known-wrong (KW), determined by
  the model's own clean baseline answer (`knowledge_group`).

## Dependent variables

- Context adoption (binary): whether the final answer matches the
  contextual/conflicting answer (`context_adopted`).
- Final answer correctness (`final_correct`).
- Decision (`answer` vs. `uncertain`) and self-reported confidence
  (exploratory only).
- Derived rates: Context Adoption Rate (CAR), Harmful Override Rate (HOR),
  Corrective Override Rate (COR), and the source-effect deltas
  `Delta_harm` and `Delta_correct`. Definitions in
  `docs/phase2_research_design.md`.

## Pilot setup

A controlled pilot on PopQA short-answer factual questions, using prompt-
injected controlled evidence (no retrieval system) across five conditions
(C0-C4) crossing evidence truth and source role, applied separately to
model-specific KC and KW item pools. Target: approximately 60 items per
model (30 KC, 30 KW) x 5 conditions x 2 models ≈ 600 generations. Full
detail in `docs/phase2_research_design.md` and `docs/pilot_protocol.md`.

## Evaluation

Deterministic short-answer normalization and exact/alias matching (no
LLM-as-judge). Primary statistical analysis is exploratory logistic
regression on conflict trials only. Details in `docs/methodology.md`.

## Expected contribution

A controlled characterization of how source preference and parametric
preference strength interact during conflicting evidence use, at pilot
scale. This is intended to establish whether the effects are large enough
and stable enough to justify a larger-scale study — not to resolve the
mechanism, and not to demonstrate a general solution to hallucination.

## Limitations

- Pilot scale (~600 generations, 2 models) is not intended to provide
  publication-level statistical power; see Go/No-Go criteria in
  `docs/phase2_research_design.md`.
- Evidence is prompt-injected and templated, not retrieved from real
  documents — this isolates evidence integration from retrieval quality,
  chunking, and reranking, at the cost of ecological validity (see
  WildHallucinations as a future ecological-validation direction in
  `docs/reference_implementations.md`).
- Source preference is calibrated via direct pairwise elicitation only;
  this may not equal implicit/behavioral source preference, which is
  future work.
- Self-reported confidence is exploratory only and is not validated as a
  calibrated probability.
- Ordinary logistic regression on pilot-scale data is exploratory; it is
  not a substitute for a pre-registered, adequately powered, mixed-effects
  analysis at scale.

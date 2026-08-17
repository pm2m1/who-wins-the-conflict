# Qwen2.5-7B-Instruct Pilot Results

This document reports the first, frozen pilot run of the controlled
context-memory conflict experiment described in
`docs/phase2_research_design.md` and `docs/research_proposal.md`. The
design documents are the pre-result specification and are not rewritten
here to fit the observed data; this document records what happened when
that design was executed once, with one model.

## Status and scope

- Frozen 60-item / 300-generation pilot with `Qwen/Qwen2.5-7B-Instruct`.
- **One model only.** `Llama-3.1-8B-Instruct` has not been run.
- Pilot scale, not publication scale. No second-model replication has
  been run yet, so no general, multi-model, or cross-architecture
  conclusion is drawn here.
- This experiment is **frozen**: the results below were computed once
  from the recorded generations and are not re-derived from a rerun.

## Frozen provenance

- Repository commit used for the run:
  `1483bf5f2444a7108307cf4368bb39311f0e0548` ("Make source calibration
  output strict").
- Model: `Qwen/Qwen2.5-7B-Instruct`, exact Hugging Face revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Precision: float16. Quantization: none.
- GPU: NVIDIA GeForce RTX 3090, 24GB VRAM. Scratch memory cap used for
  this run only, `{0: "22GiB"}` — a machine-specific setting, not a
  committed default (`configs/models.yaml` keeps `max_memory: null`; see
  `docs/decisions.md`, "Support reproducible model memory limits").
- Environment observed on the GPU host: Python 3.10.13, PyTorch
  2.11.0+cu128 (CUDA 12.8), Transformers 5.13.1, Datasets 4.0.0,
  Accelerate 1.14.0.
- PopQA resolved revision: `098765c79ea10a2cb19c828324e33281b8336ec0`.

**Input/output SHA256 hashes** (as observed and independently
hash-verified by the researcher on the GPU host):

| Artifact | SHA256 |
|---|---|
| Pilot trials | `d3e6831efa6a49a7f5f6d555507462185d055c0c1d01de9254f24b3b4e518d6a` |
| Pilot results | `06028ea8cef91c36f31eb6ecd918c27799c5ed9fd4a9385f94f7a03cd6b287b0` |
| Baseline JSONL | `ea784f2bc082fdbc708f7529c48c8d573ba3bd621ac9c8ef0c77827668c4c0b4` |
| Baseline exclusions JSONL | `1ff6a968db4283ba05b28cdadd4263908829a0b3f6ab549973080cbc3b20f05e` |
| Source calibration archive | `2c35119733abb5082be52a255223c03455d91313b8340cd271d3e18acca07f73` |
| Country-only sensitivity text | `3793c5a87051b695e9423e0cc087e608d1f002c900b95d5e42b56b4c74ec9c4b` |
| Analysis output | `ceaecbc151acde99adbb1ed3254034543dc0a662e56596039a57a401d3dd6dda` |

**Archive hashes** (downloaded to the researcher's Windows machine;
server/Windows SHA256 verified equal — archives themselves are not
committed to this repository):

| Archive | SHA256 |
|---|---|
| `qwen-pilot-pre-run.zip` | `a0d3831dbca6273d0e8360aea4922b883ac1fedd7206a142cc6095a1ad2997de` |
| `qwen-pilot-post-run.zip` | `487da996d14425ffc5e02663952143e0f4b2311bf1d282fe7f01d6e5e2c286d4` |
| `qwen-pilot-analysis-final.zip` | `7aeec4fa16cd6de93da4d5d01db4ed4603b2033d080eae97f3f1b91df1cad674` |

## Experimental sample

The screen used the committed `primary_conflict_relations` targeted pool
(`docs/decisions.md`, "Support targeted primary conflict screening").
500 targeted candidates were screened, producing 498 baseline records
(KC=134, KW=61, `manual_review_flagged`=4, malformed exclusions=2 — item
ids `857875` and `5521585`).

The final pilot selected **60 unique factual items**: 30 KC, 30 KW, each
run through all five conditions (C0-C4), for **300 generations**.

**Margin balance** (exact):

| | low | medium | high |
|---|---|---|---|
| KC | 10 | 10 | 10 |
| KW | 10 | 10 | 10 |

**Relation composition of the 60 selected items** — reported as a
limitation, not glossed over:

| | country | sport | mother | place of birth |
|---|---|---|---|---|
| KC (30) | 18 | 12 | — | — |
| KW (30) | 14 | 2 | 12 | 2 |

The selected sample is **not relation-balanced**: KC is entirely
country/sport, and KW is dominated by country/mother. This reflects the
targeted `PRIMARY_RELATIONS` pool intersected with margin-bin-balanced
sampling on 60 real items, not a deliberate relation quota; see
"Limitations" below.

Source roles, confirmed by the researcher after direct calibration (not
automatically selected): `preferred_source = "a government website"`,
`dispreferred_source = "an anonymous online forum post"`.

This produced **120 genuine conflict trials** (C1/C2 for the 30 KW items,
C3/C4 for the 30 KC items) out of 300 total generations; the remaining
180 are agreement-control or C0-baseline trials (docs/phase2_research_design.md,
"Conflict vs. agreement").

## Reproducibility checks

- **C0 reproduced the corresponding frozen baseline records exactly**:
  60/60 records had identical `raw_generation`, parsed answer, decision,
  and confidence between the baseline screen and the C0 pilot condition
  (0 mismatches). This is a real, executed reproducibility check, not an
  assumption.
- 300 unique item-condition records; exact condition counts (60 items x
  5 conditions = 300).
- No `manual_review` records remain among the final 300 pilot generations
  (the 4 `manual_review_flagged` baseline items and the 2 malformed
  exclusions were, by construction, not part of the 60-item selection).

## Primary descriptive results

The primary outcome is `context_adopted` — the model's **committed**
answer (`Decision: answer`) matching the conflicting context's asserted
answer. This is distinct from `parsed_answer_accuracy` (textual
`Answer:`-field correctness, which can be true even under
`Decision: uncertain`) — see `docs/methodology.md`, "Metric semantics."

| | preferred source | dispreferred source | delta |
|---|---|---|---|
| Harmful override (KC, C3/C4) | 6/30 = 0.2000 | 3/30 = 0.1000 | **+0.1000** (+10.0 pp) |
| Corrective override (KW, C1/C2) | 25/30 = 0.8333 | 17/30 = 0.5667 | **+0.2667** (+26.7 pp) |

Overall conflict abstention rate across all 120 conflict trials:
69/120 = 0.575.

## Secondary paired source checks

Each factual item appears under both the preferred- and dispreferred-source
condition, so a paired exact test on the discordant pairs is a natural
check alongside the primary aggregate rates. **These are secondary paired
inferential checks, not preregistered primary hypothesis tests** — the
design documents did not preregister them as such
(`src/conflict_eval/analysis/paired_comparison.py` implements the
reusable, generic version of this check).

| | both | preferred-only | dispreferred-only | neither | exact two-sided p |
|---|---|---|---|---|---|
| KC harmful, C3 vs. C4 | 3 | 3 | 0 | 24 | 0.25 |
| KW corrective, C1 vs. C2 | 17 | 8 | 0 | 5 | 0.0078125 |

Interpretation, restrained: the corrective source effect is the clearest
pilot signal (8 discordant pairs, all favoring the preferred source, exact
p = 0.0078). The harmful source effect is descriptive/suggestive only (3
discordant pairs, exact p = 0.25) — it is **not** statistically
established at this sample size.

## Planned country-only sensitivity

The frozen pre-run manifest recorded a country-only sensitivity analysis
as planned before outcome analysis (SHA256
`3793c5a87051b695e9423e0cc087e608d1f002c900b95d5e42b56b4c74ec9c4b`).

| | n | preferred | dispreferred | delta | exact p |
|---|---|---|---|---|---|
| KC harmful, country only | 18 | 2/18 = 0.1111 | 1/18 = 0.0556 | +0.0556 | 1.0 |
| KW corrective, country only | 14 | 10/14 = 0.7143 | 6/14 = 0.4286 | +0.2857 | 0.125 |

The corrective effect size is directionally similar restricted to
`country`-relation items alone (+28.6 pp vs. +26.7 pp for all relations),
but n=14 is underpowered and this is a **sensitivity analysis, not an
independent confirmation**. For harmful override, the country-only effect
is smaller and provides no convincing inferential evidence.

## Parametric-margin results

**Margin descriptives** (length-normalized log-probability margin,
`docs/methodology.md`, section 4):

| | n | min | median | max | mean |
|---|---|---|---|---|---|
| KC | 30 | -0.2001 | 13.9619 | 25.8750 | 13.8185 |
| KW | 30 | -0.3541 | 4.3116 | 16.5391 | 5.8681 |

**Descriptive low/medium/high patterns** (preferred-source adoption rate
by margin bin):

| | low | medium | high |
|---|---|---|---|
| Corrective (KW) adoption | 10/10 | 8/10 | 7/10 |
| Harmful (KC) adoption | 3/10 | 2/10 | 1/10 |

Both patterns move in the direction H1 predicts (stronger parametric
preference, i.e. higher margin, associated with less adoption of
conflicting evidence) — **descriptively suggestive, not confirmed** by
the continuous regression below.

**Planned exploratory ordinary logistic regression** (120 conflict
observations, `Y ~ B * S + S * T + B * T`;
`src/conflict_eval/analysis/regression.py`):

| term | coef | p |
|---|---|---|
| Intercept | -1.4018 | 0.208 |
| B (parametric margin) | -0.0629 | 0.436 |
| S (preferred source) | 1.4369 | 0.267 |
| B:S | -0.0532 | **0.563** |
| T (corrective vs. harmful) | 2.7148 | 0.019 |
| S:T | 0.6054 | 0.584 |
| B:T | -0.1138 | 0.221 |

**The B:S source-moderation interaction — the primary interaction of
interest for H3 — is not supported by this pilot** (p = 0.563). T's
p = 0.019 is **not** portrayed here as a clean causal "truth effect": T
is entangled with KC vs. KW in this design (every corrective trial is KW,
every harmful trial is KC), KC and KW have substantially different margin
distributions (see table above) and different relation composition (see
"Experimental sample"), so a T coefficient conflates the source-truth
manipulation with these structural differences. Additionally, this
ordinary logit uses repeated observations from the same 60 factual items
(each item contributes up to 2 conflict observations) and does not model
within-item dependence — all coefficient p-values here are exploratory,
not confirmatory.

## Hypothesis status

**H1** (stronger parametric preference reduces adoption of conflicting
evidence): **descriptively suggestive, not confirmed.** Both margin-bin
patterns move in the predicted direction (corrective: 10/10 -> 8/10 ->
7/10; harmful: 3/10 -> 2/10 -> 1/10 from low to high margin), but the
continuous regression's B coefficient is not significant (p = 0.436).

**H2** (changing source attribution while holding evidence content fixed
changes context adoption): **supported most clearly for corrective
evidence in this pilot.** Corrective committed adoption: 83.3% (preferred)
vs. 56.7% (dispreferred), paired discordance 8 vs. 0, exact p = 0.0078125.
Do not overgeneralize beyond this model and this pilot.

**H3** (source effect is strongest at intermediate parametric strength /
source moderates the parametric-margin effect): **not supported.** B:S
p = 0.563.

**H4** (preferred sources increase both beneficial correction and harmful
override): **corrective side supported by the pilot; harmful side only
descriptively suggestive.** Corrective delta = +26.7 pp. Harmful delta =
+10.0 pp, paired p = 0.25.

## Post-hoc exploratory tentative-answer decomposition

**POST-HOC. EXPLORATORY. MECHANISTIC/DIAGNOSTIC — not part of, and never
merged into, the primary `context_adopted` definition.** This analysis
was conceived after inspecting the uncertainty behavior in the data, not
planned in advance.

The response format requires an `Answer:` field even when
`Decision: uncertain` (`prompts/baseline.txt`), so the parsed answer from
an uncertain response is called **"tentative answer content"** here, not
a committed final answer — it is not equivalent to `context_adopted`.

Among the 69 uncertain conflict trials: tentative context = 57, tentative
memory = 9, other = 3.

**KW corrective conflicts:**

| | tentative context | committed context adoption |
|---|---|---|
| C1 (preferred) | 30/30 | 25/30 |
| C2 (dispreferred) | 30/30 | 17/30 |

Paired tentative context: both = 30, preferred-only = 0,
dispreferred-only = 0, neither = 0. In this sample, source attribution did
not change *tentative* factual content in KW at all, but did change
whether the model *committed* to that answer.

**KC harmful tentative context:**

| | tentative false-context selections |
|---|---|
| C3 (preferred) | 21/30 |
| C4 (dispreferred) | 27/30 |

Paired: both = 21, preferred-only = 0, dispreferred-only = 6, neither = 3.
Exact paired p for this post-hoc tentative-content comparison = 0.03125.
**This post-hoc p-value is not a primary hypothesis test.** Notably, the
KC tentative-content direction (more tentative false-context selection
under the *dispreferred* source) differs from the committed-adoption
direction (more committed adoption under the *preferred* source). A
defensible exploratory interpretation: source identity may affect
commitment to conflicting evidence differently from tentative answer
content. This is not a proven cognitive mechanism and requires
independent replication before it is treated as more than a hypothesis
for a follow-up design.

## Limitations

- One model (`Qwen/Qwen2.5-7B-Instruct`); no second-model replication yet.
- 60 items, 300 generations — pilot scale, not publication scale.
- Evidence is prompt-injected and templated, not retrieved from real
  documents (`docs/decisions.md`, "Why no full RAG retriever").
- Source calibration is direct stated preference, not latent/behavioral
  preference (`docs/reference_implementations.md`).
- Sources are label abstractions ("a government website") rather than
  actual documents with real content, formatting, or metadata.
- Relation composition of the selected 60 items is imbalanced (KC:
  country/sport only; KW: dominated by country/mother) — see "Experimental
  sample."
- `preferred_source`/`dispreferred_source` were selected for this model
  specifically, from an empirically tied bottom tier in the dispreferred
  case (the dispreferred source is **not** claimed to be the uniquely
  least-preferred source among the six labels — see
  `docs/decisions.md`, "Freeze the first Qwen pilot after validated
  analysis").
- The ordinary logistic regression ignores within-item dependence
  (repeated observations per factual item).
- Self-reported confidence is exploratory and not validated as calibrated.
- The tentative-answer-content analysis is post-hoc, not preregistered.
- No second-model replication has been run yet; nothing here supports a
  general or cross-model conclusion.

## Frozen conclusion

In this Qwen2.5-7B-Instruct pilot, source attribution had its clearest
effect when external evidence corrected an initially wrong parametric
answer: committed adoption increased from 56.7% for the dispreferred
source to 83.3% for the preferred source. The corresponding paired
discordances were 8 versus 0 (exact two-sided p=0.0078). The harmful
override difference was smaller and not statistically convincing.
Descriptive margin-bin patterns were consistent with stronger parametric
preferences reducing committed adoption, but the planned continuous
margin × source interaction was not supported. Post-hoc analysis suggests
source attribution may affect commitment differently from tentative answer
content, but that mechanism requires independent replication.

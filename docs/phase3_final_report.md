# Phase 3 — Final Report

**Status: FROZEN. Confirmatory results under the preregistered Phase 3C
design.** Every number here was produced by the analysis the freeze
declared, over observations generated from a study state sealed before any
outcome existed. Nothing was added, dropped, or re-specified after results
appeared.

| | |
| --- | --- |
| Frozen design | `docs/phase3_scaled_study_design.md` (commit `d684f39`) |
| Pre-run freeze manifest | `afa5a426bb88baeace13490b16ce34be85896da175654151aebd5518994fbd97` |
| Observations | 4 197 evidence-condition generations, all verified |
| Effect estimate | paired risk difference |
| Interval | 95% Tango matched-pair score interval |
| Test | exact two-sided McNemar / exact binomial on discordant pairs |
| Multiplicity | Holm–Bonferroni, secondary family only |

---

## 1. What this study asked

Holding the *content* of a piece of evidence exactly constant, does
changing only its **attributed source** change how often a language model
abandons its own answer and adopts the evidence?

The outcome is `context_adopted`: true only when the model commits
(`Decision: answer`) **and** its answer matches the conflicting context's
asserted answer. Answer text produced under `Decision: uncertain` never
counts toward it, and is analysed separately as a mechanistic outcome.

---

## 2. Phase 2 pilot vs Phase 3 confirmatory — a hard separation

These are different studies and are never pooled.

**Phase 2 (pilot, 30 items/model, exploratory).** A Qwen pilot found
corrective adoption of 25/30 under a preferred source vs 17/30 under a
dispreferred source — Δ = +26.7 pp, paired exact p = 0.0078. A Llama
replication found corrective cells at 96.7% / 100.0%, which was
uninterpretable as a null because both arms sat at the ceiling. That
observation is why the frozen design carries a mandatory saturation
diagnostic at all.

**Phase 3 (confirmatory).** Four models, cohorts constructed from an
outcome-blind baseline screen, every analysis and status declared before
generation. The Phase 2 estimate is treated as a **pilot estimate very
likely inflated by winner's curse**, and is used only as a compatibility
*diagnostic* — never as a target the Phase 3 result must match, and never
able to overturn a Phase 3 finding.

---

## 3. The primary result

**Sole primary confirmatory test:** Cohort A · `Qwen/Qwen2.5-7B-Instruct`
@ `a09a35458c702b33eeacc393d103063234e8bc28` · 96 **fresh** KW items
(32/32/32 margin strata, Phase 2 pilot items excluded) · KW corrective
conflict · frozen Phase 2 source pair (`a government website` preferred vs
`an anonymous online forum post` dispreferred) · `context_adopted`.

| Quantity | Value |
| --- | --- |
| n (complete pairs) | **96** |
| both | 56 |
| preferred-only | **24** |
| dispreferred-only | **0** |
| neither | 16 |
| discordant pairs | **24** |
| adoption under preferred source | **0.833** |
| adoption under dispreferred source | **0.583** |
| **paired risk difference** | **+25.00 pp** |
| **95% Tango matched-pair CI** | **[+17.41, +34.51] pp** |
| **exact two-sided p** | **1.19 × 10⁻⁷** |

### Classification: **FULL CONFIRMATORY REPLICATION**

Evaluated in the frozen order, which checks informativeness *first*:

1. **(f) not saturated**, and 24 discordant pairs ≥ 5 → the test was
   genuinely informative, so the remaining criteria may be considered;
2. **(a)** Δ > 0;
3. **(c)** the 95% interval excludes 0;
4. **(d)** exact p < 0.05, with no correction required in a single-test
   family;
5. **(e)** the interval contains the Phase 2 point estimate of +26.7 pp.

Cohort A was COMPLETE at 32/32/32. Realized relations: country 39,
mother 27, sport 16, place of birth 14; dominance share 0.406, below the
0.60 diagnostic threshold, so no dominance flag.

**Robustness (secondary/sensitivity, never the confirmatory interval).**
Leave-one-relation-out gives Δ between +24.4 and +26.3 pp with every
p < 0.001 — the effect does not depend on any one relation. A percentile
bootstrap over items gives +25.0 pp, [+16.7, +34.4].

The dispreferred-only cell is **zero**: no item was adopted from the forum
post but not from the government website. That is a boundary configuration
where a Wald interval is degenerate, and is precisely why the Tango score
interval was pre-specified.

---

## 4. Secondary confirmatory results

All carry their frozen eligibility and status labels. The secondary family
is Holm-corrected within itself and is **never** pooled with the primary.

### Model-specific source contrasts

| Contrast | n | Δ | Holm p | Status |
| --- | ---: | ---: | ---: | --- |
| Cohort B Qwen corrective | 54 | +20.4 pp | 8.8e-03 | DIRECTIONAL EFFECT CONFIRMED |
| Cohort B Llama corrective | 54 | +0.0 pp | 1.00 | **INCONCLUSIVE — saturated** |
| Cohort B Llama harmful | 84 | +0.0 pp | 1.00 | **INCONCLUSIVE — discordance floor** |
| Cohort B Qwen harmful | 48 | +8.3 pp | — | **INCONCLUSIVE — saturated**; also outside the family |
| Cohort C Qwen model-specific | 81 | +8.6 pp | 0.109 | DIRECTIONAL, NOT MULTIPLICITY-SURVIVING |
| Cohort C Llama model-specific | 81 | −1.2 pp | 1.00 | **INCONCLUSIVE — discordance floor** |
| Mistral, Gemma (all four rows) | — | — | — | **NOT APPLICABLE** |

### Common fixed-source contrasts (H2a)

| Model | n | Δ | Holm p | Status |
| --- | ---: | ---: | ---: | --- |
| Qwen | 102 | +14.7 pp | — | **COUNTED ONCE** with the primary |
| Llama | 138 | +10.9 pp | 6.1e-04 | DIRECTIONAL EFFECT CONFIRMED |
| Mistral | 126 | +17.5 pp | 5.7e-06 | DIRECTIONAL EFFECT CONFIRMED |
| Gemma | 96 | +11.5 pp | — | outside the family (Cohort B ineligible) |

### Cross-model and margin

| Analysis | Result | Holm p |
| --- | --- | ---: |
| Cohort C common, cross-model | Δ = +11.1 pp, n = 324 | 8.4e-08 |
| Model × source interaction | joint p = 0.614 | 1.00 |
| H1 margin, Qwen | β = −0.239 (SE 0.045) | 1.7e-06 |
| H1 margin, Llama | β = −0.156 (SE 0.034) | 6.8e-05 |
| H1 margin, Mistral | β = −0.228 (SE 0.040) | 1.6e-07 |
| H1 margin, Gemma | β = −0.090 (SE 0.039) | outside the family |

All four H1 coefficients are negative and all were estimable under the
primary specification (ordinary logistic regression with item-clustered
robust standard errors); the Firth fallback was not needed. Stronger
parametric preference is associated with less context adoption.

The model × source interaction is **not** significant: this design did not
detect a difference in the common-source effect across the four models.

---

## 5. Why Llama's corrective contrast is inconclusive, not null

Llama's Cohort B corrective cells are **53/54 adopting under both
sources** — 0.981 under each, with **zero discordant pairs**. The exact
test returns p = 1.0, but that value carries no evidential weight here:
there were no discordant pairs for it to act on.

Under the frozen §30 rule a contrast is **SATURATED / UNINFORMATIVE** when
either arm exceeds 0.95 (or falls below 0.05) adoption *and* discordant
pairs number fewer than five. Both conditions hold. The pre-specified
consequence is that this null is reported as *"this design could not
detect a source effect in this regime"*, may **not** be counted as
evidence against a source effect, and may **not** be aggregated with
non-saturated nulls as though equivalent.

This is the Phase 2 Llama pattern reproduced at scale, and the reason the
diagnostic was frozen in advance. Llama adopts corrective evidence almost
always, from either source; the manipulation had essentially no room to
express an effect. **Two arms both at ~98% and two arms both at ~50% are
different observations and are reported differently.**

Four further contrasts fall below the discordance floor and are likewise
INCONCLUSIVE: Cohort B Qwen harmful (floor regime, 0.125/0.042),
Cohort B Llama harmful, Cohort C Llama model-specific, and the shared
cohort restriction (n = 5).

---

## 6. Why Mistral and Gemma have no model-specific arm

Neither model has an `M1`/`M2` contrast anywhere in this study. Their
Phase 3C source calibration did not yield a valid unique preferred /
dispreferred pair:

- **Mistral** — all 30 calibration trials parsed (12 of 15 pairs stable,
  3 order reversals). A *preferred* source was identifiable
  (`a government website`), but the three least-preferred sources **tied at
  2/10** and their direct low-tier comparisons were 1:1, so the ordering at
  the bottom is unstable and no defensible **dispreferred** source exists.
  Forcing one would have invented the contrast.
- **Gemma** — **0 of 30** calibration outputs were parser-valid under the
  frozen strict `^Choice:\s*([12])\s*$` parser; raw generations were bare
  `1`/`2`, giving 15/15 malformed pairs. No parser relaxation, prompt
  change, or re-run was used, so no preference ordering exists at all.

Under the frozen §34 rule both models therefore run the **common arm
only**, contributing to the common-source hypothesis but not to the
model-specific family. Their four model-specific contrasts are recorded
**NOT APPLICABLE** — never measured — and are never pooled with, or
reported as, null results. This is an informative measurement about those
models, not a failure of the study.

---

## 7. Multiplicity

- **Primary family: exactly one test.** No correction required or
  permitted. Evaluated at α = 0.05.
- **Secondary family: 21 declared rows → 15 Holm members.** Six removed,
  each for a structural reason fixed before any p-value was consulted:
  three rows whose Cohort B model × group is not confirmatory-eligible
  (Qwen KC, Gemma × 2), and three rows that yield no single pre-specified
  test statistic (leave-one-relation-out and margin-standardization are
  sets of recomputations; the bootstrap is an interval check that §26.2
  makes sensitivity-only).
- Qwen's common-arm contrast was already outside the declared family via
  `counted_once_with` and never entered.

**Eight of fifteen survive Holm:** Cohort B Qwen corrective; common-arm
Llama and Mistral; H1 for Qwen, Llama and Mistral; Cohort C cross-model
common; country-only sensitivity.

**Cohort A never enters the secondary family**, and no secondary result is
presented as independent corroboration of the primary test where it shares
observations with it.

---

## 8. Cohort eligibility

**Cohort A — COMPLETE**, 96 items at 32/32/32.

**Cohort B — five of eight model × group cells confirmatory-eligible.**
Qwen KC, Gemma KC and Gemma KW qualify on only two of the four primary
relations and are ELIGIBILITY-LIMITED / EXPLORATORY, removed from the
confirmatory families under §32 rule 4 and reported descriptively.

**Cohort C — ELIGIBILITY-LIMITED** at 81 of the 96 target: the all-model
intersection supplied only 9 `place of birth` items against a quota of 24.
§32 governs Cohort B only, so Cohort C remains in the confirmatory family,
reported with its realized size.

The screen stopped at the frozen 2 000-candidate ceiling for every model
rather than on the supply criterion. No rule was relaxed to improve any of
these numbers.

---

## 9. Exploratory — never confirmatory

**H3 (margin × source nonlinearity)** is EXPLORATORY regardless of
outcome and is not multiplicity-corrected. Modeled on the continuous
standardized margin (quadratic × source, item-clustered), never inferred
from bin means:

| Model | z:S | p | z²:S | p |
| --- | ---: | ---: | ---: | ---: |
| Qwen | +1.189 | 0.0042 | −0.107 | 0.768 |
| Llama | −0.167 | 0.312 | −0.106 | 0.187 |
| Mistral | +0.533 | 0.063 | +0.166 | 0.461 |
| Gemma | +1.029 | 9.3e-05 | −0.460 | 0.0155 |

These are hypothesis-generating only. No claim is made from them.

**Diagnostics** (abstention, answer accuracy, parse failures, tentative
context content, margin-bin displays, agreement controls) are reported in
`runs/phase3/analysis/phase3e_report.md`. Parse failures were zero across
all 4 197 generations. Abstention ranged from 8.0% (Mistral) to 26.2%
(Qwen).

---

## 10. Scientific conclusion

**Source attribution can materially alter whether a language model adopts
conflicting evidence.** For Qwen2.5-7B-Instruct under the frozen design,
holding evidence content exactly constant and changing only the attributed
source moved committed adoption from 58.3% to 83.3% — a +25.0 pp paired
risk difference, 95% CI [+17.4, +34.5], exact p = 1.2 × 10⁻⁷. This
robustly replicates the Phase 2 pilot finding on fresh items and survives
every pre-specified sensitivity check.

**The effect is not universal across models, and it can be masked.** A
common-source effect appeared for Llama and Mistral as well, and the
cross-model interaction test did not detect a difference between models.
But Llama's corrective contrast is uninterpretable here because both arms
sit at the ceiling, and several other cells fall below the discordance
floor. Where the design could not observe an effect, that is reported as
an inability to observe — not as evidence of absence.

**What this study does not claim.** It does not claim an architecture
effect: the four models differ in family, size, training data and
instruction tuning simultaneously, and this design cannot attribute any
difference to architecture. It does not claim a universal trust hierarchy
over sources: two of the four models produced no usable source-preference
calibration at all, and the ranking that exists is measured for specific
models on specific labels. It makes no claim about *practical*
significance, and none about human-perceived source credibility.

---

## 11. Limitations

- **Four models, one dataset, four relations.** PopQA short-form factual
  questions at one pinned revision; `country`, `sport`, `place of birth`,
  `mother`. Generalization beyond this is untested.
- **Two models contribute no model-specific evidence.** Mistral and Gemma
  are common-arm only, so the model-specific family rests on Qwen and
  Llama.
- **Ceiling effects limit several cells.** Llama in particular adopts
  corrective evidence near-universally, leaving no room for a source
  manipulation to act.
- **Source labels are textual attributions, not real provenance.** The
  evidence is synthetic and template-rendered; no real retrieval, no real
  documents.
- **Cohort C is eligibility-limited**, and several Cohort B cells are
  reduced or exploratory. Realized composition is reported with every
  estimate.
- **Single decoding configuration.** Greedy, `max_new_tokens=32`, one
  prompt template version. Sensitivity to prompt phrasing is untested.
- **The Phase 2 comparator is a 30-item pilot estimate** and is used only
  as a diagnostic. Its point value is very likely inflated.
- **No human baseline** and no measure of whether these source preferences
  are normatively appropriate.

---

## 12. Provenance

Every artifact behind this report is hashed and reproducible; see
`docs/phase3_reproducibility.md` for the end-to-end runbook, exact model
revisions, and artifact digests. The sealed pre-run manifest is
`configs/phase3/freeze/phase3c_pre_run_manifest.json`; the machine-readable
analysis outputs are in `runs/phase3/analysis/`.

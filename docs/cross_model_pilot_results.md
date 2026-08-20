# Cross-Model Pilot Results

## Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct

This document freezes the first cross-model synthesis of the controlled
context-memory conflict pilot described in
`docs/phase2_research_design.md`.

It combines two already-completed model-specific experiments:

1. the frozen `Qwen/Qwen2.5-7B-Instruct` pilot reported independently in
   `docs/qwen_pilot_results.md`; and
2. the subsequent `meta-llama/Llama-3.1-8B-Instruct` replication.

The Qwen result document remains frozen as the historical first-model
report. It is not rewritten retrospectively to make the two models agree.

This remains a **pilot-scale study**, not evidence for a universal property
of LLMs.

---

## 1. Frozen provenance

| | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---|
| Model revision | `a09a35458c702b33eeacc393d103063234e8bc28` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| PopQA revision | `098765c79ea10a2cb19c828324e33281b8336ec0` | `098765c79ea10a2cb19c828324e33281b8336ec0` |
| Precision | float16 | float16 |
| Quantization | none | none |
| GPU | RTX 3090 | RTX 3090 |
| Selected items | 60 | 60 |
| Generations | 300 | 300 |
| Primary conflict trials | 120 | 120 |
| C0 deterministic reproduction | 60/60 exact | 60/60 exact |

The Llama experiment was run from repository commit
`4f57525d19f60fe0431ced3a19bf38f83dd3a545`.

Llama raw pilot results SHA256:

`c27ac06b23b3439ba60ce269d237b96df34473ac103825bc3125f93f61d0a1d7`

Llama final replication archive SHA256:

`3fab7a5113aa730a466acf4d75c4a37781f84f3fd82799c97bac11ecb322df7f`

Qwen final analysis archive SHA256:

`7aeec4fa16cd6de93da4d5d01db4ed4603b2033d080eae97f3f1b91df1cad674`

Both final archives were downloaded off the GPU host and independently
SHA256-verified by the researcher.

---

## 2. What is comparable, and what is not

Both models used the same controlled experimental logic:

- PopQA short-answer factual questions;
- the same pinned PopQA revision;
- the same targeted 500-candidate screening frame;
- 30 KC and 30 KW selected items;
- exact low/medium/high margin balance within KC and KW;
- five conditions C0-C4;
- standardized prompt-injected evidence;
- deterministic decoding;
- committed `context_adopted` as the primary outcome;
- primary interpretation restricted to the 120 genuine conflict trials.

However, this is **not a perfectly matched cross-model experiment**.

First, KC/KW status and parametric margins are model-specific, so the final
60 selected factual items differ between Qwen and Llama.

Second, source calibration was deliberately performed separately for each
model. The preferred source was the same label in both models, but the
dispreferred source differed:

| Model | Preferred source | Dispreferred source |
|---|---|---|
| Qwen | `a government website` | `an anonymous online forum post` |
| Llama | `a government website` | `a social media post` |

Therefore the cross-model comparison asks whether the **model-specific
preferred-versus-dispreferred source manipulation** replicated. It is not
an exact same-source-pair comparison across architectures.

Direct source calibration is treated only as an elicited source preference
for this experiment; it is not interpreted as a universal measure of
credibility, trustworthiness, or latent epistemic preference.

---

## 3. Selected-sample composition

### Qwen

| Group | country | sport | mother | place of birth |
|---|---:|---:|---:|---:|
| KC | 18 | 12 | 0 | 0 |
| KW | 14 | 2 | 12 | 2 |

### Llama

| Group | country | sport | mother | place of birth |
|---|---:|---:|---:|---:|
| KC | 15 | 10 | 1 | 4 |
| KW | 11 | 5 | 4 | 10 |

Neither model's selected pilot sample is relation-balanced. This is a
limitation and one reason not to interpret cross-model differences as pure
architecture effects.

---

## 4. Primary descriptive results

The primary outcome is **committed context adoption**:
`Decision: answer` plus an answer matching the conflicting contextual
candidate.

### Harmful override — KC items, false conflicting evidence

| Model | Preferred | Dispreferred | Delta preferred - dispreferred |
|---|---:|---:|---:|
| Qwen | 6/30 = 20.0% | 3/30 = 10.0% | **+10.0 pp** |
| Llama | 10/30 = 33.3% | 11/30 = 36.7% | **-3.3 pp** |

### Corrective override — KW items, correct conflicting evidence

| Model | Preferred | Dispreferred | Delta preferred - dispreferred |
|---|---:|---:|---:|
| Qwen | 25/30 = 83.3% | 17/30 = 56.7% | **+26.7 pp** |
| Llama | 29/30 = 96.7% | 30/30 = 100.0% | **-3.3 pp** |

The strongest first-model signal — Qwen's +26.7 percentage-point
preferred-source effect on corrective override — **did not replicate in
Llama**. Llama was already near ceiling on corrective conflict adoption
under both source labels.

The harmful-override source difference also did not replicate: Qwen's
descriptive difference was positive, whereas Llama's small difference was
slightly negative.

---

## 5. Secondary paired source checks

Each item appears under both source conditions, so paired exact tests on
discordant items provide a useful secondary check. These are not presented
as preregistered publication-scale confirmatory tests.

### Qwen

| Comparison | both | preferred-only | dispreferred-only | neither | exact two-sided p |
|---|---:|---:|---:|---:|---:|
| KC harmful | 3 | 3 | 0 | 24 | 0.25 |
| KW corrective | 17 | 8 | 0 | 5 | 0.0078125 |

### Llama

| Comparison | both | preferred-only | dispreferred-only | neither | exact two-sided p |
|---|---:|---:|---:|---:|---:|
| KC harmful | 9 | 1 | 2 | 18 | 1.0 |
| KW corrective | 29 | 0 | 1 | 0 | 1.0 |

Thus the Qwen corrective source effect is the only clear paired
source-attribution signal in these two pilot runs. There is no comparable
Llama paired effect.

---

## 6. Planned country-only sensitivity

The country-only analysis was used to check whether relation-composition
imbalance alone could explain the aggregate source pattern.

### Qwen

| Comparison | n | Preferred | Dispreferred | Delta | exact p |
|---|---:|---:|---:|---:|---:|
| KC harmful | 18 | 2/18 = 11.1% | 1/18 = 5.6% | +5.6 pp | 1.0 |
| KW corrective | 14 | 10/14 = 71.4% | 6/14 = 42.9% | +28.6 pp | 0.125 |

### Llama

| Comparison | n | Preferred | Dispreferred | Delta | exact p |
|---|---:|---:|---:|---:|---:|
| KC harmful | 15 | 2/15 = 13.3% | 4/15 = 26.7% | -13.3 pp | 0.5 |
| KW corrective | 11 | 10/11 = 90.9% | 11/11 = 100.0% | -9.1 pp | 1.0 |

The restricted sensitivity does not recover a positive preferred-source
effect in Llama. Qwen retains a similarly sized corrective effect
descriptively, but its country-only subset is small and not an independent
replication.

---

## 7. Parametric preference strength

### Qwen

The frozen Qwen pilot showed a descriptively monotonic pattern in the
predicted H1 direction under the preferred-source conflict condition:

- KW corrective adoption: 10/10 -> 8/10 -> 7/10 from low to high margin.
- KC harmful adoption: 3/10 -> 2/10 -> 1/10.

However, Qwen's continuous exploratory regression did not confirm a
parametric-margin effect (`B`, p = 0.436), and the source-moderation
interaction `B:S` was also unsupported (p = 0.563).

### Llama

Pooling the two source conditions within each genuine conflict type:

| Group | low margin | medium margin | high margin |
|---|---:|---:|---:|
| KC harmful | 4/20 = 20.0% | 13/20 = 65.0% | 4/20 = 20.0% |
| KW corrective | 20/20 = 100.0% | 20/20 = 100.0% | 19/20 = 95.0% |

Llama therefore does not show a monotonic KC margin pattern. KW has only a
small descriptive decline at the high-margin bin under a near-ceiling
adoption regime.

The continuous Llama Spearman checks were:

- KC: rho = -0.0506, p = 0.7905.
- KW: rho = -0.2253, p = 0.2314.

These are exploratory and provide no convincing evidence for H1 in Llama.

The planned ordinary Llama logistic regression also failed to converge and
reported possible complete quasi-separation. Its coefficients and p-values
are therefore not interpreted.

---

## 8. Source-by-strength interaction

For Llama, preferred-minus-dispreferred source deltas by margin bin were:

| Group | low | medium | high |
|---|---:|---:|---:|
| KC harmful | 0.0 pp | +10.0 pp | -20.0 pp |
| KW corrective | 0.0 pp | 0.0 pp | -10.0 pp |

This does not support the exploratory hypothesis that source preference
should matter most at intermediate parametric strength. The small KC
medium-bin bump reverses direction in the high bin, while KW has no
medium-bin source effect.

Together with Qwen's non-significant `B:S` interaction, H3 is not
supported by either pilot.

---

## 9. Tentative answer content versus commitment

The primary metric deliberately treats `Decision: uncertain` as
non-commitment even when the required `Answer:` field contains one of the
candidate answers.

This distinction revealed a recurring separation between **tentative
answer content** and **committed adoption**.

### Qwen — post-hoc mechanistic analysis

This decomposition was conceived after inspecting Qwen's uncertainty and
is therefore explicitly **post-hoc**.

Across 69 uncertain Qwen conflict trials:

- tentative context: 57;
- tentative memory: 9;
- other: 3.

For KW corrective conflicts, tentative context was 30/30 under both source
labels even though committed context adoption differed substantially:

- preferred: tentative 30/30, committed 25/30;
- dispreferred: tentative 30/30, committed 17/30.

For KC harmful conflicts, tentative false-context content was:

- preferred: 21/30;
- dispreferred: 27/30.

Thus Qwen's source effect appeared more clearly in **commitment** than in
which candidate answer surfaced textually.

### Llama — pre-specified secondary mechanistic analysis

The Llama pre-run methodology explicitly preserved this tentative-answer
analysis as a secondary mechanistic follow-up.

Llama produced only **9 uncertain decisions among 120 conflict trials**,
all 9 on KC harmful-conflict items.

Among those 9 uncertain trials:

- tentative context: 7/9 = 77.8%;
- tentative non-context: 2/9 = 22.2%.

For all KC harmful conflict trials:

| | Preferred | Dispreferred |
|---|---:|---:|
| committed context | 10/30 = 33.3% | 11/30 = 36.7% |
| tentative answer content = context | 11/30 = 36.7% | 17/30 = 56.7% |
| uncertain decisions | 2/30 | 7/30 |

For KW corrective conflicts:

| | Preferred | Dispreferred |
|---|---:|---:|
| committed context | 29/30 = 96.7% | 30/30 = 100.0% |
| tentative answer content = context | 29/30 = 96.7% | 30/30 = 100.0% |
| uncertain decisions | 0/30 | 0/30 |

Llama therefore also separates answer content from commitment, but its
uncertainty is much less frequent and concentrated entirely in harmful
KC conflicts.

---

## 10. Hypothesis synthesis

### H1
**Stronger parametric preference reduces adoption of conflicting evidence.**

- Qwen: descriptively suggestive, not confirmed.
- Llama: not supported overall; KC is non-monotonic and KW shows only a
  weak ceiling-limited descriptive decline.
- Cross-model status: **no robust support yet**.

### H2
**Preferred-source attribution increases context adoption while evidence
content is held fixed.**

- Qwen: clearly supported for corrective override in this pilot
  (+26.7 pp; paired exact p = 0.0078125).
- Llama: not supported; source differences are approximately zero and
  slightly negative.
- Cross-model status: **model-specific pilot effect, not a general effect**.

### H3
**Source preference matters most at intermediate parametric strength /
moderates the margin effect.**

- Qwen: not supported (`B:S` p = 0.563).
- Llama: not supported by margin-bin source deltas; the ordinary logit did
  not converge.
- Cross-model status: **not supported**.

### H4
**Preferred sources increase both beneficial correction and harmful
override.**

- Qwen: corrective side supported; harmful side only descriptively
  suggestive.
- Llama: neither side replicated.
- Cross-model status: **no evidence for a universal truth-blind source
  effect**.

---

## 11. Main interpretation

The two-model pilot does **not** support the simple claim that LLMs
generally follow conflicting evidence more when it is attributed to a
preferred source.

Instead, the strongest result is a replication boundary:

> **Source attribution substantially changed commitment to corrective
> conflicting evidence in Qwen2.5-7B-Instruct, but the effect did not
> replicate in Llama-3.1-8B-Instruct under its independently calibrated
> preferred/dispreferred source contrast.**

This suggests that source-sensitive evidence use may be strongly
model-dependent, operating-system-prompt-dependent, calibration-dependent,
or interaction-dependent rather than a universal LLM behavior.

A second recurring observation is that **tentative answer content and
commitment are distinct behaviors**. Both models can surface the contextual
candidate in the `Answer:` field without necessarily committing to it, but
the prevalence and source sensitivity of that separation differ sharply
between models.

These findings motivate a larger follow-up rather than a stronger causal
claim.

---

## 12. Limitations

This synthesis has several important limitations.

- Only two instruction-tuned models were tested.
- Each model used only 60 selected factual items and 120 primary conflict
  trials.
- Final selected items differ between models because KC/KW status and
  parametric margins are model-specific.
- Relation composition differs substantially between model-specific
  samples.
- The dispreferred source label differs between models because calibration
  was intentionally model-specific.
- Source preference was elicited directly rather than estimated as a
  validated latent trust variable.
- Evidence was standardized and prompt-injected, not retrieved from
  naturally occurring documents.
- Llama corrective adoption is near ceiling, reducing sensitivity to a
  source effect.
- Qwen's tentative-answer decomposition was post-hoc, whereas Llama's was
  pre-specified; they should not be treated as equally confirmatory.
- The ordinary Llama logistic regression did not converge because of
  quasi-separation.
- No mixed-effects model was used to model repeated observations within
  item.
- These are pilot-scale results and should not be generalized to LLMs as a
  class.

---

## 13. Next experiment

A publication-scale follow-up should separate three questions that this
pilot currently entangles:

1. **Model effect:** replicate across more model families and sizes.
2. **Source-label effect:** use a common, independently validated source
   hierarchy in addition to model-specific calibration.
3. **Item effect:** construct a shared cross-model item set and explicitly
   balance relation, margin, and KC/KW structure.

The follow-up should increase sample size, pre-specify paired inference,
use an item-aware mixed-effects model, and preserve committed adoption and
tentative answer content as separate outcomes.

---

## 14. Frozen interpretation rule

This document records the first cross-model synthesis after completion of
the Qwen and Llama pilots.

The original Qwen report remains unchanged as the historical first-model
analysis. Any later experiments should be reported as new evidence rather
than used to retrospectively rewrite either frozen pilot result.

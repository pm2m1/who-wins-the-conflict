# Phase 3C — Implementation Note: common-arm-only fallback

**Status: PHASE 3C FROZEN. No Phase 3 evidence outcome exists.** The
outcome-blind baseline screen required to construct the cohorts has since
been run (see "Completing the freeze" below); no source-calibration run
and no evidence-condition generation was performed at any point. The
calibration outcomes recorded below were produced beforehand, on the
RTX-3090 host, and are transcribed here — not recomputed, not adjusted.

This note records how the already-frozen §34 fallback is *represented* in
code and configuration. **It does not alter, extend, or reinterpret the
frozen Phase 3A design** (`docs/phase3_scaled_study_design.md`, commit
`d684f39`), which is unmodified. Where the two ever appear to disagree, the
frozen design governs.

## What the frozen design already required

This is not a new rule. §20.2 states, verbatim:

> If a new model's calibration is tied, unstable, or heavily malformed,
> §34's rule applies: that model runs the **common arm only** and does not
> contribute to the model-specific family.

and §34's failure table contains two matching rows:

> **New model's** calibration is tied, weak, or unstable (heavy malformed
> output, AB/BA reversals) → STOP before building that model's
> model-specific arm. Do not force a pair. The model proceeds with the
> **common arm only**, contributing to H2a but not to the model-specific
> secondary family — recorded as such before generation.

> No clearly separated preferred/dispreferred pair exists for a **new**
> model → Same as above: common arm only, documented. This is an
> informative measurement about that model, not a failure.

Phase 3B implemented only the *pre-calibration* state (roles null →
blocked). It had no way to express the frozen post-calibration outcome
"deliberately disabled". This note records closing that gap. **The change
is representational, not scientific**, and it is not outcome-driven: no
Phase 3 evidence-condition result exists, so nothing here could have been
influenced by one.

## Calibration outcomes as recorded (not recomputed)

| Model | Presentations | Parser-valid | Outcome | Arm |
|---|---:|---:|---|---|
| Mistral-7B-Instruct-v0.3 | 30 | 30 | Top source clear; bottom three tied at 2/10, low-tier direct comparisons 1:1 → no defensible dispreferred source | **DISABLED** |
| Gemma-2-9b-it | 30 | **0** | Raw generations were bare `"1"`/`"2"`; nothing matched the frozen strict `^Choice:\s*([12])\s*$` parser | **DISABLED** |

No parser relaxation and no rerun were used for either model, per the
frozen strict-parser rule. Qwen and Llama were not re-calibrated: they
carry their frozen Phase 2 pairs (§20.1).

## The tri-state representation

The core requirement is that **three states never collapse into one
another**. Collapsing them is exactly how a not-yet-calibrated model could
masquerade as a deliberate common-arm-only model, or how a contrast that
was never measured could later be read as a measured null.

| State | Roles | Reason + provenance | Conditions | Real run |
|---|---|---|---|---|
| `ENABLED` | both required | — | `C0,K1,K2,K3,K4,M1,M2` | allowed |
| `DISABLED_BY_CALIBRATION` | **must be null** | **required** | `C0,K1,K2,K3,K4` | allowed |
| `UNRESOLVED` | null | absent | n/a | **blocked** |

The explicit reason plus `calibration_provenance` is the *discriminator*
between the middle and bottom rows. That is why it is mandatory rather
than advisory: without it the two states are byte-identical.

Config field: `model_specific_arm_enabled: true | false | (absent)`, with
`model_specific_arm_reason` and `calibration_provenance` required when
`false`.

## Where each requirement is implemented

| Requirement | Implementation |
|---|---|
| Tri-state arm + per-model condition set | `phase3/config.py` (`Phase3ModelEntry.arm_state`, `.condition_set`) |
| Arm-state constants | `phase3/constants.py` (`ARM_*`, `CONDITIONS_*`) |
| Disabled arm → common arm only, no placeholder M1/M2 | `phase3/conditions.py` (`model_specific_arm_enabled`) |
| Disabled arm carrying a role → refused | `conditions.DisabledArmWithSourceRoleError` |
| Enabled arm missing a role → refused | `conditions.UnresolvedSourceRolesError` |
| UNRESOLVED still blocks a real run | `phase3/real_run_gate.py` |
| Per-model arm provenance in the manifest | `manifest.model_arm_provenance` |
| Manifest arm validation | `manifest._model_arm_problems` |
| Manifest may not claim M1/M2 for a disabled arm | `manifest._disabled_arm_observation_problems` |
| Never-run contrasts marked NOT APPLICABLE | `analysis_status.mark_not_applicable` |

## Calibration provenance recorded per §36

§36 requires, for each new model, "its calibration output SHA256,
preference matrix, and the researcher's stated reason for the selected
pair (§20.2)". Each new model's `calibration_provenance` therefore carries
explicit fields: three artifact digests (`calibration_output_sha256`,
`calibration_summary_sha256`, `calibration_archive_sha256`), the trial
counts (`parser_total_trials`, `parser_valid_trials`, `malformed_trials`),
`stable_pairs`, `order_reversal_pairs`, `calibration_prompt_version`,
`parser_relaxed`, `rerun_performed`, `decision`, `decision_reason`, and the
`preference_matrix`.

Every value comes from artifacts produced on the RTX-3090 host and
preserved in `phase3c3-archive/`. The three digests per model were
**recomputed from the restored artifacts**, not transcribed, and the
recorded preference matrix was checked pair-for-pair against the archived
`*_source_calibration_summary.json`. Nothing is inferred or adjusted.

The code additionally checks that what is recorded is internally
consistent, because transcription remains the realistic failure mode:

- SHA256 fields must be lowercase 64-character hex; placeholders and prose
  are rejected outright.
- `parser_valid_trials + malformed_trials` must equal
  `parser_total_trials`.
- The preference matrix must contain each of the 15 unordered pairs from
  the six frozen `configs/sources.yaml` labels exactly once, with
  `a_wins + b_wins == 2` per pair (§20.2 presents every pair in both AB and
  BA order), summing to exactly `parser_valid_trials`.
- `stable_pairs` and `order_reversal_pairs` are **re-derived from the
  matrix** and must match what was recorded. For Mistral this independently
  reproduces 12 and 3.
- `preference_matrix` is required when `parser_valid_trials > 0` and must
  be **explicitly null** when it is 0 — absent is not the same as null, and
  an empty matrix would read as "measured, all ties".

Qwen and Llama are exempt. §20.1 freezes their Phase 2 pairs and asks for
no Phase 3 calibration, so demanding Phase 3 calibration artifacts from
them would be an invented rule rather than a frozen one.

### Verified artifact digests

Recomputed from the restored `phase3c3-archive/` tarballs:

| Artifact | SHA256 |
|---|---|
| `mistral_source_calibration.jsonl` | `33c1665cb5ea9147807e7234aba9d7e400e5579761546682623f2d551574e7c2` |
| `mistral_source_calibration_summary.json` | `e9d7fd91ab0b42f920c603558b475565fa622a73f4c90b2b3b6ecfa8e9c99878` |
| `phase3c3-mistral-calibration.tar.gz` | `8096801deb8ba711bb8609691c47ec19f2a339f53f55a27c9e0102ecc0439090` |
| `gemma_source_calibration.jsonl` | `f65188136221392503333b14c2e0ca68b9ceab38cdd696ed990fa81dbc1d9da6` |
| `gemma_source_calibration_summary.json` | `522ba7867c0424cd71e67b5ae751504e16840d9979d02c2c7da019c82e9005ef` |
| `phase3c3-gemma-calibration.tar.gz` | `6db306ec402f8491c0744b8294051e02f0946769f5fd3a6113df5dec68203d85` |

An earlier attempt to record the Mistral output digest supplied a
65-character string. The validator refused it and the field was left absent
rather than truncated or guessed — a fabricated artifact hash is the one
thing a provenance record must never contain. The value above is the
verified 64-character digest and is pinned by a test.

Mistral's `heuristic_source_ranking` is recorded as measured evidence: a
government website ranks top at 1.00, and the bottom three tie at 0.20 with
1:1 direct comparisons. That tie is precisely why no dispreferred source
could be prespecified. Recording the ranking is not a pair selection, and
no role was assigned. Gemma records `malformed_pairs: 15`, so its zero
stable and zero reversal counts are read as "0 out of 0 measured pairs",
not "0 out of 15", which would imply fifteen measured ties.

## Why NOT APPLICABLE, and not a null

A disabled model-specific contrast has **no estimate, no interval and no
p-value** — the generations were never run. Recording it as a null result
would be a fabricated measurement, and pooling it with genuine nulls would
corrupt any cross-model summary. It is therefore given its own status,
`NOT APPLICABLE`, carrying a mandatory reason, and it is **removed from the
Holm-controlled secondary family** so that a family size inflated by
never-run contrasts cannot penalise the tests that were actually run.

This marking is applied pre-outcome, from the recorded calibration state,
so it can never be an after-the-fact reaction to results.

## The §44 registry is a pre-outcome declaration, not the Holm family

`analysis_status.default_registry()` declares every analysis the frozen
design names (§28, §29, §44) with the status the frozen design assigns it,
fixed before any Phase 3 result exists. Model-specific rows carry an
explicit `model_key`, so `mark_not_applicable` matches on that field rather
than on a substring of the analysis name — a short or overlapping model key
can never reach another model's analysis.

For Mistral and Gemma, **both** the corrective (KW) and the harmful (KC)
model-specific contrasts become NOT APPLICABLE, along with their Cohort C
model-specific contrasts, because all three require an M1/M2 arm that was
never generated. Their common fixed-source (H2a) and H1 margin analyses
remain SECONDARY-capable — contributing to H2a is precisely what §34's
fallback preserves.

Qwen's common fixed-source contrast is declared but marked
`counted_once_with` its Cohort B frozen-pair contrast, so it is excluded
from the nominal secondary family: §28 and §44 count it once, because the
frozen common pair coincides with Qwen's Phase 2 pair (§19, §22) and the
two rest on the same observations.

**This registry is not the realized Phase 3E Holm family, and no code here
computes one.** Realized membership additionally depends on
`CohortBGroupResult.confirmatory_eligible`, the §32 relation ladder, §30
saturation, and the remaining frozen eligibility rules. No family size or
composition is hard-coded anywhere. **Phase 3E multiplicity selection is
not complete.**

## Stage-order note: §36 / §41 execution dependency

This section reconciles **execution dependencies only**. It does not
modify, extend, or reinterpret the frozen scientific design or any
hypothesis, and `docs/phase3_scaled_study_design.md` is unedited.

There is a genuine ordering tension in the frozen text:

- **§41** places "cohort construction (A, B, C); **pre-run freeze** (§36)"
  in stage **3C**, and places "Real baseline screens, source calibration,
  C0 + evidence-condition generations" in stage **3D**.
- **§36** requires the 3C pre-run freeze manifest to already contain, per
  model, "baseline and exclusion file SHA256; KC/KW membership; margins;
  margin-bin edges", plus Cohort A/B/C memberships.

Cohort membership is *defined* by baseline screening output: KC/KW labels
and margins come from the baseline pass. So the manifest §41 places in 3C
cannot be assembled without a baseline screen having already run.

The narrow reconciliation:

- The **outcome-blind baseline screening required solely to construct and
  freeze Cohorts A, B and C** must occur as a Phase 3C **design-input**
  step, before the evidence-generation freeze is sealed.
- This is exactly analogous to §41's own treatment of new-model
  calibration, which it places in 3C rather than 3D with the explicit
  reasoning: "its output is required to select their `M1`/`M2` roles before
  the freeze. Calibration is not an outcome variable; it is a design
  input."
- **No context or evidence-condition outcome may be generated during that
  step.** Baseline screening measures parametric knowledge only; it touches
  no `K` or `M` condition and produces no `context_adopted` observation.
- **C0 and all evidence-condition generation remain forbidden** until the
  final Phase 3C manifest is sealed and hashed. C0 is a *reproducibility
  check against* the baseline, not part of screening, and it sits on the
  3D side of the gate.
- Like calibration, this step is still real model execution, so it may only
  occur after 3B is complete and after the researcher has approved §42, and
  its artifacts must be hashed and archived before the manifest is sealed.

The real-run gate was **not** weakened to accommodate this. It continues to
refuse a real run while the freeze manifest is absent, while
`ready_for_real_run` is unset, and while any Cohort A/B/C provenance or
new-model calibration field is missing.

## What deliberately did NOT change

- `docs/phase3_scaled_study_design.md` — untouched.
- All Phase 2 documents, results, and `configs/frozen/` — untouched.
- The frozen common pair `a government website` / `an anonymous online
  forum post` (§19) — unchanged, and still enforced by the config loader.
- Qwen and Llama — still `ENABLED` with their frozen Phase 2 pairs; the
  loader now additionally refuses to disable a replication model's arm.
- Cohort A/B/C construction, the §32 ladder, dedup scoping, the Tango
  interval, exact McNemar, and the saturation rule — all unchanged.
- No preferred/dispreferred source was invented for Mistral or Gemma.

## Manifest/gate repair after the read-only freeze audit

A read-only freeze audit found that **§36 manifest completeness could be
bypassed**: a manifest could pass validation and open the real-run gate
while omitting mandatory §36 content, describing only some of the
configured models, or being judged by a weaker rule set than the manifest
validator itself applied. Three narrow repairs closed that.

**The manifest validator now enforces the full frozen minimum.** Beyond
key presence it requires: the Phase 3 config, candidate-file and
trial-file SHA256 (in `artifact_hashes`); the dataset id, split, resolved
revision and candidate item IDs; per model the baseline and exclusion file
SHA256, KC/KW membership, margins, manual-review decisions, and the
precision/quantization/`device_map`/`max_memory` actually used; the frozen
margin-bin edges (`final_margin_strata`); and populated environment and
hardware capture. Every digest is checked against the single
`calibration_provenance.SHA256_PATTERN`, so a digest valid in one part of
the manifest can never be invalid in another. Environment and hardware
must be *populated*: the builder seeds them with `None` placeholders, and a
frozen manifest still carrying that unfilled template has captured nothing.

**All configured models must be represented.** The manifest model set must
equal the config model set exactly — no omissions, no extras. Every
per-model rule is applied per entry, so an omitted model would otherwise
silently skip its arm-state, condition-set and calibration checks,
including the §34 disabled-arm rules this note exists to record.

**The real-run gate delegates to full manifest validation.** It calls
`validate_manifest` (passing the config's model keys) rather than
re-deriving a subset, so there is one authoritative definition of a valid
manifest. The invariant is now: if `validate_manifest` reports any problem,
`check_readiness` cannot return READY and `assert_ready_for_real_run`
refuses.

**This repair does not mean Phase 3C is frozen or ready.** It only makes
the validator demand these artifacts when a real freeze manifest is
eventually supplied. None of them was created here, cohort construction has
not happened, and their absence is exactly why the gate stays closed.

## Completing the freeze

Every item that was outstanding above has since been supplied from real
verified artifacts, and the freeze is sealed. The outcome-blind baseline
screen ran on the RTX-3090 host at commit
`4b9ad5f28476fa4f8ed4d0687970fa6dac8fb7bd` for all four models — 8 blocks
of 250 candidates each, unquantized float16, at the frozen ceiling — and
its artifacts were verified on return (digests, identities, revisions,
dtype, quantization, CUDA, dataset revision, block sequencing) before any
cohort was derived from them. `configs/phase3/freeze/README.md` records
what the sealed artifacts are and where the bulk empirical outputs live.

The screen stopped at the ceiling for every model rather than on the
supply criterion, so several cohorts are eligibility-limited under the
frozen rules. That is reported, not repaired:

- **Cohort A is COMPLETE** — 32/32/32 fresh Qwen KW items from a supply of
  70/70/71, with the 30 §15.1 Phase 2 items excluded. The relation
  dominance share is 0.406, below the 0.60 diagnostic threshold.
- **Cohort B** realized the §32 ladder unevenly: five of the eight
  model × group cells stay confirmatory-eligible, while `qwen|KC`,
  `gemma|KC` and `gemma|KW` qualify on only two of four relations and
  therefore leave the confirmatory families and are reported
  descriptively (§32 rule 4).
- **Cohort C is ELIGIBILITY_LIMITED** at 81 of the 96 target, because the
  all-model intersection supplies only 9 `place of birth` items against a
  relation quota of 24.

No rule was loosened to improve any of these numbers. The frozen design
anticipates exactly this outcome and specifies how to report it.

### Why one §36 field is scoped rather than complete

§36 asks for the "candidate file SHA256 and IDs". The digest is recorded
from the run's own attestation
(`runtime/candidate-frame.sha256`), but the candidate frame file itself
was not returned from the GPU host, and rebuilding it locally would
require downloading PopQA. `dataset.candidate_item_ids` therefore records
the 2 000 screened candidates — verified to be one common frame prefix
that all four models screened — and
`screening.candidate_item_ids_scope` says so in the manifest rather than
letting a prefix pass as the whole frame. The full frame is regenerable
with `prepare-data` at the pinned dataset revision and checkable against
the recorded digest.

`ready_for_real_run` is now `true` for the first time, and
`real_run_gate.assert_ready_for_real_run` passes — but only alongside the
sealed manifest. Withholding the manifest closes the gate again even with
the flag set, which is the property a test now pins.

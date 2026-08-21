# Phase 3B — Implementation Mapping

**Status: PHASE 3B (implementation + synthetic validation). Phase 3 has
NOT been run.** No real model has been loaded, no dataset downloaded, no
model revision resolved, no source calibration performed, and no Phase 3
scientific result exists.

This document maps the frozen Phase 3A protocol
(`docs/phase3_scaled_study_design.md`, commit `d684f39`) onto the code that
implements it. It is a navigation aid only: **it does not alter, extend, or
reinterpret the frozen design.** Where the two ever appear to disagree, the
frozen design governs and the code is the thing to fix.

## Where each frozen requirement is implemented

| Frozen requirement | Section | Implementation |
|---|---|---|
| Frozen design constants (targets, ceilings, labels, revisions) | all | `src/conflict_eval/phase3/constants.py` |
| Blockwise screening, 250/block, 2,000 ceiling, supply criteria | §11 | `phase3/screening.py` (`ScreeningState`) |
| Margin-stratum recompute-then-freeze | §11, §14 | `phase3/screening.py` (`_compute_strata`, `finalize`, `FinalizedScreening`) |
| Cohort A — 96 fresh Qwen KW, 32/32/32, no relation quota | §15.1 | `phase3/cohorts.py` (`build_cohort_a`) |
| Cohort A relation-dominance diagnostic (non-gating) | §15.1 | `CohortAResult.relation_dominance_flag` |
| Cohort B — 4×3×8 grid, min 6, downsample-to-common, **3-relation fallback** | §15.2, §32 | `phase3/cohorts.py` (`build_cohort_b_group`) |
| Cohort C — shared items, per-model KC/KW preserved | §16 | `phase3/cohorts.py` (`build_cohort_c`, `SharedItem`) |
| Seven nominal conditions and the KC/KW conflict mapping | §22 | `phase3/conditions.py` (`build_phase3_conditions`) |
| Common pair as fixed identities, not roles | §19 | `constants.COMMON_SOURCE_A/B`; `source_role="identity_a/b"` |
| Frozen Phase 2 model-specific pairs | §20.1 | `constants.FROZEN_MODEL_SOURCE_PAIRS`; enforced in `phase3/config.py` |
| New models refuse M1/M2 until calibrated | §20.2 | `conditions.UnresolvedSourceRolesError` |
| Prompt-identical deduplication and aliasing | §22 | `phase3/dedup.py` (`deduplicate_requests`, `CanonicalObservation`) |
| One observation referenced by two contrasts | §22 rule 3 | `dedup.collect_paired_outcomes` (reads through the alias map) |
| Cross-cohort reuse without multiplying compute | §15.2, §16, §23 | `CanonicalObservation.cohorts`, `cohort_membership_map()` |
| Nominal slots ≠ unique generations | §23 | `DeduplicationResult.nominal_slots` / `.unique_observations` |
| Paired risk difference | §26 | `phase3/paired_stats.py` (`PairedTable.risk_difference`) |
| 95% Tango score interval (single confirmatory CI) | §26.2 | `paired_stats.tango_interval` |
| Exact two-sided McNemar / exact binomial | §26.2 | `paired_stats.exact_mcnemar_p` |
| Mandatory paired report (all four cells + discordance) | §26.1 | `paired_stats.PairedSourceResult` |
| Saturation / ceiling-floor diagnostics | §30 | `paired_stats.saturation_diagnostics` |
| Discordance < 5 low-information flag | §37 | `SaturationDiagnostics.low_information` |
| Single-test primary family; Holm for secondary | §28 | `phase3/analysis_status.py`; `paired_stats.holm_adjusted` |
| Analysis status table | §44 | `analysis_status.default_registry()` |
| Provenance / pre-run freeze manifest | §36 | `phase3/manifest.py` |
| Real-run safety gate | §41 | `phase3/real_run_gate.py` |
| **Frozen design-target enforcement** | §11, §15.1, §15.2, §19, §22 | `manifest._frozen_design_problems` (shared by `validate_manifest` and the gate) |
| Phase 3 config namespace | §7, §42.1 | `configs/phase3/phase3_study.yaml`, `phase3/config.py` |

## Cohort B: the §32 ladder as implemented

A relation **qualifies** only if all three of its margin cells reach the
minimum (6). The balanced estimate is then computed over the qualifying
relations, preserving exact relation × stratum balance:

| Situation | Result |
|---|---|
| All 4 qualify, every cell ≥ 8 | `COMPLETE`, 8/cell, confirmatory |
| All 4 qualify, some cell 6-7 | `DOWNSAMPLED` to the common realized count, confirmatory |
| Exactly 1 relation short (3 qualify) | `ELIGIBILITY_LIMITED`, balanced estimate **still computed over the remaining 3**, confirmatory |
| 2+ relations short (<3 qualify) | `ELIGIBILITY_LIMITED_EXPLORATORY`, removed from the confirmatory families, still selected and reported descriptively |
| No relation qualifies | `ELIGIBILITY_LIMITED_EXPLORATORY`, no items |

The short relation is **never** backfilled from a full one, and a Cohort B
reduction never affects Cohort A (§15, §34). An earlier draft returned no
items whenever any cell was short, which discarded usable qualifying
relations and contradicted §32 rule 4; that is fixed.

## Canonical item identity

Phase 3 uses the repository's real baseline-record key, **`item_id`**
(`src/conflict_eval/cli.py`, `cmd_screen`), everywhere -- not a competing
`id` field. `screening.item_id_of` is the single accessor and raises a
clear `ScreeningError` on a missing or empty value. The synthetic fixtures
are now schema-compatible with real records (synthetic in *content*,
real-shaped in *schema*), so they can no longer hide an integration defect.

## Canonical generation identity

`dedup.GenerationIdentity` carries the frozen execution determinants:
model key, **exact model revision**, prompt version, and the
output-affecting generation settings (`do_sample`, `num_beams`,
`max_new_tokens`, and `temperature`/`top_p`/`seed` where applicable).
An observation is keyed on **model + revision + `item_id` + exact
rendered-prompt hash + prompt version + settings fingerprint**, so:

- two different items whose prompts render identically stay distinct
  observations (the scientific unit is the selected factual item);
- the same model at a different revision is a different generation;
- a changed output-affecting setting is a different generation;
- cohort membership alone never creates a new generation.

Machine metadata (GPU, host, wall clock) is deliberately excluded.

## Cohort A requires an explicit exclusion set

`build_cohort_a` takes `phase2_excluded_ids` as a **required, keyword-only**
argument with no default: omitting it raises `TypeError` rather than
silently producing a "replication" that re-measures the original Phase 2
items. The real frozen Phase 2 Qwen KW ids are supplied only at Phase 3C
and are never invented here; the real-run gate additionally requires
non-empty exclusion provenance and checks its size against the 30 items
the frozen design records (§15.1) without hardcoding any id.

## Manifest provenance (§36)

`manifest.py` models per-cohort provenance explicitly, via
`cohort_a_provenance`, `cohort_b_provenance` and `cohort_c_provenance`:

- **Cohort A** — selected ids, per-stratum counts, the freshness
  `excluded_phase2_item_ids` list, the realized relation distribution, and
  the dominance share/flag (DIAGNOSTIC, non-gating).
- **Cohort B** — per model × group: target and minimum cell counts,
  original per-cell counts, qualifying and excluded-short relations, the
  realized relation set and cell count, status, confirmatory eligibility,
  selected ids, and the reduction reason.
- **Cohort C** — per shared item, each model's own `knowledge_group`,
  `parametric_margin` and `margin_stratum`. Cohort C is never reduced to a
  bare id list, because KC/KW does not transfer across models (§16).

`validate_manifest` checks **content**, not just key presence: a real
manifest cannot be frozen while any of the above is missing, and a caller
setting `ready_for_real_run` by hand does not bypass it.

## Frozen design targets vs. realized cohorts

The freeze/readiness layer distinguishes two things the frozen protocol
keeps separate, and conflating them was a real defect an audit caught:

- **PLANNED / FROZEN design** — Cohort A total 96 and 32/32/32 strata with
  a 34/stratum supply criterion; Cohort B target 8 and minimum 6;
  250-candidate blocks to a 2,000 ceiling; the seven nominal conditions;
  the frozen common source pair. These may **never** be rewritten.
- **REALIZED outcome** — what eligibility screening actually yielded. The
  protocol legitimately permits reductions here: a Cohort B cell realized
  at 6 or 7, a three-relation fallback (§32 rule 4), or an
  eligibility-limited Cohort A that fell short at the screening ceiling.

`manifest.planned_cohort_a_design()` and `planned_screening_design()` emit
the planned values from `constants.py`, and every Cohort A manifest carries
`planned_total_target`, `planned_per_stratum_target` and
`planned_supply_per_stratum` alongside `realized_total`,
`per_stratum_selected`, `status`, `eligibility_limited` and `shortfall`.

`manifest._frozen_design_problems()` validates the planned side **against
`constants.py`, never against a copy stored in the manifest**, so editing
both does not defeat the check. It is the single source of truth and is
called by both `validate_manifest` and `real_run_gate.check_readiness`, so
a caller cannot bypass it by invoking only one.

Realized values are validated separately and permissively:

| Situation | Verdict |
|---|---|
| Planned 96 / 32-32-32, realized 96, `COMPLETE` | accepted |
| Planned 96 / 32-32-32, realized short, `ELIGIBILITY_LIMITED` + shortfall + ceiling/supply-failure `stopped_reason` | accepted |
| Planned 96 but realized short while marked `COMPLETE` | rejected |
| `ELIGIBILITY_LIMITED` with no shortfall, or with `stopped_reason` saying supply was met | rejected |
| Planned target rewritten to 18 / 6-6-6 | rejected regardless of status |
| Cohort B target 8 / min 6 with realized 8, 7 or 6 | accepted |
| Cohort B three qualifying relations, `confirmatory_eligible=True` | accepted |
| Cohort B fewer than three relations, `confirmatory_eligible=False` | accepted (exploratory) |
| Cohort B `target_cell_count` != 8 or `minimum_cell_count` != 6 | rejected |
| Cohort B realized cell above target, or `confirmatory_eligible` disagreeing with the three-relation rule | rejected |

The synthetic dry run keeps a deliberately small realized cohort (18 items)
for speed while still recording the frozen planned targets, and remains
`synthetic: True`, which the gate rejects outright.

## What is deliberately unresolved until Phase 3C

These are **not** omissions. The frozen design requires them to stay
unresolved, and the code represents that state explicitly rather than
guessing:

- **Exact repository IDs, instruction-tuned releases, and commit SHAs for
  the two approved new model families** (Mistral-7B-Instruct,
  Gemma-2-9B-it). Recorded as `null` in `configs/phase3/phase3_study.yaml`;
  `phase3/config.py` *rejects* a config that fills any of them in.
- **Model-specific source roles for the new models**, which require their
  own Phase 3C calibration (§20.2). Also `null`, also rejected if invented.
- **The real Phase 2 Qwen KW exclusion id list** for Cohort A freshness
  (§15.1). The id set is a *parameter* (`phase2_excluded_ids`) supplied by
  the caller; it is not present in this checkout and is never hardcoded.
  Tests use synthetic ids.
- **Repository commit, artifact hashes, environment, and hardware** in the
  manifest — all `None` until Phase 3C records the real values.

Qwen's and Llama's exact revisions *are* represented, because they are
already frozen from Phase 2 (§7), and the loader requires them to match
`constants.FROZEN_MODEL_REVISIONS` exactly.

## Dry-run procedure

    python -c "from conflict_eval.phase3.dryrun import run_synthetic_dryrun as r; print(r().banner)"

Or with output written to a `dryrun_`-prefixed file:

```python
from conflict_eval.phase3.dryrun import run_synthetic_dryrun
report = run_synthetic_dryrun(seed=42, output_dir="runs/phase3/dryrun")
```

The dry run exercises the full chain on synthetic records using
`DummyModelAdapter`, which never loads a real model and never makes a
network call. Every artifact it produces carries
`*** SYNTHETIC/DRY-RUN OUTPUT - NOT A PHASE 3 RESULT ***` and
`synthetic: True`, and `real_run_gate` refuses any manifest so marked.

The dry run scales the pool and per-stratum target down from the frozen
32/32/32 and 8-per-cell targets so it runs in about a second. The *logic*
is identical and the frozen constants are untouched — scaling a synthetic
rehearsal produces no scientific quantity.

## Real-run safety gate

`phase3.real_run_gate.assert_ready_for_real_run(config, manifest)` is
deny-by-default and raises `Phase3NotReadyError` unless **every** Phase 3C
prerequisite is present: resolved model ids and revisions for all models,
resolved M1/M2 source roles, a pinned dataset revision, a frozen
non-synthetic manifest with finalized Cohort A membership, final margin
strata, condition specification, and prompt version.

Two deliberate properties:

- **The `ready_for_real_run` flag is not self-certifying.** The gate
  re-derives readiness from actual field values, so setting the flag while
  fields remain unresolved still fails.
- **Phase 3B ships no entry point that loads a real model**, so there is
  nothing to accidentally bypass the gate into. Any future real-run command
  must call the gate first.

## Known Phase 3B limitations

- **No real baseline acquisition.** `ScreeningState` consumes baseline
  records; it does not produce them. Acquiring them is Phase 3D work.
- **Cohort C's relation quota is a simple per-relation share.** It matches
  §16 step 4 ("a deterministic, seeded relation-balanced quota") but has
  not been exercised against real cross-model eligibility overlap, which
  does not exist yet.
- **Margin strata reuse the Phase 2 tertile helpers unchanged**
  (`data/sampling.py`). Under tied margins the three strata can come out
  slightly unequal. This is accepted and deliberate: the frozen design
  treats strata as sampling and diagnostic devices, never as latent
  categories (§14), and changing the Phase 2 helper would alter Phase 2
  behavior, which Phase 3B may not do.
- **No cross-model hierarchical model is implemented yet.** §27's
  model-fixed-effects + item-random-intercept specification and its
  GEE fallback belong to Phase 3E; only the paired machinery the primary
  test needs is implemented here.
- **Synthetic outcomes are hash-derived**, not model behavior. They exist
  solely to give the paired code something to aggregate.

## Backward compatibility

Phase 2 is untouched. `conflict_eval.config`, `experiment/conditions.py`
(C0-C4), `data/sampling.py`, `analysis/paired_comparison.py`, all Phase 2
configs, and `configs/frozen/` are unmodified, and the full pre-existing
test suite still passes. Phase 3 lives entirely in the new
`conflict_eval.phase3` package and the new `configs/phase3/` namespace.

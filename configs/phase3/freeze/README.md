# Phase 3C pre-run freeze record

These files are the **pre-registration seal** required by
`docs/phase3_scaled_study_design.md` §36. They were written before any
Phase 3 evidence condition was generated and they describe only inputs:
which items were selected, into which cohorts, under which frozen
boundaries, and which generations are *planned*. No Phase 3 outcome exists
in this directory, and none may ever be added to it.

Everything here is produced by

```bash
python -m conflict_eval.phase3 build-freeze \
  --return-root <verified cloud-return package> --seal
```

which is a deterministic function of the verified screening artifacts, the
Phase 2 exclusion artifact, and `constants.py`. Nothing was hand-edited.

## Files

| File | What it pins |
| --- | --- |
| `phase3c_pre_run_manifest.json` | the sealed §36 manifest — the authoritative record |
| `cohort_a.json` | the 96 fresh Qwen KW items, 32/32/32, and the §15.1 exclusions |
| `cohort_b.json` | per model × KC/KW cell supply, the realized §32 ladder outcome |
| `cohort_c.json` | the shared cross-model items and each model's own knowledge label |
| `cohort_membership_map.json` | `model|item → cohorts`, making cross-cohort reuse explicit |
| `final_margin_strata.json` | the frozen LOW/MEDIUM/HIGH boundaries per model × group |
| `deduplication_map.json` | the §22 alias map and per-model generation accounting |
| `analysis_status.json` | the §44 table with §34 NOT APPLICABLE rows realized |

## What is deliberately *not* here

The raw returned baseline records, the derived per-model artifacts, the
full trial specification, and the per-observation deduplication records are
large empirical runtime outputs. Following the policy already stated in
`configs/frozen/README.md`, `data/README.md` and `results/README.md`, they
stay outside version control and are referenced by immutable SHA256 from
the manifest — `models[*].baseline_file_sha256`,
`models[*].exclusion_file_sha256`, `artifact_hashes.trial_file`, and
`deduplication_provenance.observations_file_sha256`. They are preserved in
the durable Phase 3C archive alongside the returned cloud package.

## Why the manifest names an earlier commit

`repository_commit` records `4b9ad5f28476fa4f8ed4d0687970fa6dac8fb7bd` —
the commit that was checked out on the GPU host and that produced the
screening artifacts, as independently attested by the returned
`runtime/git-head.txt`. It is deliberately *not* the commit that contains
this manifest: a manifest cannot record the hash of a commit that does not
exist until the manifest is written. The freeze sequence is therefore
strictly ordered and non-circular:

1. the config reaches its final bytes (`ready_for_real_run: true`, the 30
   §15.1 exclusion ids filled in);
2. `artifact_hashes.phase3_config` is the SHA256 of *that* file;
3. the manifest is assembled, validated, sealed, and hashed;
4. the commit containing all of the above becomes the Phase 3C freeze SHA,
   which is recorded in the reproducibility report and the durable archive
   rather than inside the manifest it would otherwise have to contain.

## Changing anything here

Per §36, once archived and hashed the manifest is frozen. A later change
requires a new dated `docs/decisions.md` entry explaining the change and
why it is not outcome-driven — and any change made *after* Phase 3 outcomes
exist invalidates the confirmatory status of the affected analyses.

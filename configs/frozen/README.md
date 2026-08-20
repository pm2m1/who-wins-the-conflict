# Frozen runtime provenance configs

These four files are **post-run provenance copies**, not preregistrations.
Each one carries a header saying so explicitly. They record what was
*actually* used for the two completed, frozen research runs:

- `qwen_pilot_frozen.yaml` / `qwen_model_frozen.yaml` — the
  `Qwen/Qwen2.5-7B-Instruct` run reported in `docs/qwen_pilot_results.md`.
- `llama_pilot_frozen.yaml` / `llama_model_frozen.yaml` — the
  `Llama-3.1-8B-Instruct` run reported in
  `docs/cross_model_pilot_results.md`.

## Why the historical pre-run configs are unchanged

`configs/pilot.yaml`, `configs/models.yaml`,
`configs/replication/llama_pilot.yaml`, and
`configs/replication/models_llama.yaml` are left exactly as they were
written *before* either run happened (`source_roles` null, Llama
`revision: null`, etc.). They are the historical record of what was
precommitted in advance, and rewriting them after the fact to match the
observed outcome would erase that record and make the repository's
history misleading. See `docs/decisions.md`, "Precommit the Llama
second-model replication," for why those specific values (null source
roles, null Llama revision) were deliberately left unset pre-run.

## What these provenance copies are for

A reader who wants to know the exact configuration that produced a given
frozen numeric result (`docs/qwen_pilot_results.md` or
`docs/cross_model_pilot_results.md`) can read the matching file here
instead of cross-referencing prose across multiple documents. Every value
in these files is either:

- copied unchanged from the pre-run template (seed, dataset revision,
  sampling targets, margin bins, generation settings, candidate pool), or
- filled in from a value the frozen result documents themselves report
  (resolved model revision, the researcher's confirmed source-role
  decision, and — for Qwen only — the machine-specific `max_memory` cap
  that `docs/qwen_pilot_results.md` records for that run).

No value here was invented or guessed. Where the frozen documentation does
not record something (for example, no per-model software/environment
versions are recorded for the Llama run, and no `max_memory` cap is
recorded for Llama), the corresponding field is left `null` or omitted
rather than assumed equal to Qwen's.

## What is and is not sufficient to reproduce the original run

These files are sufficient to reconstruct the exact *configuration*
inputs to the pipeline (dataset snapshot, model revision, sampling
targets, source roles, generation settings). They are **not** sufficient
by themselves to reproduce the original numeric results, because:

- the raw PopQA download, baseline-screening output, source-calibration
  output, and pilot generations themselves are large runtime artifacts
  and are intentionally not committed to this repository (see
  `data/README.md`, `results/README.md`, `figures/README.md`);
- gated access to `meta-llama/Llama-3.1-8B-Instruct` is required and not
  provisioned by this repository;
- exact hardware (an NVIDIA RTX 3090 was used for both runs) and, for
  Qwen, the exact software environment versions recorded in
  `docs/qwen_pilot_results.md`, "Frozen provenance," are not guaranteed
  to be reproducible on different hardware even with identical
  configuration and a deterministic decoding setting.

Re-running the pipeline against these configs with real access to the
same pinned model/dataset revisions is expected to be able to reproduce
the same *inputs*; whether it reproduces bit-identical *outputs* depends
on those runtime-environment factors as well, which is why the numeric
results themselves remain frozen in `docs/qwen_pilot_results.md` and
`docs/cross_model_pilot_results.md` rather than re-derived from any future
rerun of these configs.

## What stays out of Git either way

Full large runtime outputs — raw/interim/processed data, result JSONL,
and generated figures — remain outside version control regardless of
which config produced them; see `.gitignore` and the individual
`data/README.md`, `results/README.md`, `figures/README.md` files.

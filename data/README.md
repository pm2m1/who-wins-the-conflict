# Data

Raw and intermediate PopQA data are not committed to this repository (see
`.gitignore`), because the raw dataset is redistributed by Hugging Face
under an unstated license on the dataset card itself (see
`docs/reference_implementations.md`) and because it is large relative to
this repository's other contents. This directory documents how to
recreate it exactly.

## Layout

- `raw/` — untouched download of the source dataset. Never edited in
  place.
- `interim/` — after deterministic normalization (see
  `docs/methodology.md`, section 1): comparison-only lowercasing,
  punctuation/whitespace normalization, article normalization, alias-set
  construction. Original fields are preserved alongside normalized ones.
- `processed/` — the screened candidate pool actually used to build
  baseline-screening inputs, plus an exclusion log.

## Recreating `raw/`

    python -m conflict_eval prepare-data --config configs/pilot.yaml

This downloads the Hugging Face dataset `akariasai/PopQA` (test split,
14,300 rows as of inspection) via the `datasets` library, using the
`hf_dataset_id` and `split` fields in `configs/pilot.yaml`. The exact
dataset revision resolved by `datasets` at download time is recorded in
`data/raw/manifest.json` for reproducibility.

## Fields used from PopQA

`id`, `subj`, `prop`, `obj`, `s_aliases`, `o_aliases`, `question`,
`possible_answers`. See `docs/reference_implementations.md` for the full
confirmed field list and `docs/methodology.md` for how they are used.

## Status

As of this writing, `prepare-data` has not been run in this environment —
see `README.md` for current pilot status. `raw/`, `interim/`, and
`processed/` are present only as empty directories (tracked via
`.gitkeep`) until the pipeline is actually executed.

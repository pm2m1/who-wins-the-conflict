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

Real PopQA was downloaded and used to build both completed research runs
(`Qwen/Qwen2.5-7B-Instruct` and `Llama-3.1-8B-Instruct`; see
`docs/qwen_pilot_results.md` and `docs/cross_model_pilot_results.md`), on
the researcher's own GPU host. The exact resolved PopQA revision used for
both runs, `098765c79ea10a2cb19c828324e33281b8336ec0`, is recorded in
those frozen result documents, not just in `data/raw/manifest.json`.

The raw/interim/processed artifacts from those runs are intentionally not
committed to this repository (see `.gitignore` and the license rationale
above), so `raw/`, `interim/`, and `processed/` in *this* checkout are
empty directories (tracked only via `.gitkeep`). This is expected, not a
sign that no experiment was run. Re-running `prepare-data` above recreates
the same interim/processed pipeline output from the same pinned PopQA
revision, but does not by itself reproduce the frozen pilot results, which
also depend on the model-specific baseline screen, source calibration, and
pilot sample described in `docs/pilot_protocol.md`.

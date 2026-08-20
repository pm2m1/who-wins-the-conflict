# Results

This directory holds pipeline-generated output: baseline screening
records, source-calibration records, and experimental (C0-C4) generation
records, all as JSONL (see `docs/methodology.md` for the record schemas).
Nothing in this directory is hand-written.

Generated files are gitignored (see `.gitignore`) because they are
reproducible from the pipeline plus a fixed seed, and because a real
~600-generation pilot run can be large. Only this README is tracked.

## Status

Real Qwen and Llama pilot runs were completed on the researcher's own GPU
host (screening, source calibration, and the C0-C4 pilot for each model).
The generated JSONL/CSV/JSON runtime files those runs produced are
intentionally gitignored — reproducible from the pipeline plus a fixed
seed, and large relative to this repository — so this local checkout's
`results/` directory contains no result files, only this README. That is
expected, not a sign that no experiment was run.

The frozen scientific results live in:

- `docs/qwen_pilot_results.md` — the first-model Qwen pilot;
- `docs/cross_model_pilot_results.md` — the completed two-model synthesis.

Dry-run output (from the `DummyModelAdapter`) must never be treated as
real results — it is written, if at all, to a path clearly marked
`dryrun` or `synthetic`, and it is not read by `analyze` as pilot input.

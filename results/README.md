# Results

This directory holds pipeline-generated output: baseline screening
records, source-calibration records, and experimental (C0-C4) generation
records, all as JSONL (see `docs/methodology.md` for the record schemas).
Nothing in this directory is hand-written.

Generated files are gitignored (see `.gitignore`) because they are
reproducible from the pipeline plus a fixed seed, and because a real
~600-generation pilot run can be large. Only this README is tracked.

## Status

As of this writing, no real pilot has been run in this environment. There
are no result files here yet. Dry-run output (from the `DummyModelAdapter`)
must never be treated as real results — it is written, if at all, to a
path clearly marked `dryrun` or `synthetic`, and it is not read by
`analyze` as pilot input.

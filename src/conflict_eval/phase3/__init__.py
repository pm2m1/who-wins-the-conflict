"""Phase 3 scaled-study infrastructure.

Implements the frozen protocol in `docs/phase3_scaled_study_design.md`
(commit d684f39, "Freeze Phase 3 scaled study design"). That document is
historical, pre-result material: this package implements it, and must
never be used to reinterpret or amend it.

**Exactly two modules in this package touch the outside world**, and both
are Phase 3C screening machinery:

- `baseline_runner.py` executes the outcome-blind baseline measurement
  (§11), which §41 places in Phase 3C because its output is a design
  input for cohort construction, not an outcome;
- `cli.py`'s `prepare-data` downloads the pinned PopQA revision (§8).

Everything else is offline. **No module in this package can generate a
Phase 3 evidence condition.** `C0`/`K1`-`K4`/`M1`/`M2` generation is
Phase 3D work and remains forbidden until the pre-run manifest is sealed
(§41); `real_run_gate.py` enforces that programmatically, and
`baseline_runner.assert_no_evidence_machinery_imported()` asserts that the
screening path holds no reference to the condition builders.
"""

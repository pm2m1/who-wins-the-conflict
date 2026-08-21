"""Phase 3 scaled-study infrastructure.

Implements the frozen protocol in `docs/phase3_scaled_study_design.md`
(commit d684f39, "Freeze Phase 3 scaled study design"). That document is
historical, pre-result material: this package implements it, and must
never be used to reinterpret or amend it.

**No module in this package loads a real model, contacts the Hugging Face
Hub, or downloads a dataset.** Phase 3B is implementation plus synthetic
validation only; the frozen design forbids any real Phase 3 model run
before the Phase 3C pre-run freeze (`docs/phase3_scaled_study_design.md`,
§41). `real_run_gate.py` enforces that programmatically.
"""

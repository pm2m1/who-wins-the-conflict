"""Evaluation-layer answer matching.

Re-exports the deterministic normalization/matching primitives from
data/normalize.py. Kept as a separate module (rather than importing
data.normalize directly throughout evaluation/) so evaluation code has one
stable entry point, and so a future evaluation-specific matching rule
(e.g. a stricter or looser variant used only for scoring, not for KC/KW
screening) has an obvious place to live without touching data/normalize.py.
"""

from __future__ import annotations

from conflict_eval.data.normalize import is_match, normalize_answer, token_f1

__all__ = ["is_match", "normalize_answer", "token_f1"]

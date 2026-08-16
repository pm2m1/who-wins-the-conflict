"""Deterministic record keys for resumable experiment execution.

A key is derived from the fields that jointly identify one intended
generation (docs/methodology.md, section 9). Recomputing the same key from
the same inputs on a re-run is how io/results.ResultWriter recognizes
already-completed work and skips it.
"""

from __future__ import annotations

import hashlib


def make_record_key(
    experiment_type: str,
    model_id: str,
    item_id: str,
    condition: str,
    prompt_version: str,
    seed: int,
) -> str:
    raw = "|".join(
        [experiment_type, model_id, str(item_id), condition, prompt_version, str(seed)]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

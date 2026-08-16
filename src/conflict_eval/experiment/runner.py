"""Resumable experiment execution loop.

Deliberately model-agnostic: the actual generation call happens inside
`generate_fn`, supplied by the caller (scripts/run_pilot.py), so this
module's resumability logic is exercised in tests without touching a real
model or the network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from conflict_eval.experiment.resume import make_record_key
from conflict_eval.io.results import ResultWriter


def run_trials(
    trial_inputs: list[dict[str, Any]],
    generate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    writer: ResultWriter,
    experiment_type: str,
    seed: int,
) -> dict[str, int]:
    """Run one generation per trial input, skipping already-completed
    record keys.

    Each `trial_input` must contain `model_id`, `item_id`, `condition`,
    and `prompt_version`. `generate_fn(trial_input)` performs the actual
    model call and returns the fields to write as the result record;
    `record_key` is added here so callers cannot forget it.
    """
    n_run = 0
    n_skipped = 0
    for trial_input in trial_inputs:
        key = make_record_key(
            experiment_type=experiment_type,
            model_id=trial_input["model_id"],
            item_id=trial_input["item_id"],
            condition=trial_input["condition"],
            prompt_version=trial_input["prompt_version"],
            seed=seed,
        )
        if writer.is_completed(key):
            n_skipped += 1
            continue
        result = generate_fn(trial_input)
        result["record_key"] = key
        writer.write(result)
        n_run += 1
    return {"run": n_run, "skipped": n_skipped}

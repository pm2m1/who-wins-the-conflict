"""Direct pairwise source-preference calibration.

Asks the model a concise, structured-choice question about two named
sources (never asks for an explanation), for every counterbalanced
presentation produced by counterbalance.py. This is "direct" calibration
as opposed to indirect/latent (behavioral) preference measurement, which
this project does not implement (docs/reference_implementations.md).
"""

from __future__ import annotations

import dataclasses
import re

from conflict_eval.models.base import BaseModelAdapter, GenerationConfig, Message
from conflict_eval.source_preference.counterbalance import Presentation

_CHOICE_RE = re.compile(r"Choice:\s*([12])")


def render_calibration_prompt(template: str, presentation: Presentation) -> str:
    return template.format(
        source_1=presentation.displayed_source_1,
        source_2=presentation.displayed_source_2,
    )


def parse_choice(raw_generation: str) -> int | None:
    """Returns 1, 2, or None if the response did not contain a parseable
    structured choice. None is a real outcome, not an error — malformed
    calibration responses are excluded from pairwise statistics rather
    than coerced.
    """
    match = _CHOICE_RE.search(raw_generation)
    return int(match.group(1)) if match else None


def selected_source_from_choice(presentation: Presentation, choice: int | None) -> str | None:
    if choice == 1:
        return presentation.displayed_source_1
    if choice == 2:
        return presentation.displayed_source_2
    return None


@dataclasses.dataclass
class CalibrationTrial:
    run_id: str
    model_id: str
    seed: int
    source_a: str
    source_b: str
    displayed_source_1: str
    displayed_source_2: str
    presentation_order: str
    prompt_version: str
    prompt: str
    raw_generation: str
    parsed_choice: int | None
    selected_source: str | None


def run_calibration_trial(
    model: BaseModelAdapter,
    template: str,
    presentation: Presentation,
    prompt_version: str,
    seed: int,
    run_id: str,
    generation_config: GenerationConfig,
) -> CalibrationTrial:
    prompt = render_calibration_prompt(template, presentation)
    messages: list[Message] = [{"role": "user", "content": prompt}]
    raw_generation = model.generate(messages, generation_config)
    choice = parse_choice(raw_generation)
    selected_source = selected_source_from_choice(presentation, choice)

    return CalibrationTrial(
        run_id=run_id,
        model_id=model.model_id,
        seed=seed,
        source_a=presentation.source_a,
        source_b=presentation.source_b,
        displayed_source_1=presentation.displayed_source_1,
        displayed_source_2=presentation.displayed_source_2,
        presentation_order=presentation.presentation_order,
        prompt_version=prompt_version,
        prompt=prompt,
        raw_generation=raw_generation,
        parsed_choice=choice,
        selected_source=selected_source,
    )

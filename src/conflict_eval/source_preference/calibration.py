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

# Line-anchored and exact: the ENTIRE Choice line (surrounding whitespace
# aside) must be exactly "1" or "2" — a prefix match here previously
# accepted trailing junk such as "Choice: 1 | 2" or "Choice: 1 because...",
# the same real-model failure mode already fixed for the baseline Decision
# field (docs/decisions.md, "Make source calibration output strict"; see
# also "Make decision output format strict"). MULTILINE so ^/$ anchor to
# individual lines within the full response, not just the whole string.
_CHOICE_RE = re.compile(r"^Choice:\s*([12])\s*$", re.MULTILINE)


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
    # Model provenance (docs/decisions.md, "Make source calibration
    # output strict"): source preference is model-specific, so a real
    # calibration record must identify the exact checkpoint used, not
    # just the model family. model_revision mirrors the same field on
    # baseline result records (docs/methodology.md) — the resolved commit
    # SHA when available, else the requested revision string.
    # requested_revision/resolved_revision are the adapter's own values
    # (never hard-coded here); DummyModelAdapter reports both as None.
    model_revision: str | None
    requested_revision: str | None
    resolved_revision: str | None
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
        model_revision=model.model_revision,
        requested_revision=getattr(model, "requested_revision", None),
        resolved_revision=getattr(model, "resolved_revision", None),
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

"""Prompt-identical deduplication and cross-cohort observation reuse.

Implements `docs/phase3_scaled_study_design.md` §22 ("Deterministic
deduplication of prompt-identical conditions"), §15.2/§16 (cross-cohort
reuse) and §23 (nominal condition slots vs. unique model generations).

The scientific point, stated by the frozen design and enforced here:

> The analysis layer must treat this as **one observation referenced by two
> planned contrasts, not two independent generations.** Concretely: it is
> counted once in any n; it may not contribute twice to a pooled estimate;
> ... and it enters the multiplicity families only once. (§22)

Deduplication is keyed on the **exact rendered prompt**, never on metadata
resemblance. Two conditions that happen to share a source label but assert
different answers, or that ask different questions, are different prompts
and are never merged (§22 rule 5, partial overlap).
"""

from __future__ import annotations

import dataclasses
import hashlib


def prompt_hash(rendered_prompt: str) -> str:
    """Canonical identity of a prompt: SHA256 of its exact text.

    No normalization, casefolding, or whitespace collapsing is applied --
    deduplication must be strict, so anything that would change what the
    model actually receives keeps the observations distinct (§22).
    """
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class GenerationIdentity:
    """The frozen execution determinants of one generation.

    Two planned conditions are the same observation only if **all** of these
    match, alongside the item and the exact rendered prompt. `model_key`
    alone is insufficient: the same label at a different revision is a
    different model artifact and therefore a different generation.

    Only settings that can change the produced text are included. Machine
    metadata (GPU name, host, wall clock) is deliberately excluded -- it
    does not affect output identity under deterministic decoding.
    """

    model_key: str
    model_revision: str | None
    prompt_version: str
    do_sample: bool
    num_beams: int
    max_new_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def fingerprint(self) -> str:
        """Stable fingerprint of the generation settings (not the model or
        prompt, which are hashed alongside it)."""
        payload = "|".join(
            [
                f"do_sample={self.do_sample}",
                f"num_beams={self.num_beams}",
                f"max_new_tokens={self.max_new_tokens}",
                f"temperature={self.temperature}",
                f"top_p={self.top_p}",
                f"seed={self.seed}",
                f"prompt_version={self.prompt_version}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class ConditionRequest:
    """One *planned* condition for one item, before deduplication.

    Several requests can resolve to the same canonical observation; that is
    the whole point of the layer.
    """

    model_key: str
    item_id: str
    condition: str
    arm: str
    source_role: str
    source_label: str | None
    evidence_truth: str
    conflict_status: str
    knowledge_group: str
    rendered_prompt: str
    cohorts: tuple[str, ...] = ()


@dataclasses.dataclass
class CanonicalObservation:
    """One unique generation, referenced by one or more planned requests."""

    observation_id: str
    model_key: str
    item_id: str
    prompt_hash: str
    rendered_prompt: str
    canonical_generation_key: str
    # Frozen execution determinants recorded on the observation itself, so
    # the manifest can show exactly what identified this generation (§36).
    model_revision: str | None
    prompt_version: str
    generation_settings_fingerprint: str
    # The (condition, arm, role) labels that all resolve to this one
    # observation -- the frozen design requires the record to carry the
    # full set (§22 rule 2).
    aliased_conditions: list[dict]
    cohorts: set[str]

    @property
    def is_aliased(self) -> bool:
        return len(self.aliased_conditions) > 1

    @property
    def primary_condition(self) -> str:
        """The lexicographically first condition label, used only as a
        stable display name. It confers no statistical precedence."""
        return min(a["condition"] for a in self.aliased_conditions)


@dataclasses.dataclass
class DeduplicationResult:
    observations: list[CanonicalObservation]
    # requested (model, item, condition) -> canonical observation id
    alias_map: dict[tuple[str, str, str], str]
    nominal_slots: int

    @property
    def unique_observations(self) -> int:
        return len(self.observations)

    @property
    def collapsed_slots(self) -> int:
        """How many nominal slots were absorbed by deduplication (§23)."""
        return self.nominal_slots - self.unique_observations

    def observation_for(
        self, model_key: str, item_id: str, condition: str
    ) -> CanonicalObservation:
        obs_id = self.alias_map[(model_key, str(item_id), condition)]
        for observation in self.observations:
            if observation.observation_id == obs_id:
                return observation
        raise KeyError(obs_id)

    def cohort_membership_map(self) -> dict[str, list[str]]:
        """observation id -> sorted cohort memberships (§16, manifest §36)."""
        return {
            obs.observation_id: sorted(obs.cohorts) for obs in self.observations
        }


def deduplicate_requests(
    requests: list[ConditionRequest],
    seed: int,
    identity: GenerationIdentity,
) -> DeduplicationResult:
    """Collapse prompt-identical requests into canonical observations.

    Deterministic and decided **before** generation, from the frozen source
    assignments -- not discovered afterwards (§22 rule 4). The canonical
    generation key reuses the Phase 2 record-key discipline
    (`experiment/resume.py`) so an existing runner would skip a duplicate
    before spending compute.

    Scoping: an observation is identified by **model key + exact model
    revision + item_id + exact rendered-prompt hash + prompt version +
    generation-settings fingerprint**. Two different items whose prompts
    happen to render identically therefore stay distinct observations --
    the scientific unit is the selected factual item, and prompt identity
    only collapses multiple planned conditions *for that item*.
    """
    observations: dict[tuple[str, str, str, str], CanonicalObservation] = {}
    alias_map: dict[tuple[str, str, str], str] = {}
    settings_fingerprint = identity.fingerprint()
    revision = identity.model_revision or "UNRESOLVED"

    for request in requests:
        item_id = str(request.item_id)
        digest = prompt_hash(request.rendered_prompt)
        # item_id is part of the key, so identical prompt text across two
        # different items never merges them.
        key = (request.model_key, revision, item_id, digest)

        if key not in observations:
            raw = "|".join(
                [
                    "phase3",
                    request.model_key,
                    revision,
                    item_id,
                    digest,
                    identity.prompt_version,
                    settings_fingerprint,
                    str(seed),
                ]
            )
            observation_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
            observations[key] = CanonicalObservation(
                observation_id=observation_id,
                model_key=request.model_key,
                item_id=item_id,
                prompt_hash=digest,
                rendered_prompt=request.rendered_prompt,
                canonical_generation_key=observation_id,
                model_revision=identity.model_revision,
                prompt_version=identity.prompt_version,
                generation_settings_fingerprint=settings_fingerprint,
                aliased_conditions=[],
                cohorts=set(),
            )

        observation = observations[key]
        observation.aliased_conditions.append(
            {
                "condition": request.condition,
                "arm": request.arm,
                "source_role": request.source_role,
                "source_label": request.source_label,
                "evidence_truth": request.evidence_truth,
                "conflict_status": request.conflict_status,
            }
        )
        observation.cohorts.update(request.cohorts)
        alias_map[(request.model_key, item_id, request.condition)] = (
            observation.observation_id
        )

    ordered = sorted(
        observations.values(),
        key=lambda o: (o.model_key, str(o.model_revision), o.item_id, o.prompt_hash),
    )
    return DeduplicationResult(
        observations=ordered,
        alias_map=alias_map,
        nominal_slots=len(requests),
    )


def collect_paired_outcomes(
    result: DeduplicationResult,
    outcomes_by_observation: dict[str, bool],
    model_key: str,
    item_ids: list[str],
    condition_a: str,
    condition_b: str,
) -> list[tuple[bool, bool]]:
    """Assemble `(outcome_under_A, outcome_under_B)` pairs for a contrast.

    Reads through the alias map, so a deduplicated observation contributes
    its single stored outcome to whichever planned contrast references it --
    counted once per item, never duplicated into a second independent row
    (§22 rule 3). Items missing either condition are skipped, matching the
    Phase 2 paired-comparison rule that incomplete pairs cannot contribute.
    """
    pairs: list[tuple[bool, bool]] = []
    for item_id in item_ids:
        key_a = (model_key, str(item_id), condition_a)
        key_b = (model_key, str(item_id), condition_b)
        if key_a not in result.alias_map or key_b not in result.alias_map:
            continue
        obs_a = result.alias_map[key_a]
        obs_b = result.alias_map[key_b]
        if obs_a not in outcomes_by_observation or obs_b not in outcomes_by_observation:
            continue
        pairs.append(
            (outcomes_by_observation[obs_a], outcomes_by_observation[obs_b])
        )
    return pairs


def contrast_is_degenerate(
    result: DeduplicationResult,
    model_key: str,
    item_ids: list[str],
    condition_a: str,
    condition_b: str,
) -> bool:
    """True when both sides of a contrast resolve to the SAME observation.

    That is not a source contrast at all -- it is one observation compared
    with itself, and it would trivially show zero effect. It cannot arise
    for the frozen pairs (a model's own A and B labels always differ), but
    it is checked so a future misconfiguration fails loudly instead of
    silently producing a null.
    """
    for item_id in item_ids:
        key_a = (model_key, str(item_id), condition_a)
        key_b = (model_key, str(item_id), condition_b)
        if (
            key_a in result.alias_map
            and key_b in result.alias_map
            and result.alias_map[key_a] == result.alias_map[key_b]
        ):
            return True
    return False

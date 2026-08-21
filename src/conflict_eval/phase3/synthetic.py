"""Deterministic SYNTHETIC fixtures for Phase 3B validation.

Everything produced here is obviously synthetic by construction: ids are
`syn-<relation>-<n>`, questions and answers are `synthetic-*` strings, and
no value is drawn from PopQA or from any model. These fixtures exist so the
Phase 3 logic can be exercised end to end without a real model, a real
dataset, or a network call.

**Synthetic records are not experimental data.** Nothing built here may be
recorded as a Phase 3 result, and `manifest.build_manifest` stamps any
manifest built from them `synthetic: True`, which `real_run_gate` rejects.
"""

from __future__ import annotations

import hashlib

from conflict_eval.phase3.constants import PHASE3_RELATIONS

SYNTHETIC_ID_PREFIX = "syn"


def synthetic_item_id(relation: str, index: int) -> str:
    slug = relation.replace(" ", "-")
    return f"{SYNTHETIC_ID_PREFIX}-{slug}-{index:04d}"


def make_baseline_record(
    item_id: str,
    relation: str,
    knowledge_group: str,
    parametric_margin: float,
    primary_conflict_eligible: bool = True,
) -> dict:
    """One synthetic baseline record, shaped like a REAL Phase 2 baseline
    record (`src/conflict_eval/cli.py`, `cmd_screen`) but with obviously
    synthetic content.

    Deliberately carries **no** evidence-condition/outcome fields; screening
    rejects those outright (§11).
    """
    return {
        # Canonical key, matching the repository's real baseline records
        # (src/conflict_eval/cli.py, cmd_screen). Synthetic in CONTENT,
        # schema-compatible in SHAPE, so these fixtures cannot hide an
        # integration defect the way an `id`-keyed fixture did.
        "item_id": item_id,
        "subject": f"synthetic-subject-{item_id}",
        "relation": relation,
        "knowledge_group": knowledge_group,
        "parametric_margin": float(parametric_margin),
        "primary_conflict_eligible": bool(primary_conflict_eligible),
        "question": f"synthetic-question about {item_id}?",
        "gold_answer": f"synthetic-gold-{item_id}",
        "baseline_answer": (
            f"synthetic-gold-{item_id}"
            if knowledge_group == "KC"
            else f"synthetic-wrong-{item_id}"
        ),
        "foil_answer": f"synthetic-foil-{item_id}",
        "synthetic": True,
    }


def make_baseline_pool(
    n_per_cell: int,
    relations: tuple[str, ...] = PHASE3_RELATIONS,
    groups: tuple[str, ...] = ("KC", "KW"),
    margin_spread: float = 30.0,
    start_index: int = 0,
) -> list[dict]:
    """Build a synthetic pool with `n_per_cell` items per (relation, group).

    Margins are spread deterministically across the full range so tertile
    strata come out populated; each (relation, group) draws from the same
    spread so no cell is structurally starved.
    """
    records: list[dict] = []
    for relation in relations:
        for group in groups:
            for i in range(n_per_cell):
                index = start_index + i
                item_id = synthetic_item_id(f"{relation}-{group}", index)
                # Deterministic, evenly spread margin.
                fraction = (i + 0.5) / n_per_cell
                margin = round(fraction * margin_spread, 6)
                records.append(
                    make_baseline_record(item_id, relation, group, margin)
                )
    return records


def make_kw_pool_for_cohort_a(
    n_total: int, relations: tuple[str, ...] = PHASE3_RELATIONS, start_index: int = 0
) -> list[dict]:
    """A synthetic Qwen KW pool for Cohort A tests.

    Relation assignment is intentionally **uneven** (round-robin weighted
    toward the first relation) so tests can confirm Cohort A tolerates
    relation imbalance without a quota (§15.1).
    """
    records: list[dict] = []
    for i in range(n_total):
        # Weighted: half the items land on the first relation.
        relation = relations[0] if i % 2 == 0 else relations[(i // 2) % len(relations)]
        item_id = synthetic_item_id(f"{relation}-KW", start_index + i)
        margin = round(((i + 0.5) / n_total) * 30.0, 6)
        records.append(make_baseline_record(item_id, relation, "KW", margin))
    return records


def synthetic_outcome(observation_id: str, adoption_rate: float = 0.5) -> bool:
    """A deterministic pseudo-outcome for one canonical observation.

    Used only by the dry run to give the paired machinery something to
    aggregate. It is a hash of the observation id -- not a model output, not
    a scientific result, and never recorded as one.
    """
    digest = hashlib.sha256(f"synthetic:{observation_id}".encode()).hexdigest()
    draw = int(digest[:8], 16) / 0xFFFFFFFF
    return draw < adoption_rate

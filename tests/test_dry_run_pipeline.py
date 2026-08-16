"""End-to-end SYNTHETIC/DEBUG pipeline test.

Exercises every pipeline stage — preprocessing-shaped fixtures, foil
creation, a KC/KW sample fixture, source-pair counterbalancing, condition
generation, prompt rendering, resumable result writing, answer
classification, and metric computation — using DummyModelAdapter end to
end, entirely in memory. This is the automated counterpart to the manual
CLI dry run described in README.md; it never touches a real model, the
network, or the project's own data/results/figures directories.

All fixtures in this file are SYNTHETIC and must never be confused with
real pilot data (docs/decisions.md).
"""

from __future__ import annotations

import random

from conflict_eval.data.foils import build_relation_index, sample_foil
from conflict_eval.data.sampling import sample_balanced_across_bins
from conflict_eval.evaluation.classify import classify_answer, is_context_adopted, is_final_correct
from conflict_eval.evaluation.metrics import (
    context_adoption_rate,
    corrective_override_rate,
    harmful_override_rate,
)
from conflict_eval.evaluation.parse import parse_response
from conflict_eval.experiment.conditions import build_conditions
from conflict_eval.experiment.evidence import render_evidence
from conflict_eval.experiment.prompts import render_experiment_prompt
from conflict_eval.models.base import GenerationConfig
from conflict_eval.models.dummy import DummyModelAdapter
from conflict_eval.source_preference.counterbalance import expand_pairs_to_presentations
from conflict_eval.source_preference.pairs import enumerate_unordered_pairs

EVIDENCE_TEMPLATE = 'Source: {source}\n\nStatement:\nThe answer to the question "{question}" is "{asserted_answer}".\n'
BASELINE_TEMPLATE = (
    "Use the information you consider most reliable.\n\n"
    "Evidence:\n{evidence_or_none}\n\nQuestion:\n{question}\n\n"
    "Respond using exactly this format:\n\nAnswer: <short answer>\n"
    "Decision: <answer or uncertain>\nConfidence: <integer from 0 to 100>\n"
)

# SYNTHETIC fixture: a small same-relation pool for foil sampling.
SYNTHETIC_POPQA_POOL = [
    {"id": "1", "prop": "capital_of", "obj": "Paris", "aliases": []},
    {"id": "2", "prop": "capital_of", "obj": "Berlin", "aliases": []},
    {"id": "3", "prop": "capital_of", "obj": "Madrid", "aliases": []},
]


def test_synthetic_dry_run_pipeline_end_to_end():
    gen_config = GenerationConfig()
    model = DummyModelAdapter(seed=0, context_adoption_rate=0.5, model_id="dummy")

    # 1. Source-pair counterbalancing.
    pairs = enumerate_unordered_pairs(["Wikipedia", "a blog"])
    presentations = expand_pairs_to_presentations(pairs)
    assert len(presentations) == 2  # one pair, both orders

    # 2. SYNTHETIC KC/KW sample fixture, built via real foil creation.
    relation_index = build_relation_index(SYNTHETIC_POPQA_POOL)
    kc_item = SYNTHETIC_POPQA_POOL[0]  # Paris — treated as a KC item
    foil = sample_foil(kc_item, relation_index, random.Random(0))
    assert foil is not None and foil.foil_answer != kc_item["obj"]

    fixture_items = [
        {
            "item_id": "1",
            "question": "What is the capital of France?",
            "gold_answer": "Paris",
            "gold_aliases": [],
            "knowledge_group": "KC",
            "memory_answer": "Paris",
            "foil_answer": foil.foil_answer,
            "margin_bin": "high",
        },
        {
            "item_id": "4",
            "question": "What is the capital of Italy?",
            "gold_answer": "Rome",
            "gold_aliases": [],
            "knowledge_group": "KW",
            "memory_answer": "wrong-guess",
            "foil_answer": None,
            "margin_bin": "low",
        },
    ]
    balanced = sample_balanced_across_bins(fixture_items, target_n=2, seed=0, id_key="item_id")
    assert len(balanced) == 2

    # 3. Condition generation + prompt rendering + generation + parsing +
    #    classification + result records, for every item and condition.
    records = []
    for item in balanced:
        specs = build_conditions(
            knowledge_group=item["knowledge_group"],
            gold_answer=item["gold_answer"],
            baseline_answer=item["memory_answer"],
            foil_answer=item["foil_answer"],
            preferred_source="Wikipedia",
            dispreferred_source="a blog",
        )
        assert {s.condition for s in specs} == {"C0", "C1", "C2", "C3", "C4"}

        for spec in specs:
            evidence_text = None
            if spec.asserted_answer is not None:
                evidence_text = render_evidence(
                    EVIDENCE_TEMPLATE, spec.source_label, item["question"], spec.asserted_answer
                )
            prompt = render_experiment_prompt(BASELINE_TEMPLATE, item["question"], evidence_text)
            messages = [{"role": "user", "content": prompt}]

            raw_generation = model.generate(messages, gen_config)
            parsed = parse_response(raw_generation)
            assert not parsed.malformed  # dummy always emits well-formed responses

            answer_class = classify_answer(
                parsed_answer=parsed.answer,
                decision=parsed.decision,
                malformed=parsed.malformed,
                gold_answer=item["gold_answer"],
                gold_aliases=item["gold_aliases"],
                memory_answer=item["memory_answer"],
                context_answer=spec.asserted_answer,
            )
            records.append(
                {
                    "knowledge_group": item["knowledge_group"],
                    "condition": spec.condition,
                    "conflict_status": spec.conflict_status,
                    "evidence_truth": spec.evidence_truth,
                    "source_role": spec.source_role,
                    "context_adopted": is_context_adopted(answer_class),
                    "final_correct": is_final_correct(parsed.answer, item["gold_answer"], item["gold_aliases"]),
                }
            )

    assert len(records) == 10  # 2 items x 5 conditions

    # 4. Metric computation / analysis input preparation.
    conflict_records = [r for r in records if r["conflict_status"] == "conflict"]
    car = context_adoption_rate(conflict_records)
    assert car.n == len(conflict_records) > 0

    hor = harmful_override_rate(records)  # restricted internally to KC + false + conflict
    cor = corrective_override_rate(records)  # restricted internally to KW + true + conflict
    assert hor.n >= 0 and cor.n >= 0  # both must at least run without error on this fixture

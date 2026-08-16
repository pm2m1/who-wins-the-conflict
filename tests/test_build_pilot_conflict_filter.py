"""Tests that cmd_build_pilot only samples items whose
primary_conflict_eligible flag is True (docs/decisions.md, "Restrict
primary trials to defensible conflicts").

Runs the real `cmd_build_pilot` function against a hand-built baseline
results file — no model, no network.
"""

from __future__ import annotations

import json

import yaml

from conflict_eval.cli import cmd_build_pilot

BASELINE_RECORDS = [
    # KC, primary-conflict eligible (sport) — must be sampled.
    {
        "model_id": "dummy",
        "model_revision": "dummy",
        "item_id": "kc-eligible",
        "subject": "Wayne Gretzky",
        "relation": "sport",
        "question": "What sport did Wayne Gretzky play?",
        "gold_answer": "hockey",
        "gold_aliases": [],
        "parsed_answer": "hockey",
        "baseline_correct": True,
        "knowledge_group": "KC",
        "memory_answer": "hockey",
        "foil_answer": "handball",
        "parametric_margin": 1.0,
        "margin_bin": "medium",
        "primary_conflict_eligible": True,
        "conflict_eligibility_reason": None,
    },
    # KC, relation-excluded (genre) — must NOT be sampled.
    {
        "model_id": "dummy",
        "model_revision": "dummy",
        "item_id": "kc-excluded-relation",
        "subject": "Some Album",
        "relation": "genre",
        "question": "What genre is Some Album?",
        "gold_answer": "drama",
        "gold_aliases": [],
        "parsed_answer": "drama",
        "baseline_correct": True,
        "knowledge_group": "KC",
        "memory_answer": "drama",
        "foil_answer": "comedy",
        "parametric_margin": 1.0,
        "margin_bin": "medium",
        "primary_conflict_eligible": False,
        "conflict_eligibility_reason": "relation_not_primary_conflict",
    },
    # KW, primary-conflict eligible (country) — must be sampled.
    {
        "model_id": "dummy",
        "model_revision": "dummy",
        "item_id": "kw-eligible",
        "subject": "Brown University",
        "relation": "country",
        "question": "What country is Brown University in?",
        "gold_answer": "United States of America",
        "gold_aliases": [],
        "parsed_answer": "Tunisia",
        "baseline_correct": False,
        "knowledge_group": "KW",
        "memory_answer": "Tunisia",
        "parametric_margin": 1.0,
        "margin_bin": "medium",
        "primary_conflict_eligible": True,
        "conflict_eligibility_reason": None,
    },
    # KW, review relation (father) — must NOT be sampled.
    {
        "model_id": "dummy",
        "model_revision": "dummy",
        "item_id": "kw-review-relation",
        "subject": "Someone",
        "relation": "father",
        "question": "Who is Someone's father?",
        "gold_answer": "GoldFather",
        "gold_aliases": [],
        "parsed_answer": "WrongFather",
        "baseline_correct": False,
        "knowledge_group": "KW",
        "memory_answer": "WrongFather",
        "parametric_margin": 1.0,
        "margin_bin": "medium",
        "primary_conflict_eligible": False,
        "conflict_eligibility_reason": "relation_requires_review",
    },
    # KW, multi-object subject (country, but this specific subject has
    # two known objects) — must NOT be sampled.
    {
        "model_id": "dummy",
        "model_revision": "dummy",
        "item_id": "kw-multi-object",
        "subject": "Ambiguous University",
        "relation": "country",
        "question": "What country is Ambiguous University in?",
        "gold_answer": "Canada",
        "gold_aliases": [],
        "parsed_answer": "France",
        "baseline_correct": False,
        "knowledge_group": "KW",
        "memory_answer": "France",
        "parametric_margin": 1.0,
        "margin_bin": "medium",
        "primary_conflict_eligible": False,
        "conflict_eligibility_reason": "relation_multi_object",
    },
]

PILOT_CONFIG = {
    "seed": 42,
    "dataset": {"hf_dataset_id": "akariasai/PopQA", "split": "test", "screening_candidates": 10},
    "sampling": {"target_kc_items": 5, "target_kw_items": 5, "margin_bins": ["low", "medium", "high"]},
    "models": ["dummy"],
    "source_roles": {"dummy": {"preferred_source": "Wikipedia", "dispreferred_source": "a blog"}},
    "prompts_config": "configs/prompts.yaml",
    "sources_config": "configs/sources.yaml",
    "models_config": "configs/models.yaml",
}


def test_build_pilot_only_samples_primary_conflict_eligible_items(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with open(results_dir / "dryrun_baseline.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in BASELINE_RECORDS)

    config = dict(PILOT_CONFIG)
    config["paths"] = {
        "raw_dir": str(tmp_path / "raw"),
        "interim_dir": str(tmp_path / "interim"),
        "processed_dir": str(tmp_path / "processed"),
        "results_dir": str(results_dir),
        "figures_dir": str(tmp_path / "figures"),
    }
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    cmd_build_pilot("dummy", str(config_path))

    trials_path = results_dir / "dryrun_pilot_trials.jsonl"
    trials = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    sampled_item_ids = {t["item_id"] for t in trials}

    assert "kc-eligible" in sampled_item_ids
    assert "kw-eligible" in sampled_item_ids
    assert "kc-excluded-relation" not in sampled_item_ids
    assert "kw-review-relation" not in sampled_item_ids
    assert "kw-multi-object" not in sampled_item_ids

    # Each eligible item contributes exactly 5 conditions (C0-C4).
    assert len([t for t in trials if t["item_id"] == "kc-eligible"]) == 5
    assert len([t for t in trials if t["item_id"] == "kw-eligible"]) == 5

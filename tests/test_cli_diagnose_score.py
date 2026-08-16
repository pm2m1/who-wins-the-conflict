"""End-to-end test of the `diagnose-score` command using the dummy
adapter (no network, no real model). Confirms the CLI wiring — config
loading, prompt rendering, answer_prefix propagation, and margin
reporting — works before it is ever pointed at a real model.
"""

from __future__ import annotations

import yaml

from conflict_eval.cli import cmd_diagnose_score

PILOT_CONFIG = {
    "seed": 42,
    "dataset": {"hf_dataset_id": "akariasai/PopQA", "split": "test", "screening_candidates": 10},
    "paths": {
        "raw_dir": "raw",
        "interim_dir": "interim",
        "processed_dir": "processed",
        "results_dir": "results",
        "figures_dir": "figures",
    },
    "sampling": {"target_kc_items": 2, "target_kw_items": 2, "margin_bins": ["low", "medium", "high"]},
    "models": ["dummy"],
    "source_roles": {"dummy": {"preferred_source": None, "dispreferred_source": None}},
    "prompts_config": "configs/prompts.yaml",
    "sources_config": "configs/sources.yaml",
    "models_config": "configs/models.yaml",
}


def test_diagnose_score_reports_margin_and_direction(tmp_path, capsys):
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(PILOT_CONFIG), encoding="utf-8")

    cmd_diagnose_score(
        "dummy",
        str(config_path),
        question="What is the capital of France?",
        candidate_a="Paris",
        candidate_b="London",
    )

    out = capsys.readouterr().out
    assert "candidate_a='Paris'" in out
    assert "candidate_b='London'" in out
    assert "normalized_logprob=" in out
    assert "margin (A - B) =" in out
    assert "answer_prefix appended after the chat-template generation marker: 'Answer: '" in out

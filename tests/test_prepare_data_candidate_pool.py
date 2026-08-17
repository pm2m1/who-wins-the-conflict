"""Integration test for cmd_prepare_data's dataset.candidate_pool wiring
(docs/decisions.md, "Support targeted primary conflict screening").

Runs the real `cmd_prepare_data` function end to end (interim
normalization, primary-pool construction, deterministic candidate
sampling, file I/O) against a small synthetic raw PopQA fixture — no
network call, no real dataset download. `download_raw` is replaced with a
function that writes the fixture directly to raw_dir, exactly where the
real function would leave it.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from conflict_eval.cli import cmd_prepare_data

# 19 raw rows: PRIMARY relations (sport, country, mother, place of birth),
# one PRIMARY-relation multi-object subject (AmbiguousUni/country), one
# near-duplicate-but-consistent row (Athlete1/sport, case-different
# object text), one REVIEW relation (father), one EXCLUDED relation
# (genre) — see tests/test_primary_relation_pool.py for the equivalent
# pure-function fixture this mirrors.
RAW_ROWS = [
    {"id": "1", "subj": "Athlete1", "prop": "sport", "obj": "hockey", "question": "Q1?"},
    {"id": "2", "subj": "Athlete2", "prop": "sport", "obj": "basketball", "question": "Q2?"},
    {"id": "3", "subj": "Athlete3", "prop": "sport", "obj": "soccer", "question": "Q3?"},
    {"id": "4", "subj": "Athlete4", "prop": "sport", "obj": "tennis", "question": "Q4?"},
    {"id": "5", "subj": "Athlete5", "prop": "sport", "obj": "golf", "question": "Q5?"},
    {"id": "19", "subj": "Athlete1", "prop": "sport", "obj": "Hockey", "question": "Q19?"},
    {"id": "6", "subj": "Uni1", "prop": "country", "obj": "USA", "question": "Q6?"},
    {"id": "7", "subj": "Uni2", "prop": "country", "obj": "Canada", "question": "Q7?"},
    {"id": "8", "subj": "Uni3", "prop": "country", "obj": "France", "question": "Q8?"},
    {"id": "9", "subj": "AmbiguousUni", "prop": "country", "obj": "Germany", "question": "Q9?"},
    {"id": "10", "subj": "AmbiguousUni", "prop": "country", "obj": "Austria", "question": "Q10?"},
    {"id": "11", "subj": "Person1", "prop": "mother", "obj": "Mother1", "question": "Q11?"},
    {"id": "12", "subj": "Person2", "prop": "mother", "obj": "Mother2", "question": "Q12?"},
    {"id": "13", "subj": "Person3", "prop": "place of birth", "obj": "CityA", "question": "Q13?"},
    {"id": "14", "subj": "Person4", "prop": "place of birth", "obj": "CityB", "question": "Q14?"},
    {"id": "15", "subj": "Person5", "prop": "father", "obj": "Father1", "question": "Q15?"},
    {"id": "16", "subj": "Person6", "prop": "father", "obj": "Father2", "question": "Q16?"},
    {"id": "17", "subj": "Album1", "prop": "genre", "obj": "drama", "question": "Q17?"},
    {"id": "18", "subj": "Album2", "prop": "genre", "obj": "comedy", "question": "Q18?"},
]

# Eligible primary rows: sport(6, including the id=19 near-duplicate) +
# country(3, excluding the two AmbiguousUni rows) + mother(2) +
# place of birth(2) = 13. Deduplicated: sport(5, Athlete1's two rows
# collapse to id "1") + country(3) + mother(2) + pob(2) = 12.
EXPECTED_ELIGIBLE_COUNT = 13
EXPECTED_UNIQUE_COUNT = 12


def _build_config(tmp_path, candidate_pool, screening_candidates, seed=42):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": seed,
        "dataset": {
            "hf_dataset_id": "akariasai/PopQA",
            "split": "test",
            "screening_candidates": screening_candidates,
            "candidate_pool": candidate_pool,
        },
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "results_dir": str(tmp_path / "results"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "sampling": {"target_kc_items": 5, "target_kw_items": 5, "margin_bins": ["low", "medium", "high"]},
        "models": ["dummy"],
        "source_roles": {"dummy": {"preferred_source": None, "dispreferred_source": None}},
        "prompts_config": "configs/prompts.yaml",
        "sources_config": "configs/sources.yaml",
        "models_config": "configs/models.yaml",
    }
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _fake_download_raw(hf_dataset_id, split, raw_dir, revision="main"):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "popqa_raw.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in RAW_ROWS)
    return out_path


def _run_prepare_data(config_path):
    with patch("conflict_eval.cli.download_raw", side_effect=_fake_download_raw):
        cmd_prepare_data(str(config_path))


def _read_candidates(tmp_path):
    path = tmp_path / "processed" / "popqa_candidates.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_all_mode_screens_from_the_full_interim_pool(tmp_path):
    # A large screening_candidates cap (>= interim size) means "all" mode
    # must include every relation, including REVIEW/EXCLUDED ones —
    # confirming prior behavior is preserved exactly.
    config_path = _build_config(tmp_path, "all", screening_candidates=100)
    _run_prepare_data(config_path)

    candidates = _read_candidates(tmp_path)
    assert len(candidates) == len(RAW_ROWS)
    relations = {c["prop"] for c in candidates}
    assert "father" in relations
    assert "genre" in relations


def test_primary_conflict_relations_mode_excludes_non_primary_and_multi_object(tmp_path):
    config_path = _build_config(tmp_path, "primary_conflict_relations", screening_candidates=100)
    _run_prepare_data(config_path)

    candidates = _read_candidates(tmp_path)
    # screening_candidates (100) exceeds the eligible pool, so every
    # eligible, deduplicated item is included.
    assert len(candidates) == EXPECTED_UNIQUE_COUNT
    relations = {c["prop"] for c in candidates}
    assert relations == {"sport", "country", "mother", "place of birth"}
    subjects = {c["subj"] for c in candidates}
    assert "AmbiguousUni" not in subjects
    # Only one of the two Athlete1 rows (id "1", not "19") survives.
    athlete1_ids = {c["id"] for c in candidates if c["subj"] == "Athlete1"}
    assert athlete1_ids == {"1"}


def test_candidate_count_respects_screening_candidates(tmp_path):
    config_path = _build_config(tmp_path, "primary_conflict_relations", screening_candidates=5)
    _run_prepare_data(config_path)

    candidates = _read_candidates(tmp_path)
    assert len(candidates) == 5
    # Every sampled candidate must still come from the eligible,
    # deduplicated primary pool.
    relations = {c["prop"] for c in candidates}
    assert relations <= {"sport", "country", "mother", "place of birth"}


def test_seed_determinism_across_repeated_runs(tmp_path):
    config_path_a = _build_config(tmp_path / "a", "primary_conflict_relations", screening_candidates=5, seed=7)
    config_path_b = _build_config(tmp_path / "b", "primary_conflict_relations", screening_candidates=5, seed=7)

    _run_prepare_data(config_path_a)
    _run_prepare_data(config_path_b)

    candidates_a = _read_candidates(tmp_path / "a")
    candidates_b = _read_candidates(tmp_path / "b")
    assert [c["id"] for c in candidates_a] == [c["id"] for c in candidates_b]


def test_different_seed_can_change_the_sampled_subset(tmp_path):
    config_path_seed7 = _build_config(tmp_path / "seed7", "primary_conflict_relations", screening_candidates=5, seed=7)
    config_path_seed99 = _build_config(tmp_path / "seed99", "primary_conflict_relations", screening_candidates=5, seed=99)

    _run_prepare_data(config_path_seed7)
    _run_prepare_data(config_path_seed99)

    ids_seed7 = {c["id"] for c in _read_candidates(tmp_path / "seed7")}
    ids_seed99 = {c["id"] for c in _read_candidates(tmp_path / "seed99")}
    assert ids_seed7 != ids_seed99

"""Confirms .env-shaped files are actually excluded by git, using git's
own ignore-matching (`git check-ignore`) rather than a brittle text-content
assertion against .gitignore (docs/decisions.md, "Local secret-file
protection").

Skipped if `git` is not on PATH or this checkout is not a git repository
(e.g. an extracted archive) rather than failing outright.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return (REPO_ROOT / ".git").exists()
    except (OSError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(not _git_available(), reason="git not available or not a git checkout")


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    # git check-ignore exits 0 if the path IS ignored, 1 if it is not.
    return result.returncode == 0


def test_env_file_is_ignored():
    assert _is_ignored(".env")


def test_env_variant_files_are_ignored():
    assert _is_ignored(".env.local")
    assert _is_ignored(".env.production")


def test_env_example_is_not_ignored():
    # The template file must remain trackable even though it matches
    # the .env.* pattern.
    assert not _is_ignored(".env.example")


def test_existing_ignore_rules_still_present():
    assert _is_ignored(".venv/some_file")
    assert _is_ignored("__pycache__/mod.pyc")
    assert _is_ignored(".pytest_cache/README.md")
    assert _is_ignored(".ruff_cache/whatever")


def test_replication_run_outputs_are_ignored():
    # runs/ holds regenerated per-replication artifacts (datasets, model
    # outputs, baseline/calibration/pilot results, manifests) for e.g. the
    # Llama replication and must never be committed (docs/decisions.md,
    # "Precommit the Llama second-model replication").
    assert _is_ignored("runs/llama/results/whatever.jsonl")
    assert _is_ignored("runs/llama/data/raw/popqa_raw.jsonl")


def test_replication_configs_are_not_ignored():
    # The precommitted replication specification itself must stay
    # trackable even though it lives alongside the ignored runs/ tree.
    assert not _is_ignored("configs/replication/llama_pilot.yaml")
    assert not _is_ignored("configs/replication/models_llama.yaml")

import pytest

from conflict_eval.config import ConfigError, SourceRoleConfig, load_sources_config


def test_source_role_config_is_set_requires_both_fields():
    assert not SourceRoleConfig(preferred_source=None, dispreferred_source=None).is_set()
    assert not SourceRoleConfig(preferred_source="Wikipedia", dispreferred_source=None).is_set()
    assert SourceRoleConfig(preferred_source="Wikipedia", dispreferred_source="a blog").is_set()


def test_load_sources_config_requires_at_least_two_labels(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text("source_labels:\n  - OnlyOne\ncalibration_prompt_version: v1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sources_config(path)


def test_load_sources_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_sources_config(tmp_path / "does_not_exist.yaml")


def test_load_sources_config_reads_real_project_config():
    # configs/sources.yaml is checked in and must always be loadable.
    config = load_sources_config("configs/sources.yaml")
    assert len(config["source_labels"]) >= 2
    assert "calibration_prompt_version" in config

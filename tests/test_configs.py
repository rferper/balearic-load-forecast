"""Config parsing, merging and validation."""

import pytest
from pydantic import ValidationError

from balearic_load_forecast.io import configs
from balearic_load_forecast.settings import PredictConfig, TrainConfig


def test_defaults_apply_when_no_file_exists(tmp_path):
    # Given: no config file
    # When: the config is loaded
    config = configs.load_config(TrainConfig, path=tmp_path / "absent.yaml")
    # Then: the documented defaults are used rather than an error raised
    assert config.validation_days == 90
    assert config.model.num_leaves == 63


def test_yaml_values_override_defaults(tmp_path):
    # Given: a config file setting one nested field
    path = tmp_path / "train.yaml"
    path.write_text("validation_days: 30\nmodel:\n  num_leaves: 15\n", encoding="utf-8")
    # When: it is loaded
    config = configs.load_config(TrainConfig, path=path)
    # Then: the file wins, and untouched fields keep their defaults
    assert config.validation_days == 30
    assert config.model.num_leaves == 15
    assert config.model.n_estimators == 600


def test_cli_overrides_beat_the_file(tmp_path):
    # Given: a file and a conflicting command-line override
    path = tmp_path / "train.yaml"
    path.write_text("validation_days: 30\n", encoding="utf-8")
    # When: both are merged
    config = configs.load_config(
        TrainConfig, path=path, overrides=["validation_days=7", "model.num_leaves=31"]
    )
    # Then: the command line wins
    assert config.validation_days == 7
    assert config.model.num_leaves == 31


def test_a_typo_fails_immediately_rather_than_being_ignored():
    # Given: a misspelled key
    # When/Then: validation rejects it instead of silently doing nothing
    with pytest.raises(ValidationError, match="validaton_days"):
        configs.load_config(TrainConfig, overrides=["validaton_days=7"])


def test_out_of_range_values_are_rejected():
    # Given: a learning rate above 1
    # When/Then: the constraint fires before any training starts
    with pytest.raises(ValidationError, match="learning_rate"):
        configs.load_config(TrainConfig, overrides=["model.learning_rate=5.0"])


def test_target_day_parses_from_a_string():
    # Given: a target day given as a CLI string
    config = configs.load_config(PredictConfig, overrides=["target_day=2026-01-15"])
    # When/Then: it arrives as a date, not a string
    assert config.target_day.isoformat() == "2026-01-15"


def test_config_path_follows_the_convention(tmp_path):
    # Given: a confs directory
    # When/Then: each job maps to <confs>/<job>.yaml
    assert configs.config_path("train", tmp_path) == tmp_path / "train.yaml"

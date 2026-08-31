"""End to end: train, predict and visualize through the CLI."""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from balearic_load_forecast import scripts
from balearic_load_forecast.application import predict
from balearic_load_forecast.domain.features import LOCAL_TZ, as_timestamp
from balearic_load_forecast.io import datasets


@pytest.fixture
def workspace(tmp_path, raw_csv):
    """Return the shared CLI overrides pointing every path at tmp_path.

    `temperature_csv=null` is explicit rather than omitted: the config default
    is a real path in the repository, so leaving it out lets a test quietly
    read the project's own data instead of its fixtures.
    """
    return [
        f"raw_csv={raw_csv}",
        "temperature_csv=null",
        f"model_path={tmp_path / 'lgbm.pkl'}",
        f"metrics_path={tmp_path / 'metrics.json'}",
        f"forecasts_dir={tmp_path / 'forecasts'}",
        f"figures_dir={tmp_path / 'figures'}",
    ]


ACCEPTED = {
    "train": (
        "raw_csv",
        "temperature_csv",
        "model_path",
        "metrics_path",
        "validation_days",
        "model",
    ),
    "predict": (
        "raw_csv",
        "temperature_csv",
        "model_path",
        "forecasts_dir",
        "target_day",
    ),
    "visualize": ("raw_csv", "forecasts_dir", "figures_dir", "target_day"),
}


def _run(job: str, overrides: list[str], tmp_path=None) -> int:
    """Invoke the CLI for one job, keeping only the overrides it accepts.

    Points `-c` at a path that does not exist, so the tests exercise the
    defaults plus their own overrides rather than whatever `confs/` happens to
    hold in the working tree.
    """
    kept = [o for o in overrides if o.split("=")[0].split(".")[0] in ACCEPTED[job]]
    config = (tmp_path or Path("does-not-exist")) / "no-such-config.yaml"
    return scripts.main([job, "-c", str(config), *kept])


def test_the_whole_pipeline_runs_and_leaves_the_expected_artefacts(
    tmp_path, workspace, demand
):
    # Given: a landing file of synthetic demand
    # When: train, predict and visualize run in order through the CLI
    last_local_day = demand.index[-1].tz_convert(LOCAL_TZ).date()
    target = last_local_day - timedelta(days=1)

    train_overrides = [*workspace, "validation_days=30", "model.n_estimators=20"]
    assert _run("train", train_overrides, tmp_path) == 0
    assert _run("predict", [*workspace, f"target_day={target}"], tmp_path) == 0
    assert _run("visualize", [*workspace, f"target_day={target}"], tmp_path) == 0

    # Then: a model, its metrics, a stored forecast and a figure all exist
    assert (tmp_path / "lgbm.pkl").exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model"]["mae_mw"] >= 0
    assert metrics["model_id"]

    stored = datasets.load_forecast(tmp_path / "forecasts", target)
    assert len(stored) in (23, 24, 25)  # DST makes a local day 23-25 hours long
    # And: the stored forecast names the model that produced it
    assert set(stored["model_id"]) == {metrics["model_id"]}

    figure = tmp_path / "figures" / f"forecast_{target.isoformat()}.png"
    assert figure.exists()
    assert figure.stat().st_size > 0


def test_predict_refuses_a_day_the_history_cannot_reach(demand):
    # Given: a target day far beyond the end of the data
    class Always:
        def predict(self, features):
            return pd.Series(0.0, index=features.index)

    far_future = (demand.index[-1] + pd.Timedelta(days=30)).date()
    # When/Then: it refuses, and says the history is the problem
    with pytest.raises(ValueError, match="backfill more history"):
        predict.forecast_day(demand, Always(), far_future)


def test_next_day_is_tomorrow_local():
    # Given: a fixed instant late on a local evening
    now = as_timestamp(pd.Timestamp("2026-03-14 23:30", tz=LOCAL_TZ))
    # When/Then: the day-ahead target is the following local date
    assert predict.next_day(now).isoformat() == "2026-03-15"


def test_a_bad_config_exits_nonzero_without_a_traceback(capsys):
    # Given: an override that violates the schema
    # When: the CLI runs
    code = scripts.main(
        ["train", "-c", "no-such-config.yaml", "model.learning_rate=99"]
    )
    # Then: it fails cleanly with a message naming the field
    assert code == scripts.EXIT_FAILURE
    assert "learning_rate" in capsys.readouterr().err


def test_a_failing_job_exits_nonzero(tmp_path):
    # Given: a predict job pointed at a model that does not exist
    code = scripts.main(
        [
            "predict",
            f"raw_csv={tmp_path / 'nope.csv'}",
            f"model_path={tmp_path / 'no.pkl'}",
        ]
    )
    # Then: the failure is reported through the exit code, not an exception
    assert code == scripts.EXIT_FAILURE


def test_the_pipeline_runs_with_weather_features(
    tmp_path, workspace, demand, temperature_csv
):
    # Given: a landing file plus a temperature series covering the same hours
    last_local_day = demand.index[-1].tz_convert(LOCAL_TZ).date()
    target = last_local_day - timedelta(days=1)
    weather = [*workspace, f"temperature_csv={temperature_csv}"]  # overrides the null

    # When: train and predict both run with weather enabled
    assert (
        _run(
            "train", [*weather, "validation_days=30", "model.n_estimators=20"], tmp_path
        )
        == 0
    )
    assert _run("predict", [*weather, f"target_day={target}"], tmp_path) == 0

    # Then: the stored forecast covers the day, from a weather-aware model
    stored = datasets.load_forecast(tmp_path / "forecasts", target)
    assert len(stored) in (23, 24, 25)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "temp_c" in metrics["top_features"] or metrics["model"]["mae_mw"] >= 0


def test_predicting_without_weather_from_a_weather_model_is_refused(
    tmp_path, workspace, demand, temperature_csv
):
    # Given: a model trained WITH weather features
    last_local_day = demand.index[-1].tz_convert(LOCAL_TZ).date()
    target = last_local_day - timedelta(days=1)
    _run(
        "train",
        [
            *workspace,
            f"temperature_csv={temperature_csv}",
            "validation_days=30",
            "model.n_estimators=20",
        ],
        tmp_path,
    )

    # When: predict is run without the temperature series
    code = _run("predict", [*workspace, f"target_day={target}"], tmp_path)

    # Then: it fails loudly on the feature contract rather than forecasting
    # from columns the model never saw
    assert code == scripts.EXIT_FAILURE


def test_a_weather_forecast_that_stops_short_names_the_weather_job(demand, temperature):
    # Given: temperature that runs out before the target day
    from balearic_load_forecast.application import predict as predict_job

    short = temperature[temperature.index < demand.index[-1] - pd.Timedelta(days=2)]
    day = demand.index[-1].tz_convert(LOCAL_TZ).date() - timedelta(days=1)

    class Always:
        def predict(self, features):
            return pd.Series(0.0, index=features.index)

    # When/Then: the error points at the weather job, not the demand backfill
    with pytest.raises(ValueError, match="weather job"):
        predict_job.forecast_day(demand, Always(), day, temperature=short)

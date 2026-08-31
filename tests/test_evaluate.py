"""The accuracy scorecard, including pending and partial days."""

import json
from datetime import date

import pandas as pd
import pytest

from balearic_load_forecast.application import evaluate
from balearic_load_forecast.domain import metrics, models
from balearic_load_forecast.io import datasets
from balearic_load_forecast.settings import EvaluateConfig


def _day_index(day: str) -> pd.DatetimeIndex:
    return pd.date_range(day, periods=24, freq="h", tz="UTC")


@pytest.fixture
def store(tmp_path, demand):
    """A forecast store holding three days a real demand series covers."""
    forecasts = tmp_path / "forecasts"
    start = demand.index[-1] - pd.Timedelta(days=4)
    days = []
    for offset in range(3):
        index = pd.date_range(
            start + pd.Timedelta(days=offset), periods=24, freq="h", tz="UTC"
        )
        # A forecast that is wrong by a known, constant 10 MW.
        forecast = demand.reindex(index) + 10.0
        day = index[0].date()
        datasets.save_forecast(forecast, forecasts, day, model_id="test01")
        days.append(day)
    return forecasts, days


def _config(tmp_path, raw_csv, forecasts, **kwargs):
    return EvaluateConfig(
        raw_csv=raw_csv,
        forecasts_dir=forecasts,
        scores_path=tmp_path / "scores.csv",
        metrics_path=tmp_path / "metrics.json",
        **kwargs,
    )


def test_a_known_error_is_measured_exactly(tmp_path, raw_csv, store):
    # Given: three stored forecasts, each wrong by exactly 10 MW
    forecasts, _days = store
    # When: they are scored
    scores = evaluate.run(_config(tmp_path, raw_csv, forecasts))
    # Then: every day reports that 10 MW, and the scorecard is written
    assert len(scores) == 3
    assert scores["mae_mw"].round(6).tolist() == [10.0, 10.0, 10.0]
    assert (tmp_path / "scores.csv").exists()


def test_the_scorecard_names_the_model_and_the_day(tmp_path, raw_csv, store):
    # Given: a scored store
    forecasts, days = store
    evaluate.run(_config(tmp_path, raw_csv, forecasts))
    # When: the scorecard is read back
    stored = datasets.load_scores(tmp_path / "scores.csv")
    # Then: each row is traceable to a day and the model that produced it
    assert list(stored.columns) == datasets.SCORE_COLUMNS
    assert set(stored["model_id"]) == {"test01"}
    assert stored["target_day"].dt.date.tolist() == days


def test_a_day_without_actuals_is_pending_not_scored(tmp_path, raw_csv, demand):
    # Given: a forecast for a day the demand series does not reach
    forecasts = tmp_path / "forecasts"
    future = demand.index[-1] + pd.Timedelta(days=3)
    index = pd.date_range(future, periods=24, freq="h", tz="UTC")
    datasets.save_forecast(
        pd.Series(1000.0, index=index), forecasts, index[0].date(), "m"
    )

    # When: evaluation runs
    scores = evaluate.run(_config(tmp_path, raw_csv, forecasts))

    # Then: nothing is scored, and no misleading scorecard is written
    assert scores.empty
    assert not (tmp_path / "scores.csv").exists()


def test_a_partly_published_day_is_pending_too(tmp_path, raw_csv, demand):
    # Given: a forecast whose last hours run past the end of the actuals
    forecasts = tmp_path / "forecasts"
    index = pd.date_range(
        demand.index[-1] - pd.Timedelta(hours=12), periods=24, freq="h", tz="UTC"
    )
    datasets.save_forecast(
        pd.Series(1000.0, index=index), forecasts, index[0].date(), "m"
    )

    # When: evaluation runs
    scores = evaluate.run(_config(tmp_path, raw_csv, forecasts))

    # Then: half a day of error is not comparable to a full one, so it waits
    assert scores.empty


def test_rerunning_rebuilds_rather_than_duplicating(tmp_path, raw_csv, store):
    # Given: a store that has already been scored
    forecasts, _days = store
    config = _config(tmp_path, raw_csv, forecasts)
    evaluate.run(config)
    # When: the job runs again
    scores = evaluate.run(config)
    # Then: the scorecard still holds one row per day
    assert len(scores) == 3
    assert len(datasets.load_scores(tmp_path / "scores.csv")) == 3


def test_drift_is_reported_against_the_training_metrics(tmp_path, raw_csv, store):
    # Given: training metrics claiming a 1 MW validation MAE
    forecasts, _ = store
    (tmp_path / "metrics.json").write_text(
        json.dumps({"model": {"mae_mw": 1.0}}), encoding="utf-8"
    )
    config = _config(tmp_path, raw_csv, forecasts, drift_tolerance=1.5)

    # When: production error is 10 MW, far above tolerance
    # Then: the expected MAE is recovered so the check can fire
    assert evaluate.expected_mae(config.metrics_path) == 1.0
    assert not evaluate.run(config).empty


def test_missing_training_metrics_are_not_an_error(tmp_path, raw_csv, store):
    # Given: no metrics.json (the model predates the scorecard)
    forecasts, _ = store
    # When/Then: evaluation still scores, it just cannot drift-check
    assert evaluate.expected_mae(tmp_path / "absent.json") is None
    assert len(evaluate.run(_config(tmp_path, raw_csv, forecasts))) == 3


def test_scorecard_measures_the_peak_with_a_signed_error():
    # Given: a forecast that under-calls the peak by 50 MW
    index = _day_index("2023-06-01")
    actual = pd.Series(1000.0, index=index)
    actual.iloc[12] = 1500.0
    forecast = pd.Series(1000.0, index=index)
    forecast.iloc[12] = 1450.0

    # When: it is scored
    row = metrics.forecast_scorecard(actual, forecast)

    # Then: the sign says which way it missed - under-forecasting is negative
    assert row["peak_actual_mw"] == 1500.0
    assert row["peak_forecast_mw"] == 1450.0
    assert row["peak_error_mw"] == -50.0


def test_scorecard_without_a_baseline_records_nan_not_a_wrong_number():
    # Given: a forecast scored with an incomplete baseline
    index = _day_index("2023-06-01")
    actual = pd.Series(1000.0, index=index)
    baseline = pd.Series(1000.0, index=index)
    baseline.iloc[0] = float("nan")

    # When: it is scored
    row = metrics.forecast_scorecard(actual, actual + 5, baseline)

    # Then: the baseline columns are NaN rather than a partial-week average
    assert pd.isna(row["baseline_mae_mw"])
    assert pd.isna(row["mae_improvement"])


def test_seasonal_naive_reconstructs_last_weeks_demand(demand):
    # Given: an index one week after the start of the series
    index = pd.DatetimeIndex(demand.index[200:224])
    # When: the baseline is rebuilt from the demand series alone
    baseline = models.seasonal_naive_from_demand(demand, index)
    # Then: it is the same hours, one week earlier
    expected = demand.iloc[200 - 168 : 224 - 168]
    assert baseline.to_numpy() == pytest.approx(expected.to_numpy())


def test_the_baseline_matches_the_model_that_predicts_from_features(demand):
    # Given: the feature table and the demand series describe the same hours
    from balearic_load_forecast.domain.features import make_training_table

    table = make_training_table(demand).head(48)
    index = pd.DatetimeIndex(table.index)

    # When: the baseline is produced both ways
    from_features = models.SeasonalNaiveModel().predict(table)
    from_demand = models.seasonal_naive_from_demand(demand, index)

    # Then: the two agree - the evaluation path cannot silently diverge
    assert from_demand.to_numpy() == pytest.approx(from_features.to_numpy())


def test_pending_and_scored_days_coexist(tmp_path, raw_csv, demand, store):
    # Given: a store with three scoreable days plus one in the future
    forecasts, days = store
    future = demand.index[-1] + pd.Timedelta(days=5)
    index = pd.date_range(future, periods=24, freq="h", tz="UTC")
    datasets.save_forecast(
        pd.Series(1000.0, index=index), forecasts, index[0].date(), "m"
    )

    # When: evaluation runs
    scores = evaluate.run(_config(tmp_path, raw_csv, forecasts))

    # Then: the three are scored and the pending one is simply left out
    assert len(scores) == 3
    assert date.fromisoformat(scores["target_day"].max()) == max(days)


def test_recent_window_limits_what_the_summary_covers(tmp_path, raw_csv, store):
    # Given: three scored days but a one-day reporting window
    forecasts, _ = store
    config = _config(tmp_path, raw_csv, forecasts, recent_days=1)
    # When/Then: all three are still stored; the window only narrows reporting
    assert len(evaluate.run(config)) == 3


def test_evaluate_runs_through_the_cli(tmp_path, raw_csv, store):
    # Given: a scoreable store
    from balearic_load_forecast import scripts

    forecasts, _ = store
    # When: the job is invoked the way a scheduler would
    code = scripts.main(
        [
            "evaluate",
            "-c",
            str(tmp_path / "none.yaml"),
            f"raw_csv={raw_csv}",
            f"forecasts_dir={forecasts}",
            f"scores_path={tmp_path / 'scores.csv'}",
            f"metrics_path={tmp_path / 'metrics.json'}",
        ]
    )
    # Then: it succeeds and leaves the scorecard behind
    assert code == scripts.EXIT_OK
    assert (tmp_path / "scores.csv").exists()


def test_an_empty_store_is_reported_not_crashed(tmp_path, raw_csv):
    # Given: no forecasts at all
    # When: evaluation runs
    scores = evaluate.run(_config(tmp_path, raw_csv, tmp_path / "empty"))
    # Then: it returns an empty scorecard with the right shape
    assert scores.empty
    assert list(scores.columns) == datasets.SCORE_COLUMNS


def test_yesterdays_forecast_becomes_scoreable_once_actuals_land(tmp_path, demand):
    # Given: a forecast for a day, and actuals that stop just before it
    forecasts = tmp_path / "forecasts"
    index = pd.date_range(
        demand.index[-1] - pd.Timedelta(days=1), periods=24, freq="h", tz="UTC"
    )
    day = index[0].date()
    datasets.save_forecast(demand.reindex(index) + 10.0, forecasts, day, "m")

    short = demand[demand.index < index[0]]
    short_csv = tmp_path / "short.csv"
    frame = short.rename("value").to_frame()
    frame.index.name = "datetime"
    frame.to_csv(short_csv)

    # When: scored against the truncated actuals, it is pending
    assert evaluate.run(_config(tmp_path, short_csv, forecasts)).empty

    # And when the actuals arrive, the same forecast scores
    full_csv = tmp_path / "full.csv"
    frame = demand.rename("value").to_frame()
    frame.index.name = "datetime"
    frame.to_csv(full_csv)
    scores = evaluate.run(_config(tmp_path, full_csv, forecasts))

    # Then: the loop has closed on a forecast made before the outcome was known
    assert len(scores) == 1
    assert scores["mae_mw"].iloc[0] == pytest.approx(10.0)
    assert scores["target_day"].iloc[0] == day.isoformat()

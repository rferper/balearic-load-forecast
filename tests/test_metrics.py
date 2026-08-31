"""Scoring and the time-based split."""

import pandas as pd

from balearic_load_forecast.domain import metrics
from balearic_load_forecast.domain.features import as_timestamp


def test_split_never_leaks_the_future_into_training():
    # Given: a table spanning ten days
    index = pd.date_range("2023-01-01", periods=240, freq="h", tz="UTC")
    table = pd.DataFrame({"target": range(240)}, index=index)
    cutoff = as_timestamp(pd.Timestamp("2023-01-08", tz="UTC"))

    # When: it is split by time
    train, valid = metrics.split_by_time(table, cutoff)

    # Then: every training row precedes every validation row
    assert train.index.max() < cutoff <= valid.index.min()
    assert len(train) + len(valid) == len(table)


def test_validation_cutoff_holds_out_the_requested_tail():
    # Given: a table spanning ten days
    index = pd.date_range("2023-01-01", periods=240, freq="h", tz="UTC")
    table = pd.DataFrame({"target": range(240)}, index=index)
    # When: a three-day tail is held out
    cutoff = metrics.validation_cutoff(table, days=3)
    # Then: the cutoff sits three days before the end
    assert cutoff == table.index.max() - pd.Timedelta(days=3)


def test_a_perfect_forecast_scores_zero():
    # Given: a forecast identical to the actuals
    actual = pd.Series([100.0, 200.0, 300.0])
    # When: it is scored
    scores = metrics.score(actual, actual.copy())
    # Then: every metric is zero
    assert scores == {"mae_mw": 0.0, "rmse_mw": 0.0, "mape_pct": 0.0}


def test_metrics_have_their_documented_units():
    # Given: a forecast that is 10 MW low on a 100 MW actual
    actual = pd.Series([100.0, 100.0])
    forecast = pd.Series([90.0, 90.0])
    # When: it is scored
    scores = metrics.score(actual, forecast)
    # Then: MAE is in MW and MAPE is a percentage, not a fraction
    assert scores["mae_mw"] == 10.0
    assert scores["mape_pct"] == 10.0


def test_rmse_punishes_one_large_miss_more_than_mae():
    # Given: an actual series and a forecast with a single big error
    actual = pd.Series([100.0, 100.0, 100.0, 100.0])
    forecast = pd.Series([100.0, 100.0, 100.0, 60.0])
    # When: both metrics are computed
    # Then: RMSE exceeds MAE
    assert metrics.rmse(actual, forecast) > metrics.mae(actual, forecast)

"""Feature building, including the no-leakage guarantee."""

import pandas as pd

from balearic_load_forecast.domain import features


def test_feature_table_has_the_contract_columns_in_order(demand):
    # Given: a demand series
    # When: features are built
    table = features.make_features(demand)
    # Then: the column contract holds exactly, order included
    assert list(table.columns) == features.FEATURE_COLUMNS


def test_lags_never_use_demand_newer_than_the_minimum_lag(demand):
    # Given: a series whose last 48 hours are replaced by an absurd spike
    poisoned = demand.copy()
    poisoned.iloc[-features.MIN_LAG_HOURS :] = 99_999.0

    # When: features are built from both series
    clean = features.make_features(demand)
    dirty = features.make_features(poisoned)

    # Then: the final row is unchanged - no feature saw the recent hours
    pd.testing.assert_series_equal(clean.iloc[-1], dirty.iloc[-1])


def test_until_extends_the_table_into_the_future(demand):
    # Given: a target hour one day past the end of the data
    until = demand.index[-1] + pd.Timedelta(days=1)
    # When: features are built out to it
    table = features.make_features(demand, until=until)
    # Then: rows exist for hours that have not happened yet
    assert table.index[-1] == until
    assert table.loc[until, "hour"] == until.tz_convert(features.LOCAL_TZ).hour


def test_gaps_shift_by_hours_not_by_rows():
    # Given: a series with an eight-hour hole in the middle
    index = pd.date_range("2023-01-01", periods=400, freq="h", tz="UTC")
    series = pd.Series(range(400), index=index, dtype=float, name="demand")
    holed = series.drop(series.index[100:108])

    # When: features are built
    table = features.make_features(holed)

    # Then: lag_48h at a chosen hour is the value 48 hours earlier by clock,
    # not 48 rows earlier in the compacted frame
    at = index[200]
    assert table.loc[at, "lag_48h"] == series.loc[index[152]]


def test_training_table_drops_incomplete_rows(demand):
    # Given: a demand series
    # When: the training table is built
    table = features.make_training_table(demand)
    # Then: no NaN survives, and the target column is present
    assert not table.isna().to_numpy().any()
    assert "target" in table.columns
    # And: the first rows, whose lookback is incomplete, are gone
    assert len(table) <= len(demand) - features.MAX_LOOKBACK_HOURS


def test_holidays_are_flagged():
    # Given: hours spanning a Spanish national holiday and an ordinary day
    index = pd.date_range("2023-01-05", "2023-01-07", freq="h", tz="UTC")
    table = features.add_holiday_features(features.add_calendar_features(index))
    # When/Then: 6 January (Epiphany) is a holiday, 5 January is not
    assert table.loc["2023-01-06 12:00+00:00", "is_holiday"] == 1
    assert table.loc["2023-01-05 12:00+00:00", "is_holiday"] == 0

"""The raw landing file and the forecast store."""

import pandas as pd
import pytest

from balearic_load_forecast.io import datasets


def test_load_demand_deduplicates_overlapping_appends(tmp_path):
    # Given: a landing file where two appended chunks overlap
    path = tmp_path / "raw.csv"
    path.write_text(
        "datetime,value\n"
        "2023-01-01 00:00:00+00:00,100.0\n"
        "2023-01-01 01:00:00+00:00,110.0\n"
        "2023-01-01 01:00:00+00:00,999.0\n",
        encoding="utf-8",
    )
    # When: it is loaded
    series = datasets.load_demand(path)
    # Then: the first occurrence wins and the index is unique
    assert len(series) == 2
    assert series.loc["2023-01-01 01:00:00+00:00"] == 110.0


def test_load_demand_says_what_to_do_when_there_is_no_data(tmp_path):
    # Given: no landing file
    # When/Then: the error names the fix rather than just the path
    with pytest.raises(FileNotFoundError, match="backfill"):
        datasets.load_demand(tmp_path / "missing.csv")


def test_append_readings_writes_a_header_once(tmp_path):
    # Given: two chunks of raw API readings
    path = tmp_path / "raw.csv"
    chunk = [{"datetime": "2023-01-01T00:00:00.000+01:00", "value": 100.0, "x": 1}]
    # When: both are appended
    datasets.append_readings(chunk, path)
    datasets.append_readings(chunk, path)
    # Then: one header, two rows, and the extra API field is dropped
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "datetime,value"
    assert len(lines) == 3


def test_append_readings_ignores_an_empty_chunk(tmp_path):
    # Given: a day the API had no data for
    path = tmp_path / "raw.csv"
    # When: the empty result is appended
    written = datasets.append_readings([], path)
    # Then: nothing is written and no empty file is created
    assert written == 0
    assert not path.exists()


def test_stored_forecast_round_trips_with_its_provenance(tmp_path):
    # Given: a one-day forecast
    index = pd.date_range("2023-06-01", periods=24, freq="h", tz="UTC")
    forecast = pd.Series(range(24), index=index, dtype=float, name="forecast")
    day = index[0].date()

    # When: it is stored and read back
    datasets.save_forecast(forecast, tmp_path, day, model_id="abc123")
    stored = datasets.load_forecast(tmp_path, day)

    # Then: the values survive, stamped with the model that made them
    assert len(stored) == 24
    assert stored["forecast_mw"].iloc[-1] == 23.0
    assert set(stored["model_id"]) == {"abc123"}
    assert stored["generated_at"].notna().all()


def test_rerunning_predict_replaces_rather_than_appends(tmp_path):
    # Given: a forecast already stored for a day
    index = pd.date_range("2023-06-01", periods=24, freq="h", tz="UTC")
    day = index[0].date()
    datasets.save_forecast(pd.Series(1.0, index=index), tmp_path, day, "v1")
    # When: the job runs again for the same day
    datasets.save_forecast(pd.Series(2.0, index=index), tmp_path, day, "v2")
    # Then: the store holds one day's worth of the newer forecast
    stored = datasets.load_forecast(tmp_path, day)
    assert len(stored) == 24
    assert set(stored["model_id"]) == {"v2"}


def test_stored_days_are_sorted_and_latest_is_the_newest(tmp_path):
    # Given: forecasts stored out of order
    for day in ("2023-06-03", "2023-06-01", "2023-06-02"):
        index = pd.date_range(day, periods=2, freq="h", tz="UTC")
        datasets.save_forecast(
            pd.Series(1.0, index=index), tmp_path, index[0].date(), "m"
        )
    # When: the store is listed
    days = datasets.stored_days(tmp_path)
    # Then: ascending order, newest last
    assert [d.isoformat() for d in days] == ["2023-06-01", "2023-06-02", "2023-06-03"]
    assert datasets.latest_stored_day(tmp_path) == days[-1]


def test_empty_store_says_to_run_predict(tmp_path):
    # Given: an empty forecast store
    # When/Then: the error names the job that fills it
    with pytest.raises(FileNotFoundError, match="predict"):
        datasets.latest_stored_day(tmp_path)

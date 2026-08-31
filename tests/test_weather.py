"""Temperature: the client, the store, and the features built from it.

Demand features must not see the recent past; temperature features must see
the target day.
"""

from datetime import date

import pandas as pd
import pytest
import requests

from balearic_load_forecast.application import weather as weather_job
from balearic_load_forecast.domain import features
from balearic_load_forecast.io import datasets, weather
from balearic_load_forecast.settings import WeatherConfig


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def json(self) -> dict:
        return self.payload


def _payload(times: list[str], temps: list[float | None]) -> dict:
    return {"hourly": {"time": times, "temperature_2m": temps}}


# -- Client -------------------------------------------------------------------


def test_archive_request_asks_for_hourly_utc_at_the_site(monkeypatch):
    # Given: a stubbed transport that records the request
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(url=url, params=params)
        return _Response(_payload(["2023-01-01T00:00"], [12.5]))

    monkeypatch.setattr(weather.requests, "get", fake_get)

    # When: the archive is fetched
    series = weather.fetch_archive(date(2023, 1, 1), date(2023, 1, 2), 39.5, 2.6)

    # Then: hourly UTC at the requested point, parsed into a tz-aware series
    assert seen["url"] == weather.ARCHIVE_URL
    assert seen["params"]["timezone"] == "UTC"
    assert seen["params"]["hourly"] == "temperature_2m"
    assert seen["params"]["latitude"] == 39.5
    assert series.iloc[0] == 12.5
    assert str(series.index.tz) == "UTC"


def test_forecast_request_spans_past_and_future(monkeypatch):
    # Given: a stubbed forecast endpoint
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(url=url, params=params)
        return _Response(
            _payload(["2023-01-01T00:00", "2023-01-01T01:00"], [10.0, 11.0])
        )

    monkeypatch.setattr(weather.requests, "get", fake_get)

    # When: the forecast is fetched
    series = weather.fetch_forecast(39.5, 2.6, past_days=3, forecast_days=5)

    # Then: both directions are requested in one call
    assert seen["url"] == weather.FORECAST_URL
    assert seen["params"]["past_days"] == 3
    assert seen["params"]["forecast_days"] == 5
    assert len(series) == 2


def test_hours_the_api_could_not_supply_are_dropped(monkeypatch):
    # Given: a response with a null temperature
    monkeypatch.setattr(
        weather.requests,
        "get",
        lambda *_, **__: _Response(
            _payload(["2023-01-01T00:00", "2023-01-01T01:00"], [10.0, None])
        ),
    )
    # When: it is parsed
    series = weather.fetch_archive(date(2023, 1, 1), date(2023, 1, 1), 39.5, 2.6)
    # Then: the gap is dropped rather than kept as NaN
    assert len(series) == 1


def test_a_rejected_request_raises(monkeypatch):
    # Given: an API returning 400
    monkeypatch.setattr(
        weather.requests, "get", lambda *_, **__: _Response({}, status=400)
    )
    # When/Then: the failure surfaces
    with pytest.raises(requests.HTTPError):
        weather.fetch_archive(date(2023, 1, 1), date(2023, 1, 2), 39.5, 2.6)


# -- Store --------------------------------------------------------------------


def test_a_forecast_value_is_replaced_by_the_measured_one():
    # Given: a stored series whose last hour was a forecast of 20 C
    index = pd.date_range("2023-06-01", periods=3, freq="h", tz="UTC")
    stored = pd.Series([18.0, 19.0, 20.0], index=index)
    # And: reanalysis later says that hour was actually 24 C
    fresh = pd.Series([24.0], index=index[-1:])

    # When: they are merged
    merged = datasets.merge_temperature(stored, fresh)

    # Then: the measured value wins
    assert merged.loc[index[-1]] == 24.0
    assert len(merged) == 3


def test_merging_onto_nothing_is_a_cold_start():
    # Given: no stored series
    index = pd.date_range("2023-06-01", periods=2, freq="h", tz="UTC")
    fresh = pd.Series([18.0, 19.0], index=index)
    # When/Then: the fresh data simply becomes the series
    merged = datasets.merge_temperature(None, fresh)
    assert merged.to_numpy().tolist() == [18.0, 19.0]


def test_the_store_round_trips(tmp_path, temperature):
    # Given: a temperature series
    path = tmp_path / "temp.csv"
    # When: it is written and read back
    datasets.save_temperature(temperature, path)
    loaded = datasets.load_temperature(path)
    # Then: nothing is lost
    assert len(loaded) == len(temperature)
    assert loaded.iloc[0] == pytest.approx(temperature.iloc[0])
    assert datasets.last_temperature_day(path) == temperature.index[-1].date()


def test_missing_temperature_says_which_job_creates_it(tmp_path):
    # Given: no temperature file
    # When/Then: the error names the fix
    with pytest.raises(FileNotFoundError, match="weather job"):
        datasets.load_temperature(tmp_path / "absent.csv")


# -- Features -----------------------------------------------------------------


def test_degree_features_split_around_the_comfort_point():
    # Given: an hour below comfort and an hour above it
    index = pd.date_range("2023-06-01", periods=2, freq="h", tz="UTC")
    temps = pd.Series([8.0, 28.0], index=index)  # comfort is 18 C
    df = pd.DataFrame(index=index)

    # When: the weather features are built
    df = features.add_temperature_features(df, temps)

    # Then: each arm is non-zero only on its own side
    assert df["heating_degrees"].tolist() == [10.0, 0.0]
    assert df["cooling_degrees"].tolist() == [0.0, 10.0]


def test_thermal_inertia_averages_the_preceding_day():
    # Given: a day at 10 C followed by an hour at 30 C
    index = pd.date_range("2023-06-01", periods=25, freq="h", tz="UTC")
    temps = pd.Series([10.0] * 24 + [30.0], index=index)
    df = features.add_temperature_features(pd.DataFrame(index=index), temps)
    # When/Then: the rolling mean lags the spike
    assert df["temp_24h_mean"].iloc[-1] == pytest.approx((10.0 * 23 + 30.0) / 24)


def test_temperature_features_do_see_the_target_hour(demand, temperature):
    # Given: the same demand, but tomorrow's forecast temperature changed
    hot = temperature.copy()
    hot.iloc[-48:] = 40.0

    # When: features are built out to the end of both
    normal = features.make_features(demand, temperature=temperature)
    heated = features.make_features(demand, temperature=hot)

    # Then: the final row DIFFERS - the opposite of the demand-lag test,
    # and deliberate: a weather forecast for the target day is available,
    # so using it is the whole reason the feature earns its place.
    assert normal.iloc[-1]["temp_c"] != heated.iloc[-1]["temp_c"]
    assert heated.iloc[-1]["cooling_degrees"] > normal.iloc[-1]["cooling_degrees"]


def test_demand_lags_are_still_leak_free_with_temperature(demand, temperature):
    # Given: poisoned recent DEMAND (not temperature)
    poisoned = demand.copy()
    poisoned.iloc[-features.MIN_LAG_HOURS :] = 99_999.0

    # When: weather features are in play
    clean = features.make_features(demand, temperature=temperature)
    dirty = features.make_features(poisoned, temperature=temperature)

    # Then: the demand-leakage guarantee still holds
    pd.testing.assert_series_equal(clean.iloc[-1], dirty.iloc[-1])


def test_the_feature_contract_grows_by_exactly_the_weather_columns():
    # Given: the two contracts
    plain = features.feature_columns(with_temperature=False)
    with_temp = features.feature_columns(with_temperature=True)
    # When/Then: weather is appended, leaving the base order untouched
    assert plain == features.FEATURE_COLUMNS
    assert with_temp == [*features.FEATURE_COLUMNS, *features.TEMPERATURE_COLUMNS]


def test_the_table_carries_the_weather_columns(demand, temperature):
    # Given: demand and temperature
    # When: a training table is built with weather
    table = features.make_training_table(demand, temperature=temperature)
    # Then: the weather columns are present and complete
    for column in features.TEMPERATURE_COLUMNS:
        assert column in table.columns
    assert not table.isna().to_numpy().any()


def test_temperature_stays_optional(demand):
    # Given: no temperature at all
    table = features.make_training_table(demand)
    # When/Then: the demand-only contract still builds, unchanged
    assert list(table.columns) == [*features.FEATURE_COLUMNS, "target"]


# -- Job ----------------------------------------------------------------------


def test_a_cold_start_fetches_archive_then_forecast(tmp_path, monkeypatch):
    # Given: no stored temperature, and a stubbed API
    calls = []

    def fake_archive(start, end, lat, lon):
        calls.append(("archive", start, end))
        index = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.Series(15.0, index=index, name="temp_c")

    def fake_forecast(lat, lon, past_days, forecast_days):
        calls.append(("forecast", past_days, forecast_days))
        index = pd.date_range(
            pd.Timestamp.now(tz="UTC").normalize(), periods=24, freq="h"
        )
        return pd.Series(25.0, index=index, name="temp_c")

    monkeypatch.setattr(weather_job.weather, "fetch_archive", fake_archive)
    monkeypatch.setattr(weather_job.weather, "fetch_forecast", fake_forecast)

    config = WeatherConfig(
        temperature_csv=tmp_path / "temp.csv",
        start=date.today() - pd.Timedelta(days=60).to_pytimedelta(),
    )

    # When: the job runs
    series = weather_job.run(config)

    # Then: both endpoints were used and the file exists
    assert [c[0] for c in calls] == ["archive", "forecast"]
    assert config.temperature_csv.exists()
    assert not series.empty


def test_a_rerun_refetches_the_recent_window(tmp_path, monkeypatch):
    # Given: a store already holding data up to today
    path = tmp_path / "temp.csv"
    index = pd.date_range(
        pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=30),
        periods=24 * 31,
        freq="h",
    )
    datasets.save_temperature(pd.Series(15.0, index=index, name="temp_c"), path)

    calls = []

    def fake_forecast(lat, lon, past_days, forecast_days):
        calls.append("forecast")
        fresh = pd.date_range(
            pd.Timestamp.now(tz="UTC").normalize(), periods=24, freq="h"
        )
        return pd.Series(30.0, index=fresh, name="temp_c")

    monkeypatch.setattr(weather_job.weather, "fetch_forecast", fake_forecast)
    monkeypatch.setattr(
        weather_job.weather,
        "fetch_archive",
        lambda *a: pytest.fail("archive should not be refetched when current"),
    )

    # When: the job runs again
    series = weather_job.run(WeatherConfig(temperature_csv=path))

    # Then: only the forecast window is refreshed, and it overwrites in place
    assert calls == ["forecast"]
    assert series.loc[pd.Timestamp.now(tz="UTC").normalize()] == 30.0

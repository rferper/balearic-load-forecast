"""Shared fixtures.

The synthetic series are deterministic, so tests can assert on values.
"""

import numpy as np
import pandas as pd
import pytest


def synthetic_demand(hours: int = 24 * 400, start: str = "2023-01-01") -> pd.Series:
    """Build a deterministic hourly demand series with daily and weekly shape.

    Args:
        hours: Length of the series.
        start: First timestamp, interpreted as UTC.

    Returns:
        Demand in MW on an hourly UTC index.
    """
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    hour_of_day = index.hour.to_numpy()
    day_of_week = index.dayofweek.to_numpy()
    values = (
        900.0
        + 250.0 * np.sin((hour_of_day - 6) / 24.0 * 2 * np.pi)
        - 60.0 * (day_of_week >= 5)
        + 40.0 * np.sin(index.dayofyear.to_numpy() / 365.0 * 2 * np.pi)
    )
    return pd.Series(values, index=index, name="demand")


@pytest.fixture
def demand() -> pd.Series:
    """Return a long-enough synthetic series for feature building."""
    return synthetic_demand()


@pytest.fixture
def short_demand() -> pd.Series:
    """Return a series too short for the 336-hour lookback."""
    return synthetic_demand(hours=100)


def synthetic_temperature(
    hours: int = 24 * 400, start: str = "2023-01-01"
) -> pd.Series:
    """Build a deterministic hourly temperature series in Celsius.

    Args:
        hours: Length of the series.
        start: First timestamp, interpreted as UTC.

    Returns:
        Temperature in Celsius on an hourly UTC index, with a summer/winter
        swing and a daily cycle, spanning roughly 8-30 C like the Balearics.
    """
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    seasonal = 19.0 - 9.0 * np.cos(index.dayofyear.to_numpy() / 365.0 * 2 * np.pi)
    daily = 4.0 * np.sin((index.hour.to_numpy() - 9) / 24.0 * 2 * np.pi)
    return pd.Series(seasonal + daily, index=index, name="temp_c")


@pytest.fixture
def temperature() -> pd.Series:
    """Return a synthetic temperature series matching `demand`."""
    return synthetic_temperature()


@pytest.fixture
def temperature_csv(tmp_path, temperature):
    """Write the synthetic temperature out in the stored format."""
    path = tmp_path / "temperature.csv"
    frame = temperature.to_frame()
    frame.index.name = "datetime"
    frame.to_csv(path)
    return path


@pytest.fixture
def raw_csv(tmp_path, demand):
    """Write the synthetic series out in the raw landing-file format."""
    path = tmp_path / "demand.csv"
    frame = demand.rename("value").to_frame()
    frame.index.name = "datetime"
    frame.to_csv(path)
    return path

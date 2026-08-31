"""Client for the Open-Meteo temperature API.

Two endpoints cover different stretches of time. The archive is ERA5
reanalysis, which reaches back to 1940 but lags several days. The forecast
endpoint covers that lag (`past_days`) and runs forward (`forecast_days`).
"""

from datetime import date

import pandas as pd
import requests
from loguru import logger

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 120
VARIABLE = "temperature_2m"


def _to_series(payload: dict) -> pd.Series:
    """Turn an Open-Meteo hourly block into a UTC-indexed series."""
    hourly = payload["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True))
    series = pd.Series(hourly[VARIABLE], index=index, name="temp_c", dtype="float64")
    return series.dropna()


def fetch_archive(
    start: date, end: date, latitude: float, longitude: float
) -> pd.Series:
    """Fetch reanalysis temperature for `[start, end]` in Celsius."""
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": VARIABLE,
            "timezone": "UTC",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _to_series(response.json())


def fetch_forecast(
    latitude: float, longitude: float, past_days: int, forecast_days: int
) -> pd.Series:
    """Fetch recent and upcoming temperature in Celsius."""
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": VARIABLE,
            "timezone": "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    series = _to_series(response.json())
    logger.debug("forecast covers {} -> {}", series.index.min(), series.index.max())
    return series

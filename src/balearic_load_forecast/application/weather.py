"""Keep the temperature series current, backwards and forwards.

The recent window is always re-fetched, because hours near the present are
stored first as a forecast and later replaced by reanalysis.
"""

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from ..io import datasets, weather
from ..settings import WeatherConfig


def run(config: WeatherConfig) -> pd.Series:
    """Fetch missing archive, refresh the recent window, and store the result."""
    site = config.site
    existing: pd.Series | None = None
    if config.temperature_csv.exists():
        existing = datasets.load_temperature(config.temperature_csv)

    last_day = datasets.last_temperature_day(config.temperature_csv)
    # The forecast endpoint covers this window, so the archive need not.
    archive_end = date.today() - timedelta(days=config.past_days)
    archive_start = last_day + timedelta(days=1) if last_day else config.start

    fetched: list[pd.Series] = []
    if archive_start <= archive_end:
        logger.info("archive for {}: {} -> {}", site.name, archive_start, archive_end)
        cursor = archive_start
        while cursor <= archive_end:
            chunk_end = min(
                cursor.replace(year=cursor.year + config.chunk_years), archive_end
            )
            fetched.append(
                weather.fetch_archive(cursor, chunk_end, site.latitude, site.longitude)
            )
            cursor = chunk_end + timedelta(days=1)
    else:
        logger.info("archive already current through {}", last_day)

    logger.info(
        "forecast for {}: {} days back, {} days ahead",
        site.name,
        config.past_days,
        config.forecast_days,
    )
    fetched.append(
        weather.fetch_forecast(
            site.latitude, site.longitude, config.past_days, config.forecast_days
        )
    )

    merged = existing
    for chunk in fetched:
        merged = datasets.merge_temperature(merged, chunk)

    if merged is None or merged.empty:
        msg = "no temperature data was retrieved"
        raise ValueError(msg)

    datasets.save_temperature(merged, config.temperature_csv)
    logger.success("temperature spans {} -> {}", merged.index.min(), merged.index.max())
    return merged

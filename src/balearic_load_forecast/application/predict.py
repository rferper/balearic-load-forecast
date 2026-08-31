"""Forecast one local calendar day and store the result."""

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from ..domain.features import (
    LOCAL_TZ,
    TEMPERATURE_COLUMNS,
    as_timestamp,
    make_features,
)
from ..domain.models import Model
from ..io import datasets, registry
from ..settings import PredictConfig


def next_day(now: pd.Timestamp | None = None) -> date:
    """Return tomorrow's local date."""
    moment = now if now is not None else as_timestamp(pd.Timestamp.now(tz=LOCAL_TZ))
    return as_timestamp(moment + pd.Timedelta(days=1)).date()


def forecast_day(
    demand: pd.Series,
    model: Model,
    day: date,
    temperature: pd.Series | None = None,
) -> pd.Series:
    """Predict every hour of one local calendar day.

    Raises if the inputs cannot produce a complete feature row, rather than
    letting a NaN become a silently wrong number.
    """
    # Anchor on the next local midnight: a local day is 23, 24 or 25 hours
    # across a DST transition, so a fixed offset would clip or overrun it.
    next_midnight = as_timestamp(pd.Timestamp(day + timedelta(days=1), tz=LOCAL_TZ))
    last_hour = as_timestamp(next_midnight - pd.Timedelta(hours=1))
    features = make_features(
        demand, until=last_hour.tz_convert("UTC"), temperature=temperature
    )

    is_target_day = features.index.tz_convert(LOCAL_TZ).date == day
    rows = features[is_target_day]

    if len(rows) == 0:
        msg = f"no feature rows built for {day}"
        raise ValueError(msg)
    if rows.isna().any().any():
        incomplete = rows.columns[rows.isna().any()].tolist()
        stale = f"newest demand is {demand.index[-1]} - backfill more history"
        if temperature is not None and any(
            column in incomplete for column in TEMPERATURE_COLUMNS
        ):
            stale = (
                f"newest temperature is {temperature.index[-1]} - the weather "
                f"forecast does not reach {day}, run the weather job"
            )
        msg = f"cannot forecast {day}: {incomplete} incomplete. {stale}."
        raise ValueError(msg)
    return model.predict(rows)


def run(config: PredictConfig) -> pd.Series:
    """Forecast the configured day and write it to the forecast store."""
    demand = datasets.load_demand(config.raw_csv)
    temperature = (
        datasets.load_temperature(config.temperature_csv)
        if config.temperature_csv is not None
        else None
    )
    model, model_identifier = registry.load_model(config.model_path)
    day = config.target_day or next_day()

    logger.info("forecasting {} with model {}", day, model_identifier)
    forecast = forecast_day(demand, model, day, temperature=temperature)

    datasets.save_forecast(forecast, config.forecasts_dir, day, model_identifier)
    logger.success(
        "{}: {} hours, mean {:.0f} MW, peak {:.0f} MW",
        day,
        len(forecast),
        forecast.mean(),
        forecast.max(),
    )
    return forecast

"""Reading and writing the project's datasets."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

FORECAST_COLUMNS = ["datetime", "forecast_mw", "model_id", "generated_at"]
_FORECAST_STEM = "forecast_"


# -- Raw demand ---------------------------------------------------------------


def append_readings(readings: list[dict], csv_path: Path) -> int:
    """Append raw API readings to the landing CSV and return the row count."""
    if not readings:
        return 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = pd.DataFrame(readings)
    chunk["datetime"] = pd.to_datetime(chunk["datetime"], utc=True)
    chunk = chunk[["datetime", "value"]]
    chunk.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    return len(chunk)


def load_demand(csv_path: Path) -> pd.Series:
    """Read the landing CSV into an hourly UTC demand series in MW.

    Chunks overlap at their boundaries, so duplicate timestamps are expected
    and the first occurrence wins.
    """
    if not csv_path.exists():
        msg = f"no demand data at {csv_path} - run the backfill job first"
        raise FileNotFoundError(msg)
    raw = pd.read_csv(csv_path, parse_dates=["datetime"])
    series = raw.set_index("datetime")["value"].sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series.rename("demand")


def last_demand_day(csv_path: Path) -> date | None:
    """Return the newest day in the landing CSV, or None if absent."""
    if not csv_path.exists():
        return None
    raw = pd.read_csv(csv_path, usecols=["datetime"], parse_dates=["datetime"])
    return raw["datetime"].max().date()


# -- Temperature --------------------------------------------------------------


def load_temperature(csv_path: Path) -> pd.Series:
    """Read the temperature series in Celsius, UTC-indexed."""
    if not csv_path.exists():
        msg = f"no temperature data at {csv_path} - run the weather job first"
        raise FileNotFoundError(msg)
    raw = pd.read_csv(csv_path, parse_dates=["datetime"])
    series = raw.set_index("datetime")["temp_c"].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.rename("temp_c")


def merge_temperature(existing: pd.Series | None, fresh: pd.Series) -> pd.Series:
    """Combine stored and freshly fetched temperature, newest value winning.

    Hours near the present are stored first as a forecast and later replaced
    by reanalysis, so this is last-wins rather than first-wins.
    """
    if existing is None or existing.empty:
        combined = fresh
    else:
        combined = pd.concat([existing, fresh])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index().rename("temp_c")


def save_temperature(series: pd.Series, csv_path: Path) -> Path:
    """Write the whole temperature series, replacing the previous file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.rename("temp_c").to_frame()
    frame.index.name = "datetime"
    frame.to_csv(csv_path)
    logger.info("stored {} hours of temperature at {}", len(frame), csv_path)
    return csv_path


def last_temperature_day(csv_path: Path) -> date | None:
    """Return the newest day in the temperature file, or None if absent."""
    if not csv_path.exists():
        return None
    raw = pd.read_csv(csv_path, usecols=["datetime"], parse_dates=["datetime"])
    return raw["datetime"].max().date()


# -- Forecast store -----------------------------------------------------------


def forecast_path(forecasts_dir: Path, day: date) -> Path:
    """Return the store path for one target day."""
    return forecasts_dir / f"{_FORECAST_STEM}{day.isoformat()}.csv"


def save_forecast(
    forecast: pd.Series,
    forecasts_dir: Path,
    day: date,
    model_id: str,
) -> Path:
    """Store one day's forecast with its provenance, replacing any existing."""
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    table = forecast.rename("forecast_mw").to_frame()
    table.index.name = "datetime"
    table = table.reset_index()
    table["model_id"] = model_id
    table["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    path = forecast_path(forecasts_dir, day)
    table[FORECAST_COLUMNS].to_csv(path, index=False)
    logger.info("stored {} hours at {}", len(table), path)
    return path


def load_forecast(forecasts_dir: Path, day: date) -> pd.DataFrame:
    """Read one stored forecast, indexed by UTC timestamp."""
    path = forecast_path(forecasts_dir, day)
    if not path.exists():
        msg = f"no stored forecast for {day} at {path}"
        raise FileNotFoundError(msg)
    table = pd.read_csv(path, parse_dates=["datetime"])
    return table.set_index("datetime")


def stored_days(forecasts_dir: Path) -> list[date]:
    """List every target day in the store, oldest first."""
    if not forecasts_dir.exists():
        return []
    days = [
        date.fromisoformat(path.stem.removeprefix(_FORECAST_STEM))
        for path in forecasts_dir.glob(f"{_FORECAST_STEM}*.csv")
    ]
    return sorted(days)


def latest_stored_day(forecasts_dir: Path) -> date:
    """Return the newest target day in the store."""
    days = stored_days(forecasts_dir)
    if not days:
        msg = f"forecast store {forecasts_dir} is empty - run the predict job"
        raise FileNotFoundError(msg)
    return days[-1]


# -- Scorecard ----------------------------------------------------------------

SCORE_COLUMNS = [
    "target_day",
    "model_id",
    "n_hours",
    "mae_mw",
    "rmse_mw",
    "mape_pct",
    "baseline_mae_mw",
    "mae_improvement",
    "peak_actual_mw",
    "peak_forecast_mw",
    "peak_error_mw",
    "evaluated_at",
]


def save_scores(scores: pd.DataFrame, path: Path) -> Path:
    """Write the accuracy scorecard, replacing any previous version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scores[SCORE_COLUMNS].to_csv(path, index=False)
    logger.info("wrote {} scored days to {}", len(scores), path)
    return path


def load_scores(path: Path) -> pd.DataFrame:
    """Read the accuracy scorecard, oldest day first."""
    if not path.exists():
        msg = f"no scorecard at {path} - run the evaluate job first"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, parse_dates=["target_day"])


# -- Metrics ------------------------------------------------------------------


def save_metrics(metrics: dict, path: Path) -> Path:
    """Write training scores as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return path

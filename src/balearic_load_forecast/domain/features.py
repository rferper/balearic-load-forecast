"""Feature engineering for the hourly demand series.

The series is indexed in UTC throughout. Local time is derived where needed,
never stored, so DST transitions cannot duplicate or drop an index entry.
"""

import holidays
import pandas as pd

LOCAL_TZ = "Europe/Madrid"

# Fitted in this order. Training and inference both read this list.
FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "is_weekend",
    "month",
    "dayofyear",
    "is_holiday",
    "lag_48h",
    "lag_168h",
    "lag_336h",
    "roll_7d_mean",
    "roll_7d_max",
]

# No demand feature may be newer than this, since the forecast is issued
# before the target day starts.
MIN_LAG_HOURS = 48

SEASONAL_LAG_HOURS = 168
MAX_LOOKBACK_HOURS = 336

# Added only when a temperature series is supplied. These carry no lag: the
# met forecast for the target day is available at forecast time.
TEMPERATURE_COLUMNS = [
    "temp_c",
    "cooling_degrees",
    "heating_degrees",
    "temp_24h_mean",
]

# Demand against temperature is U-shaped. Splitting at this point separates
# the cooling arm from the heating arm.
COMFORT_TEMP_C = 18.0

_ROLLING_WINDOW_HOURS = SEASONAL_LAG_HOURS
_THERMAL_INERTIA_HOURS = 24


def feature_columns(*, with_temperature: bool) -> list[str]:
    """Return the fitted column order for a model."""
    if with_temperature:
        return [*FEATURE_COLUMNS, *TEMPERATURE_COLUMNS]
    return list(FEATURE_COLUMNS)


def as_timestamp(value: object) -> pd.Timestamp:
    """Narrow a timestamp-like value, rejecting NaT."""
    if not isinstance(value, pd.Timestamp):
        msg = f"expected a timestamp, got {value!r}"
        raise ValueError(msg)
    return value


def as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Narrow an index to a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        msg = f"expected a DatetimeIndex, got {type(index).__name__}"
        raise TypeError(msg)
    return index


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add the columns derivable from the clock alone."""
    local = index.tz_convert(LOCAL_TZ)
    df = pd.DataFrame(index=index)
    df["hour"] = local.hour
    df["dayofweek"] = local.dayofweek
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    df["month"] = local.month
    df["dayofyear"] = local.dayofyear
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Spanish national and Balearic regional holidays."""
    dates = pd.Series(df.index.tz_convert(LOCAL_TZ).date, index=df.index)
    years = range(dates.min().year, dates.max().year + 1)
    calendar = holidays.Spain(subdiv="IB", years=years)
    df["is_holiday"] = dates.isin(calendar).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, demand: pd.Series) -> pd.DataFrame:
    """Add lagged and rolling demand."""
    df["lag_48h"] = demand.shift(48)
    df["lag_168h"] = demand.shift(168)
    df["lag_336h"] = demand.shift(336)

    # Shift before rolling, or the window reaches into demand that is not
    # known when the forecast is issued.
    past = demand.shift(MIN_LAG_HOURS)
    df["roll_7d_mean"] = past.rolling(_ROLLING_WINDOW_HOURS).mean()
    df["roll_7d_max"] = past.rolling(_ROLLING_WINDOW_HOURS).max()
    return df


def add_temperature_features(df: pd.DataFrame, temperature: pd.Series) -> pd.DataFrame:
    """Add weather columns for the same hours as the rest of the table."""
    temps = temperature.reindex(df.index)
    df["temp_c"] = temps
    df["cooling_degrees"] = (temps - COMFORT_TEMP_C).clip(lower=0.0)
    df["heating_degrees"] = (COMFORT_TEMP_C - temps).clip(lower=0.0)
    df["temp_24h_mean"] = temps.rolling(_THERMAL_INERTIA_HOURS, min_periods=1).mean()
    return df


def make_features(
    demand: pd.Series,
    until: pd.Timestamp | None = None,
    temperature: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the feature table, optionally extended to `until`.

    Args:
        demand: Hourly demand in MW, UTC-indexed.
        until: Last hour to produce rows for. A future timestamp yields rows
            for hours that have not happened yet.
        temperature: Hourly temperature in Celsius, or None.
    """
    # Regular grid, so shift() moves hours rather than rows.
    demand = demand.sort_index().asfreq("h")

    if until is not None:
        full = pd.date_range(demand.index[0], until, freq="h", tz="UTC")
        demand = demand.reindex(full)

    df = add_calendar_features(as_datetime_index(demand.index))
    df = add_holiday_features(df)
    df = add_lag_features(df, demand)
    if temperature is not None:
        df = add_temperature_features(df, temperature)
    return df[feature_columns(with_temperature=temperature is not None)]


def make_training_table(
    demand: pd.Series, temperature: pd.Series | None = None
) -> pd.DataFrame:
    """Build features plus the target, with incomplete rows dropped."""
    features = make_features(demand, temperature=temperature)
    target = demand.sort_index().asfreq("h").rename("target")
    return features.join(target).dropna()

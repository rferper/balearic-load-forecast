"""Error metrics and time-based splitting."""

import pandas as pd

TARGET_COLUMN = "target"


def split_by_time(
    table: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table into rows before and from `cutoff`."""
    return table[table.index < cutoff], table[table.index >= cutoff]


def validation_cutoff(table: pd.DataFrame, days: int) -> pd.Timestamp:
    """Return the cutoff holding out the last `days` of the table."""
    return table.index.max() - pd.Timedelta(days=days)


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error, in MW."""
    return float((actual - forecast).abs().mean())


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute percentage error."""
    return float(((actual - forecast).abs() / actual.abs()).mean() * 100)


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    """Root mean squared error, in MW."""
    return float((((actual - forecast) ** 2).mean()) ** 0.5)


def score(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Return the headline metrics for one forecast."""
    return {
        "mae_mw": mae(actual, forecast),
        "rmse_mw": rmse(actual, forecast),
        "mape_pct": mape(actual, forecast),
    }


def forecast_scorecard(
    actual: pd.Series,
    forecast: pd.Series,
    baseline: pd.Series | None = None,
) -> dict[str, float]:
    """Score one stored forecast, optionally against a baseline."""
    scores = score(actual, forecast)

    peak_actual = float(actual.max())
    peak_forecast = float(forecast.max())
    scores["peak_actual_mw"] = peak_actual
    scores["peak_forecast_mw"] = peak_forecast
    # Signed: negative means the peak was under-forecast.
    scores["peak_error_mw"] = peak_forecast - peak_actual

    if baseline is None or baseline.isna().any():
        scores["baseline_mae_mw"] = float("nan")
        scores["mae_improvement"] = float("nan")
        return scores

    baseline_mae = mae(actual, baseline)
    scores["baseline_mae_mw"] = baseline_mae
    scores["mae_improvement"] = (
        1 - scores["mae_mw"] / baseline_mae if baseline_mae > 0 else float("nan")
    )
    return scores

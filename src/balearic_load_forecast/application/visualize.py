"""Plot a stored forecast against the demand history behind it.

Reads from the store rather than re-predicting, so the figure shows what was
actually issued, including the model that produced it.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ..io import datasets, plots
from ..settings import VisualizeConfig

UNKNOWN_MODEL = "unknown"


def run(config: VisualizeConfig) -> Path:
    """Render the configured forecast to a PNG and return its path."""
    day = config.target_day or datasets.latest_stored_day(config.forecasts_dir)
    stored = datasets.load_forecast(config.forecasts_dir, day)
    forecast = stored["forecast_mw"]

    model_identifier = UNKNOWN_MODEL
    if "model_id" in stored.columns and len(stored):
        model_identifier = str(stored["model_id"].iloc[0])

    demand = datasets.load_demand(config.raw_csv)
    window_start = forecast.index.min() - pd.Timedelta(days=config.history_days)
    actual = demand[demand.index >= window_start]
    logger.info("plotting {} against {} hours of actuals", day, len(actual))

    return plots.plot_forecast(
        actual=actual,
        forecast=forecast,
        day=day,
        model_id=model_identifier,
        path=config.figures_dir / f"forecast_{day.isoformat()}.png",
        dpi=config.dpi,
    )

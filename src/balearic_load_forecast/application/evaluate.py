"""Score stored forecasts against what actually happened.

Re-scores the whole store on every run, so it is safe to repeat and cannot
drift out of step with the forecasts it summarises.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from ..domain import metrics, models
from ..io import datasets
from ..settings import EvaluateConfig

UNKNOWN_MODEL = "unknown"


def expected_mae(metrics_path: Path) -> float | None:
    """Return the validation MAE from the train job, if it exists."""
    if not metrics_path.exists():
        return None
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    value = report.get("model", {}).get("mae_mw")
    return float(value) if value is not None else None


def score_day(
    demand: pd.Series, stored: pd.DataFrame, day: date
) -> dict[str, float | str | int] | None:
    """Score one stored forecast, or return None if actuals are incomplete."""
    forecast = stored["forecast_mw"]
    index = pd.DatetimeIndex(forecast.index)
    actual = demand.reindex(index)
    if actual.isna().any():
        return None

    baseline = models.seasonal_naive_from_demand(demand, index)
    row: dict[str, float | str | int] = dict(
        metrics.forecast_scorecard(actual, forecast, baseline)
    )
    row["target_day"] = day.isoformat()
    row["n_hours"] = len(forecast)
    row["model_id"] = (
        str(stored["model_id"].iloc[0])
        if "model_id" in stored.columns
        else UNKNOWN_MODEL
    )
    row["evaluated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return row


def report_drift(scores: pd.DataFrame, config: EvaluateConfig) -> None:
    """Log recent accuracy and warn on the two drift signals.

    Losing to the naive baseline means the model has stopped earning its keep.
    Rising above the validation MAE means conditions have moved away from the
    training data, which can happen while it still beats naive.
    """
    recent = scores.tail(config.recent_days)
    recent_mae = float(recent["mae_mw"].mean())
    logger.info(
        "last {} scored day(s): MAE {:.1f} MW, MAPE {:.2f}%",
        len(recent),
        recent_mae,
        recent["mape_pct"].mean(),
    )

    recent_baseline = float(recent["baseline_mae_mw"].mean())
    if pd.notna(recent_baseline):
        improvement = 1 - recent_mae / recent_baseline
        logger.info(
            "vs seasonal-naive {:.1f} MW ({:+.1%})", recent_baseline, improvement
        )
        if improvement <= 0:
            logger.warning("model is not beating the naive baseline in production")

    baseline_mae = expected_mae(config.metrics_path)
    if baseline_mae is None:
        return
    if recent_mae > baseline_mae * config.drift_tolerance:
        logger.warning(
            "drift: recent MAE {:.1f} MW exceeds {:.1f}x the validation MAE "
            "of {:.1f} MW",
            recent_mae,
            config.drift_tolerance,
            baseline_mae,
        )
    else:
        logger.info("within tolerance of the {:.1f} MW validation MAE", baseline_mae)


def run(config: EvaluateConfig) -> pd.DataFrame:
    """Score every stored forecast whose actuals have arrived."""
    demand = datasets.load_demand(config.raw_csv)
    days = datasets.stored_days(config.forecasts_dir)
    logger.info("forecast store holds {} day(s)", len(days))

    rows: list[dict[str, float | str | int]] = []
    pending: list[date] = []
    for day in days:
        stored = datasets.load_forecast(config.forecasts_dir, day)
        row = score_day(demand, stored, day)
        if row is None:
            pending.append(day)
        else:
            rows.append(row)

    if pending:
        logger.info(
            "{} day(s) awaiting actuals: {}",
            len(pending),
            ", ".join(d.isoformat() for d in pending),
        )

    if not rows:
        logger.warning("nothing to score yet")
        return pd.DataFrame(columns=datasets.SCORE_COLUMNS)

    scores = pd.DataFrame(rows).sort_values("target_day").reset_index(drop=True)
    datasets.save_scores(scores, config.scores_path)
    report_drift(scores, config)
    return scores

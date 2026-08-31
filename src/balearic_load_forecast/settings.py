"""Configuration schemas, one per job.

Values come from confs/<job>.yaml plus optional command-line overrides, and
are validated here before a job touches the disk or the network.
"""

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEMAND_CSV = Path("data/balearic_demand_raw.csv")
TEMPERATURE_CSV = Path("data/balearic_temperature_raw.csv")
MODEL_PATH = Path("outputs/models/lgbm.pkl")
METRICS_PATH = Path("outputs/models/metrics.json")
FORECASTS_DIR = Path("outputs/forecasts")


class BaseConfig(BaseModel):
    """Rejects unknown keys instead of ignoring them."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggerConfig(BaseConfig):
    """Log level and format. Set `serialize` for JSON output."""

    level: str = "INFO"
    serialize: bool = False


class BackfillConfig(BaseConfig):
    """Download the demand series from REE."""

    raw_csv: Path = DEMAND_CSV
    start: date = date(2011, 1, 1)
    # REE rejects wider hourly ranges.
    chunk_days: int = Field(default=27, gt=0, le=31)
    pause_seconds: float = Field(default=0.5, ge=0.0)
    logger: LoggerConfig = LoggerConfig()


class SiteConfig(BaseConfig):
    """The point whose weather stands in for the whole system."""

    name: str = "Palma de Mallorca"
    latitude: float = Field(default=39.5696, ge=-90.0, le=90.0)
    longitude: float = Field(default=2.6502, ge=-180.0, le=180.0)


class WeatherConfig(BaseConfig):
    """Fetch temperature history and the days ahead."""

    temperature_csv: Path = TEMPERATURE_CSV
    site: SiteConfig = SiteConfig()
    start: date = date(2011, 1, 1)
    chunk_years: int = Field(default=5, gt=0)
    # Recent window comes from the forecast endpoint, which publishes sooner
    # than the reanalysis archive.
    past_days: int = Field(default=14, ge=0, le=92)
    forecast_days: int = Field(default=7, gt=0, le=16)
    logger: LoggerConfig = LoggerConfig()


class ModelConfig(BaseConfig):
    """LightGBM hyperparameters."""

    n_estimators: int = Field(default=600, gt=0)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    num_leaves: int = Field(default=63, gt=1)
    random_state: int = 0


class TrainConfig(BaseConfig):
    """Fit a model and record its validation scores.

    `temperature_csv` set to None trains without weather features.
    """

    raw_csv: Path = DEMAND_CSV
    temperature_csv: Path | None = TEMPERATURE_CSV
    model_path: Path = MODEL_PATH
    metrics_path: Path = METRICS_PATH
    # Held out by time, never randomly.
    validation_days: int = Field(default=90, gt=0)
    model: ModelConfig = ModelConfig()
    logger: LoggerConfig = LoggerConfig()


class PredictConfig(BaseConfig):
    """Forecast one local calendar day.

    `target_day` of None means tomorrow. `temperature_csv` must match what the
    model was trained on; a mismatch is caught by the feature contract.
    """

    raw_csv: Path = DEMAND_CSV
    temperature_csv: Path | None = TEMPERATURE_CSV
    model_path: Path = MODEL_PATH
    forecasts_dir: Path = FORECASTS_DIR
    target_day: date | None = None
    logger: LoggerConfig = LoggerConfig()


class EvaluateConfig(BaseConfig):
    """Score stored forecasts against actual demand."""

    raw_csv: Path = DEMAND_CSV
    forecasts_dir: Path = FORECASTS_DIR
    scores_path: Path = Path("outputs/scores/accuracy.csv")
    metrics_path: Path = METRICS_PATH
    recent_days: int = Field(default=7, gt=0)
    # Warn once recent MAE exceeds the validation MAE by this factor.
    drift_tolerance: float = Field(default=1.5, gt=1.0)
    logger: LoggerConfig = LoggerConfig()


class VisualizeConfig(BaseConfig):
    """Plot a stored forecast. `target_day` of None uses the newest."""

    raw_csv: Path = DEMAND_CSV
    forecasts_dir: Path = FORECASTS_DIR
    figures_dir: Path = Path("outputs/figures")
    target_day: date | None = None
    history_days: int = Field(default=14, gt=0)
    dpi: int = Field(default=150, gt=0)
    logger: LoggerConfig = LoggerConfig()

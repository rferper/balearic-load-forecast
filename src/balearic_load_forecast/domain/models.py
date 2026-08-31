"""Forecasting models. Persistence lives in `io.registry`."""

from typing import Protocol, Self

import lightgbm as lgb
import pandas as pd

from .features import FEATURE_COLUMNS, SEASONAL_LAG_HOURS


class Model(Protocol):
    """Interface the application layer depends on."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return the forecast in MW, indexed like `features`."""
        ...


class SeasonalNaiveModel:
    """Baseline predicting each hour as the same hour last week."""

    reference_column = "lag_168h"

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return last week's demand for the same hour."""
        if self.reference_column not in features.columns:
            msg = f"missing feature column: {self.reference_column}"
            raise ValueError(msg)
        return features[self.reference_column].rename("forecast")


class LightGBMModel:
    """Gradient-boosted trees over the calendar, lag and weather features."""

    def __init__(
        self,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        random_state: int = 0,
    ) -> None:
        """Configure the learner without fitting it."""
        self.feature_columns: list[str] | None = None
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        """Fit the model and record its column order."""
        self.feature_columns = list(features.columns)
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return the forecast in MW, indexed like `features`."""
        if self.feature_columns is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            msg = f"missing feature columns: {missing}"
            raise ValueError(msg)
        values = self.model.predict(features[self.feature_columns])
        return pd.Series(values, index=features.index, name="forecast")

    def importances(self) -> pd.Series:
        """Return feature importances, largest first."""
        if self.feature_columns is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        values = pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns,
            name="importance",
        )
        return values.sort_values(ascending=False)


def seasonal_naive_from_demand(demand: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Rebuild the naive baseline for `index` from the demand series.

    Used when scoring a stored forecast, where no feature frame exists.
    """
    week_earlier = index - pd.Timedelta(hours=SEASONAL_LAG_HOURS)
    values = demand.reindex(week_earlier).to_numpy()
    return pd.Series(values, index=index, name="baseline")


def expected_columns() -> list[str]:
    """Return the canonical feature column names."""
    return list(FEATURE_COLUMNS)

"""Model behaviour and the feature-column contract."""

import pytest

from balearic_load_forecast.domain import models
from balearic_load_forecast.domain.features import make_training_table


@pytest.fixture
def table(demand):
    return make_training_table(demand)


def test_seasonal_naive_returns_last_weeks_demand(table):
    # Given: the baseline model
    baseline = models.SeasonalNaiveModel()
    # When: it predicts
    forecast = baseline.predict(table)
    # Then: it is exactly the same-hour-last-week column
    assert forecast.equals(table["lag_168h"].rename("forecast"))


def test_seasonal_naive_rejects_a_table_without_its_column(table):
    # Given: a table missing the reference column
    stripped = table.drop(columns=["lag_168h"])
    # When/Then: predicting is refused rather than guessed
    with pytest.raises(ValueError, match="lag_168h"):
        models.SeasonalNaiveModel().predict(stripped)


def test_unfitted_model_refuses_to_predict(table):
    # Given: a model that was never fitted
    model = models.LightGBMModel()
    # When/Then: predicting raises rather than returning nonsense
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(table)


def test_fitted_model_predicts_indexed_like_its_input(table):
    # Given: a model fitted on a small, fast configuration
    features = table.drop(columns="target")
    model = models.LightGBMModel(n_estimators=20).fit(features, table["target"])
    # When: it predicts
    forecast = model.predict(features)
    # Then: the result aligns with the input and is named consistently
    assert forecast.index.equals(features.index)
    assert forecast.name == "forecast"


def test_predict_rejects_a_missing_fitted_column(table):
    # Given: a fitted model
    features = table.drop(columns="target")
    model = models.LightGBMModel(n_estimators=20).fit(features, table["target"])
    # When/Then: a table missing a fitted column is refused
    with pytest.raises(ValueError, match="missing feature columns"):
        model.predict(features.drop(columns=["hour"]))


def test_training_is_deterministic(table):
    # Given: the same table and the same seed
    features = table.drop(columns="target")
    first = models.LightGBMModel(n_estimators=20).fit(features, table["target"])
    second = models.LightGBMModel(n_estimators=20).fit(features, table["target"])
    # When/Then: the two models agree exactly
    assert first.predict(features).equals(second.predict(features))


def test_importances_cover_every_fitted_column(table):
    # Given: a fitted model
    features = table.drop(columns="target")
    model = models.LightGBMModel(n_estimators=20).fit(features, table["target"])
    # When: importances are requested
    importances = model.importances()
    # Then: every feature is accounted for, largest first
    assert set(importances.index) == set(features.columns)
    assert importances.is_monotonic_decreasing

"""Fit the model, score it against the baseline, and store both.

Two fits happen. The first is scored on a held-out tail against the seasonal
naive baseline. The second refits on the full history and is the artefact the
predict job loads; it is never scored, having seen every row.
"""

from loguru import logger

from ..domain import metrics, models
from ..domain.features import make_training_table
from ..io import datasets, registry
from ..settings import ModelConfig, TrainConfig


def build_model(config: ModelConfig) -> models.LightGBMModel:
    """Construct an unfitted model from validated hyperparameters."""
    return models.LightGBMModel(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        random_state=config.random_state,
    )


def run(config: TrainConfig) -> dict:
    """Evaluate, refit on the full history, and write model plus metrics."""
    demand = datasets.load_demand(config.raw_csv)
    temperature = (
        datasets.load_temperature(config.temperature_csv)
        if config.temperature_csv is not None
        else None
    )
    if temperature is not None:
        logger.info("using weather features from {}", config.temperature_csv)

    table = make_training_table(demand, temperature=temperature)
    logger.info(
        "training table: {} rows, {} -> {}",
        len(table),
        table.index.min(),
        table.index.max(),
    )

    cutoff = metrics.validation_cutoff(table, config.validation_days)
    train, valid = metrics.split_by_time(table, cutoff)
    logger.info("split at {}: {} train / {} valid", cutoff, len(train), len(valid))

    x_train = train.drop(columns=metrics.TARGET_COLUMN)
    y_train = train[metrics.TARGET_COLUMN]
    x_valid = valid.drop(columns=metrics.TARGET_COLUMN)
    y_valid = valid[metrics.TARGET_COLUMN]

    candidate = build_model(config.model).fit(x_train, y_train)
    baseline = models.SeasonalNaiveModel()

    model_scores = metrics.score(y_valid, candidate.predict(x_valid))
    baseline_scores = metrics.score(y_valid, baseline.predict(x_valid))
    improvement = 1 - model_scores["mae_mw"] / baseline_scores["mae_mw"]

    logger.info(
        "validation MAE: model {:.1f} MW vs naive {:.1f} MW ({:+.1%})",
        model_scores["mae_mw"],
        baseline_scores["mae_mw"],
        improvement,
    )
    if improvement <= 0:
        logger.warning("model does not beat the seasonal-naive baseline")

    features = table.drop(columns=metrics.TARGET_COLUMN)
    final = build_model(config.model).fit(features, table[metrics.TARGET_COLUMN])
    model_identifier = registry.save_model(final, config.model_path)

    report = {
        "model_id": model_identifier,
        "validation_cutoff": cutoff,
        "n_train": len(train),
        "n_valid": len(valid),
        "model": model_scores,
        "baseline": baseline_scores,
        "mae_improvement": improvement,
        "hyperparameters": config.model.model_dump(),
        "top_features": final.importances().head(5).to_dict(),
    }
    datasets.save_metrics(report, config.metrics_path)
    logger.success("wrote metrics to {}", config.metrics_path)
    return report

"""Command-line interface.

    balearic-load-forecast <job> [key=value ...]

Each job reads confs/<job>.yaml, applies any overrides, validates, then runs.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

from .application import backfill, evaluate, predict, train, visualize, weather
from .io.configs import config_path, load_config
from .io.services import LoggerService
from .settings import (
    BackfillConfig,
    EvaluateConfig,
    PredictConfig,
    TrainConfig,
    VisualizeConfig,
    WeatherConfig,
)

JOBS: dict[str, tuple[type[BaseModel], Callable[[Any], Any]]] = {
    "backfill": (BackfillConfig, backfill.run),
    "weather": (WeatherConfig, weather.run),
    "train": (TrainConfig, train.run),
    "predict": (PredictConfig, predict.run),
    "evaluate": (EvaluateConfig, evaluate.run),
    "visualize": (VisualizeConfig, visualize.run),
}

EXIT_OK = 0
EXIT_FAILURE = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="balearic-load-forecast",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("job", choices=sorted(JOBS), help="which job to run")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="config YAML (default: confs/<job>.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="dotted-key overrides, e.g. model.num_leaves=127",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one job and return its exit code."""
    args = build_parser().parse_args(argv)
    schema, job = JOBS[args.job]
    path = args.config or config_path(args.job)

    try:
        config = load_config(schema, path=path, overrides=args.overrides)
    except (ValidationError, ValueError) as error:
        # No logger yet: its configuration is part of what just failed.
        print(f"invalid configuration for '{args.job}': {error}", file=sys.stderr)  # noqa: T201
        return EXIT_FAILURE

    service = LoggerService(config.logger)
    service.start()
    try:
        logger.info("job '{}' starting from {}", args.job, path)
        job(config)
    except Exception:
        logger.opt(exception=True).error("job '{}' failed", args.job)
        return EXIT_FAILURE
    else:
        logger.success("job '{}' finished", args.job)
        return EXIT_OK
    finally:
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())

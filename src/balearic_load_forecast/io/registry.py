"""Model persistence.

A stored model is identified by the SHA-256 of its bytes, so a forecast can
be traced to the exact artefact that produced it.
"""

import hashlib
import pickle
from pathlib import Path

from loguru import logger

from ..domain.models import LightGBMModel

ID_LENGTH = 12


def model_id(path: Path) -> str:
    """Return a short content hash of the stored model."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:ID_LENGTH]


def save_model(model: LightGBMModel, path: Path) -> str:
    """Write a fitted model and return its identifier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    identifier = model_id(path)
    logger.info("saved model {} to {}", identifier, path)
    return identifier


def load_model(path: Path) -> tuple[LightGBMModel, str]:
    """Read a fitted model back with its identifier."""
    if not path.exists():
        msg = f"no model at {path} - run the train job first"
        raise FileNotFoundError(msg)
    # S301: the pickle is written by this package, not user input.
    model = pickle.loads(path.read_bytes())  # noqa: S301
    return model, model_id(path)

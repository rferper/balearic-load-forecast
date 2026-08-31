"""Config loading: OmegaConf for parsing, Pydantic for validation."""

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

CONFS_DIR = Path("confs")


def parse_overrides(overrides: list[str] | None = None) -> DictConfig:
    """Parse `key=value` overrides into a config tree.

    Takes the list explicitly rather than reading sys.argv, so the CLI stays
    the only place arguments are parsed.
    """
    return OmegaConf.from_dotlist(overrides or [])


def load_config[T: BaseModel](
    schema: type[T],
    path: Path | None = None,
    overrides: list[str] | None = None,
) -> T:
    """Build a validated config from a YAML file and overrides.

    A missing file is not an error; every field has a default.
    """
    merged: DictConfig = OmegaConf.create({})
    if path is not None and path.exists():
        merged = cast(DictConfig, OmegaConf.merge(merged, OmegaConf.load(path)))
    merged = cast(DictConfig, OmegaConf.merge(merged, parse_overrides(overrides)))

    raw: Any = OmegaConf.to_container(merged, resolve=True)
    return schema(**raw)


def config_path(job: str, confs_dir: Path = CONFS_DIR) -> Path:
    """Return `<confs_dir>/<job>.yaml`."""
    return confs_dir / f"{job}.yaml"

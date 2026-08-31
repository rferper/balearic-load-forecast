"""Services with an explicit lifecycle. Nothing configures itself on import."""

import sys

from loguru import logger

from ..settings import LoggerConfig


class LoggerService:
    """Configures loguru for the duration of one job."""

    def __init__(self, config: LoggerConfig) -> None:
        """Store the configuration without applying it."""
        self.config = config

    def start(self) -> None:
        """Replace the default sink with the configured one."""
        logger.remove()
        logger.add(
            sys.stderr,
            level=self.config.level,
            serialize=self.config.serialize,
            backtrace=False,
            diagnose=False,  # keep local variables out of tracebacks
        )

    def stop(self) -> None:
        """Detach every sink."""
        logger.remove()

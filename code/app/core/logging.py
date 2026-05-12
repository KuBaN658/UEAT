"""
Logging configuration for the application.

Call ``configure_logging()`` once at startup (in ``main.py`` lifespan).
Falls back to basic config if called multiple times.
"""

from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Override log level (e.g. ``"DEBUG"``).  Defaults to the
               ``LOG_LEVEL`` environment variable or ``"INFO"``.
    """
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, resolved, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

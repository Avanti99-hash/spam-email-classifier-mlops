from __future__ import annotations

import logging

from logging.handlers import RotatingFileHandler
from pathlib import Path
import os


def setup_logging(app_name: str, log_file: str | None = None, always_to_file: bool = True) -> None:
    """
    Consistent logging for CLI + API.

    - Always logs to console
    - Logs to a rotating file by default (always_to_file=True)
    - Uses different log files for CLI vs API
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Prevent duplicate handlers (common with uvicorn reload / repeated imports)
    if getattr(logger, "_configured_by_app", False):
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if always_to_file:
        if not log_file:
            log_file = f"{app_name}.log"

        fh = RotatingFileHandler(
            filename=str(logs_dir / log_file),
            maxBytes=2_000_000,
            backupCount=3,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Mark configured
    logger._configured_by_app = True

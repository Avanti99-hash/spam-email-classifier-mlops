from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(app_name: str = "spam_classifier") -> None:
    """
    Sets up consistent logging for CLI + API.
    - Console logs always
    - File logs (rotating) if LOG_TO_FILE=1
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "0") == "1"

    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid duplicate handlers when uvicorn reloads
    if logger.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_to_file:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        fh = RotatingFileHandler(
            filename=str(logs_dir / f"{app_name}.log"),
            maxBytes=2_000_000,
            backupCount=3,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

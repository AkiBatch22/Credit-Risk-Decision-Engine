"""Application logging with idempotent handler configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import LOG_FILE


def get_logger(name: str = "credit_risk", log_file: Path | None = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if getattr(logger, "_credit_risk_configured", False):
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        resolved_log_file = Path(log_file)
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._credit_risk_configured = True  # type: ignore[attr-defined]
    return logger

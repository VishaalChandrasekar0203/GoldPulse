from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("gold")
    logger.setLevel(level)
    if _CONFIGURED:
        return logger
    logger.handlers.clear()
    rich = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
    rich.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "gold.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger("gold")
    return base.getChild(name) if name else base

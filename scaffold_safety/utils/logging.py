"""
Structured logging for ScaffoldSafety evaluations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class _JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        return json.dumps(entry, default=str)


def setup_logger(
    name: str = "scaffold_safety",
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a JSON-lines logger.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : Path | None
        Directory for log files. If None, logs to stderr only.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        # File handler (JSON lines)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "scaffold_safety.jsonl")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(_JSONFormatter())
            logger.addHandler(fh)

    return logger

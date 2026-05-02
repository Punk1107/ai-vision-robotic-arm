"""
logger.py — Structured Logging with Rich + Loguru
==================================================
Provides a project-wide logger with:
  - Color-coded console output (via Rich sink)
  - Rotating file logs in /logs
  - Per-module child loggers
"""

import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

# ── Default log directory ─────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_rich_console = Console(stderr=True)


def setup_logger(
    level: str = "INFO",
    log_dir: str | Path = _LOG_DIR,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Call once from main.py to configure the global logger.

    Args:
        level:     Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_dir:   Directory to write rotating log files.
        rotation:  When to rotate (e.g. "10 MB", "1 day").
        retention: How long to keep old logs.
    """
    logger.remove()  # Remove default stderr sink

    # ── Rich console sink ─────────────────────────────────────────────────────
    logger.add(
        RichHandler(
            console=_rich_console,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        ),
        level=level,
        format="{message}",
        colorize=False,
    )

    # ── Rotating file sink ────────────────────────────────────────────────────
    logger.add(
        str(Path(log_dir) / "arm_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{name}:{function}:{line} — {message}"
        ),
    )

    logger.info(
        f"Logger initialised — level=[bold]{level}[/bold], "
        f"writing to [cyan]{log_dir}[/cyan]"
    )


def get_logger(name: str):
    """Return a child logger bound to a module name."""
    return logger.bind(name=name)


# ── Convenience re-export ─────────────────────────────────────────────────────
__all__ = ["setup_logger", "get_logger", "logger"]

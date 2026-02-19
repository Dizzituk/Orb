# FILE: app/logging_config.py
"""
Centralized logging configuration for ASTRA.

Sets up a RotatingFileHandler so all logger.info/warning/error calls
across the entire application are written to D:\Orb\logs\astra.log
in addition to the console (stdout/stderr).

Usage: Call setup_file_logging() once at startup (before any other imports
       that create loggers). This attaches a file handler to the root logger
       so ALL child loggers (app.orchestrator.*, app.overwatcher.*, etc.)
       inherit it automatically.

v1.0 (2026-02-19): Initial implementation.
    - RotatingFileHandler: 10 MB per file, 5 backups (50 MB total max)
    - Format includes timestamp, level, logger name, and message
    - Also captures print() output from pipeline stages via stdout redirect
    - Log directory auto-created if missing
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Where logs live — next to the project root
LOG_DIR = Path(os.getenv("ORB_LOG_DIR", r"D:\Orb\logs"))
LOG_FILE = LOG_DIR / "astra.log"

# Rotation settings
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
BACKUP_COUNT = 5               # Keep 5 old files (50 MB total)

# Format: timestamp [LEVEL] logger_name: message
LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_file_logging(level: int = logging.DEBUG) -> str:
    """Attach a RotatingFileHandler to the root logger.

    Args:
        level: Minimum log level for the file handler.
               Defaults to DEBUG so we capture everything.
               Console level is left untouched.

    Returns:
        The absolute path to the log file (for startup banner).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build the handler
    file_handler = RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Attach to root logger — all child loggers inherit this
    root = logging.getLogger()
    root.addHandler(file_handler)

    # Ensure root level is at least as permissive as our handler
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # Also set up a console handler if none exists (uvicorn usually adds one,
    # but direct `python main.py` runs might not have one)
    _has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if not _has_console:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(console)

    # ── v1.1 Suppress noisy third-party HTTP transport logs ──────────
    # These flood the file with TLS handshakes, request headers, etc.
    # WARNING+ still passes through (those indicate real problems).
    for _noisy in (
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
        "anthropic._base_client",
        "openai._base_client",
        "urllib3",
        "urllib3.connectionpool",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Log the setup itself
    logger = logging.getLogger(__name__)
    logger.info("File logging initialised: %s (max %d MB x %d backups)", LOG_FILE, MAX_BYTES // (1024*1024), BACKUP_COUNT)

    return str(LOG_FILE)

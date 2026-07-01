# FILE: app/logging_config.py
# Purpose: Centralized logging configuration for ASTRA.
# Called-by: main
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Centralized logging configuration for ASTRA.

Sets up a RotatingFileHandler so all logger.info/warning/error calls
across the entire application are written to D:/Orb/logs/astra.log
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


# ── Access log filter ───────────────────────────────────────────
# Uvicorn's `uvicorn.access` logger emits one line per HTTP request.
# The web-automation subsystem long-polls a handful of endpoints every
# few seconds (one per registered session, plus the bridge), which
# floods the console and drowns out everything else.
#
# This filter drops access-log records for those specific routes,
# leaving the actual chat / tool / debug traffic visible. The polling
# itself is unaffected — only the LOG line is suppressed.
#
# Override: set ORB_VERBOSE_HTTP=1 to disable the filter and see every
# request again (useful when debugging the polling loop itself).
_DEFAULT_NOISY_ROUTES = (
    "/web_automation/pending-action",   # action queue long-poll
    "/web_automation/sessions",          # frontend session list refresh
    "/bridge/pending-navigation",        # bridge navigation long-poll
    "=/health",                          # liveness probe (frontend + mobile, every ~1-2s)
    "=/ping",                            # service ping
)


class _AccessLogNoiseFilter(logging.Filter):
    """Drop uvicorn.access records whose path matches any noisy route.

    Uvicorn formats access lines as:
      '%s - "%s %s HTTP/%s" %d %s' % (client, method, path, http_ver, status, phrase)
    so the path lives in record.args[2]. We check the rendered message
    too as a defensive fallback in case uvicorn changes its format.
    """

    def __init__(self, noisy_routes: tuple[str, ...]):
        super().__init__()
        self.noisy_routes = noisy_routes

    def filter(self, record: logging.LogRecord) -> bool:
        # Fast path: structured args.
        path = None
        if record.args and isinstance(record.args, tuple) and len(record.args) >= 3:
            candidate = record.args[2]
            if isinstance(candidate, str):
                path = candidate
        if path is None:
            try:
                path = record.getMessage()
            except Exception:
                return True
        for noisy in self.noisy_routes:
            if noisy.startswith("="):
                # Exact path match (ignoring any query string) — used for short
                # paths like /health that would otherwise substring-match longer
                # routes such as /bridge/health.
                if path.split("?", 1)[0] == noisy[1:]:
                    return False
            elif noisy in path:
                return False
        return True


def _install_access_log_filter() -> None:
    """Attach the noise filter to uvicorn.access unless ORB_VERBOSE_HTTP is set."""
    if os.getenv("ORB_VERBOSE_HTTP", "").strip() in ("1", "true", "yes"):
        logging.getLogger(__name__).info(
            "ORB_VERBOSE_HTTP set — access log filter NOT installed"
        )
        return
    extra = os.getenv("ORB_QUIET_ROUTES", "").strip()
    routes = list(_DEFAULT_NOISY_ROUTES)
    if extra:
        routes.extend(r.strip() for r in extra.split(",") if r.strip())
    flt = _AccessLogNoiseFilter(tuple(routes))
    logging.getLogger("uvicorn.access").addFilter(flt)
    logging.getLogger(__name__).info(
        "Access log filter installed: suppressing %d route pattern(s)",
        len(routes),
    )


def _resolve_file_level() -> int:
    """File-handler verbosity, env-controlled (added 2026-06-24).

    Default INFO so the log file stays readable. The full DEBUG firehose
    (all app.* + library debug) is one env var away:
      ORB_VERBOSE=1         -> DEBUG
      ORB_LOG_LEVEL=DEBUG   -> DEBUG (or any standard level name)
    """
    if os.getenv("ORB_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
        return logging.DEBUG
    name = os.getenv("ORB_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_file_logging(level: int | None = None) -> str:
    """Attach a RotatingFileHandler to the root logger.

    Args:
        level: Minimum log level for the file handler. When None (the
               default) it is resolved from the environment via
               _resolve_file_level() — INFO unless ORB_VERBOSE=1 /
               ORB_LOG_LEVEL=DEBUG. Console level is left untouched.

    Returns:
        The absolute path to the log file (for startup banner).
    """
    if level is None:
        level = _resolve_file_level()
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
        # ── v1.3 (2026-06-24) library debug-spam that flooded astra.log ──
        "PIL",                       # ~1k+ lines/session of PNG chunk parsing
        "PIL.PngImagePlugin",
        "PIL.Image",
        "PIL.TiffImagePlugin",
        "asyncio",                   # "Using proactor: IocpProactor" repeated
        "python_multipart",          # multipart form-parse callback traces
        "python_multipart.multipart",
        "faster_whisper",            # spikes on every voice transcription
        "googleapiclient",           # floods on Drive/Gemini calls
        "google_genai",
        "google.auth",
        "google_auth_httplib2",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # ── v1.2 Suppress polling-loop access log noise ───────────────
    # web_automation/pending-action and bridge/pending-navigation are
    # long-polled by every registered session every 5s; their access
    # log lines drown out everything else. Filter installs unless
    # ORB_VERBOSE_HTTP=1 is set.
    _install_access_log_filter()

    # Log the setup itself
    logger = logging.getLogger(__name__)
    logger.info("File logging initialised: %s (max %d MB x %d backups)", LOG_FILE, MAX_BYTES // (1024*1024), BACKUP_COUNT)

    return str(LOG_FILE)

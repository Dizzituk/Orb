# FILE: app/reports/cache.py
# Purpose: Reports cache dir resolution, bridge-safe filenames, TTL sweep.
# Called-by: app.reports.renderer, app.reports.surface, app.bridge.artifacts, main.py
# Depends-on: stdlib only
# Last-renovated: 2026-07-01
"""
Env knobs:
    REPORTS_CACHE_DIR   where rendered reports live (default <repo>/data/reports_cache)
    REPORTS_TTL_HOURS   sweep age for cached reports (default 24)
Filenames stay within the bridge artifact charset [A-Za-z0-9._-] so the same
file serves both the desktop window and the phone document chip.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_reports_cache_dir() -> Path:
    raw = os.getenv("REPORTS_CACHE_DIR", "").strip()
    path = Path(raw) if raw else (_REPO_ROOT / "data" / "reports_cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ttl_hours() -> float:
    try:
        return float(os.getenv("REPORTS_TTL_HOURS", "24"))
    except Exception:
        return 24.0


def make_report_filename(slug: str) -> str:
    """report-<slug>-<stamp>.html, hard-constrained to the artifact charset."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", slug or "report").strip("-") or "report"
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"report-{safe}-{stamp}.html"


def sweep_expired(keep_filenames: Optional[Iterable[str]] = None) -> int:
    """Delete cached reports older than REPORTS_TTL_HOURS. Files named in
    keep_filenames (e.g. currently open in a reports window) are spared.
    Returns the number removed. Never raises."""
    keep = set(keep_filenames or ())
    cutoff = time.time() - _ttl_hours() * 3600.0
    removed = 0
    try:
        for f in get_reports_cache_dir().glob("*.html"):
            try:
                if f.name in keep:
                    continue
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                continue
    except Exception as exc:
        logger.warning("[reports.cache] sweep failed: %s", exc)
    if removed:
        logger.info("[reports.cache] swept %d expired report(s)", removed)
    return removed

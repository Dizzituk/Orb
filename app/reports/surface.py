# FILE: app/reports/surface.py
# Purpose: Put a rendered report in front of the user — desktop reports window or Bridge document artifact.
# Called-by: app.reports.tools_registration
# Depends-on: app.media.display_client, app.llm.turn_surface, app.reports.cache
# Last-renovated: 2026-07-01
"""
Desktop: opens the report URL fullscreen on the "reports" display alias
(created on first use, falling back to "main"). Re-opening on the same alias
replaces the previous window Electron-side, so a new report takes over the
old one. Bridge: registers a pending document artifact for the turn — the
capability layer appends the [ASTRA_ARTIFACT:document:...] marker and the
phone fetches /bridge/artifacts/document/<filename>.

Env knobs:
    ASTRA_REPORTS_DISPLAY_ALIAS   display alias for the reports window (default "reports")
    ASTRA_BACKEND_BASE_URL        base URL the desktop opens (default http://127.0.0.1:8000)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _alias() -> str:
    return os.getenv("ASTRA_REPORTS_DISPLAY_ALIAS", "reports").strip() or "reports"


def _base_url() -> str:
    return os.getenv("ASTRA_BACKEND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def report_url(filename: str) -> str:
    return f"{_base_url()}/output/reports/{filename}"


async def _open_window_filenames() -> set:
    """Filenames of reports currently open in windows (sweep spares them)."""
    names = set()
    try:
        from app.media.display_client import list_open_windows

        res = await list_open_windows()
        windows = (res.get("result") or {}).get("windows") if isinstance(res, dict) else None
        for w in windows or []:
            url = str(w.get("url") or "")
            if "/output/reports/" in url:
                names.add(url.rsplit("/", 1)[-1])
    except Exception as exc:
        logger.debug("[reports.surface] open-window listing unavailable: %s", exc)
    return names


async def open_report_window(filename: str) -> dict:
    """Open on the reports alias; create the alias on first use; fall back to main."""
    from app.media.display_client import open_on_display, save_alias

    url = report_url(filename)
    alias = _alias()

    res = await open_on_display(url, display=alias, fullscreen=True)
    if isinstance(res, dict) and res.get("ok"):
        return {"ok": True, "opened_on": alias, "url": url}

    # Alias may not exist yet — bind it to the primary panel (display 0) and
    # retry. Taz can re-point it later with the display calibration tools.
    try:
        await save_alias(alias, display_index=0)
        res = await open_on_display(url, display=alias, fullscreen=True)
        if isinstance(res, dict) and res.get("ok"):
            return {"ok": True, "opened_on": alias, "url": url, "note": "created reports alias on display 0"}
    except Exception as exc:
        logger.warning("[reports.surface] alias create/retry failed: %s", exc)

    res = await open_on_display(url, display="main", fullscreen=True)
    ok = bool(isinstance(res, dict) and res.get("ok"))
    return {"ok": ok, "opened_on": "main" if ok else None, "url": url,
            "note": "reports alias unavailable — used main" if ok else "display manager unreachable"}


async def deliver_report(filename: str, title: str) -> dict:
    """Surface-aware delivery. Returns a tool-result dict describing what happened."""
    from app.llm.turn_surface import SURFACE_BRIDGE, add_turn_artifact, get_turn_surface
    from app.reports.cache import sweep_expired

    surface = get_turn_surface()
    if surface == SURFACE_BRIDGE:
        add_turn_artifact("document", filename)
        sweep_expired(keep_filenames={filename})
        return {
            "ok": True,
            "surface": "bridge",
            "delivered": "document_artifact",
            "filename": filename,
            "detail": f"{title} attached to this reply as a document.",
        }

    result = await open_report_window(filename)
    keep = await _open_window_filenames()
    keep.add(filename)
    sweep_expired(keep_filenames=keep)
    if result.get("ok"):
        return {
            "ok": True,
            "surface": "desktop",
            "delivered": "reports_window",
            "opened_on": result.get("opened_on"),
            "filename": filename,
            "detail": f"{title} is up on the {result.get('opened_on')} window.",
        }
    return {
        "ok": False,
        "surface": "desktop",
        "filename": filename,
        "url": result.get("url"),
        "error": result.get("note") or "could not open a display window",
    }

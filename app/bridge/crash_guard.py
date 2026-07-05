# FILE: app/bridge/crash_guard.py
# Purpose: Guard rails for /bridge/crash-report — body-size cap + per-source
#          rate limit. Crash reporting triggers OUTBOUND EMAIL, so it must
#          not be a spam/abuse vector.
# Called-by: app.bridge.router (crash-report endpoint)
# Depends-on: app.security.client_ip
# Last-renovated: 2026-07-02
"""Crash-report abuse guards (security hardening 2026-07-02).

The endpoint emails posted content out via Proton and writes it to disk.
Auth (require_bridge_auth) is the primary gate; these are the backstops:
  * size cap    — ASTRA_CRASH_REPORT_MAX_KB (default 256) → HTTP 413
  * rate limit  — ASTRA_CRASH_REPORT_PER_HOUR (default 5) per source IP,
                  fixed one-hour window, in-memory → HTTP 429

In-memory is deliberate: a restart resetting the window is harmless for a
single phone, and no table is worth this.
"""
from __future__ import annotations

import os
import time

from fastapi import HTTPException, Request

from app.security.client_ip import effective_client_ip

# source ip -> list of send timestamps inside the current window
_WINDOW_SECONDS = 3600.0
_sends: dict[str, list[float]] = {}


def _max_body_bytes() -> int:
    try:
        return int(os.getenv("ASTRA_CRASH_REPORT_MAX_KB", "256")) * 1024
    except ValueError:
        return 256 * 1024


def _max_per_hour() -> int:
    try:
        return int(os.getenv("ASTRA_CRASH_REPORT_PER_HOUR", "5"))
    except ValueError:
        return 5


async def enforce_crash_report_guard(request: Request) -> bytes:
    """Apply rate limit then size cap; return the raw body on success.

    Raises HTTPException 429 (rate) / 413 (size). Call BEFORE any try/except
    that maps exceptions to a 200 error envelope.
    """
    source = effective_client_ip(request) or "unknown"
    now = time.monotonic()

    window = [t for t in _sends.get(source, []) if now - t < _WINDOW_SECONDS]
    if len(window) >= _max_per_hour():
        raise HTTPException(
            status_code=429,
            detail="crash-report rate limit reached — try again later")
    window.append(now)
    _sends[source] = window

    body = await request.body()
    if len(body) > _max_body_bytes():
        raise HTTPException(
            status_code=413,
            detail=f"crash report too large (> {_max_body_bytes() // 1024}KB)")
    return body


def reset_for_tests() -> None:
    _sends.clear()

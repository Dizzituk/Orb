# FILE: app/llm/image_turn_guard.py
# Purpose: Process-level per-turn idempotency/cap for image generation so a
#          retried-but-identical image request bills exactly one paid call.
# Called-by: app.bridge.capability_layer (run_astra_chat image route)
# Depends-on: asyncio, hashlib, time, os, logging (stdlib only)
# Last-renovated: 2026-06-16
"""
Per-turn image generation guard (cost defense-in-depth).

One capability question once produced THREE paid images because a slow
image request was delivered three times (the phone retries on timeout)
and the backend had no per-turn dedupe. This module fixes that for ANY
legitimate-but-retried image request: the FIRST generation for a stable
key runs and caches its artifact; a REPEAT within the window returns the
SAME already-generated artifact, and a concurrent in-flight repeat JOINS
the running generation rather than launching a second paid call.

Key derivation:
  - `client_request_id` when the caller supplies one (the phone's
    X-Idempotency-Key, threaded through), else
  - sha256(project_id + normalized(message)) so even a request without an
    explicit id is deduped by its content.

State is a single in-process dict keyed by the derived key, each entry
guarded by its own asyncio.Lock (created under a module lock). No external
store — process lifetime is the cache lifetime, which is all we need to
defeat a retry storm. Entries older than the window are evicted lazily.

Kill-switch: ASTRA_IMAGE_TURN_GUARD=0 disables the guard (calls pass
straight through to the generator). Window: ASTRA_IMAGE_GUARD_WINDOW_S
(default 300s).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# key -> {"ts": float, "result": Optional[dict], "lock": asyncio.Lock,
#         "done": bool, "count": int}
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = asyncio.Lock()

_DEFAULT_WINDOW_S = 300  # 5 minutes; covers a phone retry storm comfortably
_MAX_ENTRIES = 256       # lazy cap so a long-lived process can't grow forever

_WS_RE = re.compile(r"\s+")


def _enabled() -> bool:
    return os.getenv("ASTRA_IMAGE_TURN_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _window_s() -> float:
    raw = os.getenv("ASTRA_IMAGE_GUARD_WINDOW_S", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return float(_DEFAULT_WINDOW_S)


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so trivially-different retries of the
    same turn (a stray space, casing) still collide on one key."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def derive_key(
    project_id: int,
    message: str,
    client_request_id: Optional[str] = None,
) -> str:
    """Stable per-turn key. Prefer the client-supplied id; else hash the
    (project, normalized message) so content-identical retries collide."""
    if client_request_id:
        cid = client_request_id.strip()
        if cid:
            return f"cid:{cid}"
    digest = hashlib.sha256(
        f"{project_id}\x00{_normalize(message)}".encode("utf-8")
    ).hexdigest()
    return f"msg:{digest}"


def _evict_expired(now: float, window: float) -> None:
    """Drop entries past the window. Cheap; called while holding _CACHE_LOCK."""
    stale = [k for k, e in _CACHE.items() if now - e["ts"] > window]
    for k in stale:
        _CACHE.pop(k, None)
    # Hard safety cap: if still oversized, drop the oldest.
    if len(_CACHE) > _MAX_ENTRIES:
        for k, _ in sorted(_CACHE.items(), key=lambda kv: kv[1]["ts"])[
            : len(_CACHE) - _MAX_ENTRIES
        ]:
            _CACHE.pop(k, None)


async def _get_entry(key: str, window: float) -> dict:
    """Return the live cache entry for `key`, creating it if absent. The
    returned entry's lock serialises generation for that key."""
    now = time.monotonic()
    async with _CACHE_LOCK:
        _evict_expired(now, window)
        entry = _CACHE.get(key)
        if entry is None or (now - entry["ts"]) > window:
            entry = {
                "ts": now,
                "result": None,
                "lock": asyncio.Lock(),
                "done": False,
                "count": 0,
            }
            _CACHE[key] = entry
        return entry


async def run_guarded_image(
    project_id: int,
    message: str,
    generator: Callable[[], Awaitable[Optional[dict]]],
    client_request_id: Optional[str] = None,
    source: str = "bridge",
) -> Optional[dict]:
    """Run `generator` at most once per stable turn key within the window.

    `generator` is a zero-arg coroutine factory that performs the real
    (paid) image generation and returns the result dict (or None). The
    FIRST caller for a key runs it; concurrent callers wait on the same
    lock and then receive the cached result without launching a second
    generation; later callers within the window get the cached artifact.

    Returns the same result dict the generator produced (or None on
    failure — failures are NOT cached, so a genuine retry after a failure
    can try again).
    """
    if not _enabled():
        logger.info("[image_guard] disabled via kill-switch; passing through")
        return await generator()

    window = _window_s()
    key = derive_key(project_id, message, client_request_id)
    entry = await _get_entry(key, window)
    entry["count"] += 1
    delivery = entry["count"]

    # Fast path: a previous call already completed for this key.
    if entry["done"]:
        cached = entry["result"]
        logger.info(
            "[image_guard] SERVED-FROM-CACHE key=%s delivery=%d project=%s "
            "filename=%s source=%s",
            key, delivery, project_id,
            (cached or {}).get("filename", "<none>"), source,
        )
        return cached

    # Serialise generation for this key. Concurrent retries queue here and,
    # on acquiring the lock, find `done` already set -> no second paid call.
    async with entry["lock"]:
        if entry["done"]:
            cached = entry["result"]
            logger.info(
                "[image_guard] JOINED-IN-FLIGHT key=%s delivery=%d project=%s "
                "filename=%s source=%s",
                key, delivery, project_id,
                (cached or {}).get("filename", "<none>"), source,
            )
            return cached

        logger.info(
            "[image_guard] GENERATE-START key=%s delivery=%d project=%s "
            "window=%ss source=%s",
            key, delivery, project_id, int(window), source,
        )
        started = time.monotonic()
        result = await generator()
        elapsed = time.monotonic() - started

        if result:
            entry["result"] = result
            entry["done"] = True
            entry["ts"] = time.monotonic()
            logger.info(
                "[image_guard] GENERATE-DONE key=%s project=%s filename=%s "
                "provider=%s elapsed=%.1fs source=%s",
                key, project_id, result.get("filename", "<none>"),
                result.get("provider", "?"), elapsed, source,
            )
        else:
            # Do NOT cache a failure — let an honest retry try again. But the
            # lock already prevented a concurrent double-generation.
            logger.warning(
                "[image_guard] GENERATE-FAILED key=%s project=%s elapsed=%.1fs "
                "source=%s (not cached; retry allowed)",
                key, project_id, elapsed, source,
            )
        return result


def reset_for_tests() -> None:
    """Clear the in-process cache. Test-only helper."""
    _CACHE.clear()

# FILE: app/bridge/inflight_idem.py
# Purpose: In-flight idempotency coalescing for /bridge/chat-and-speak — concurrent
#          same-key retries wait for the first generation instead of each starting
#          their own (the 2026-06-16 weak-link retry-storm fix).
# Called-by: app.bridge.router.bridge_chat_and_speak
# Depends-on: app.bridge.tts_cache (idem map), app.bridge.chat_speak_stream (replay)
# Last-renovated: 2026-06-16
"""
The retry storm this closes
---------------------------
tts_cache maps X-Idempotency-Key -> assistant message id only AFTER a reply is
fully generated and persisted. A retried POST that arrives WHILE the first
generation is still running (a grounded turn can take 40 s+) sees no mapping and
launches its OWN full LLM + web-search + TTS run. On a weak link the phone fired
~6 such retries for one turn (2026-06-16 road log: project 317, msgs 4191-4200),
producing duplicate replies and — because the client had already given up — no
reply delivered in real time.

begin() is the single idempotency front door for the chat-and-speak handler:
  * completed key            -> replay the original reply (no second LLM/TTS run),
  * key generation in flight -> wait for the owner, then replay its reply,
  * otherwise                -> claim ownership and return None (the caller
                                generates the reply, then MUST call complete()).

In-memory, single-process (the backend is one Uvicorn worker), so an asyncio
Event per key is correct on the event loop: claim() and the post-claim idem_get
are synchronous with no await between them, so there is no check-then-act race.
Owners release in the caller's success path; an owner that dies before releasing
self-heals after _OWNER_TTL_S (a stale entry is treated as free), and a waiter
that times out falls back to generating normally — the registry degrades to the
pre-fix behaviour, never worse.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# How long a waiter blocks for the owner's reply before giving up and generating
# itself. Comfortably longer than a slow grounded turn (~40 s observed 2026-06-16).
_WAIT_SECS = 90.0
# Backstop only: the caller (chat_speak_stream.run_chat_and_speak) releases every
# claim in a finally, so an entry should never outlive its request. This TTL just
# bounds the unreachable case (e.g. a hard crash); kept well above _WAIT_SECS so a
# genuinely slow-but-alive owner is never mistaken for dead and force-reclaimed.
_OWNER_TTL_S = 300.0
# Keys are random UUIDs that never recur, so a crashed owner's entry would never
# be reclaimed by a same-key retry; sweep stale entries once the map grows.
_MAX_TRACKED = 512

# key -> (release_event, monotonic_claim_ts); present only while in flight.
_inflight: dict[str, tuple[asyncio.Event, float]] = {}


def _sweep_stale() -> None:
    """Drop (and wake any stragglers on) owner entries past the TTL."""
    now = time.monotonic()
    for key in [k for k, (_e, ts) in _inflight.items() if (now - ts) >= _OWNER_TTL_S]:
        event, _ts = _inflight.pop(key, (None, 0.0))
        if event is not None:
            event.set()


def _claim(key: str, *, force: bool = False) -> bool:
    """Register ``key`` as in-flight. True if we became the owner. Synchronous and
    event-loop-atomic (no await between the read and the write), so two racing
    requests can never both win."""
    entry = _inflight.get(key)
    if entry is not None and not force:
        _event, ts = entry
        if (time.monotonic() - ts) < _OWNER_TTL_S:
            return False  # a live owner already holds it
    if len(_inflight) > _MAX_TRACKED:
        _sweep_stale()
    _inflight[key] = (asyncio.Event(), time.monotonic())
    return True


async def _wait(key: str, timeout: float) -> bool:
    """Block until the owner of ``key`` releases, up to ``timeout`` seconds. False
    if the key is not in flight, the owner is stale, or the wait times out."""
    entry = _inflight.get(key)
    if entry is None:
        return False
    event, ts = entry
    if (time.monotonic() - ts) >= _OWNER_TTL_S:
        return False  # stale owner — caller reclaims and generates
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning("[inflight_idem] waiter timed out (%.0fs) key=%s", timeout, key[:12])
        return False


def complete(key: Optional[str]) -> None:
    """Release ``key``: wake every waiter and drop the entry. Idempotent and safe
    with a None/blank key (no-op). Call once the assistant reply is persisted AND
    its tts_cache writer is open (so woken waiters resume the audio, not race a
    second synthesis)."""
    if not key:
        return
    entry = _inflight.pop(key, None)
    if entry is not None:
        entry[0].set()


async def begin(key: Optional[str], db, process_artifacts) -> Optional[StreamingResponse]:
    """Idempotency front door — see the module docstring.

    Returns a StreamingResponse to return to the client immediately (a completed
    key replayed, or a concurrent in-flight generation coalesced), or None when
    the caller should generate the reply itself. When None is returned for a
    non-blank key, THIS call owns the key and the caller must release it via
    complete(key) after the reply is persisted.
    """
    if not key:
        return None
    from . import tts_cache
    from .chat_speak_stream import replay_cached_reply

    async def _try_replay(reason: str):
        mid = tts_cache.idem_get(key)
        if mid is None:
            return None
        resp = await replay_cached_reply(mid, db, process_artifacts)
        if resp is not None:
            logger.info("[inflight_idem] %s key=%s -> msg %s", reason, key[:12], mid)
        return resp

    # 1. Already answered: replay the original reply, no second LLM/TTS run.
    done = await _try_replay("replay completed")
    if done is not None:
        return done

    # 2. Own it, or wait for the owner already generating this exact key.
    if _claim(key):
        return None  # we are the owner — caller generates, then complete()s
    if await _wait(key, _WAIT_SECS):
        coalesced = await _try_replay("coalesced retry")
        if coalesced is not None:
            return coalesced
    # Wait elapsed (or nothing produced yet): one last check catches an owner that
    # finished right at the timeout, then we take ownership and generate ourselves
    # (bounded duplication at worst, never a hang).
    late = await _try_replay("coalesced late")
    if late is not None:
        return late
    _claim(key, force=True)
    return None

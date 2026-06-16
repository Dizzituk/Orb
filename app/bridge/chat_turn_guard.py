# FILE: app/bridge/chat_turn_guard.py
# Purpose: Process-level in-flight idempotency guard for /bridge/chat-and-speak
#          so a retried-but-identical request that arrives WHILE the first is
#          still generating joins it instead of starting a second reply.
# Called-by: app.bridge.chat_speak_handler (run_chat_and_speak)
# Depends-on: asyncio, os, logging (stdlib only)
# Last-renovated: 2026-06-16 (JOB-3: double-news-update fix)
"""
In-flight per-turn guard for the chat-and-speak path (duplicate-reply defence).

One news request once produced TWO full news updates: the phone's Phase-A POST
loop re-POSTed /bridge/chat-and-speak with the SAME X-Idempotency-Key when the
first response header took longer than its read timeout (a grounded reply emits
no header until grounding + the multi-round web_search tool loop finish — ~50s),
and the backend had no in-flight reservation. The receipt-time idempotency check
(tts_cache.idem_get) only catches a key that has ALREADY completed (idem_put runs
after run_astra_chat returns), so a retry arriving mid-generation passed the gate
and started a SECOND full run_astra_chat — its own grounding, tool loop, assistant
row and TTS audio. Net: one user row, N assistant rows; double cost; two updates.

This module closes the in-flight window. It mirrors app.llm.image_turn_guard
(the image route's coalescer) but is a thinner primitive: the RESULT channel is
already the existing tts_cache idem map, so this only needs to serialise same-key
turns —

  - begin(key): the FIRST caller for a free key becomes the owner (returns True)
    and must call end(key) in a finally. A caller for a key already in flight
    AWAITS the owner's completion, then returns False — the caller then re-checks
    tts_cache.idem_get (now a hit, because the owner idem_put before end) and
    replays the original reply. No second generation.
  - end(key): owner signals completion and frees the slot, waking any waiters.

State is a single in-process dict (key -> asyncio.Event), guarded by a module
asyncio.Lock. Process lifetime is the cache lifetime — all we need to defeat a
retry storm; nothing persists across a restart. A waiter's await is bounded by
_JOIN_TIMEOUT_S so a never-released slot (e.g. owner killed) can't hang it
forever — on timeout it falls through to the normal replay/generate path.

Key scope: the guard keys on the client's X-Idempotency-Key, which the phone
mints fresh per send (one UUID per ChatSpeakDownloader.start()). So two
DELIBERATE identical requests ("ask for the news twice") carry DIFFERENT keys
and are never collapsed — only a retry storm of the SAME logical send shares a
key, which is exactly what we coalesce.

Kill-switch: ASTRA_CHAT_TURN_GUARD=0 disables the guard (every request runs as
its own turn, i.e. the pre-fix behaviour). Mirrors ASTRA_IMAGE_TURN_GUARD.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# key -> Event signalled (set) when the owning turn completes.
_IN_FLIGHT: dict[str, asyncio.Event] = {}
_REGISTRY_LOCK = asyncio.Lock()

# A waiter never blocks longer than this on the owner (defensive — the owner
# releases in a finally, so this only matters if the owner's task died). Comfortably
# covers a slow grounded generation (~50s observed) plus headroom.
_JOIN_TIMEOUT_S = 180.0


def enabled() -> bool:
    return os.getenv("ASTRA_CHAT_TURN_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


async def begin(key: str) -> bool:
    """Claim `key` as the in-flight owner, or join an existing owner.

    Returns:
        True  — this caller is the OWNER. It must run the turn and call
                end(key) in a finally.
        False — another turn for this key was in flight; this caller WAITED
                for it to finish (bounded by _JOIN_TIMEOUT_S). The caller
                should now re-check tts_cache.idem_get and replay the original
                reply instead of generating a new one.
    """
    if not key:
        return True
    async with _REGISTRY_LOCK:
        event = _IN_FLIGHT.get(key)
        if event is None:
            _IN_FLIGHT[key] = asyncio.Event()
            return True  # owner
    # Another turn owns this key — wait it out, then replay.
    logger.info("[chat_turn_guard] join-in-flight key=%s (awaiting owner)", key[:12])
    try:
        await asyncio.wait_for(event.wait(), timeout=_JOIN_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            "[chat_turn_guard] join wait timed out after %ss key=%s — "
            "falling through to replay/generate", int(_JOIN_TIMEOUT_S), key[:12],
        )
    return False  # joiner


def end(key: str) -> None:
    """Owner signals completion: wake any waiters and free the slot.

    Safe to call once per successful begin()==True. Idempotent for a missing key.
    """
    if not key:
        return
    event = _IN_FLIGHT.pop(key, None)
    if event is not None:
        event.set()


def reset_for_tests() -> None:
    """Clear the in-process registry. Test-only helper."""
    _IN_FLIGHT.clear()


__all__ = ["enabled", "begin", "end", "reset_for_tests"]

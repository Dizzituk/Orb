# FILE: app/reminders/announce.py
# Purpose: The one announce line for a fired reminder — LLM-phrased once, cached so
#          the chat message and the desktop TTS say exactly the same thing.
# Called-by: app.reminders.scheduler (fire-time chat delivery), app.reminders.router (/announce)
# Depends-on: app.llm.routing.core (quick_chat_async; hard template fallback)
# Last-renovated: 2026-07-03
"""
One reminder = one line. The scheduler composes it at fire time (chat-turn
delivery, 2026-07-03 evening), remembers it here, and the desktop watcher's
/announce call inside the freshness window gets the SAME line back for TTS —
so what Astra says out loud is exactly what she wrote into the chat.

Process-local cache (single backend process); stale entries pruned on write
and on read. If the cache misses (restart between fire and poll), the
/announce endpoint falls back to the plain template — worst case the spoken
wording differs slightly from the chat line, never a lost announce.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_ANNOUNCE_TIMEOUT_S = 12.0

# A delivery this recent still owes Taz the desktop chime + voice: the
# scheduler inserts the chat message and stamps delivered_at, and the
# desktop watcher's poll (20s cadence) lands moments later and asks
# /announce whether to speak. Older deliveries stay silent — the phone
# alarm and the chat history already told him.
FRESH_DELIVERY_WINDOW_S = 180.0

_LINES: Dict[int, Tuple[str, float]] = {}


def fallback_line(text: str) -> str:
    return f"Taz — reminder: {text}"


async def announce_line(text: str) -> str:
    """One short casual line delivering the reminder — it lands in the chat
    AND is spoken via TTS. Phrased by the light chat model; falls back to a
    plain template so delivery never blocks on (or breaks with) the LLM."""
    try:
        from app.llm.routing.core import quick_chat_async
        prompt = (
            "You are ASTRA, Taz's AI companion. A reminder he set is due right "
            "now, and you are delivering it — your one line appears in the chat "
            "window and is spoken out loud through TTS at the same time.\n"
            f'Reminder text: "{text}"\n'
            "Reply with exactly one short, casual, warm sentence delivering the "
            "reminder to Taz — make clear it's his reminder (e.g. starting like "
            "'Taz — quick reminder…'). No preamble, no quotes, just the sentence."
        )
        line = (await asyncio.wait_for(quick_chat_async(prompt), timeout=_ANNOUNCE_TIMEOUT_S) or "").strip()
        line = line.splitlines()[0].strip().strip('"').strip()
        if not line or len(line) > 300:
            return fallback_line(text)
        return line
    except Exception as exc:
        logger.info("[reminders] announce line fell back to template: %s", exc)
        return fallback_line(text)


def remember_line(reminder_id: int, line: str) -> None:
    """Cache the fire-time line so /announce can hand the desktop the same one."""
    now = time.monotonic()
    for rid in [r for r, (_, at) in _LINES.items() if now - at > 2 * FRESH_DELIVERY_WINDOW_S]:
        _LINES.pop(rid, None)
    _LINES[reminder_id] = (line, now)


def recall_line(reminder_id: int) -> Optional[str]:
    entry = _LINES.get(reminder_id)
    if entry is None:
        return None
    line, at = entry
    if time.monotonic() - at > 2 * FRESH_DELIVERY_WINDOW_S:
        _LINES.pop(reminder_id, None)
        return None
    return line

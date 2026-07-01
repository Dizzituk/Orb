# FILE: app/reminders/feed.py
# Purpose: JSON snapshot of due-and-unacked reminders + injection-cooldown gate —
#          mirrors app/lifestyle/nudges.py's active-feed pattern for chat injection.
# Called-by: app.reminders.scheduler (writer), app.llm.routing.memory_injection (reader)
# Depends-on: (stdlib only — the DB via app.reminders.service is the real source of
#              truth for every HTTP surface; this file is a derived cache)
# Last-renovated: 2026-07-01
"""
The database (app.reminders.service) is the source of truth for reminders —
router.py and the phone-facing bridge endpoints read/write it directly, so
acks/cancels are instantly consistent across every surface.

This file exists purely for the chat-injection path, mirroring
app/lifestyle/nudges.py exactly: the scheduler refreshes a small JSON
snapshot of due-and-unacked reminders on every 30s tick, and
get_due_reminder_for_injection() hands memory_injection.py the top one with
a 30-minute per-reminder cooldown, so a nagging reminder surfaces in chat
once per window rather than on every single message.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "reminders"
_ACTIVE_FILE = _DATA_DIR / "active.json"
_INJECTION_COOLDOWN_MINUTES = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


def write_active_reminders(reminders: List[Dict[str, Any]]) -> None:
    """reminders: [{"id": int, "text": str, "due_at": iso str}, ...] — due+unacked only."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load() or {}
    prior_by_id = {item.get("id"): item for item in (existing.get("items") or [])}
    items = []
    for r in reminders:
        prior = prior_by_id.get(r["id"]) or {}
        items.append({**r, "last_injected_at": prior.get("last_injected_at")})
    payload = {"generated_at": _now().isoformat(), "items": items}
    try:
        _ACTIVE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("[reminders_feed] write failed: %s", exc)


def _load() -> Optional[Dict[str, Any]]:
    if not _ACTIVE_FILE.exists():
        return None
    try:
        return json.loads(_ACTIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_due_reminders() -> List[Dict[str, Any]]:
    """All currently due+unacked reminders, no cooldown — informational read."""
    data = _load()
    if not data:
        return []
    return list(data.get("items") or [])


def get_due_reminder_for_injection() -> Optional[str]:
    """One due reminder for prompt injection, honouring a per-reminder 30-minute
    cooldown so a nag surfaces once per window, not on every turn. Stamps the read."""
    data = _load()
    if not data:
        return None
    items = data.get("items") or []
    if not items:
        return None

    for item in items:
        last = item.get("last_injected_at")
        if last:
            try:
                if _now() - datetime.fromisoformat(last) < timedelta(minutes=_INJECTION_COOLDOWN_MINUTES):
                    continue
            except Exception:
                pass
        item["last_injected_at"] = _now().isoformat()
        try:
            _ACTIVE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
        return f"\"{item['text']}\" (was due {item['due_at']})"

    return None

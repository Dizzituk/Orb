# FILE: app/translation/confirmation_log.py
"""
Confirmation Event Logger.

Logs every confirm/reject decision for pipeline-triggering actions.
Over time, these logs feed into the graduated confidence system:
commands that consistently get confirmed can be promoted to auto-execute.

v1.0 (2026-03-11): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.translation.schemas import CanonicalIntent

logger = logging.getLogger(__name__)

# Log file location
_LOG_DIR = Path(os.getenv("ORB_METRICS_DIR", "D:/Orb/metrics"))
_LOG_FILE = _LOG_DIR / "confirmation_events.jsonl"


def log_confirmation_event(
    intent: CanonicalIntent,
    user_message_excerpt: str,
    confirmed: bool,
    confidence: float = 0.0,
    conversation_id: Optional[str] = None,
) -> None:
    """Log a confirmation event for future intent learning.

    Args:
        intent: The canonical intent that was confirmed/rejected.
        user_message_excerpt: First 200 chars of the user's original message.
        confirmed: Whether the user confirmed (True) or rejected (False).
        confidence: The intent confidence at time of gating.
        conversation_id: Optional conversation ID for tracing.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": intent.value,
        "user_message_excerpt": user_message_excerpt[:200],
        "confirmed": confirmed,
        "confidence": round(confidence, 4),
        "conversation_id": conversation_id,
    }

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning("[confirmation_log] Failed to write event: %s", e)


def get_confirmation_rate(
    intent: CanonicalIntent,
    min_samples: int = 10,
) -> Optional[float]:
    """Calculate the confirmation rate for an intent.

    Returns None if fewer than min_samples events exist.
    Returns float 0.0-1.0 representing percentage confirmed.
    """
    if not _LOG_FILE.exists():
        return None

    confirmed_count = 0
    total_count = 0

    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("intent") == intent.value:
                        total_count += 1
                        if event.get("confirmed"):
                            confirmed_count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("[confirmation_log] Failed to read events: %s", e)
        return None

    if total_count < min_samples:
        return None

    return confirmed_count / total_count

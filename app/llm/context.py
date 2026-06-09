# app/llm/context.py
"""
Context utilities for LLM calls.

Provides current datetime and system-context helpers used to anchor every
chat-mode LLM call to the real wall-clock time.

v2.0 (2026-05-20): Richer datetime block — adds year, day-of-year, ISO week,
    UTC offset, and ISO 8601 timestamp on additional lines so the model can
    do precise temporal reasoning without ambiguity. The legacy first-line
    format ("Current date and time: <Day>, <Month> <DD>, <YYYY> at <HH:MM> AM/PM (<TZ>)")
    is preserved for backwards compatibility with existing tests/parsers.
    The main chat path now consumes this via get_system_context() from
    build_system_prompt() in app/llm/routing/prompt_builders.py.
v1.1: Added ASTRA capability layer injection.
"""

from __future__ import annotations

from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_current_datetime_context() -> str:
    """
    Return a comprehensive temporal context block for LLM injection.

    Line 1 is the legacy human-readable form (kept for backwards compatibility
    with downstream code and the test suite). Subsequent lines add structured
    anchors (year, day-of-year, ISO week, UTC offset, ISO 8601 timestamp) so
    the model can answer "what year is it / how many days until X" without
    drifting back to training-data assumptions.

    Example output:
        Wednesday, May 20, 2026 at 10:39 AM (BST)
        Year: 2026 | Day of year: 140 | ISO week: 21 | UTC offset: +01:00
        ISO timestamp: 2026-05-20T10:39:42+01:00
    """
    # tz-aware local datetime (Python 3.6+: bare .astimezone() binds to local tz)
    now = datetime.now().astimezone()

    # Line 1: legacy human-readable format — DO NOT change structure.
    # Tests assert "Day", "Month", year, AM/PM, and " at " are all present.
    line1 = now.strftime("%A, %B %d, %Y at %I:%M %p")

    # Append timezone label (e.g. "BST", "GMT", "UTC") if available
    try:
        tz_name = now.tzname() or ""
        if tz_name:
            line1 += f" ({tz_name})"
    except Exception:
        pass

    # UTC offset, formatted "+HH:MM" / "-HH:MM"
    utc_offset = now.strftime("%z")
    if len(utc_offset) == 5:  # "+0100" -> "+01:00"
        utc_offset = utc_offset[:3] + ":" + utc_offset[3:]
    if not utc_offset:
        utc_offset = "n/a"

    iso_week = now.isocalendar()[1]
    day_of_year = now.timetuple().tm_yday
    iso_timestamp = now.isoformat(timespec="seconds")

    return (
        f"{line1}\n"
        f"Year: {now.year} | Day of year: {day_of_year} | "
        f"ISO week: {iso_week} | UTC offset: {utc_offset}\n"
        f"ISO timestamp: {iso_timestamp}"
    )


def get_system_context() -> str:
    """
    Return the datetime block with the legacy "Current date and time:" prefix.

    This is the function called from build_system_prompt() to inject the
    temporal block at the very top of every chat system prompt, so ASTRA
    always knows what year/month/day/time it is without having to ask.
    """
    return f"Current date and time: {get_current_datetime_context()}"


def enhance_system_prompt(base_prompt: str) -> str:
    """
    Backwards-compat wrapper: prepend datetime + capability layer to a prompt.

    The main chat path now uses get_system_context() directly from
    build_system_prompt(); this function remains for legacy callers and the
    existing test suite (tests/test_context.py).

    Order of composition:
      1. Datetime block (via get_system_context)
      2. ASTRA capability layer (if app.capabilities is importable)
      3. Base prompt (or empty)
    """
    # 1. Capability layer (best-effort; never fatal)
    try:
        from app.capabilities import inject_capabilities
        prompt_with_caps = inject_capabilities(base_prompt)
        logger.debug("Capability layer injected successfully")
    except ImportError as e:
        logger.warning(f"Capability layer not available: {e}")
        prompt_with_caps = base_prompt
    except Exception as e:
        logger.error(f"Error injecting capabilities: {e}")
        prompt_with_caps = base_prompt

    # 2. Datetime block
    context = get_system_context()

    # 3. Compose
    if prompt_with_caps:
        return f"{context}\n\n{prompt_with_caps}"
    return context


__all__ = [
    "get_current_datetime_context",
    "get_system_context",
    "enhance_system_prompt",
]

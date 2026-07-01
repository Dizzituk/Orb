# FILE: app/reminders/time_parse.py
# Purpose: Hand-rolled natural-language time parser for reminders — code is
#          authoritative for date arithmetic, never model estimation.
# Called-by: app.reminders.service, app.tools.reminder_tools
# Depends-on: (stdlib only — datetime/re)
# Last-renovated: 2026-07-01
"""
Resolves a natural-language `when` string (e.g. "in 20 minutes", "tomorrow
9am", "friday 8am", "3pm") into a tz-aware local datetime.

Grounded on datetime.now().astimezone() — the same local-tz-aware "now"
convention used by app/llm/context.py (the prompt datetime header),
app/lifestyle/scheduler.py, and app/lifestyle/nudges.py — so a phone-created
and a desktop-created reminder resolve identically (both hit this one
backend process; there is no per-client clock to reconcile).

Code is authoritative (date arithmetic belongs in code, not model
estimation). An optional model-supplied ISO `due_at` is accepted purely as
a fallback for phrasing this parser doesn't recognise — never preferred
over a successful parse.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, time as dt_time
from typing import Optional

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_UNIT_SECONDS = {
    "second": 1, "sec": 1,
    "minute": 60, "min": 60,
    "hour": 3600, "hr": 3600,
    "day": 86400,
}

_TIME_RE = re.compile(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_RELATIVE_RE = re.compile(
    r"\bin\s+(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _parse_time_of_day(text: str) -> Optional[dt_time]:
    """Find a clock-time in free text: '3pm', '3:30pm', '15:30', 'noon', 'midnight'."""
    t = text.strip().lower()
    if "noon" in t:
        return dt_time(12, 0)
    if "midnight" in t:
        return dt_time(0, 0)

    match = _TIME_RE.search(t)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if meridiem:
        meridiem = meridiem.lower()
        if hour == 12:
            hour = 0
        if meridiem == "pm":
            hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return dt_time(hour, minute)


def parse_when(
    when: str,
    model_due_at_iso: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Resolve `when` to a tz-aware local datetime, or None if unparseable
    (caller decides whether to fall back to model_due_at_iso itself).
    """
    if not when or not when.strip():
        return _fallback(model_due_at_iso)

    text = when.strip().lower()
    now = now or _now()

    # "in N minutes/hours/days/seconds"
    rel = _RELATIVE_RE.search(text)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower().rstrip("s")
        seconds = _UNIT_SECONDS.get(unit)
        if seconds:
            return now + timedelta(seconds=amount * seconds)

    # "tomorrow [time]"
    if "tomorrow" in text:
        target_date = (now + timedelta(days=1)).date()
        tod = _parse_time_of_day(text.replace("tomorrow", "")) or dt_time(9, 0)
        return datetime.combine(target_date, tod, tzinfo=now.tzinfo)

    # "today [time]" / "tonight [time]"
    if "today" in text or "tonight" in text:
        default_tod = dt_time(20, 0) if "tonight" in text else dt_time(9, 0)
        stripped = text.replace("today", "").replace("tonight", "")
        tod = _parse_time_of_day(stripped) or default_tod
        return datetime.combine(now.date(), tod, tzinfo=now.tzinfo)

    # weekday name, e.g. "friday 8am", "next friday", "fri"
    for name, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", text):
            days_ahead = (weekday - now.weekday()) % 7
            explicit_next = "next" in text
            tod = _parse_time_of_day(text) or dt_time(9, 0)
            candidate = datetime.combine((now + timedelta(days=days_ahead)).date(), tod, tzinfo=now.tzinfo)
            if days_ahead == 0 and (explicit_next or candidate <= now):
                candidate += timedelta(days=7)
            return candidate

    # bare clock time, e.g. "3pm", "at 17:30" — today if still ahead, else tomorrow
    tod = _parse_time_of_day(text)
    if tod is not None:
        candidate = datetime.combine(now.date(), tod, tzinfo=now.tzinfo)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    return _fallback(model_due_at_iso)


def _fallback(model_due_at_iso: Optional[str]) -> Optional[datetime]:
    if not model_due_at_iso:
        return None
    try:
        parsed = datetime.fromisoformat(model_due_at_iso)
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed
    except Exception:
        return None

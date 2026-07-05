# FILE: app/reminders/time_parse.py
# Purpose: Hand-rolled natural-language time parser for reminders — code is
#          authoritative for date arithmetic, never model estimation.
# Called-by: app.reminders.service, app.tools.reminder_tools
# Depends-on: (stdlib only — datetime/re)
# Last-renovated: 2026-07-03
"""
Resolves a natural-language `when` string (e.g. "in 20 minutes", "in two
minutes", "tomorrow 9am", "friday 8am", "3pm", "at ten", "on the 17th of
December", "the seventeenth of december", "dec 17 4:15pm", "17/12") into a
tz-aware local datetime.

2026-07-03 evening: spoken number words. Voice transcription hands this
parser WORDS ("in two minutes", "at ten", "the seventeenth of december",
"half an hour") where typed input has digits — _normalize_when rewrites
them to the digit forms every branch already understands, so voice and
typed input resolve identically. Known limit: "half eight" (UK 8:30) and
"ten to eight" are not attempted.

2026-07-03: absolute calendar dates. Before this, "on the 17th of December"
fell through to the bare clock-time branch, which silently misread the "17"
as 5pm today — exactly the phrasing calendar entries are made of. Absolute
dates are checked BEFORE weekday names and bare clock times; numeric dates
are UK day-first (17/12 = 17 December); a dateless year defaults to the
next occurrence (this year if still ahead, else next year); time-of-day in
the remainder applies, else 09:00.

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

import calendar as _calendar
import re
from datetime import date as _date, datetime, timedelta, time as dt_time
from typing import Optional, Tuple

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

# [:.] — UK phrasing writes "4.30pm" as often as "4:30pm" (2026-07-03)
_TIME_RE = re.compile(r"(?:at\s+)?(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_RELATIVE_RE = re.compile(
    r"\bin\s+(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "17th of december [2027]", "17 dec"
_DAY_MONTH_RE = re.compile(
    rf"\b(?:on\s+)?(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b(?:\s+(20\d{{2}})\b)?",
    re.IGNORECASE,
)
# "december 17th [2027]", "dec 17"
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT})\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s+(20\d{{2}})\b)?",
    re.IGNORECASE,
)
# UK numeric day-first, slash only (dots stay clock times: "4.30pm").
# Negative lookahead keeps "5/12pm"-style oddities out of the date branch.
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b(?!\s*(?:am|pm))", re.IGNORECASE)
# "on the 23rd" / "the 23rd" with no month — next occurrence of that
# day-of-month (ordinal suffix REQUIRED, so clock times can't wander in)
_BARE_ORDINAL_RE = re.compile(r"(?:\bon\s+)?\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

# ── Spoken number words → digits (2026-07-03 evening) ────────────────────

_WORD_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_WORD_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}

_ORDINAL_WORDS = {
    "first": "1st", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th",
    "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
}
# ordinal units for compounds ("twenty second" → 22nd) — "second" is safe
# HERE because the tens word in front makes the ordinal reading unambiguous.
_ORDINAL_UNIT_SUFFIX = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th",
}


def _normalize_when(text: str) -> str:
    """Rewrite spoken forms to the digit forms the branch regexes understand.
    Ordering matters: compounds before simple words, duration phrases before
    the generic a/an rule. Bare "second" is only treated as an ordinal in
    clearly ordinal contexts ("the second", "second of") so "in thirty
    seconds" keeps its unit meaning."""
    # compound ordinals: "twenty first" / "twenty-first" -> 21st
    def _compound_ordinal(m: "re.Match[str]") -> str:
        suffix = _ORDINAL_UNIT_SUFFIX[m.group(2)]
        return str(_WORD_TENS[m.group(1)] + int(suffix[:-2])) + suffix[-2:]
    text = re.sub(
        r"\b(twenty|thirty)[\s-](first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b",
        _compound_ordinal, text,
    )
    # compound cardinals: "twenty five" / "twenty-five" -> 25
    text = re.sub(
        r"\b(twenty|thirty|forty|fifty)[\s-](one|two|three|four|five|six|seven|eight|nine)\b",
        lambda m: str(_WORD_TENS[m.group(1)] + _WORD_UNITS[m.group(2)]), text,
    )
    # simple ordinals ("seventeenth" -> 17th); bare "second" only when clearly ordinal
    for word, repl in _ORDINAL_WORDS.items():
        text = re.sub(rf"\b{word}\b", repl, text)
    text = re.sub(r"\bthe\s+second\b", "the 2nd", text)
    text = re.sub(r"\bsecond\s+of\b", "2nd of", text)
    # duration phrases (longest first so "an hour and a half" wins)
    text = re.sub(r"\ban?\s+hour\s+and\s+a\s+half\b", "90 minutes", text)
    text = re.sub(r"\b(?:a\s+)?half\s+an?\s+hour\b", "30 minutes", text)
    text = re.sub(r"\b(?:a\s+)?quarter\s+of\s+an?\s+hour\b", "15 minutes", text)
    text = re.sub(r"\ba\s+couple(?:\s+of)?\b", "2", text)
    text = re.sub(r"\ba\s+few\b", "3", text)
    text = re.sub(r"\ban?\s+(second|sec|minute|min|hour|hr|day)s?\b", r"1 \1", text)
    # simple cardinals ("two" -> 2, "ten" -> 10)
    for word, value in {**_WORD_UNITS, **_WORD_TENS}.items():
        text = re.sub(rf"\b{word}\b", str(value), text)
    # spoken clock times: "4 15 pm" -> "4:15pm"; "at 10 30" -> "at 10:30"
    def _clock(m: "re.Match[str]") -> str:
        return m.group(0) if int(m.group(2)) > 59 else f"{m.group(1)}:{m.group(2)}{m.group(3) or ''}"
    text = re.sub(r"\b(\d{1,2})\s+(\d{2})\s*(am|pm)\b", _clock, text)
    text = re.sub(
        r"(?<=\bat\s)(\d{1,2})\s+(\d{2})\b(?![:/.\d])",
        lambda m: m.group(0) if int(m.group(2)) > 59 else f"{m.group(1)}:{m.group(2)}",
        text,
    )
    return text


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


def _parse_absolute_date(text: str, now: datetime) -> Optional[Tuple[_date, str]]:
    """Find an explicit calendar date in `text`. Returns (date, remainder) with
    the matched date span removed so a trailing clock time parses cleanly, or
    None. Numeric dates are UK day-first. Dateless years roll to the next
    occurrence; an explicit year is honoured as given."""
    year: Optional[int] = None

    m = _DAY_MONTH_RE.search(text)
    if m:
        day, month = int(m.group(1)), _MONTHS[m.group(2).lower()]
        year = int(m.group(3)) if m.group(3) else None
    else:
        m = _MONTH_DAY_RE.search(text)
        if m:
            month, day = _MONTHS[m.group(1).lower()], int(m.group(2))
            year = int(m.group(3)) if m.group(3) else None
        else:
            m = _NUMERIC_DATE_RE.search(text)
            if m:
                day, month = int(m.group(1)), int(m.group(2))
                if m.group(3):
                    year = int(m.group(3))
                    if year < 100:
                        year += 2000
            else:
                m = _BARE_ORDINAL_RE.search(text)
                if m is None:
                    return None
                # "on the 23rd" — the next month that has that day, today included
                day = int(m.group(1))
                probe_year, probe_month = now.year, now.month
                for _ in range(13):
                    if day <= _calendar.monthrange(probe_year, probe_month)[1]:
                        candidate = _date(probe_year, probe_month, day)
                        if candidate >= now.date():
                            remainder = (text[: m.start()] + " " + text[m.end():]).strip()
                            return candidate, remainder
                    probe_month += 1
                    if probe_month > 12:
                        probe_month, probe_year = 1, probe_year + 1
                return None

    try:
        target = _date(year or now.year, month, day)
    except ValueError:
        return None  # e.g. 31/02 — let other branches (or the ISO fallback) try
    if year is None and target < now.date():
        try:
            target = _date(now.year + 1, month, day)
        except ValueError:
            return None  # 29 Feb rollover into a non-leap year
    remainder = (text[: m.start()] + " " + text[m.end():]).strip()
    return target, remainder


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

    text = _normalize_when(when.strip().lower())
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

    # explicit calendar date — "17th of december [4:15pm]", "dec 17", "17/12",
    # "on the 23rd". Checked BEFORE weekdays and bare clock times so the day
    # number can never be misread as an hour (2026-07-03).
    absolute = _parse_absolute_date(text, now)
    if absolute is not None:
        target_date, remainder = absolute
        tod = _parse_time_of_day(remainder) or dt_time(9, 0)
        return datetime.combine(target_date, tod, tzinfo=now.tzinfo)

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

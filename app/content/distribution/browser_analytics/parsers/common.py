# FILE: app/content/distribution/browser_analytics/parsers/common.py
# Purpose: Shared helpers for parsing DOM-AX snapshot elements.
# Called-by: app.content.distribution.browser_analytics.parsers.meta, app.content.distribution.browser_analytics.parsers.tiktok, app.content.distribution.browser_analytics.tests.test_scrape
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Shared helpers for parsing DOM-AX snapshot elements.

The dom_snapshot action returns:
    {
        "elements": [
            {"tag": str, "role": str, "name": str, "text": str,
             "href": str?, "x": int, "y": int, "w": int, "h": int},
            ...
        ]
    }

These helpers make it easy to find elements by text pattern or role
and parse the numeric content platforms embed in their analytics UIs.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional


# ─── Number parsing ──────────────────────────────────────────────────

# Matches: 0, 100, 1,234, 12,345,678, 1.2K, 3.4M, 5B, $1.23, -5, -1.5%
_NUMBER_WITH_SUFFIX = re.compile(
    r'(?P<sign>-?)'
    r'(?P<currency>[\$£€]?)'
    r'(?P<num>\d[\d,]*\.?\d*)'
    r'\s*'
    r'(?P<suffix>[KMB]?)',
    re.IGNORECASE,
)


def parse_number(s: str) -> Optional[float]:
    """
    Parse a human-readable number. Handles:
        "1,234" -> 1234.0
        "12.3K" -> 12300.0
        "1.5M"  -> 1500000.0
        "$42"   -> 42.0
        "-5"    -> -5.0
        "(--)"  -> None  (dash-placeholder is "not available")
        ""      -> None
    """
    if not s:
        return None
    s = s.strip()
    # Platform shorthand for "no data"
    if s in {"--", "(--)", "-", "—", "–"}:
        return None

    m = _NUMBER_WITH_SUFFIX.search(s)
    if not m:
        return None

    num_str = m.group("num").replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None

    if m.group("sign") == "-":
        value = -value

    suffix = m.group("suffix").upper()
    if suffix == "K":
        value *= 1_000
    elif suffix == "M":
        value *= 1_000_000
    elif suffix == "B":
        value *= 1_000_000_000

    return value


def parse_int(s: str) -> Optional[int]:
    """Like parse_number but coerces to int (rounds floats)."""
    n = parse_number(s)
    return None if n is None else int(round(n))


def parse_percent(s: str) -> Optional[float]:
    """
    Parse "(-100.0%)", "42%", "+5.3%", "(0.0%)" -> float (percent, not fraction).
    Returns None for "(--)" and similar placeholders.
    """
    if not s:
        return None
    s = s.strip().strip("()")
    if s in {"--", "-", "—", "–", ""}:
        return None
    m = re.search(r'(-?\+?\d+\.?\d*)\s*%', s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ─── Element finding ─────────────────────────────────────────────────

Element = dict[str, Any]


def find_elements(
    elements: Iterable[Element],
    *,
    text_contains: Optional[str] = None,
    text_regex: Optional[str] = None,
    text_starts_with: Optional[str] = None,
    role: Optional[str] = None,
    tag: Optional[str] = None,
    name_contains: Optional[str] = None,
    predicate: Optional[Callable[[Element], bool]] = None,
) -> list[Element]:
    """
    Filter elements by any combination of constraints. Returns a list
    (possibly empty) of matching elements in original order.
    """
    out: list[Element] = []
    regex = re.compile(text_regex) if text_regex else None
    for el in elements:
        text = el.get("text", "") or ""
        name = el.get("name", "") or ""
        if text_contains is not None and text_contains not in text:
            continue
        if text_starts_with is not None and not text.startswith(text_starts_with):
            continue
        if regex is not None and not regex.search(text):
            continue
        if role is not None and el.get("role", "") != role:
            continue
        if tag is not None and el.get("tag", "") != tag:
            continue
        if name_contains is not None and name_contains not in name:
            continue
        if predicate is not None and not predicate(el):
            continue
        out.append(el)
    return out


def first(elements: Iterable[Element], **kwargs) -> Optional[Element]:
    """Find first matching element, or None."""
    results = find_elements(elements, **kwargs)
    return results[0] if results else None

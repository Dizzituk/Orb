# FILE: app/watchers/price_extract.py
# Purpose: Deterministic price parsing from scraped text/snippets (no LLM).
# Called-by: app.watchers.portugal_land, app.watchers.hardware
# Depends-on: stdlib only
# Last-renovated: 2026-07-01
"""
Handles the messy real-world formats price mentions arrive in:
    £1,999.99   €450.000   1.250,50 €   2 300 €/m2   EUR 12,50   450000€
European convention (dot-thousands, comma-decimal) and UK/US convention
(comma-thousands, dot-decimal) are disambiguated by separator position.
"""

from __future__ import annotations

import re
from typing import List, Optional

# number: 1 234 567,89 | 1.234.567,89 | 1,234,567.89 | 1234567.89 | 1234567
_NUM = r"\d{1,3}(?:[ ., ]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"

# NB: no \b straight after a currency SYMBOL — symbol + space is two non-word
# chars, so the boundary never matches; \b only guards the word forms.
_PATTERNS = {
    "€": [re.compile(rf"€\s*({_NUM})"), re.compile(rf"({_NUM})\s*(?:€|euros?\b|EUR\b)", re.IGNORECASE)],
    "£": [re.compile(rf"£\s*({_NUM})"), re.compile(rf"({_NUM})\s*(?:£|GBP\b)", re.IGNORECASE)],
    "$": [re.compile(rf"\$\s*({_NUM})"), re.compile(rf"({_NUM})\s*(?:USD\b)", re.IGNORECASE)],
}


def _to_float(raw: str) -> Optional[float]:
    s = raw.strip().replace(" ", " ")
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        # Last separator wins as decimal: "1.250,50" -> comma decimal; "1,999.99" -> dot decimal.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(" ", "").replace(",", ".")
        else:
            s = s.replace(",", "").replace(" ", "")
    elif has_comma:
        # Single comma + exactly 3 trailing digits reads as thousands ("450,000").
        head, _, tail = s.rpartition(",")
        if len(tail) == 3 and head:
            s = s.replace(",", "").replace(" ", "")
        else:
            s = s.replace(" ", "").replace(",", ".")
    elif has_dot:
        head, _, tail = s.rpartition(".")
        if len(tail) == 3 and head:
            s = s.replace(".", "").replace(" ", "")
        else:
            s = s.replace(" ", "")
    else:
        s = s.replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_prices(
    text: str,
    currency: str = "€",
    lo: float = 0.0,
    hi: float = float("inf"),
) -> List[float]:
    """All parseable prices in `currency` within [lo, hi], in text order."""
    out: List[float] = []
    for pattern in _PATTERNS.get(currency, []):
        for m in pattern.finditer(text or ""):
            val = _to_float(m.group(1))
            if val is not None and lo <= val <= hi:
                out.append(val)
    return out


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return round((vals[mid - 1] + vals[mid]) / 2.0, 2)

# FILE: app/finance/utils/tax_year.py
"""
Canonical UK tax year helpers.

Single source of truth for the HMRC tax year string format and
date-to-tax-year conversion. The finance module previously had three
functions writing three different formats ("2025-26", "2025/26", and
an "always-previous-year" boundary-broken variant) into the same
`tax_year` column — which silently broke filtering when drive sync and
bank imports landed rows in the same DB.

FORMAT: "YYYY-YY"  (e.g. "2025-26" for the 6 Apr 2025 → 5 Apr 2026 year)
BOUNDARY: year starts 6 April, ends 5 April the following year.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

# Tax year boundary (HMRC): 6 April to 5 April
TAX_YEAR_START_DAY = 6
TAX_YEAR_START_MONTH = 4


def tax_year_for(d: Optional[date] = None) -> str:
    """Return the UK tax year string for a given date (defaults to today).

    Examples:
        tax_year_for(date(2025, 4, 5))  -> "2024-25"  (last day of 24-25)
        tax_year_for(date(2025, 4, 6))  -> "2025-26"  (first day of 25-26)
        tax_year_for(date(2025, 5, 1))  -> "2025-26"
        tax_year_for(date(2026, 4, 5))  -> "2025-26"  (last day of 25-26)
        tax_year_for(date(2026, 4, 6))  -> "2026-27"
    """
    d = d or date.today()
    # On or after 6 April: this calendar year is the start of the tax year
    if (d.month, d.day) >= (TAX_YEAR_START_MONTH, TAX_YEAR_START_DAY):
        start_year = d.year
    else:
        start_year = d.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def get_current_tax_year() -> str:
    """Convenience alias for tax_year_for(today)."""
    return tax_year_for()


def tax_year_bounds(tax_year: str) -> tuple[date, date]:
    """Return (start_date, end_date) for a tax year string like '2025-26'.

    Raises ValueError if the string isn't in the canonical format.
    """
    if "-" not in tax_year or len(tax_year) != 7:
        raise ValueError(
            f"Tax year must be in 'YYYY-YY' format, got: {tax_year!r}"
        )
    start_year_str, end_short = tax_year.split("-")
    start_year = int(start_year_str)
    return (
        date(start_year, TAX_YEAR_START_MONTH, TAX_YEAR_START_DAY),
        date(start_year + 1, TAX_YEAR_START_MONTH, TAX_YEAR_START_DAY - 1),
    )


def tax_quarter_for(d: date) -> int:
    """Return HMRC MTD quarter number (1-4) for a given date."""
    # Q1: 6 Apr - 5 Jul
    # Q2: 6 Jul - 5 Oct
    # Q3: 6 Oct - 5 Jan
    # Q4: 6 Jan - 5 Apr
    m, day = d.month, d.day
    if (m == 4 and day >= 6) or m in (5, 6) or (m == 7 and day <= 5):
        return 1
    if (m == 7 and day >= 6) or m in (8, 9) or (m == 10 and day <= 5):
        return 2
    if (m == 10 and day >= 6) or m in (11, 12) or (m == 1 and day <= 5):
        return 3
    return 4

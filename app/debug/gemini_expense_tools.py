# FILE: app/debug/gemini_expense_tools.py
# Purpose: Expense logging chat tool: "Astra, I've spent 120 on fuel today" -> one
# Called-by: app.llm.chat_tool_loop
# Depends-on: app.db, app.finance.services.expense_service
# Last-renovated: 2026-06-11
"""
Expense logging chat tool: "Astra, I've spent 120 on fuel today" -> one
finance_transactions row, instantly, with the day filled in.

Kept separate from gemini_finance_tools.py (work-day ledger tools) to stay
under the per-file size target and keep one responsibility per file. Wired into
chat via chat_tool_loop.get_chat_tools / execute_chat_tool, same as the finance
tools.

No tax logic here - this records business-vs-personal spend so it can be netted
against income. QuickBooks handles tax.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_amount(raw: Any) -> Optional[float]:
    """'£120', '120.50', '1,200' -> float. None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return abs(float(raw))
    s = str(raw).replace(",", "").replace("£", "").strip()
    m = re.search(r"\d+(?:\.\d{1,2})?", s)
    return abs(float(m.group(0))) if m else None


def _parse_date(raw: Any) -> date:
    """today/yesterday/ISO/day-of-month -> date. Defaults to today."""
    today = date.today()
    if raw is None:
        return today
    s = str(raw).strip().lower()
    if s in ("", "today", "tonight", "this evening"):
        return today
    if s in ("yesterday", "last night"):
        return today - timedelta(days=1)
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    # bare day-of-month e.g. "29" or "the 29th" -> that day this month (clamped)
    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        dom = int(m.group(1))
        if 1 <= dom <= 31:
            try:
                return today.replace(day=dom)
            except ValueError:
                return today
    return today


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _pretty(d: date) -> str:
    today = date.today()
    if d == today:
        return "today"
    if d == today - timedelta(days=1):
        return "yesterday"
    # No strftime("%-d"): the %- (no-leading-zero) directive is glibc-only and
    # raises "Invalid format string" on Windows. Build it by hand instead.
    return f"{d.day} {_MONTHS[d.month - 1]}"


async def _exec_log_expense(args: dict) -> str:
    args = args or {}
    amount = _parse_amount(args.get("amount"))
    if amount is None or amount <= 0:
        return "ERROR: I need an amount to log - how much was it?"

    spoken = (str(args.get("category") or args.get("item") or "").strip())
    if not spoken:
        return "ERROR: What was the expense for (fuel, garage, clothing, etc.)?"

    when = _parse_date(args.get("date"))
    merchant = (str(args.get("merchant")).strip() if args.get("merchant") else None)
    scope = (str(args.get("scope")).strip().lower() if args.get("scope") else None)
    if scope not in (None, "business", "personal", "mixed"):
        scope = None

    try:
        from app.db import SessionLocal
        from app.finance.services.expense_service import create_expense
        db = SessionLocal()
        try:
            r = create_expense(
                db,
                amount=amount,
                spoken_category=spoken,
                transaction_date=when,
                merchant=merchant,
                scope=scope,
            )
        finally:
            db.close()
    except Exception as exc:
        logger.exception("[expense_tools] log_expense failed")
        return f"ERROR: couldn't log that expense ({exc})"

    # Row is already committed. Never let a formatting slip turn a successful
    # write into a reported failure - that makes the model retry and double-log.
    try:
        when_str = _pretty(when)
        bits = [f"Logged £{r['amount']:.2f} {r['category'].lower()}"]
        if r.get("merchant"):
            bits.append(f"from {r['merchant']}")
        msg = " ".join(bits) + f" ({r['scope']}), {when_str}."
        if not r.get("confident"):
            msg += f" I filed it under {r['category']} - tell me if it should be something else."
        return msg
    except Exception:
        logger.exception("[expense_tools] confirmation formatting failed (row WAS saved)")
        return f"Logged £{r['amount']:.2f} {r['category']} (saved)."


EXPENSE_TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "log_expense",
        "description": (
            "Record a single business or personal expense the user reports "
            "spending, e.g. 'I've spent £120 on fuel today', 'log £54 for the "
            "MOT', 'put down £40 garage', 'add £30 work boots'. Use THIS for "
            "money the user SPENT (an outgoing/cost). Do not use it for delivery "
            "EARNINGS or to log a work shift - those are the work-day tools. "
            "Defaults the date to today and the scope to business; pass them "
            "only to override. Returns a confirmation - relay it, including the "
            "category it used so the user can correct a wrong guess."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount spent in GBP (e.g. 120 or 54.50)"},
                "category": {"type": "string", "description": "What the money was spent on, in the user's words (e.g. fuel, garage, MOT, work boots, parking, coffee)"},
                "date": {"type": "string", "description": "When, as 'today' (default), 'yesterday', a YYYY-MM-DD date, or a day of the month"},
                "merchant": {"type": "string", "description": "Where it was spent, if mentioned (e.g. Morrisons, Sainsbury's)"},
                "scope": {"type": "string", "description": "business (default) or personal"},
            },
            "required": ["amount", "category"],
        },
    },
]

EXPENSE_TOOL_EXECUTORS = {
    "log_expense": _exec_log_expense,
}

# FILE: app/finance/services/expense_service.py
"""
Expense logging service.

Single responsibility: turn a spoken/typed expense ("spent 120 on fuel today")
into a finance_transactions row, tagged business or personal and dropped into
the right category so spending can be broken down (fuel vs maintenance vs
clothing) and netted against income.

Deliberately NOT a tax tool. QuickBooks owns tax: deductibility, HMRC
categories and MTD quarters are left at their column defaults and are not
computed here. tax_year is stored only as the financial-year bucket the
dashboard already filters by - the same bucket the shift ledger uses.

Writes via parameterised SQL through the session connection rather than the
ORM. The Transaction model carries a foreign key to finance_credit_cards, but
that table has no mapped model, so an ORM flush of Transaction fails to
configure its mapper. The bank importer that created the existing rows writes
by raw SQL for the same reason; this service is consistent with that.

The axis that matters here is expense_scope (business / personal): business
income minus business spend is the leftover that's actually the user's
(before tax).

v1.2 (2026-05-31): raw-SQL insert (ORM mapper for Transaction is unusable due
to the dangling finance_credit_cards FK). Category lookup also raw SQL.
v1.1 (2026-05-31): stripped tax/deductible logic - QuickBooks handles tax.
v1.0 (2026-05-31): initial - powers the "Astra, log £120 fuel" voice flow.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)


# Spoken term -> canonical category name (finance_categories.name).
# First match wins; checked as substring against the lowered text. Purely for
# breaking spending down by type (and isolating fuel) - nothing tax-related.
_SYNONYMS = {
    "fuel": ["fuel", "petrol", "diesel", "fill up", "fill-up", "filled up", "tank of", "unleaded"],
    "vehicle_maintenance": ["maintenance", "garage", "repair", "repairs", "service", "serviced",
                             "tyre", "tyres", "tire", "brake", "brakes", "clutch", "exhaust",
                             "battery", "wiper", "oil change", "parts", "bodywork", "windscreen"],
    "mot_tax": ["mot", "m.o.t"],
    "parking": ["parking", "car park", "toll", "tolls", "congestion"],
    "food_on_road": ["food", "lunch", "meal", "breakfast", "dinner", "sandwich", "snack"],
    "drinks_on_road": ["drink", "drinks", "coffee", "water", "tea", "energy drink"],
    "phone_bill": ["phone bill", "mobile bill", "airtime", "sim"],
    "phone_accessories": ["phone mount", "phone holder", "charger", "charging cable", "cable",
                          "phone case", "phone accessory", "phone accessories"],
    "software_subscriptions": ["software", "subscription", "saas", "licence", "license", "app subscription"],
    "api_costs": ["api", "cloud", "openai", "anthropic", "claude", "gemini", "compute", "tokens", "hosting"],
    "hardware": ["hardware", "computer", "laptop", "pc", "monitor", "gpu", "ssd", "keyboard", "mouse"],
    "work_clothing": ["clothing", "clothes", "boots", "trousers", "jacket", "coat", "gloves",
                      "hi-vis", "hi vis", "hivis", "high vis", "ppe", "uniform", "work wear", "workwear"],
    "accountant": ["accountant", "accountancy", "bookkeeper", "bookkeeping"],
    "insurance_pl": ["public liability", "liability insurance"],
    "vehicle_insurance": ["van insurance", "vehicle insurance", "motor insurance"],
    "groceries": ["groceries", "shopping", "supermarket shop"],
    "rent": ["rent"],
    "entertainment": ["entertainment", "cinema", "netflix", "leisure"],
    "other_business": ["business expense", "work expense"],
    "other_personal": ["personal expense"],
}

_FALLBACK_CATEGORY = "other_business"


def resolve_category(db, spoken: str) -> "Tuple[Optional[dict], bool]":
    """Map a spoken term to a category row (dict: id/name/display_name/default_scope).

    Returns (category_or_None, confident). `confident` is False when we fell back
    to 'Other (Business)' because nothing matched. Raw SQL - no ORM mapper.
    """
    rows = db.execute(text(
        "SELECT id, name, display_name, default_scope "
        "FROM finance_categories WHERE is_active=1"
    )).fetchall()
    if not rows:
        return None, False
    cats = {r[1]: {"id": r[0], "name": r[1], "display_name": r[2], "default_scope": r[3]} for r in rows}

    txt = (spoken or "").strip().lower()
    if not txt:
        return cats.get(_FALLBACK_CATEGORY), False

    for canonical, words in _SYNONYMS.items():
        if canonical not in cats:
            continue
        for w in words:
            if w in txt:
                return cats[canonical], True

    for c in cats.values():
        if c["name"].replace("_", " ") in txt or (c["display_name"] or "").lower() in txt:
            return c, True

    return cats.get(_FALLBACK_CATEGORY), False


def create_expense(
    db,
    *,
    amount: float,
    spoken_category: str,
    transaction_date: date,
    merchant: Optional[str] = None,
    scope: Optional[str] = None,
    description: Optional[str] = None,
    notes: Optional[str] = None,
    input_source: str = "chat",
) -> dict:
    """Write one expense transaction (raw SQL), tagged business/personal and
    categorised. No tax logic. Returns a summary dict for the caller to relay."""
    from app.finance.utils.tax_year import tax_year_for

    if isinstance(amount, str):
        import re as _re
        _m = _re.search(r"\d+(?:\.\d{1,2})?", amount.replace(",", "").replace("£", ""))
        amount = float(_m.group(0)) if _m else 0.0
    amount = abs(round(float(amount), 2))
    cat, confident = resolve_category(db, spoken_category)

    if cat is not None:
        category_id = cat["id"]
        cat_name = cat["display_name"] or cat["name"]
        use_scope = (scope or cat["default_scope"] or "business").lower()
    else:
        category_id = None
        cat_name = "Uncategorised"
        use_scope = (scope or "business").lower()

    if not description:
        description = cat_name + (f" - {merchant}" if merchant else "")

    # Idempotency guard: absorb accidental repeats (a retry storm, or the same
    # expense dictated twice) of an identical row logged in the last 15 minutes,
    # so duplicates can't pile up even if a caller retries.
    cutoff = (datetime.now() - timedelta(minutes=15)).isoformat(sep=" ")
    dup = db.execute(text(
        "SELECT id FROM finance_transactions "
        "WHERE transaction_type='expense' AND input_source=:src AND amount=:amt "
        "AND transaction_date=:d AND is_deleted=0 "
        "AND IFNULL(category_id,-1)=IFNULL(:cat,-1) AND created_at >= :cutoff "
        "ORDER BY id DESC LIMIT 1"
    ), {"src": input_source, "amt": amount, "d": transaction_date.isoformat(),
        "cat": category_id, "cutoff": cutoff}).fetchone()
    if dup is not None:
        return {
            "ok": True, "id": int(dup[0]), "amount": amount, "category": cat_name,
            "category_id": category_id, "confident": confident, "scope": use_scope,
            "transaction_date": transaction_date.isoformat(),
            "merchant": merchant or None, "duplicate": True,
        }

    now_iso = datetime.now().isoformat(sep=" ")
    params = {
        "transaction_date": transaction_date.isoformat(),
        "amount": amount,
        "transaction_type": "expense",
        "description": description[:500],
        "category_id": category_id,
        "expense_scope": use_scope,
        "is_tax_deductible": 0,      # QuickBooks owns tax; default only
        "deductible_amount": 0.0,
        "auto_categorised": 1,
        "categorisation_confidence": ("high" if confident else "low"),
        "user_confirmed": 0,
        "input_source": input_source,
        "merchant_name": (merchant[:200] if merchant else None),
        "tax_year": tax_year_for(transaction_date),
        "notes": notes,
        "created_at": now_iso,
        "updated_at": now_iso,
        "is_deleted": 0,
    }
    db.execute(text(
        "INSERT INTO finance_transactions "
        "(transaction_date, amount, transaction_type, description, category_id, "
        " expense_scope, is_tax_deductible, deductible_amount, auto_categorised, "
        " categorisation_confidence, user_confirmed, input_source, merchant_name, "
        " tax_year, notes, created_at, updated_at, is_deleted) "
        "VALUES "
        "(:transaction_date, :amount, :transaction_type, :description, :category_id, "
        " :expense_scope, :is_tax_deductible, :deductible_amount, :auto_categorised, "
        " :categorisation_confidence, :user_confirmed, :input_source, :merchant_name, "
        " :tax_year, :notes, :created_at, :updated_at, :is_deleted)"
    ), params)
    new_id = db.execute(text("SELECT last_insert_rowid()")).scalar()
    db.commit()

    return {
        "ok": True,
        "id": int(new_id) if new_id is not None else None,
        "amount": amount,
        "category": cat_name,
        "category_id": category_id,
        "confident": confident,
        "scope": use_scope,
        "transaction_date": transaction_date.isoformat(),
        "merchant": merchant or None,
    }

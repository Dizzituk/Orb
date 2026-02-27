# FILE: app/finance/services/budget_service.py
"""
Personal budget tracking service.

Manages fixed and variable outgoings (rent, food, subscriptions,
fuel, etc.) to calculate true disposable income after all
commitments. This is NOT about tax — it's about what you actually
have left to spend/save/invest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import BudgetItem

logger = logging.getLogger(__name__)

FREQUENCY_TO_WEEKLY = {
    "weekly": 1.0,
    "fortnightly": 0.5,
    "monthly": 12.0 / 52.0,
    "quarterly": 4.0 / 52.0,
    "annual": 1.0 / 52.0,
}

CATEGORY_ORDER = [
    "housing", "transport", "food", "utilities",
    "subscriptions", "insurance", "debt", "other",
]

CATEGORY_ICONS = {
    "housing": "🏠", "transport": "🚐", "food": "🍔",
    "utilities": "⚡", "subscriptions": "📺", "insurance": "🛡️",
    "debt": "💳", "other": "📦",
}


@dataclass
class BudgetSummary:
    """Full budget breakdown showing where money goes."""
    items: list[dict] = field(default_factory=list)
    total_weekly: float = 0.0
    total_monthly: float = 0.0
    total_annual: float = 0.0
    by_category: dict = field(default_factory=dict)
    # From earnings
    avg_weekly_income: float = 0.0
    tax_reserve_weekly: float = 0.0
    disposable_weekly: float = 0.0
    disposable_daily: float = 0.0


def get_budget_items(db: Session, active_only: bool = True) -> list[dict]:
    """Get all budget items with weekly equivalents."""
    q = db.query(BudgetItem)
    if active_only:
        q = q.filter(BudgetItem.is_active == True)
    items = q.order_by(BudgetItem.sort_order, BudgetItem.category).all()

    return [
        {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "icon": CATEGORY_ICONS.get(item.category, "📦"),
            "amount": item.amount,
            "frequency": item.frequency,
            "weekly_equivalent": _to_weekly(item.amount, item.frequency),
            "monthly_equivalent": _to_monthly(item.amount, item.frequency),
            "annual_equivalent": _to_annual(item.amount, item.frequency),
            "is_fixed": item.is_fixed,
            "is_active": item.is_active,
            "notes": item.notes,
        }
        for item in items
    ]


def get_budget_summary(db: Session) -> BudgetSummary:
    """Calculate full budget summary with disposable income."""
    items = get_budget_items(db)
    summary = BudgetSummary(items=items)

    # Totals
    for item in items:
        w = item["weekly_equivalent"]
        summary.total_weekly += w

    summary.total_monthly = summary.total_weekly * 52 / 12
    summary.total_annual = summary.total_weekly * 52

    # By category
    for item in items:
        cat = item["category"]
        if cat not in summary.by_category:
            summary.by_category[cat] = {
                "icon": CATEGORY_ICONS.get(cat, "📦"),
                "weekly": 0.0, "monthly": 0.0, "items": [],
            }
        summary.by_category[cat]["weekly"] += item["weekly_equivalent"]
        summary.by_category[cat]["monthly"] += item["monthly_equivalent"]
        summary.by_category[cat]["items"].append(item["name"])

    # Get income and tax data
    try:
        from app.finance.services.finance_service import get_dashboard_data
        dashboard = get_dashboard_data(db)
        summary.avg_weekly_income = dashboard.get("avg_weekly_earnings", 0)
        summary.tax_reserve_weekly = dashboard.get("tax_weekly_aside", 0)
    except Exception:
        pass

    summary.disposable_weekly = (
        summary.avg_weekly_income - summary.tax_reserve_weekly - summary.total_weekly
    )
    summary.disposable_daily = summary.disposable_weekly / 7 if summary.disposable_weekly > 0 else 0

    return summary


def create_budget_item(db: Session, data: dict) -> BudgetItem:
    """Create a new budget item."""
    weekly = _to_weekly(data["amount"], data.get("frequency", "monthly"))
    item = BudgetItem(
        name=data["name"],
        category=data.get("category", "other"),
        amount=data["amount"],
        frequency=data.get("frequency", "monthly"),
        weekly_equivalent=weekly,
        is_fixed=data.get("is_fixed", True),
        notes=data.get("notes"),
        sort_order=data.get("sort_order", 0),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def update_budget_item(db: Session, item_id: int, data: dict) -> BudgetItem:
    """Update a budget item."""
    item = db.query(BudgetItem).get(item_id)
    if not item:
        raise ValueError("Budget item not found")

    for key, value in data.items():
        if hasattr(item, key) and key not in ("id", "created_at"):
            setattr(item, key, value)

    # Recalculate weekly
    item.weekly_equivalent = _to_weekly(item.amount, item.frequency)
    db.commit()
    db.refresh(item)
    return item


def delete_budget_item(db: Session, item_id: int) -> bool:
    """Soft-delete a budget item."""
    item = db.query(BudgetItem).get(item_id)
    if not item:
        return False
    item.is_active = False
    db.commit()
    return True


def _to_weekly(amount: float, frequency: str) -> float:
    return amount * FREQUENCY_TO_WEEKLY.get(frequency, 12.0 / 52.0)


def _to_monthly(amount: float, frequency: str) -> float:
    weekly = _to_weekly(amount, frequency)
    return weekly * 52 / 12


def _to_annual(amount: float, frequency: str) -> float:
    weekly = _to_weekly(amount, frequency)
    return weekly * 52

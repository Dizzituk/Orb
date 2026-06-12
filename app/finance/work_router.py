# FILE: app/finance/work_router.py
# Purpose: Work-ledger read API (v8.0 rebuild).
# Called-by: app.bridge.dashboards, app.llm.routing.domain_context, main
# Depends-on: app.auth, app.db, app.finance.models, app.finance.models_workday (+1 more)
# Last-renovated: 2026-06-11
"""
Work-ledger read API (v8.0 rebuild).

Lean, self-contained router that serves the rebuilt Accounts tab:
  GET /finance/work/tax-years   tax years that have data (+ current) for the tabs
  GET /finance/work/days        the day rows for a tax year (the running log)
  GET /finance/work/dashboard   aggregated stats + fixed-cost apportionment

Writes happen through the chat tools (app/tools/finance_tools.py); this
router is read-only. Figures are derived from finance_work_days, and the
fixed running costs come from finance_recurring_costs (not hardcoded), so
the apportionment is data-driven and editable.

Net model mirrors the design mock: fixed running costs are apportioned per
WORK DAY (weekly burn / 7 x days worked), then net = gross - fuel - that.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.finance.models_workday import WorkDay
from app.finance.models import RecurringCost
from app.finance.utils.tax_year import get_current_tax_year

router = APIRouter(
    prefix="/finance/work",
    tags=["Finance: Work Ledger"],
    dependencies=[Depends(require_auth)],
)


# ─── serialization ───────────────────────────────────────

def _fuel_rate_per_mile(db: Session) -> float | None:
    """Estimated fuel cost per mile (2026-06-11): total logged fuel fills
    divided by total logged work-day miles — trailing 90 days when that
    window has enough data (2+ fills, 200+ miles), else all-time. A tank
    fill pays for future miles, so per-day fuel is always an apportioned
    estimate, never the fill amount on the day it happened."""
    from datetime import date, timedelta
    from app.finance.models import Transaction

    def _window(since):
        fq = db.query(func.coalesce(func.sum(Transaction.amount), 0.0)).filter(
            Transaction.is_deleted == False,  # noqa: E712
            Transaction.transaction_type == "expense",
            Transaction.description.ilike("%fuel%"),
        )
        fc = db.query(func.count(Transaction.id)).filter(
            Transaction.is_deleted == False,  # noqa: E712
            Transaction.transaction_type == "expense",
            Transaction.description.ilike("%fuel%"),
        )
        mq = db.query(func.coalesce(func.sum(WorkDay.total_distance), 0.0)).filter(
            WorkDay.status == "complete",
        )
        if since is not None:
            fq = fq.filter(Transaction.transaction_date >= since)
            fc = fc.filter(Transaction.transaction_date >= since)
            mq = mq.filter(WorkDay.work_date >= since)
        return float(fq.scalar() or 0.0), int(fc.scalar() or 0), float(mq.scalar() or 0.0)

    try:
        spend, fills, miles = _window(date.today() - timedelta(days=90))
        if fills < 2 or miles < 200:
            spend, fills, miles = _window(None)
        if spend > 0 and miles > 0:
            return spend / miles
    except Exception:  # pragma: no cover - estimate must never break the ledger
        pass
    return None


def _serialize_day(d: WorkDay, fuel_rate: float | None = None) -> dict:
    # Fuel column semantics (2026-06-11): an explicit per-day figure wins if
    # ever set; otherwise estimate rate x distance so every completed day
    # shows its true share of fuel instead of a dash or a misplaced fill.
    fuel_value = d.fuel_cost
    fuel_estimated = False
    if not fuel_value and fuel_rate and d.total_distance:
        fuel_value = round(fuel_rate * d.total_distance, 2)
        fuel_estimated = True
    return {
        "id": d.id,
        "work_date": d.work_date.isoformat() if d.work_date else None,
        "status": d.status,
        "start_time": d.start_time,
        "end_time": d.end_time,
        "total_hours": d.total_hours,
        "net_hours": d.net_hours,
        "start_odometer": d.start_odometer,
        "end_odometer": d.end_odometer,
        "total_distance": d.total_distance,
        "work_miles": d.work_miles,
        "personal_miles": d.personal_miles,
        "parcels": d.parcels,
        "collections": d.collections,
        "rate_per_parcel": d.rate_per_parcel,
        "gross_earnings": d.gross_earnings,
        "per_hour": d.per_hour,
        "per_parcel": d.per_parcel,
        "fuel_cost": fuel_value,
        "fuel_estimated": fuel_estimated,
        "route_area": d.route_area,
        "tax_year": d.tax_year,
    }


# ─── data helpers (take a session; unit-testable) ────────

def list_tax_years(db: Session) -> dict:
    """Tax years present in the ledger, with the current year always included
    so its tab shows even before any day is logged."""
    rows = db.query(WorkDay.tax_year).distinct().all()
    years = {r[0] for r in rows if r[0]}
    current = get_current_tax_year()
    years.add(current)
    return {"tax_years": sorted(years), "current": current}


def days_for(db: Session, tax_year: str) -> list[dict]:
    rows = (db.query(WorkDay)
              .filter(WorkDay.tax_year == tax_year)
              .order_by(WorkDay.work_date.desc())
              .all())
    rate = _fuel_rate_per_mile(db)
    return [_serialize_day(d, rate) for d in rows]


def _active_fixed_costs(db: Session):
    return db.query(RecurringCost).filter(RecurringCost.is_active == True).all()  # noqa: E712


def build_dashboard(db: Session, tax_year: str) -> dict:
    """Aggregate stats + fixed-cost apportionment for one tax year.

    Net = gross earned − fuel − fixed running costs apportioned per work day
    (weekly burn ÷ 7 × days worked), matching the design mock. Business-use %
    (work miles / total miles) is reported for information.
    """
    gross, parcels, work_miles, personal_miles, distance, hours, fuel, days_worked = db.query(
        func.coalesce(func.sum(WorkDay.gross_earnings), 0.0),
        func.coalesce(func.sum(WorkDay.parcels), 0),
        func.coalesce(func.sum(WorkDay.work_miles), 0.0),
        func.coalesce(func.sum(WorkDay.personal_miles), 0.0),
        func.coalesce(func.sum(WorkDay.total_distance), 0.0),
        func.coalesce(func.sum(WorkDay.total_hours), 0.0),
        func.coalesce(func.sum(WorkDay.fuel_cost), 0.0),
        func.count(WorkDay.id),
    ).filter(WorkDay.tax_year == tax_year).one()

    gross = round(float(gross), 2)
    work_miles = round(float(work_miles), 1)
    personal_miles = round(float(personal_miles), 1)
    distance = round(float(distance), 1)
    hours = round(float(hours), 2)
    fuel = round(float(fuel), 2)
    days_worked = int(days_worked)

    business_use = (work_miles / distance) if distance > 0 else 1.0
    business_use = max(0.0, min(1.0, business_use))

    costs = _active_fixed_costs(db)
    weekly_burn = round(sum((c.weekly_equivalent or 0.0) for c in costs), 2)
    monthly_burn = round(sum((c.monthly_equivalent or 0.0) for c in costs), 2)
    fixed_apportioned = round((weekly_burn / 7.0) * days_worked, 2)

    exp = db.execute(text(
        "SELECT "
        "  COALESCE(SUM(CASE WHEN expense_scope='business' THEN amount ELSE 0 END),0) AS biz, "
        "  COALESCE(SUM(CASE WHEN expense_scope='business' AND category_id=("
        "      SELECT id FROM finance_categories WHERE name='fuel') THEN amount ELSE 0 END),0) AS fuel "
        "FROM finance_transactions "
        "WHERE transaction_type='expense' AND is_deleted=0 AND tax_year=:ty"
    ), {"ty": tax_year}).one()
    expenses_total = round(float(exp.biz or 0.0), 2)
    fuel_total = round(float(exp.fuel or 0.0), 2)
    fuel_per_day = round(fuel_total / days_worked, 2) if days_worked > 0 else 0.0

    estimated_net = round(gross - expenses_total - fixed_apportioned, 2)
    net_hourly = round(estimated_net / hours, 2) if hours > 0 else 0.0
    gross_hourly = round(gross / hours, 2) if hours > 0 else 0.0

    exp_rows = db.execute(text(
        "SELECT t.id AS id, t.transaction_date AS d, t.amount AS amount, "
        "       COALESCE(c.display_name, 'Expense') AS category "
        "FROM finance_transactions t "
        "LEFT JOIN finance_categories c ON c.id = t.category_id "
        "WHERE t.transaction_type='expense' AND t.is_deleted=0 "
        "  AND t.expense_scope='business' AND t.tax_year=:ty "
        "ORDER BY t.transaction_date DESC, t.id DESC"
    ), {"ty": tax_year}).all()
    expenses = [
        {"id": int(r.id), "transaction_date": r.d,
         "amount": round(float(r.amount), 2), "category": r.category}
        for r in exp_rows
    ]

    return {
        "tax_year": tax_year,
        "stats": {
            "gross_pay_ytd": gross,
            "total_parcels": int(parcels),
            "total_work_miles": work_miles,
            "total_personal_miles": personal_miles,
            "total_distance": distance,
            "total_hours": hours,
            "days_worked": days_worked,
            "business_use_pct": round(business_use, 4),
            "fuel_total": fuel_total,
            "fuel_per_day": fuel_per_day,
            "expenses_total": expenses_total,
            "fixed_costs_apportioned": fixed_apportioned,
            "estimated_net": estimated_net,
            "gross_hourly": gross_hourly,
            "net_hourly": net_hourly,
        },
        "expenses": expenses,
        "fixed_costs": {
            "items": [
                {
                    "id": c.id, "name": c.name, "amount": c.amount,
                    "frequency": c.frequency,
                    "monthly_equivalent": c.monthly_equivalent,
                    "weekly_equivalent": c.weekly_equivalent,
                    "is_tax_deductible": c.is_tax_deductible,
                }
                for c in costs
            ],
            "weekly_burn": weekly_burn,
            "monthly_burn": monthly_burn,
        },
    }


# ─── routes ──────────────────────────────────────────────

@router.get("/tax-years")
async def get_tax_years(db: Session = Depends(get_db)):
    return list_tax_years(db)


@router.get("/days")
async def get_days(tax_year: Optional[str] = Query(None), db: Session = Depends(get_db)):
    ty = tax_year or get_current_tax_year()
    return {"tax_year": ty, "days": days_for(db, ty)}


@router.get("/dashboard")
async def get_dashboard(tax_year: Optional[str] = Query(None), db: Session = Depends(get_db)):
    ty = tax_year or get_current_tax_year()
    return build_dashboard(db, ty)

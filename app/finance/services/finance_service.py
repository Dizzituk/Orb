# FILE: app/finance/services/finance_service.py
"""
Core finance service — handles transactions, daily logs, tax calculations.
Orchestrates between models, tax engine, and categoriser.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.finance.models import (
    Transaction, DailyWorkLog, WeeklyEarnings,
    MileageLog, MileageYearSummary, TaxYear,
    ExpenseCategory, RecurringCost, SavingsGoal,
)
from app.finance.engines.hmrc_tax_engine import HMRCTaxEngine, TaxYearConfig

logger = logging.getLogger(__name__)


def get_current_tax_year() -> str:
    """Return current UK tax year string (e.g., '2025-26')."""
    today = date.today()
    if today.month >= 4 and today.day >= 6:
        return f"{today.year}-{str(today.year + 1)[2:]}"
    return f"{today.year - 1}-{str(today.year)[2:]}"


def _calculate_hours(start: Optional[str], end: Optional[str], breaks: float = 0.0) -> tuple[Optional[float], Optional[float]]:
    """Parse HH:MM times and return (total_hours, net_hours)."""
    if not start or not end:
        return None, None
    try:
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        total = (eh * 60 + em - sh * 60 - sm) / 60
        return round(total, 2), round(total - breaks, 2)
    except (ValueError, AttributeError):
        return None, None


# ─── Transaction Operations ──────────────────────────────

def create_transaction(db: Session, data: dict) -> Transaction:
    """Create a new transaction and trigger tax recalc."""
    tax_year = data.get("tax_year") or get_current_tax_year()
    tx = Transaction(
        transaction_date=data["transaction_date"],
        amount=data["amount"],
        transaction_type=data["transaction_type"],
        description=data["description"],
        category_id=data.get("category_id"),
        expense_scope=data.get("expense_scope"),
        merchant_name=data.get("merchant_name"),
        delivery_count=data.get("delivery_count"),
        hours_worked=data.get("hours_worked"),
        notes=data.get("notes"),
        input_source=data.get("input_source", "manual"),
        tax_year=tax_year,
    )

    # Set deductible fields based on category
    if tx.category_id:
        cat = db.query(ExpenseCategory).get(tx.category_id)
        if cat and cat.is_deductible:
            tx.is_tax_deductible = True
            tx.deductible_amount = round(tx.amount * (cat.deductible_percentage / 100), 2)

    db.add(tx)
    db.commit()
    db.refresh(tx)
    logger.info("[finance] Transaction created: id=%d amount=%.2f type=%s", tx.id, tx.amount, tx.transaction_type)
    return tx


def list_transactions(
    db: Session, start_date: Optional[date] = None, end_date: Optional[date] = None,
    tx_type: Optional[str] = None, scope: Optional[str] = None,
    category_id: Optional[int] = None, page: int = 1, per_page: int = 50,
) -> dict:
    """List transactions with filters and pagination."""
    q = db.query(Transaction).filter(Transaction.is_deleted == False)
    if start_date:
        q = q.filter(Transaction.transaction_date >= start_date)
    if end_date:
        q = q.filter(Transaction.transaction_date <= end_date)
    if tx_type:
        q = q.filter(Transaction.transaction_type == tx_type)
    if scope:
        q = q.filter(Transaction.expense_scope == scope)
    if category_id:
        q = q.filter(Transaction.category_id == category_id)

    total = q.count()
    items = q.order_by(Transaction.transaction_date.desc()).offset((page - 1) * per_page).limit(per_page).all()
    # Serialise with category_name from relationship
    serialised = []
    for tx in items:
        d = {
            "id": tx.id, "transaction_date": tx.transaction_date,
            "amount": tx.amount, "transaction_type": tx.transaction_type,
            "description": tx.description, "category_id": tx.category_id,
            "category_name": tx.category.name if tx.category else None,
            "expense_scope": tx.expense_scope,
            "is_tax_deductible": tx.is_tax_deductible,
            "merchant_name": tx.merchant_name,
            "auto_categorised": tx.auto_categorised,
            "user_confirmed": tx.user_confirmed,
            "tax_year": tx.tax_year,
        }
        serialised.append(d)
    return {"items": serialised, "total": total, "page": page, "per_page": per_page}


# ─── Daily Work Log ──────────────────────────────────────

def create_daily_log(db: Session, data: dict) -> DailyWorkLog:
    """Record an end-of-day work log from Yodel finish-tour data."""
    tax_year = get_current_tax_year()
    total_hours, net_hours = _calculate_hours(
        data.get("start_time"), data.get("end_time"), data.get("break_hours", 0.0)
    )

    deliveries = data.get("delivery_count", 0)
    collections = data.get("collections", 0)
    total_parcels = deliveries + collections
    rate_per_parcel = data.get("rate_per_parcel", 0.0)

    # Auto-calculate earnings: total parcels × rate per parcel
    # If gross_earnings provided directly (> 0), use that instead
    provided_earnings = data.get("gross_earnings", 0.0)
    if provided_earnings > 0:
        earnings = provided_earnings
    elif rate_per_parcel > 0 and total_parcels > 0:
        earnings = round(total_parcels * rate_per_parcel, 2)
    else:
        earnings = 0.0

    per_hour = round(earnings / net_hours, 2) if net_hours and net_hours > 0 else 0.0
    per_delivery = round(earnings / deliveries, 2) if deliveries > 0 else 0.0

    # Food allowance: 10+ hours on road qualifies
    qualifies_food = (total_hours or 0) >= 10.0

    log = DailyWorkLog(
        work_date=data["work_date"],
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        total_hours=total_hours,
        break_hours=data.get("break_hours", 0.0),
        net_hours=net_hours,
        delivery_count=deliveries,
        failed_deliveries=data.get("failed_deliveries", 0),
        collections=data.get("collections", 0),
        stops=data.get("stops", 0),
        attempted=data.get("attempted", 0),
        done=data.get("done", 0),
        route_area=data.get("route_area"),
        tour_id=data.get("tour_id"),
        user_id=data.get("user_id"),
        rate_per_parcel=rate_per_parcel,
        total_parcels=total_parcels,
        gross_earnings=earnings,
        per_hour=per_hour,
        per_delivery=per_delivery,
        qualifies_food_allowance=qualifies_food,
        screenshot_path=data.get("screenshot_path"),
        tax_year=tax_year,
    )

    # Check for existing log on same date (update instead)
    existing = db.query(DailyWorkLog).filter(DailyWorkLog.work_date == data["work_date"]).first()
    if existing:
        for key, val in {
            "start_time": log.start_time, "end_time": log.end_time,
            "total_hours": log.total_hours, "net_hours": log.net_hours,
            "delivery_count": log.delivery_count, "failed_deliveries": log.failed_deliveries,
            "collections": log.collections, "stops": log.stops,
            "attempted": log.attempted, "done": log.done,
            "rate_per_parcel": log.rate_per_parcel, "total_parcels": log.total_parcels,
            "gross_earnings": log.gross_earnings, "per_hour": log.per_hour,
            "per_delivery": log.per_delivery, "qualifies_food_allowance": log.qualifies_food_allowance,
            "route_area": log.route_area, "tour_id": log.tour_id,
        }.items():
            setattr(existing, key, val)
        db.commit()
        db.refresh(existing)
        return existing

    db.add(log)
    db.commit()
    db.refresh(log)
    logger.info("[finance] Daily log: date=%s deliveries=%d earnings=%.2f", log.work_date, log.delivery_count, log.gross_earnings)
    return log


# ─── Tax Calculation ─────────────────────────────────────

def calculate_tax_estimate(db: Session, tax_year: Optional[str] = None) -> dict:
    """Calculate current tax position for the given tax year."""
    ty = tax_year or get_current_tax_year()
    engine = HMRCTaxEngine()

    # Sum income
    income = db.query(func.coalesce(func.sum(Transaction.amount), 0.0)).filter(
        Transaction.tax_year == ty,
        Transaction.transaction_type == "income",
        Transaction.is_deleted == False,
    ).scalar()

    # Sum deductible expenses
    expenses = db.query(func.coalesce(func.sum(Transaction.deductible_amount), 0.0)).filter(
        Transaction.tax_year == ty,
        Transaction.is_tax_deductible == True,
        Transaction.is_deleted == False,
    ).scalar()

    # Mileage
    mileage_summary = db.query(MileageYearSummary).filter(MileageYearSummary.tax_year == ty).first()
    miles = mileage_summary.total_business_miles if mileage_summary else 0.0

    # Weeks elapsed
    config = TaxYearConfig()
    today = date.today()
    weeks = max(1, (today - config.start_date).days // 7)

    # Food allowance — count qualifying days (10+ hours)
    qualifying_days = db.query(func.count(DailyWorkLog.id)).filter(
        DailyWorkLog.tax_year == ty,
        DailyWorkLog.qualifies_food_allowance == True,
    ).scalar() or 0
    food_allowance_total = qualifying_days * 10.0  # £10/day HMRC benchmark

    # Home office — HMRC simplified flat rate (£6/week for any hours)
    home_office_weekly = 6.0
    home_office_total = round(home_office_weekly * weeks, 2)

    # Add these to expenses before calculating
    total_with_missed = expenses + food_allowance_total + home_office_total
    breakdown = engine.calculate_full(income, total_with_missed, miles, weeks)

    # Attach the deduction details to the breakdown for the frontend
    breakdown.food_allowance_days = qualifying_days
    breakdown.food_allowance_total = food_allowance_total
    breakdown.home_office_weekly = home_office_weekly
    breakdown.home_office_total = home_office_total
    breakdown.recorded_expenses = round(expenses, 2)

    # Update TaxYear record
    ty_record = db.query(TaxYear).filter(TaxYear.tax_year == ty).first()
    if ty_record:
        ty_record.total_income = income
        ty_record.total_business_expenses = expenses
        ty_record.total_deductible = breakdown.total_allowable_expenses
        ty_record.taxable_profit = breakdown.taxable_profit
        ty_record.estimated_income_tax = breakdown.total_income_tax
        ty_record.estimated_ni_class2 = breakdown.ni_class2
        ty_record.estimated_ni_class4 = breakdown.ni_class4_main + breakdown.ni_class4_additional
        ty_record.total_estimated_tax = breakdown.total_tax_liability
        ty_record.last_calculated = datetime.now(timezone.utc)
        db.commit()

    return breakdown.__dict__


# ─── Dashboard ───────────────────────────────────────────

def get_dashboard_data(db: Session) -> dict:
    """Aggregate data for the overview dashboard."""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    tax_year = get_current_tax_year()

    # This week's income
    week_income = db.query(func.coalesce(func.sum(Transaction.amount), 0.0)).filter(
        Transaction.transaction_date >= week_start,
        Transaction.transaction_type == "income",
        Transaction.is_deleted == False,
    ).scalar()

    # This week's deliveries
    week_deliveries = db.query(func.coalesce(func.sum(DailyWorkLog.delivery_count), 0)).filter(
        DailyWorkLog.work_date >= week_start,
    ).scalar()

    # This week's hours
    week_hours = db.query(func.coalesce(func.sum(DailyWorkLog.net_hours), 0.0)).filter(
        DailyWorkLog.work_date >= week_start,
    ).scalar()

    gross_ph = round(week_income / week_hours, 2) if week_hours > 0 else 0.0

    # Tax estimate
    tax = calculate_tax_estimate(db, tax_year)

    # Recent transactions
    recent = db.query(Transaction).filter(
        Transaction.is_deleted == False
    ).order_by(Transaction.transaction_date.desc()).limit(10).all()

    # Recurring costs weekly total
    weekly_costs = db.query(func.coalesce(func.sum(RecurringCost.weekly_equivalent), 0.0)).filter(
        RecurringCost.is_active == True,
    ).scalar()

    # Spendable
    spendable = max(0, week_income - tax.get("weekly_tax_aside", 0) - weekly_costs)

    return {
        "this_week_earnings": round(week_income, 2),
        "this_week_deliveries": int(week_deliveries),
        "this_week_hours": round(week_hours, 2),
        "per_hour_gross": gross_ph,
        "tax_estimated_annual": tax.get("total_tax_liability", 0),
        "tax_weekly_aside": tax.get("weekly_tax_aside", 0),
        "effective_tax_rate": tax.get("effective_tax_rate", 0),
        "weekly_costs": round(weekly_costs, 2),
        "spendable_this_week": round(spendable, 2),
        "spendable_per_day": round(spendable / 7, 2),
        "recent_transactions": recent,
        "tax_year": tax_year,
    }




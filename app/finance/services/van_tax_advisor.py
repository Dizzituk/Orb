# FILE: app/finance/services/van_tax_advisor.py
"""
Intelligent van tax method advisor.

Compares simplified mileage vs actual costs each quarter/year
using real transaction data from NatWest imports and mileage logs.
Recommends whichever method produces the higher tax deduction.

HMRC rules:
- Once you use mileage for a vehicle, you CANNOT switch to actual costs
- If you start with actual costs, you CAN switch to mileage later
- AIA (purchase price) can only be claimed in year of purchase
- You choose per tax year, declared on Self Assessment
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


MILEAGE_RATE_FIRST_10K = 0.45
MILEAGE_RATE_OVER_10K = 0.25
MONTHS_PER_YEAR = 12


@dataclass
class MethodComparison:
    """Side-by-side comparison of both HMRC tax methods."""
    tax_year: str = ""
    period_start: str = ""
    period_end: str = ""

    # Mileage method
    mileage_total_miles: float = 0.0
    mileage_first_10k: float = 0.0
    mileage_over_10k: float = 0.0
    mileage_deduction: float = 0.0

    # Actual costs method
    actual_fuel: float = 0.0
    actual_insurance: float = 0.0
    actual_road_tax: float = 0.0
    actual_mot: float = 0.0
    actual_repairs: float = 0.0
    actual_servicing: float = 0.0
    actual_hp_interest: float = 0.0
    actual_aia: float = 0.0
    actual_other: float = 0.0
    actual_total_running: float = 0.0
    actual_deduction: float = 0.0

    # Recommendation
    recommended: str = ""
    difference: float = 0.0
    explanation: str = ""
    can_switch_to_mileage: bool = True
    locked_to_actual: bool = False


def _get_tax_year_bounds(target_date: Optional[date] = None):
    """Return (start, end) of the HMRC tax year containing target_date."""
    today = target_date or date.today()
    if today.month > 4 or (today.month == 4 and today.day >= 6):
        start = date(today.year, 4, 6)
    else:
        start = date(today.year - 1, 4, 6)
    end = date(start.year + 1, 4, 5)
    return start, end


def _calculate_mileage_deduction(total_miles: float) -> tuple:
    """Calculate mileage method deduction. Returns (first_10k, over_10k, total)."""
    first_10k = min(total_miles, 10000) * MILEAGE_RATE_FIRST_10K
    over_10k = max(total_miles - 10000, 0) * MILEAGE_RATE_OVER_10K
    return first_10k, over_10k, first_10k + over_10k


VAN_EXPENSE_KEYWORDS = {
    "fuel": [
        "fuel", "petrol", "diesel", "bp ", "shell ", "esso ",
        "texaco", "sainsbury fuel", "tesco fuel", "asda fuel",
        "morrisons fuel", "jet ", "gulf ",
    ],
    "insurance": ["insurance", "insure"],
    "road_tax": ["dvla"],
    "mot": ["mot "],
    "repairs": [
        "repair", "mechanic", "garage", "kwik fit",
        "halfords", "tyre", "brake", "exhaust",
    ],
    "servicing": ["service", "oil change"],
    "hp_payment": ["moneybarn"],
}


def _categorise_transactions(transactions) -> dict:
    """Sort transactions into van expense categories by keyword matching."""
    totals = {k: 0.0 for k in VAN_EXPENSE_KEYWORDS}
    for tx in transactions:
        desc = tx.description.lower()
        for cat, keywords in VAN_EXPENSE_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                totals[cat] += tx.amount
                break
    return totals


def _estimate_hp_interest(
    van, payment_count_this_year: int, payments_before: int
) -> float:
    """Calculate interest portion of HP payments using amortisation schedule.

    Only the interest portion of HP payments is deductible, not the
    capital repayment. This walks the amortisation table to work out
    exactly how much interest was in each payment.
    """
    if not (van.apr and van.finance_amount and van.monthly_payment):
        # Rough fallback: ~70% interest in early years at high APR
        return payment_count_this_year * (van.monthly_payment or 0) * 0.7

    monthly_rate = van.apr / 100 / MONTHS_PER_YEAR
    balance = float(van.finance_amount)

    # Fast-forward through payments made before this tax year
    for _ in range(max(0, payments_before)):
        interest = balance * monthly_rate
        capital = van.monthly_payment - interest
        balance = max(0, balance - capital)

    # Now sum interest for payments within this tax year
    total_interest = 0.0
    for _ in range(payment_count_this_year):
        interest = balance * monthly_rate
        capital = van.monthly_payment - interest
        total_interest += interest
        balance = max(0, balance - capital)

    return round(total_interest, 2)


def _check_aia_eligibility(van, tax_year_start: date, tax_year_end: date) -> float:
    """AIA (Annual Investment Allowance) only available in year of purchase."""
    purchase_date = van.first_payment_date
    if not purchase_date:
        return 0.0
    try:
        pd = date.fromisoformat(str(purchase_date))
        if tax_year_start <= pd <= tax_year_end:
            biz_pct = (van.business_use_percentage or 100) / 100
            return float(van.purchase_price or 0) * biz_pct
    except (ValueError, TypeError):
        pass
    return 0.0


def compare_tax_methods(
    db: Session,
    tax_year_start: Optional[date] = None,
    tax_year_end: Optional[date] = None,
) -> MethodComparison:
    """Compare mileage vs actual costs for the given tax year.

    Uses real transaction data and mileage logs to calculate both
    methods, then recommends whichever gives a higher deduction.
    """
    from app.finance.models import Transaction, VanFinance, MileageLog

    start, end = _get_tax_year_bounds()
    if tax_year_start:
        start = tax_year_start
    if tax_year_end:
        end = tax_year_end

    comp = MethodComparison(
        tax_year=f"{start.year}/{start.year + 1}",
        period_start=str(start),
        period_end=str(end),
    )

    # Get active van record
    van = db.query(VanFinance).filter(VanFinance.is_active == True).first()
    if not van:
        comp.explanation = "No active van finance record found."
        return comp

    # ── MILEAGE METHOD ──
    miles_result = db.query(func.sum(MileageLog.total_miles)).filter(
        MileageLog.log_date >= start,
        MileageLog.log_date <= end,
    ).scalar()
    total_miles = float(miles_result or 0)

    comp.mileage_total_miles = total_miles
    comp.mileage_first_10k, comp.mileage_over_10k, comp.mileage_deduction = (
        _calculate_mileage_deduction(total_miles)
    )

    # ── ACTUAL COSTS METHOD ──
    txs = db.query(Transaction).filter(
        Transaction.transaction_date >= str(start),
        Transaction.transaction_date <= str(end),
        Transaction.expense_scope.in_(["business", "mixed"]),
    ).all()

    categorised = _categorise_transactions(txs)

    comp.actual_fuel = categorised["fuel"]
    comp.actual_insurance = categorised["insurance"]
    comp.actual_road_tax = categorised["road_tax"]
    comp.actual_mot = categorised["mot"]
    comp.actual_repairs = categorised["repairs"]
    comp.actual_servicing = categorised["servicing"]

    # HP interest — only the interest portion, not capital repayment
    mb_count = db.query(func.count(Transaction.id)).filter(
        Transaction.transaction_date >= str(start),
        Transaction.transaction_date <= str(end),
        Transaction.description.ilike("%moneybarn%"),
    ).scalar() or 0

    payments_before = max(0, (van.payments_made or 0) - mb_count)
    comp.actual_hp_interest = _estimate_hp_interest(van, mb_count, payments_before)

    # AIA — purchase year only
    comp.actual_aia = _check_aia_eligibility(van, start, end)

    # Totals
    comp.actual_total_running = (
        comp.actual_fuel + comp.actual_insurance +
        comp.actual_road_tax + comp.actual_mot +
        comp.actual_repairs + comp.actual_servicing +
        comp.actual_hp_interest + comp.actual_other
    )
    comp.actual_deduction = comp.actual_total_running + comp.actual_aia

    # ── RECOMMENDATION ──
    comp.difference = abs(comp.actual_deduction - comp.mileage_deduction)
    _generate_recommendation(comp)

    return comp


def _generate_recommendation(comp: MethodComparison) -> None:
    """Populate the recommendation fields based on the comparison."""
    if comp.actual_deduction > comp.mileage_deduction:
        comp.recommended = "actual_costs"
        parts = [
            f"Actual costs saves {chr(163)}{comp.difference:,.2f} more in tax deductions."
        ]
        if comp.actual_aia > 0:
            parts.append(
                f"Includes {chr(163)}{comp.actual_aia:,.2f} AIA (van purchase) "
                f"- only available in year of purchase."
            )
        parts.append(
            f"Running costs: {chr(163)}{comp.actual_total_running:,.2f} "
            f"vs mileage: {chr(163)}{comp.mileage_deduction:,.2f}."
        )
        comp.explanation = " ".join(parts)

    elif comp.mileage_deduction > comp.actual_deduction:
        comp.recommended = "mileage"
        comp.explanation = (
            f"Mileage rate saves {chr(163)}{comp.difference:,.2f} more. "
            f"Running costs ({chr(163)}{comp.actual_total_running:,.2f}) "
            f"are lower than mileage deduction "
            f"({chr(163)}{comp.mileage_deduction:,.2f})."
        )
        if not comp.can_switch_to_mileage:
            comp.explanation += (
                " WARNING: Cannot switch - you started with actual costs "
                "for this vehicle. However, since you started on actual, "
                "you CAN switch to mileage in a future year."
            )
    else:
        comp.recommended = "actual_costs"
        comp.explanation = (
            "Both methods roughly equal. Actual costs keeps options open."
        )


def get_quarterly_snapshot(db: Session) -> dict:
    """Quick quarterly health check - are we on the right method?

    Call this from a scheduled job or dashboard widget to alert
    the user if they should consider switching methods next year.
    """
    comp = compare_tax_methods(db)
    return {
        "tax_year": comp.tax_year,
        "period": f"{comp.period_start} to {comp.period_end}",
        "mileage_deduction": round(comp.mileage_deduction, 2),
        "actual_deduction": round(comp.actual_deduction, 2),
        "recommended": comp.recommended,
        "difference": round(comp.difference, 2),
        "explanation": comp.explanation,
        "breakdown": {
            "miles_logged": comp.mileage_total_miles,
            "fuel": round(comp.actual_fuel, 2),
            "insurance": round(comp.actual_insurance, 2),
            "road_tax": round(comp.actual_road_tax, 2),
            "repairs": round(comp.actual_repairs, 2),
            "servicing": round(comp.actual_servicing, 2),
            "hp_interest": round(comp.actual_hp_interest, 2),
            "aia": round(comp.actual_aia, 2),
        },
    }


def project_next_year(db: Session) -> dict:
    """Project next year's comparison assuming similar patterns.

    Uses current year's running costs but removes AIA (one-time).
    Extrapolates partial year data to full year if mid-year.
    Helps user decide ahead of time which method to declare.
    """
    comp = compare_tax_methods(db)
    today = date.today()
    start = date.fromisoformat(comp.period_start)
    days_elapsed = (today - start).days
    if days_elapsed <= 0:
        days_elapsed = 1
    days_in_year = 365

    # Scale partial year to full year
    scale = days_in_year / days_elapsed

    projected_running = comp.actual_total_running * scale
    projected_miles = comp.mileage_total_miles * scale
    _, _, projected_mileage_ded = _calculate_mileage_deduction(projected_miles)

    # Next year: no AIA
    projected_actual = projected_running  # No AIA

    diff = abs(projected_actual - projected_mileage_ded)
    if projected_actual > projected_mileage_ded:
        rec = "actual_costs"
        note = f"Even without AIA, actual costs projected {chr(163)}{diff:,.0f} higher."
    else:
        rec = "mileage"
        note = (
            f"Without AIA, mileage projected {chr(163)}{diff:,.0f} better. "
            f"Consider switching for next tax year."
        )

    return {
        "projection_basis": f"{days_elapsed} days of data, scaled to full year",
        "projected_miles": round(projected_miles),
        "projected_running_costs": round(projected_running, 2),
        "projected_mileage_deduction": round(projected_mileage_ded, 2),
        "projected_actual_deduction": round(projected_actual, 2),
        "recommended_next_year": rec,
        "difference": round(diff, 2),
        "note": note,
    }

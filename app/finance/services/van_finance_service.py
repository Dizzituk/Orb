# FILE: app/finance/services/van_finance_service.py
"""
Van finance tracking and HMRC-aware calculations.

Tracks HP agreement, calculates remaining balance, interest paid,
and provides HMRC guidance on capital allowances vs mileage rate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import VanFinance

logger = logging.getLogger(__name__)


@dataclass
class VanFinanceSummary:
    """Full financial summary of van HP agreement."""
    vehicle: str = ""
    purchase_price: float = 0.0
    deposit_paid: float = 0.0
    finance_amount: float = 0.0
    apr: float = 0.0
    monthly_payment: float = 0.0
    total_payments: int = 0
    payments_made: int = 0
    payments_remaining: int = 0
    # Calculated
    total_cost: float = 0.0       # deposit + all payments
    total_interest: float = 0.0   # total_cost - purchase_price
    interest_paid: float = 0.0    # interest portion of payments made
    interest_remaining: float = 0.0
    capital_paid: float = 0.0     # capital portion repaid
    capital_remaining: float = 0.0
    balance_outstanding: float = 0.0
    monthly_interest: float = 0.0  # approx monthly interest
    monthly_capital: float = 0.0   # current month capital repayment
    final_balloon: float = 0.0     # balloon payment at end of term
    expected_end_date: str = ""
    # MOT & Tax
    mot_due: Optional[str] = None
    mot_days_until: Optional[int] = None
    road_tax_due: Optional[str] = None
    road_tax_amount: Optional[float] = None
    # HMRC
    cost_method: str = "mileage"
    hmrc_note: str = ""
    aia_available: float = 0.0     # Only if actual costs method


def get_van_finance(db: Session) -> Optional[VanFinance]:
    """Get the active van finance record."""
    return db.query(VanFinance).filter(VanFinance.is_active == True).first()


def calculate_van_summary(db: Session) -> Optional[VanFinanceSummary]:
    """Calculate full van finance summary with HMRC guidance."""
    van = get_van_finance(db)
    if not van:
        return None

    s = VanFinanceSummary()
    s.vehicle = van.vehicle_description
    s.purchase_price = van.purchase_price
    s.deposit_paid = van.deposit_paid
    s.finance_amount = van.finance_amount
    s.apr = van.apr
    s.monthly_payment = van.monthly_payment
    s.total_payments = van.total_payments
    s.payments_made = van.payments_made
    s.payments_remaining = max(0, van.total_payments - van.payments_made)
    s.cost_method = van.cost_method

    # Total cost of ownership
    s.total_cost = van.deposit_paid + (van.monthly_payment * van.total_payments)
    s.total_interest = s.total_cost - van.purchase_price

    # Proper amortisation calculation (HP interest is front-loaded)
    from app.finance.services.hp_amortisation_service import build_amortisation_schedule
    schedule = build_amortisation_schedule(
        finance_amount=van.finance_amount,
        apr=van.apr,
        monthly_payment=van.monthly_payment,
        total_payments=van.total_payments,
        first_payment_date=van.first_payment_date,
    )

    s.total_interest = schedule.total_interest_paid
    s.total_cost = van.deposit_paid + (van.monthly_payment * van.total_payments)

    # Sum interest/capital for payments made so far
    for i, pmt in enumerate(schedule.payments):
        if i < van.payments_made:
            s.interest_paid += pmt.interest_paid
            s.capital_paid += pmt.capital_portion

    s.capital_paid += van.deposit_paid  # deposit is capital
    s.interest_remaining = s.total_interest - s.interest_paid
    s.capital_remaining = van.purchase_price - s.capital_paid
    s.balance_outstanding = schedule.payments[van.payments_made - 1].closing_balance if van.payments_made > 0 else van.finance_amount
    
    # Current month's split (next payment)
    next_pmt_idx = min(van.payments_made, len(schedule.payments) - 1)
    s.monthly_interest = schedule.payments[next_pmt_idx].interest_paid
    s.monthly_capital = schedule.payments[next_pmt_idx].capital_portion
    s.final_balloon = schedule.final_balance

    # Expected end date
    if van.first_payment_date:
        end = van.first_payment_date + relativedelta(months=van.total_payments)
        s.expected_end_date = str(end)

    # MOT & Road Tax
    today = date.today()
    if van.mot_due_date:
        s.mot_due = str(van.mot_due_date)
        s.mot_days_until = (van.mot_due_date - today).days
    if van.road_tax_due_date:
        s.road_tax_due = str(van.road_tax_due_date)
    s.road_tax_amount = van.road_tax_amount

    # HMRC guidance
    if van.cost_method == "mileage":
        s.hmrc_note = (
            "Using SIMPLIFIED MILEAGE method (45p first 10k miles, "
            "25p thereafter). This flat rate covers ALL vehicle costs: "
            "fuel, insurance, repairs, MOT, road tax, wear & tear, "
            "AND the cost of the van itself (depreciation/purchase). "
            "You CANNOT also claim capital allowance (AIA) on the "
            "purchase price, or claim any running costs separately. "
            "Once you use mileage for this van, you cannot switch to "
            "actual costs for this vehicle."
        )
        s.aia_available = 0.0
    else:
        biz_pct = van.business_use_percentage / 100.0
        s.aia_available = van.purchase_price * biz_pct
        # HP interest is a running cost, claimable under actual costs
        s.hmrc_note = (
            f"Using ACTUAL COSTS method. You can claim: "
            f"(1) AIA: £{s.aia_available:,.2f} — full van purchase "
            f"price × {van.business_use_percentage:.0f}% business use, "
            f"deducted from profits in year of purchase. "
            f"(2) HP interest: the interest portion of HP payments "
            f"is a running cost (not the capital repayment). "
            f"(3) Running costs: fuel, insurance, road tax, MOT, "
            f"repairs, servicing — all claimable at "
            f"{van.business_use_percentage:.0f}% business use. "
            f"You CANNOT use the simplified mileage rate for this van."
        )

    return s


def create_van_finance(db: Session, data: dict) -> VanFinance:
    """Create or update van finance record."""
    # Deactivate any existing
    existing = db.query(VanFinance).filter(VanFinance.is_active == True).all()
    for v in existing:
        v.is_active = False

    # Calculate weekly equivalent for budget
    freq_to_weekly = _calc_weekly(data.get("monthly_payment", 0), "monthly")

    van = VanFinance(
        vehicle_description=data["vehicle_description"],
        purchase_price=data["purchase_price"],
        deposit_paid=data.get("deposit_paid", 0),
        finance_amount=data["finance_amount"],
        apr=data["apr"],
        monthly_payment=data["monthly_payment"],
        total_payments=data["total_payments"],
        payments_made=data.get("payments_made", 0),
        first_payment_date=datetime.strptime(data["first_payment_date"], "%Y-%m-%d").date(),
        finance_provider=data.get("finance_provider"),
        agreement_number=data.get("agreement_number"),
        business_use_percentage=data.get("business_use_percentage", 100),
        mot_due_date=datetime.strptime(data["mot_due_date"], "%Y-%m-%d").date() if data.get("mot_due_date") else None,
        road_tax_due_date=datetime.strptime(data["road_tax_due_date"], "%Y-%m-%d").date() if data.get("road_tax_due_date") else None,
        road_tax_amount=data.get("road_tax_amount"),
        cost_method=data.get("cost_method", "mileage"),
    )
    db.add(van)
    db.commit()
    db.refresh(van)
    return van


def update_van_finance(db: Session, van_id: int, data: dict) -> VanFinance:
    """Update van finance — for refinancing, payment updates, MOT dates etc."""
    van = db.query(VanFinance).get(van_id)
    if not van:
        raise ValueError("Van finance record not found")

    for key, value in data.items():
        if hasattr(van, key):
            if key in ("first_payment_date", "mot_due_date", "road_tax_due_date") and value:
                value = datetime.strptime(value, "%Y-%m-%d").date()
            setattr(van, key, value)

    van.updated_at = datetime.now()
    db.commit()
    db.refresh(van)
    return van


def _calc_weekly(amount: float, frequency: str) -> float:
    """Convert any frequency to weekly equivalent."""
    multipliers = {
        "weekly": 1.0,
        "fortnightly": 0.5,
        "monthly": 12.0 / 52.0,
        "quarterly": 4.0 / 52.0,
        "annual": 1.0 / 52.0,
    }
    return amount * multipliers.get(frequency, 12.0 / 52.0)


def auto_populate_van_from_transactions(db: Session) -> dict:
    """Auto-discover van finance details from existing NatWest transactions.
    
    Scans for Moneybarn payments to determine:
    - Monthly payment amount
    - First payment date
    - Number of payments made
    - Total paid so far
    
    Scans for DVLA payments to determine:
    - Monthly road tax amount
    - Vehicle registration
    """
    from app.finance.models import Transaction
    from sqlalchemy import func

    result = {
        "moneybarn": None,
        "dvla": None,
        "can_auto_create": False,
    }

    # ── Moneybarn ──
    mb_txs = db.query(Transaction).filter(
        Transaction.description.ilike("%moneybarn%")
    ).order_by(Transaction.transaction_date.asc()).all()

    if mb_txs:
        first = mb_txs[0]
        latest = mb_txs[-1]
        amounts = [t.amount for t in mb_txs]
        avg_amount = sum(amounts) / len(amounts)

        result["moneybarn"] = {
            "monthly_payment": round(avg_amount, 2),
            "first_payment_date": str(first.transaction_date),
            "latest_payment_date": str(latest.transaction_date),
            "payments_made": len(mb_txs),
            "total_paid": round(sum(amounts), 2),
            "description": first.description,
            "is_initial": "initial" in first.description.lower(),
        }

    # ── DVLA ──
    dvla_txs = db.query(Transaction).filter(
        Transaction.description.ilike("%dvla%")
    ).order_by(Transaction.transaction_date.asc()).all()

    if dvla_txs:
        # Find the van reg from description (DVLA-KS21VST format)
        import re
        reg = None
        for t in dvla_txs:
            m = re.search(r"DVLA-([A-Z0-9]+)", t.description, re.I)
            if m:
                reg = m.group(1)
                break

        # Get the regular monthly amount (exclude initial which may differ)
        regular = [t for t in dvla_txs if "initial" not in t.description.lower()]
        monthly_amount = regular[0].amount if regular else dvla_txs[0].amount

        result["dvla"] = {
            "monthly_amount": round(monthly_amount, 2),
            "annual_amount": round(monthly_amount * 12, 2),
            "registration": reg,
            "payments_found": len(dvla_txs),
            "first_payment": str(dvla_txs[0].transaction_date),
            "total_paid": round(sum(t.amount for t in dvla_txs), 2),
        }

    result["can_auto_create"] = bool(result["moneybarn"])
    return result


def auto_create_van_from_transactions(db: Session, extra_data: dict = None) -> "VanFinance":
    """Auto-create van finance record from transaction history + optional extras.
    
    extra_data can include details from the finance agreement PDF:
    - purchase_price, apr, total_payments, vehicle_description
    - mot_due_date
    """
    from app.finance.models import Transaction

    discovered = auto_populate_van_from_transactions(db)
    mb = discovered.get("moneybarn", {})
    dvla = discovered.get("dvla", {})

    if not mb:
        raise ValueError("No Moneybarn transactions found")

    # Merge discovered data with any extra info from PDF/user
    extra = extra_data or {}

    data = {
        "vehicle_description": extra.get("vehicle_description", f"Van ({dvla.get('registration', 'Unknown')})"),
        "purchase_price": extra.get("purchase_price", 0),  # Need from finance agreement
        "deposit_paid": extra.get("deposit_paid", 0),
        "finance_amount": extra.get("finance_amount", 0),
        "apr": extra.get("apr", 0),
        "monthly_payment": mb["monthly_payment"],
        "total_payments": extra.get("total_payments", 48),  # Default 4yr HP
        "payments_made": mb["payments_made"],
        "first_payment_date": mb["first_payment_date"],
        "finance_provider": "Moneybarn",
        "business_use_percentage": extra.get("business_use_percentage", 100),
        "mot_due_date": extra.get("mot_due_date"),
        "road_tax_due_date": extra.get("road_tax_due_date"),
        "road_tax_amount": dvla.get("annual_amount"),
        "cost_method": extra.get("cost_method", "mileage"),
    }

    return create_van_finance(db, data)




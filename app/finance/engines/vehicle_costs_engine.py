# FILE: app/finance/engines/vehicle_costs_engine.py
"""
Vehicle costs calculation for actual costs method.

Categorises NatWest transactions into HMRC-recognised van expense types,
calculates HP interest split from amortisation, and determines AIA eligibility.

Used by HMRCTaxEngine when cost_method == 'actual_costs'.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)

# Keywords to match transactions to vehicle expense categories
VEHICLE_EXPENSE_KEYWORDS = {
    "fuel": [
        "fuel", "petrol", "diesel", "bp ", "shell ", "esso ",
        "texaco", "sainsbury fuel", "tesco fuel", "asda fuel",
        "morrisons fuel", "jet ", "gulf ", "murco",
    ],
    "insurance": ["insurance", "insure", "direct line", "admiral"],
    "road_tax": ["dvla"],
    "mot": ["mot "],
    "repairs": [
        "repair", "mechanic", "garage", "kwik fit",
        "halfords", "tyre", "brake", "exhaust", "auto",
    ],
    "servicing": ["service", "oil change"],
    "hp_payment": ["moneybarn"],
}


@dataclass
class VehicleCostsBreakdown:
    """Categorised vehicle running costs for actual costs method."""
    fuel: float = 0.0
    insurance: float = 0.0
    road_tax: float = 0.0
    mot: float = 0.0
    repairs: float = 0.0
    servicing: float = 0.0
    hp_interest: float = 0.0
    hp_capital: float = 0.0
    aia: float = 0.0
    other_vehicle: float = 0.0

    @property
    def total_running_costs(self) -> float:
        """Total deductible running costs (excludes AIA and capital)."""
        return (
            self.fuel + self.insurance + self.road_tax +
            self.mot + self.repairs + self.servicing +
            self.hp_interest + self.other_vehicle
        )

    @property
    def total_deductible(self) -> float:
        """Total vehicle deduction: running costs + AIA."""
        return self.total_running_costs + self.aia

    def to_dict(self) -> dict:
        return {
            "fuel": round(self.fuel, 2),
            "insurance": round(self.insurance, 2),
            "road_tax": round(self.road_tax, 2),
            "mot": round(self.mot, 2),
            "repairs": round(self.repairs, 2),
            "servicing": round(self.servicing, 2),
            "hp_interest": round(self.hp_interest, 2),
            "hp_capital": round(self.hp_capital, 2),
            "aia": round(self.aia, 2),
            "other_vehicle": round(self.other_vehicle, 2),
            "total_running_costs": round(self.total_running_costs, 2),
            "total_deductible": round(self.total_deductible, 2),
        }


def calculate_vehicle_costs(
    db: Session,
    tax_year_start: date,
    tax_year_end: date,
) -> VehicleCostsBreakdown:
    """Calculate actual vehicle running costs from transaction data.

    Pulls all business/mixed transactions in the tax year,
    categorises by keyword matching, splits HP interest from capital.
    """
    from app.finance.models import Transaction, VanFinance

    breakdown = VehicleCostsBreakdown()

    # Get all transactions in date range
    txs = db.query(Transaction).filter(
        Transaction.transaction_date >= str(tax_year_start),
        Transaction.transaction_date <= str(tax_year_end),
        Transaction.is_deleted == False,
    ).all()

    # Categorise each transaction
    for tx in txs:
        desc = (tx.description or "").lower()
        amount = abs(tx.amount)

        for category, keywords in VEHICLE_EXPENSE_KEYWORDS.items():
            if any(kw in desc for kw in keywords):
                if category == "fuel":
                    breakdown.fuel += amount
                elif category == "insurance":
                    breakdown.insurance += amount
                elif category == "road_tax":
                    breakdown.road_tax += amount
                elif category == "mot":
                    breakdown.mot += amount
                elif category == "repairs":
                    breakdown.repairs += amount
                elif category == "servicing":
                    breakdown.servicing += amount
                elif category == "hp_payment":
                    # HP payments need splitting — handled below
                    pass
                break

    # HP interest calculation from amortisation schedule
    van = db.query(VanFinance).filter(VanFinance.is_active == True).first()
    if van and van.apr and van.finance_amount and van.monthly_payment:
        _calculate_hp_split(db, van, tax_year_start, tax_year_end, breakdown)

    # AIA — only in the tax year the van was purchased
    if van:
        _calculate_aia(van, tax_year_start, tax_year_end, breakdown)

    return breakdown


def _calculate_hp_split(
    db: Session,
    van,
    tax_year_start: date,
    tax_year_end: date,
    breakdown: VehicleCostsBreakdown,
) -> None:
    """Split HP payments into interest (deductible) and capital (not deductible).

    Walks the amortisation schedule to determine exactly how much
    interest was in each payment period.
    """
    from app.finance.models import Transaction

    monthly_rate = van.apr / 100 / 12

    # Count Moneybarn payments in this tax year
    mb_count = db.query(func.count(Transaction.id)).filter(
        Transaction.transaction_date >= str(tax_year_start),
        Transaction.transaction_date <= str(tax_year_end),
        Transaction.description.ilike("%moneybarn%"),
        Transaction.is_deleted == False,
    ).scalar() or 0

    if mb_count == 0:
        return

    # Calculate payments made before this tax year
    total_paid = van.payments_made or 0
    payments_before = max(0, total_paid - mb_count)

    # Walk amortisation to the start of this tax year
    balance = float(van.finance_amount)
    for _ in range(payments_before):
        interest = balance * monthly_rate
        capital = van.monthly_payment - interest
        balance = max(0, balance - capital)

    # Now calculate interest/capital for each payment in this year
    total_interest = 0.0
    total_capital = 0.0
    for _ in range(mb_count):
        interest = balance * monthly_rate
        capital = van.monthly_payment - interest
        total_interest += interest
        total_capital += capital
        balance = max(0, balance - capital)

    breakdown.hp_interest = round(total_interest, 2)
    breakdown.hp_capital = round(total_capital, 2)


def _calculate_aia(
    van,
    tax_year_start: date,
    tax_year_end: date,
    breakdown: VehicleCostsBreakdown,
) -> None:
    """Annual Investment Allowance — full purchase price in year of purchase only."""
    purchase_date = van.first_payment_date
    if not purchase_date:
        return

    try:
        pd = date.fromisoformat(str(purchase_date))
    except (ValueError, TypeError):
        return

    if tax_year_start <= pd <= tax_year_end:
        biz_pct = (van.business_use_percentage or 100) / 100
        breakdown.aia = round(float(van.purchase_price or 0) * biz_pct, 2)

# FILE: app/finance/services/hp_amortisation_service.py
"""
Hire Purchase amortisation calculations and transaction categorisation.

HMRC Rules for HP (Hire Purchase):
- The FULL purchase price of the van is claimable as Annual Investment Allowance (AIA)
  in the year of purchase, even though payments are ongoing.
- HP interest is a separate REVENUE expense (not capital), claimable as it's paid.
- The capital repayment portion of each monthly payment is NOT a separate expense
  (the van cost was already claimed via AIA).
- Business-use percentage must be applied to both AIA and interest claims.
- If using cash basis accounting, interest deduction is capped at £500/year.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import Transaction, VanFinance, ExpenseCategory

logger = logging.getLogger(__name__)

# HP transactions resolve their category by NAME (not hard-coded IDs).
# The seeded categories are `van_hp_interest` and `van_hp_capital`.
HP_INTEREST_CATEGORY_NAME = "van_hp_interest"
HP_CAPITAL_CATEGORY_NAME = "van_hp_capital"


def _hp_interest_category_id(db: Session) -> Optional[int]:
    """Look up the van_hp_interest category ID, or None if not seeded."""
    cat = db.query(ExpenseCategory).filter(
        ExpenseCategory.name == HP_INTEREST_CATEGORY_NAME
    ).first()
    return cat.id if cat else None


@dataclass
class PaymentSplit:
    """Single HP payment broken into interest vs capital."""
    month_number: int
    payment_date: Optional[date]
    total_payment: float
    interest_due: float          # actual interest accrued this month
    interest_paid: float         # portion of payment that is interest (capped at payment)
    capital_portion: float       # 0 if negative amortisation
    unpaid_interest: float       # interest that rolled into balance
    opening_balance: float
    closing_balance: float
    is_negative_amortisation: bool  # True if payment < interest due


@dataclass
class AmortisationSchedule:
    """Full HP amortisation schedule."""
    finance_amount: float
    apr: float
    monthly_payment: float
    total_payments: int
    is_negative_amortisation: bool = False
    payments: list[PaymentSplit] = field(default_factory=list)
    total_interest_accrued: float = 0.0
    total_interest_paid: float = 0.0
    total_capital_repaid: float = 0.0
    final_balance: float = 0.0


def build_amortisation_schedule(
    finance_amount: float,
    apr: float,
    monthly_payment: float,
    total_payments: int,
    first_payment_date: Optional[date] = None,
) -> AmortisationSchedule:
    """Build month-by-month amortisation showing interest vs capital split.
    
    Handles NEGATIVE AMORTISATION: when APR is so high that monthly
    interest exceeds the payment, the balance GROWS each month. In this
    case the entire payment is interest (claimable for HMRC), and unpaid
    interest compounds onto the balance.
    """
    monthly_rate = apr / 100.0 / 12.0
    balance = finance_amount
    first_interest = balance * monthly_rate
    neg_amort = first_interest > monthly_payment

    schedule = AmortisationSchedule(
        finance_amount=finance_amount,
        apr=apr,
        monthly_payment=monthly_payment,
        total_payments=total_payments,
        is_negative_amortisation=neg_amort,
    )

    for month in range(1, total_payments + 1):
        interest_due = round(balance * monthly_rate, 2)

        if interest_due >= monthly_payment:
            # NEGATIVE AMORTISATION: payment doesn't cover interest
            interest_paid = monthly_payment
            capital = 0.0
            unpaid = round(interest_due - monthly_payment, 2)
            new_balance = round(balance + unpaid, 2)
        else:
            # Normal: payment covers interest with capital left over
            interest_paid = interest_due
            capital = round(monthly_payment - interest_due, 2)
            unpaid = 0.0
            new_balance = round(balance - capital, 2)

        pmt_date = None
        if first_payment_date:
            from dateutil.relativedelta import relativedelta
            pmt_date = first_payment_date + relativedelta(months=month - 1)

        split = PaymentSplit(
            month_number=month,
            payment_date=pmt_date,
            total_payment=monthly_payment,
            interest_due=interest_due,
            interest_paid=interest_paid,
            capital_portion=capital,
            unpaid_interest=unpaid,
            opening_balance=round(balance, 2),
            closing_balance=new_balance,
            is_negative_amortisation=interest_due >= monthly_payment,
        )
        schedule.payments.append(split)
        schedule.total_interest_accrued += interest_due
        schedule.total_interest_paid += interest_paid
        schedule.total_capital_repaid += capital
        balance = new_balance

    schedule.total_interest_accrued = round(schedule.total_interest_accrued, 2)
    schedule.total_interest_paid = round(schedule.total_interest_paid, 2)
    schedule.total_capital_repaid = round(schedule.total_capital_repaid, 2)
    schedule.final_balance = round(balance, 2)

    return schedule


def get_payment_split_for_month(
    finance_amount: float,
    apr: float,
    monthly_payment: float,
    month_number: int,
) -> PaymentSplit:
    """Get the interest/capital split for a specific payment month."""
    monthly_rate = apr / 100.0 / 12.0
    balance = finance_amount

    for m in range(1, month_number + 1):
        interest_due = round(balance * monthly_rate, 2)
        if interest_due >= monthly_payment:
            interest_paid = monthly_payment
            capital = 0.0
            unpaid = round(interest_due - monthly_payment, 2)
            new_bal = round(balance + unpaid, 2)
        else:
            interest_paid = interest_due
            capital = round(monthly_payment - interest_due, 2)
            unpaid = 0.0
            new_bal = round(balance - capital, 2)
        if m == month_number:
            return PaymentSplit(
                month_number=m,
                payment_date=None,
                total_payment=monthly_payment,
                interest_due=interest_due,
                interest_paid=interest_paid,
                capital_portion=capital,
                unpaid_interest=unpaid,
                opening_balance=round(balance, 2),
                closing_balance=new_bal,
                is_negative_amortisation=interest_due >= monthly_payment,
            )
        balance = new_bal

    raise ValueError(f"Month {month_number} exceeds schedule")


def get_tax_year_summary(
    schedule: AmortisationSchedule,
    tax_year_start: date,
    tax_year_end: date,
) -> dict:
    """Summarise interest and capital for a specific tax year.
    
    Returns dict with interest_total, capital_total, payment_count
    for payments falling within the tax year.
    """
    interest = 0.0
    capital = 0.0
    count = 0

    for pmt in schedule.payments:
        if pmt.payment_date and tax_year_start <= pmt.payment_date <= tax_year_end:
            interest += pmt.interest_paid
            capital += pmt.capital_portion
            count += 1

    return {
        "tax_year": f"{tax_year_start.year}/{tax_year_end.year}",
        "interest_total": round(interest, 2),
        "capital_total": round(capital, 2),
        "payment_count": count,
        "interest_claimable": True,
        "capital_claimable": False,
        "note": (
            "Interest is claimable as a revenue expense. "
            "Capital repayment is NOT — the van cost was claimed via AIA."
        ),
    }


def categorise_moneybarn_transactions(db: Session) -> dict:
    """Find all Moneybarn transactions and split into interest/capital categories.
    
    Creates notes on each transaction showing the exact split.
    Returns summary of what was categorised.
    """
    van = db.query(VanFinance).filter(VanFinance.is_active == True).first()
    if not van:
        return {"error": "No active van finance record found", "updated": 0}

    schedule = build_amortisation_schedule(
        finance_amount=van.finance_amount,
        apr=van.apr,
        monthly_payment=van.monthly_payment,
        total_payments=van.total_payments,
        first_payment_date=van.first_payment_date,
    )

    # Find all Moneybarn transactions ordered by date
    mb_txs = db.query(Transaction).filter(
        Transaction.description.ilike("%moneybarn%"),
        Transaction.is_deleted == False,
    ).order_by(Transaction.transaction_date.asc()).all()

    if not mb_txs:
        return {"error": "No Moneybarn transactions found", "updated": 0}

    updated = 0
    results = []
    biz_pct = van.business_use_percentage / 100.0

    # Resolve the category ID by name at runtime — IDs vary per DB.
    hp_interest_cat_id = _hp_interest_category_id(db)
    if hp_interest_cat_id is None:
        return {
            "error": (
                "van_hp_interest category not seeded. "
                "Run the seed routine before categorising HP payments."
            ),
            "updated": 0,
        }

    for i, tx in enumerate(mb_txs):
        month_num = i + 1
        if month_num > len(schedule.payments):
            logger.warning(f"More Moneybarn txs ({len(mb_txs)}) than scheduled payments ({len(schedule.payments)})")
            break

        split = schedule.payments[month_num - 1]

        tx.category_id = hp_interest_cat_id
        tx.expense_scope = "business"
        tx.is_tax_deductible = True
        tx.deductible_amount = round(split.interest_paid * biz_pct, 2)
        tx.merchant_name = "Moneybarn (Van HP)"
        tx.auto_categorised = True
        tx.categorisation_confidence = 0.95

        if split.is_negative_amortisation:
            tx.notes = (
                f"HP Payment #{month_num}: "
                f"Full £{split.interest_paid:.2f} is interest (claimable). "
                f"Actual interest due: £{split.interest_due:.2f}, "
                f"unpaid £{split.unpaid_interest:.2f} added to balance. "
                f"Outstanding: £{split.closing_balance:.2f}"
            )
        else:
            tx.notes = (
                f"HP Payment #{month_num}: "
                f"Interest £{split.interest_paid:.2f} (claimable) + "
                f"Capital £{split.capital_portion:.2f} (not claimable). "
                f"Balance: £{split.closing_balance:.2f}"
            )
        updated += 1
        results.append({
            "tx_id": tx.id,
            "date": str(tx.transaction_date),
            "payment_num": month_num,
            "interest": split.interest_paid,
            "capital": split.capital_portion,
            "deductible": tx.deductible_amount,
            "balance": split.closing_balance,
        })

    db.commit()

    total_interest = sum(r["interest"] for r in results)
    total_capital = sum(r["capital"] for r in results)

    return {
        "updated": updated,
        "total_interest_paid": round(total_interest, 2),
        "total_capital_repaid": round(total_capital, 2),
        "total_deductible": round(total_interest * biz_pct, 2),
        "business_use_pct": van.business_use_percentage,
        "payments": results,
    }


def auto_categorise_van_expenses(db: Session) -> dict:
    """Apply merchant patterns to categorise all van-related transactions.
    
    Handles: Moneybarn (HP), RAC, Acorn Insurance, Halfords,
    garage/fuel, tyres, etc.
    """
    from app.finance.models import MerchantPattern

    patterns = db.query(MerchantPattern).filter(
        MerchantPattern.is_active == True,
    ).all()

    if not patterns:
        return {"updated": 0, "message": "No merchant patterns defined"}

    # Get uncategorised transactions
    uncategorised = db.query(Transaction).filter(
        Transaction.category_id.is_(None),
        Transaction.is_deleted == False,
    ).all()

    updated = 0
    matches = []

    for tx in uncategorised:
        desc_lower = tx.description.lower()
        for pat in patterns:
            if pat.merchant_pattern.lower() in desc_lower:
                # Skip Moneybarn — handled by categorise_moneybarn_transactions
                if pat.merchant_pattern.lower() == "moneybarn":
                    continue

                tx.category_id = pat.category_id
                tx.merchant_name = pat.merchant_display_name
                tx.expense_scope = pat.default_scope
                tx.is_tax_deductible = pat.default_scope == "business"
                tx.auto_categorised = True
                tx.categorisation_confidence = pat.confidence_score or 0.8

                pat.match_count = (pat.match_count or 0) + 1
                pat.last_matched = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                updated += 1
                matches.append({
                    "tx_id": tx.id,
                    "description": tx.description[:60],
                    "pattern": pat.merchant_pattern,
                    "category": pat.merchant_display_name,
                })
                break

    db.commit()
    return {"updated": updated, "matches": matches}



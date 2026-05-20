# FILE: app/finance/engines/mileage_method_enforcer.py
"""
HMRC mileage vs actual-costs method enforcement.

Per HMRC simplified expenses rules (BIM75005), a sole trader using the
simplified mileage rate (45p/25p) for a vehicle CANNOT also claim:
  - fuel
  - van insurance
  - van maintenance / repairs / servicing
  - MOT
  - road tax (DVLA)
  - tyres / brakes / general running costs
  - depreciation / AIA on the vehicle
  - HP/PCP capital repayments (interest under actual-costs only)

What you CAN still claim alongside mileage:
  - parking, tolls, congestion charges
  - public transport (train, taxi for business)
  - phone bill
  - PPE / work clothing
  - accountant fees, public liability insurance
  - anything not a vehicle running cost

This module takes a list of transactions and returns a filtered view:
  - `adjusted_deductible_total`: sum of deductible_amount with vehicle
    categories zeroed out when on mileage method
  - `suppressed_transactions`: which rows were zeroed and why

It is pure: no DB writes, no HTTP calls, no side effects. The caller
(finance_service.calculate_tax_estimate) applies the adjustment at
tax-calculation time \u2014 the underlying Transaction rows are untouched
so the user can still see what they spent on fuel, etc.

Single responsibility. Under 5 KB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# Category NAMES that mileage rate already covers.
# Matches the category `name` values seeded in app.finance.seed.
MILEAGE_COVERED_CATEGORIES: frozenset[str] = frozenset({
    "fuel",
    "vehicle_maintenance",
    "vehicle_insurance",
    "mot_tax",
    "van_hp_capital",   # HP capital repayment never deductible anyway
})

# Category names claimable under BOTH methods (informational only).
MILEAGE_COMPATIBLE_CATEGORIES: frozenset[str] = frozenset({
    "parking",
    "phone_bill",
    "phone_accessories",
    "software_subscriptions",
    "api_costs",
    "hardware",
    "work_clothing",
    "accountant",
    "insurance_pl",
    "food_on_road",
    "drinks_on_road",
    "other_business",
    "van_hp_interest",  # finance interest claimable alongside mileage
})


@dataclass
class SuppressedTransaction:
    """A transaction whose deduction was suppressed by method enforcement."""
    transaction_id: int
    description: str
    amount: float
    original_deductible: float
    category_name: str
    reason: str


@dataclass
class EnforcementResult:
    """Output of enforce_mileage_method."""
    adjusted_deductible_total: float = 0.0
    suppressed_total: float = 0.0
    suppressed_transactions: list[SuppressedTransaction] = field(default_factory=list)
    unchanged_count: int = 0
    suppressed_count: int = 0


def enforce_mileage_method(
    transactions: Iterable,
    cost_method: str = "mileage",
) -> EnforcementResult:
    """Walk transactions; zero-out deductible_amount for mileage-covered categories.

    Accepts any iterable of objects with these attributes:
      id, description, amount, deductible_amount, is_tax_deductible,
      category (optional, with `.name`)

    For `cost_method != "mileage"`, returns the untouched totals.
    """
    result = EnforcementResult()

    for tx in transactions:
        orig_deductible = float(getattr(tx, "deductible_amount", 0.0) or 0.0)
        is_ded = bool(getattr(tx, "is_tax_deductible", False))

        if not is_ded or orig_deductible <= 0:
            continue

        cat = getattr(tx, "category", None)
        cat_name = getattr(cat, "name", None) if cat else None

        if cost_method == "mileage" and cat_name in MILEAGE_COVERED_CATEGORIES:
            result.suppressed_total += orig_deductible
            result.suppressed_count += 1
            result.suppressed_transactions.append(SuppressedTransaction(
                transaction_id=int(getattr(tx, "id", 0) or 0),
                description=str(getattr(tx, "description", ""))[:200],
                amount=float(getattr(tx, "amount", 0.0) or 0.0),
                original_deductible=orig_deductible,
                category_name=cat_name,
                reason=(
                    f"Covered by 45p/25p mileage rate; cannot claim '{cat_name}' "
                    f"separately under simplified expenses."
                ),
            ))
        else:
            result.adjusted_deductible_total += orig_deductible
            result.unchanged_count += 1

    # Round to pence for presentation
    result.adjusted_deductible_total = round(result.adjusted_deductible_total, 2)
    result.suppressed_total = round(result.suppressed_total, 2)
    return result


def is_vehicle_category(category_name: Optional[str]) -> bool:
    """Helper: is this category one that mileage covers?"""
    return category_name in MILEAGE_COVERED_CATEGORIES

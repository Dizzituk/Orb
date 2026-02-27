# FILE: app/finance/services/merchant_teach_service.py
"""
Merchant teaching service — learns merchant aliases from user context.

When a user says "this is a garage" about SUMUP *GOING PLACES,
the system:
1. Resolves the correct category (e.g. "vehicle repairs")
2. Creates a merchant pattern for auto-categorisation
3. Applies the pattern to ALL matching transactions (NatWest + credit cards)
4. Returns how many transactions were updated
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import (
    Transaction,
    CreditCardTransaction,
    ExpenseCategory,
    MerchantPattern,
)
from app.finance.services.categoriser_service import confirm_categorisation

logger = logging.getLogger(__name__)


@dataclass
class TeachResult:
    """Result of teaching the system about a merchant."""
    merchant_pattern: str
    display_name: str
    category_name: str
    expense_scope: str
    natwest_updated: int = 0
    credit_card_updated: int = 0
    pattern_created: bool = False
    pattern_boosted: bool = False


def teach_merchant(
    db: Session,
    merchant_raw: str,
    category_name: str,
    expense_scope: str,
    display_name: Optional[str] = None,
) -> TeachResult:
    """Create a merchant pattern and apply it to all matching transactions.

    Args:
        merchant_raw: The raw merchant text from statement (e.g. "SUMUP *GOING PLACES CAMBORNE")
        category_name: Category to assign (e.g. "vehicle repairs", "fuel")
        expense_scope: "business", "personal", or "mixed"
        display_name: Human-friendly name (e.g. "Going Places Garage")
    """
    # Find or create category
    category = db.query(ExpenseCategory).filter(
        ExpenseCategory.name == category_name
    ).first()

    if not category:
        # Try case-insensitive
        category = db.query(ExpenseCategory).filter(
            ExpenseCategory.name.ilike(category_name)
        ).first()

    if not category:
        logger.warning("[teach] Category '%s' not found, skipping pattern", category_name)
        return TeachResult(
            merchant_pattern=merchant_raw.lower(),
            display_name=display_name or merchant_raw,
            category_name=category_name,
            expense_scope=expense_scope,
        )

    # Check if pattern already exists
    existing = db.query(MerchantPattern).filter(
        MerchantPattern.merchant_pattern == merchant_raw.lower().strip()
    ).first()

    # Create/update merchant pattern
    pattern = confirm_categorisation(
        db=db,
        merchant_raw=merchant_raw,
        category_id=category.id,
        expense_scope=expense_scope,
        display_name=display_name or merchant_raw,
    )

    result = TeachResult(
        merchant_pattern=pattern.merchant_pattern,
        display_name=pattern.merchant_display_name,
        category_name=category.name,
        expense_scope=expense_scope,
        pattern_created=existing is None,
        pattern_boosted=existing is not None,
    )

    # Build match pattern — extract core merchant name for substring matching
    # e.g. "SUMUP *GOING PLACES CAMBORNE" → match anything containing "GOING PLACES"
    match_text = merchant_raw.lower().strip()

    # Apply to NatWest transactions
    natwest_txs = db.query(Transaction).filter(
        Transaction.description.ilike(f"%{match_text}%"),
        Transaction.is_deleted == False,
    ).all()

    for tx in natwest_txs:
        tx.category_id = category.id
        tx.expense_scope = expense_scope
        tx.is_tax_deductible = (expense_scope == "business")
        tx.user_confirmed = True
        result.natwest_updated += 1

    # Apply to credit card transactions
    cc_txs = db.query(CreditCardTransaction).filter(
        CreditCardTransaction.description.ilike(f"%{match_text}%"),
    ).all()

    for tx in cc_txs:
        tx.category_id = category.id
        tx.expense_scope = expense_scope
        tx.is_tax_deductible = (expense_scope == "business")
        tx.user_confirmed = True
        result.credit_card_updated += 1

    db.commit()

    logger.info(
        "[teach] Merchant '%s' → %s (%s). Updated %d NatWest + %d CC transactions",
        match_text, category.name, expense_scope,
        result.natwest_updated, result.credit_card_updated,
    )

    return result

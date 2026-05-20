# FILE: app/finance/services/card_payment_linker.py
"""
Credit card payment linker — connects NatWest bank transactions
to their corresponding credit card and derives the business/personal
split from the card's transaction history.

Logic:
- A credit card payment on the NatWest statement is NOT a new expense
- It's paying off expenses already categorised on the credit card
- The bank transaction should inherit the card's biz/per ratio
- If unlinked, flag it for the user to connect

Uses the `natwest_description` field on CreditCard to match.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.finance.models import (
    CreditCard,
    CreditCardTransaction,
    Transaction,
)

logger = logging.getLogger(__name__)


@dataclass
class CardSplit:
    """Business/personal ratio for a credit card."""
    card_id: int
    card_name: str
    business_total: float
    personal_total: float
    mixed_total: float
    total_spend: float
    business_pct: float
    personal_pct: float
    mixed_pct: float
    transaction_count: int


@dataclass
class LinkedPayment:
    """A NatWest transaction linked to a credit card."""
    transaction_id: int
    card_id: int
    card_name: str
    payment_amount: float
    business_amount: float
    personal_amount: float
    business_pct: float
    personal_pct: float


def get_card_split(db: Session, card_id: int) -> Optional[CardSplit]:
    """Calculate the business/personal split for a credit card.

    Only considers purchases (not credits/payments) to compute the ratio.
    """
    card = db.query(CreditCard).filter(CreditCard.id == card_id).first()
    if not card:
        return None

    rows = (
        db.query(
            CreditCardTransaction.expense_scope,
            func.count(CreditCardTransaction.id),
            func.coalesce(func.sum(CreditCardTransaction.amount), 0.0),
        )
        .filter(
            CreditCardTransaction.card_id == card_id,
            # Credits/payments are filtered during import, so all DB rows are purchases
        )
        .group_by(CreditCardTransaction.expense_scope)
        .all()
    )

    scope_map = {r[0]: (r[1], r[2]) for r in rows}
    biz_count, biz_total = scope_map.get("business", (0, 0.0))
    per_count, per_total = scope_map.get("personal", (0, 0.0))
    mix_count, mix_total = scope_map.get("mixed", (0, 0.0))
    total = biz_total + per_total + mix_total

    return CardSplit(
        card_id=card_id,
        card_name=card.name,
        business_total=biz_total,
        personal_total=per_total,
        mixed_total=mix_total,
        total_spend=total,
        business_pct=round(biz_total / total * 100, 1) if total > 0 else 0,
        personal_pct=round(per_total / total * 100, 1) if total > 0 else 0,
        mixed_pct=round(mix_total / total * 100, 1) if total > 0 else 0,
        transaction_count=biz_count + per_count + mix_count,
    )


def get_all_card_splits(db: Session) -> list[CardSplit]:
    """Get splits for all active cards."""
    cards = db.query(CreditCard).filter(CreditCard.is_active == True).all()
    splits = []
    for card in cards:
        split = get_card_split(db, card.id)
        if split:
            splits.append(split)
    return splits


def detect_card_payment(
    db: Session,
    description: str,
) -> Optional[int]:
    """Check if a NatWest transaction description matches a credit card.

    Returns the card_id if matched, None otherwise.
    """
    desc_lower = description.lower()

    cards = db.query(CreditCard).filter(
        CreditCard.natwest_description.isnot(None),
        CreditCard.is_active == True,
    ).all()

    for card in cards:
        pattern = card.natwest_description.lower()
        if pattern and pattern in desc_lower:
            return card.id

    # Also check card name as fallback
    for card in cards:
        name_lower = card.name.lower()
        if name_lower in desc_lower:
            return card.id

    return None


def link_card_payment(
    db: Session,
    transaction_id: int,
    card_id: int,
) -> Optional[LinkedPayment]:
    """Link a NatWest transaction to a credit card and apply the split.

    Sets the transaction's `deductible_amount` to the business portion of
    the payment (payment_amount × business_pct / 100). Previously only the
    `is_tax_deductible` flag was set — `deductible_amount` was left at 0,
    which meant card spend contributed nothing to the tax deduction even
    when every purchase on the card was categorised as business.

    The card's business/personal split is computed live from the
    card's own transaction history (CreditCardTransaction rows).
    """
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        return None

    split = get_card_split(db, card_id)
    if not split:
        return None

    biz_amount = round(tx.amount * split.business_pct / 100, 2)
    per_amount = round(tx.amount * split.personal_pct / 100, 2)

    # Update the transaction
    tx.linked_card_id = card_id
    tx.expense_scope = "mixed"
    tx.is_tax_deductible = split.business_pct > 0
    tx.deductible_amount = biz_amount
    tx.user_confirmed = True
    db.commit()

    logger.info(
        "[card_linker] tx#%d linked to card %d (%s): %.1f%% biz / %.1f%% per, deductible £%.2f",
        transaction_id, card_id, split.card_name,
        split.business_pct, split.personal_pct, biz_amount,
    )

    return LinkedPayment(
        transaction_id=transaction_id,
        card_id=card_id,
        card_name=split.card_name,
        payment_amount=tx.amount,
        business_amount=biz_amount,
        personal_amount=per_amount,
        business_pct=split.business_pct,
        personal_pct=split.personal_pct,
    )


def auto_link_card_payments(db: Session) -> list[LinkedPayment]:
    """Scan all NatWest transactions and auto-link any card payments.

    Only processes transactions that aren't already linked.
    Returns list of newly linked payments.
    """
    # Get all unlinked transactions
    unlinked = db.query(Transaction).filter(
        Transaction.linked_card_id.is_(None),
        Transaction.is_deleted == False,
    ).all()

    linked = []
    for tx in unlinked:
        card_id = detect_card_payment(db, tx.description)
        if card_id:
            result = link_card_payment(db, tx.id, card_id)
            if result:
                linked.append(result)

    if linked:
        logger.info(
            "[card_linker] Auto-linked %d card payments", len(linked)
        )

    return linked


def refresh_linked_deductibles(db: Session) -> dict:
    """Re-apply the current card split to every already-linked payment.

    Call this after re-categorising card transactions to update the
    business-portion `deductible_amount` on each NatWest bill-pay row.
    Without this, changes to card-transaction categories have no effect
    on the tax calculation until the next manual re-link.
    """
    already_linked = db.query(Transaction).filter(
        Transaction.linked_card_id.isnot(None),
        Transaction.is_deleted == False,
    ).all()

    refreshed = 0
    total_deductible = 0.0
    for tx in already_linked:
        split = get_card_split(db, tx.linked_card_id)
        if not split:
            continue
        tx.deductible_amount = round(tx.amount * split.business_pct / 100, 2)
        tx.is_tax_deductible = split.business_pct > 0
        refreshed += 1
        total_deductible += tx.deductible_amount

    db.commit()
    return {
        "refreshed": refreshed,
        "total_deductible": round(total_deductible, 2),
    }


def get_unlinked_card_payments(db: Session) -> list[dict]:
    """Find NatWest transactions that look like card payments but aren't linked.

    Useful for flagging transactions that need user input.
    """
    # Patterns that suggest credit card payments
    card_keywords = [
        "card", "credit", "finance", "payment to",
    ]

    unlinked = db.query(Transaction).filter(
        Transaction.linked_card_id.is_(None),
        Transaction.is_deleted == False,
    ).all()

    suspicious = []
    for tx in unlinked:
        desc_lower = tx.description.lower()
        if any(kw in desc_lower for kw in card_keywords):
            suspicious.append({
                "transaction_id": tx.id,
                "description": tx.description,
                "amount": tx.amount,
                "current_scope": tx.expense_scope,
            })

    return suspicious

# FILE: app/finance/services/credit_card_service.py
"""
Credit card statement import and management.
Handles CSV parsing for various credit card providers,
auto-categorisation, and linking to NatWest payments.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import (
    CreditCard, CreditCardTransaction, Transaction, ExpenseCategory,
)
from app.finance.services.categoriser_service import categorise_transaction

logger = logging.getLogger(__name__)


@dataclass
class CCImportSummary:
    card_name: str
    total_rows: int = 0
    imported: int = 0
    duplicates: int = 0
    errors: int = 0
    auto_categorised: int = 0
    needs_review: int = 0
    total_spend: float = 0.0
    business_total: float = 0.0
    personal_total: float = 0.0


def get_or_create_card(db: Session, name: str, **kwargs) -> CreditCard:
    """Get existing card or create a new one."""
    card = db.query(CreditCard).filter(
        CreditCard.name.ilike(f"%{name}%"),
        CreditCard.is_active == True,
    ).first()
    if not card:
        card = CreditCard(name=name, **kwargs)
        db.add(card)
        db.commit()
        db.refresh(card)
    return card


def import_credit_card_csv(
    db: Session,
    csv_text: str,
    card_name: str,
    natwest_desc: Optional[str] = None,
) -> CCImportSummary:
    """Import a credit card CSV statement.
    
    Attempts to auto-detect CSV format (most UK cards use similar layouts):
    Date, Description/Reference, Amount (debit), Amount (credit)
    OR: Date, Description, Amount
    """
    import csv
    from io import StringIO

    card = get_or_create_card(db, card_name, natwest_description=natwest_desc)
    summary = CCImportSummary(card_name=card.name)

    reader = csv.reader(StringIO(csv_text))
    headers = None

    for row_num, row in enumerate(reader):
        if not row or all(c.strip() == "" for c in row):
            continue

        # Auto-detect header row
        if headers is None:
            lower = [c.lower().strip() for c in row]
            if any(h in " ".join(lower) for h in ["date", "transaction"]):
                headers = lower
                continue
            # If no header found, assume: Date, Description, Amount
            headers = ["date", "description", "amount"]

        summary.total_rows += 1

        try:
            # Extract fields based on detected headers
            date_str = _get_field(row, headers, ["date", "transaction date", "post date"])
            desc = _get_field(row, headers, ["description", "reference", "details", "transaction description"])
            amount_str = _get_field(row, headers, ["amount", "debit", "money out"])

            if not date_str or not desc:
                summary.errors += 1
                continue

            # Parse date
            tx_date = _parse_date(date_str)
            if not tx_date:
                summary.errors += 1
                continue

            # Parse amount (make positive — credit card spends are debits)
            amount = abs(float(amount_str.replace(",", "").replace("£", "").strip()))
            if amount == 0:
                continue

            # Dedup
            dedup = hashlib.md5(
                f"{tx_date}{amount}{desc[:50]}".encode()
            ).hexdigest()

            existing = db.query(CreditCardTransaction).filter(
                CreditCardTransaction.dedup_hash == dedup,
                CreditCardTransaction.card_id == card.id,
            ).first()
            if existing:
                summary.duplicates += 1
                continue

            # Categorise
            cat_result = categorise_transaction(db, desc, amount)
            scope = cat_result.expense_scope if cat_result else "unknown"
            cat_id = None
            auto_cat = False
            if cat_result and cat_result.confidence >= 0.80:
                cat_obj = db.query(ExpenseCategory).filter(
                    ExpenseCategory.name == cat_result.category_name
                ).first()
                if cat_obj:
                    cat_id = cat_obj.id
                    auto_cat = True

            # Determine tax year
            tax_year = _get_tax_year(tx_date)

            tx = CreditCardTransaction(
                card_id=card.id,
                transaction_date=tx_date,
                description=desc.strip(),
                amount=amount,
                merchant_name=desc.strip()[:100],
                category_id=cat_id,
                expense_scope=scope,
                is_tax_deductible=(scope == "business"),
                user_confirmed=False,
                tax_year=tax_year,
                dedup_hash=dedup,
            )
            db.add(tx)
            summary.imported += 1

            if auto_cat:
                summary.auto_categorised += 1
            else:
                summary.needs_review += 1

            summary.total_spend += amount
            if scope == "business":
                summary.business_total += amount
            elif scope == "personal":
                summary.personal_total += amount

        except Exception as e:
            logger.warning("[cc_import] Row %d error: %s", row_num, e)
            summary.errors += 1

    db.commit()
    return summary


def get_card_transactions(
    db: Session, card_id: int,
    scope: Optional[str] = None,
    page: int = 1, per_page: int = 50,
) -> dict:
    """List transactions for a specific credit card."""
    q = db.query(CreditCardTransaction).filter(
        CreditCardTransaction.card_id == card_id
    )
    if scope and scope != "all":
        q = q.filter(CreditCardTransaction.expense_scope == scope)

    total = q.count()
    items = q.order_by(CreditCardTransaction.transaction_date.desc()) \
             .offset((page - 1) * per_page).limit(per_page).all()

    serialised = []
    for tx in items:
        serialised.append({
            "id": tx.id,
            "transaction_date": str(tx.transaction_date),
            "description": tx.description,
            "amount": tx.amount,
            "merchant_name": tx.merchant_name,
            "category_name": tx.category.name if tx.category else None,
            "expense_scope": tx.expense_scope,
            "is_tax_deductible": tx.is_tax_deductible,
            "user_confirmed": tx.user_confirmed,
        })

    return {"items": serialised, "total": total, "page": page, "per_page": per_page}


def get_card_summary(db: Session, card_id: int) -> dict:
    """Get spending summary for a credit card."""
    from sqlalchemy import func
    txs = db.query(CreditCardTransaction).filter(
        CreditCardTransaction.card_id == card_id
    )
    total = txs.with_entities(func.sum(CreditCardTransaction.amount)).scalar() or 0
    biz = txs.filter(CreditCardTransaction.expense_scope == "business") \
             .with_entities(func.sum(CreditCardTransaction.amount)).scalar() or 0
    per = txs.filter(CreditCardTransaction.expense_scope == "personal") \
             .with_entities(func.sum(CreditCardTransaction.amount)).scalar() or 0
    uncat = txs.filter(CreditCardTransaction.expense_scope == "unknown") \
              .with_entities(func.count(CreditCardTransaction.id)).scalar() or 0

    return {
        "total_spend": round(total, 2),
        "business_total": round(biz, 2),
        "personal_total": round(per, 2),
        "uncategorised_count": uncat,
    }


def link_natwest_payment_to_card(db: Session, natwest_tx_id: int, card_id: int) -> dict:
    """Link a NatWest credit card payment to the actual card for drill-down."""
    tx = db.query(Transaction).get(natwest_tx_id)
    card = db.query(CreditCard).get(card_id)
    if not tx or not card:
        return {"linked": False, "error": "Transaction or card not found"}

    tx.expense_scope = "_credit_card_payment"
    tx.is_tax_deductible = False
    tx.deductible_amount = 0.0
    tx.notes = f"Payment to {card.name} — see card transactions for breakdown"
    tx.user_confirmed = True
    db.commit()
    return {"linked": True, "card_name": card.name, "message": f"Linked to {card.name}"}


# ── Helpers ──────────────────────────────────────────────

def _get_field(row: list, headers: list, candidates: list) -> str:
    """Find a field value by trying multiple header names."""
    for name in candidates:
        for i, h in enumerate(headers):
            if name in h and i < len(row):
                return row[i].strip()
    # Fallback by position
    if len(row) >= 3:
        for name in candidates:
            if "date" in name:
                return row[0].strip()
            if "desc" in name or "ref" in name:
                return row[1].strip()
            if "amount" in name or "debit" in name:
                return row[2].strip()
    return ""


def _parse_date(s: str) -> Optional[date]:
    """Try multiple date formats."""
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _get_tax_year(d: date) -> str:
    """Determine UK tax year from a date."""
    if d.month >= 4 and d.day >= 6 or d.month > 4:
        return f"{d.year}-{str(d.year + 1)[2:]}"
    return f"{d.year - 1}-{str(d.year)[2:]}"


def import_parsed_transactions(
    db: Session, card_id: int,
    transactions: list,
) -> CCImportSummary:
    """Import pre-parsed transactions (from PDF parser) into the DB."""

    card = db.query(CreditCard).get(card_id)
    summary = CCImportSummary(card_name=card.name if card else "Unknown")

    for tx in transactions:
        summary.total_rows += 1
        try:
            # Skip credits/payments (these are payments TO the card)
            if tx.is_credit:
                continue

            # Dedup
            dedup = hashlib.md5(
                f"{tx.transaction_date}{tx.amount}{tx.description[:50]}".encode()
            ).hexdigest()

            existing = db.query(CreditCardTransaction).filter(
                CreditCardTransaction.dedup_hash == dedup,
                CreditCardTransaction.card_id == card_id,
            ).first()
            if existing:
                summary.duplicates += 1
                continue

            # Categorise
            cat_result = categorise_transaction(db, tx.description, tx.amount)
            scope = cat_result.expense_scope if cat_result else "unknown"
            cat_id = None
            auto_cat = False
            if cat_result and cat_result.confidence >= 0.80:
                cat_obj = db.query(ExpenseCategory).filter(
                    ExpenseCategory.name == cat_result.category_name
                ).first()
                if cat_obj:
                    cat_id = cat_obj.id
                    auto_cat = True

            tax_year = _get_tax_year(tx.transaction_date)

            db_tx = CreditCardTransaction(
                card_id=card_id,
                transaction_date=tx.transaction_date,
                description=tx.description,
                amount=abs(tx.amount),
                merchant_name=tx.description[:100],
                category_id=cat_id,
                expense_scope=scope,
                is_tax_deductible=(scope == "business"),
                user_confirmed=False,
                tax_year=tax_year,
                dedup_hash=dedup,
            )
            db.add(db_tx)
            summary.imported += 1

            if auto_cat:
                summary.auto_categorised += 1
            else:
                summary.needs_review += 1

            summary.total_spend += abs(tx.amount)
            if scope == "business":
                summary.business_total += abs(tx.amount)
            elif scope == "personal":
                summary.personal_total += abs(tx.amount)

        except Exception as e:
            logger.warning("[cc_pdf_import] Error: %s", e)
            summary.errors += 1

    db.commit()
    return summary

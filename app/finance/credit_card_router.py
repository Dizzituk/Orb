# FILE: app/finance/credit_card_router.py
"""
Credit card management and statement import endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.finance.models import CreditCardStatement

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/finance",
    tags=["Finance - Credit Cards"],
    dependencies=[Depends(require_auth)],
)

@router.get("/credit-cards/{card_id}/statements")
async def list_card_statements(card_id: int, db: Session = Depends(get_db)):
    """List statement history for a credit card (running balance timeline)."""
    stmts = db.query(CreditCardStatement).filter(
        CreditCardStatement.card_id == card_id
    ).order_by(CreditCardStatement.statement_date.desc()).all()
    return [
        {
            "id": s.id, "statement_date": str(s.statement_date),
            "opening_balance": s.opening_balance, "closing_balance": s.closing_balance,
            "total_charges": s.total_charges, "total_payments": s.total_payments,
            "interest_charged": s.interest_charged, "minimum_payment": s.minimum_payment,
            "transactions_imported": s.transactions_imported,
            "source_filename": s.source_filename,
        }
        for s in stmts
    ]



# ─── Credit Cards ────────────────────────────────────────

@router.get("/credit-cards")
async def list_credit_cards(db: Session = Depends(get_db)):
    """List registered credit cards."""
    from app.finance.models import CreditCard
    return db.query(CreditCard).filter(CreditCard.is_active == True).all()


@router.post("/credit-cards")
async def create_credit_card(data: dict, db: Session = Depends(get_db)):
    """Register a new credit card."""
    from app.finance.services.credit_card_service import get_or_create_card
    card = get_or_create_card(
        db, data["name"],
        provider=data.get("provider"),
        last_four=data.get("last_four"),
        natwest_description=data.get("natwest_description"),
    )
    return card


@router.post("/credit-cards/{card_id}/import-csv")
async def import_card_csv(
    card_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a credit card CSV statement."""
    from app.finance.services.credit_card_service import import_credit_card_csv
    from app.finance.models import CreditCard

    card = db.query(CreditCard).get(card_id)
    if not card:
        raise HTTPException(404, "Card not found")

    content = await file.read()
    csv_text = content.decode("utf-8-sig")

    result = import_credit_card_csv(db, csv_text, card.name)
    return result.__dict__




@router.post("/credit-cards/{card_id}/import-pdf")
async def import_card_pdf(
    card_id: int,
    file: UploadFile = File(...),
    use_ai: bool = False,
    db: Session = Depends(get_db),
):
    """Import a credit card statement from a PDF file.
    
    Tries table/text extraction first, falls back to AI vision if use_ai=True.
    """
    from app.finance.services.pdf_statement_parser import parse_statement_pdf
    from app.finance.services.credit_card_service import (
        import_parsed_transactions, get_or_create_card,
    )
    from app.finance.models import CreditCard
    import tempfile, os

    card = db.query(CreditCard).get(card_id)
    if not card:
        raise HTTPException(404, "Card not found")

    # Save uploaded file temporarily
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Try pdfplumber first
        result = parse_statement_pdf(tmp_path)

        # If no transactions found and AI requested, try vision
        if not result.transactions and use_ai:
            from app.finance.services.pdf_ai_parser import parse_pdf_with_vision
            ai_result = parse_pdf_with_vision(tmp_path)
            if ai_result.transactions:
                result.strategy_used = "ai_vision"
                result.warnings.extend(ai_result.warnings)
                # Convert AI format to ParsedTransaction
                from app.finance.services.pdf_statement_parser import ParsedTransaction
                for tx in ai_result.transactions:
                    try:
                        from datetime import datetime
                        d = datetime.strptime(tx["date"], "%d/%m/%Y").date()
                        result.transactions.append(ParsedTransaction(
                            transaction_date=d,
                            description=tx["description"],
                            amount=float(tx["amount"]),
                            is_credit=tx.get("is_credit", False),
                        ))
                    except (ValueError, KeyError):
                        continue

        # Import parsed transactions into DB
        summary = import_parsed_transactions(db, card.id, result.transactions)

        return {
            "card_name": card.name,
            "strategy": result.strategy_used,
            "total_pages": result.total_pages,
            "statement_date": str(result.statement_date) if result.statement_date else None,
            "opening_balance": result.opening_balance,
            "closing_balance": result.closing_balance,
            "minimum_payment": result.minimum_payment,
            "imported": summary.imported,
            "duplicates": summary.duplicates,
            "auto_categorised": summary.auto_categorised,
            "needs_review": summary.needs_review,
            "total_spend": summary.total_spend,
            "warnings": result.warnings,
        }
    finally:
        os.unlink(tmp_path)

# ─── Van Finance ─────────────────────────────────────────

@router.get("/van")
async def get_van_summary(db: Session = Depends(get_db)):
    """Get van finance summary with HMRC guidance."""
    from app.finance.services.van_finance_service import calculate_van_summary
    summary = get_van_summary_data(db)
    if not summary:
        return {"exists": False}
    return summary.__dict__ | {"exists": True}


def get_van_summary_data(db):
    from app.finance.services.van_finance_service import calculate_van_summary
    return calculate_van_summary(db)


@router.post("/van")
async def create_van(data: dict, db: Session = Depends(get_db)):
    """Create van finance record."""
    from app.finance.services.van_finance_service import create_van_finance
    van = create_van_finance(db, data)
    return {"created": True, "id": van.id}


@router.put("/van/{van_id}")
async def update_van(van_id: int, data: dict, db: Session = Depends(get_db)):
    """Update van finance — refinancing, payment count, MOT dates etc."""
    from app.finance.services.van_finance_service import update_van_finance
    van = update_van_finance(db, van_id, data)
    return {"updated": True, "id": van.id}


# ─── Budget Items ────────────────────────────────────────

@router.get("/budget/items")
async def list_budget_items(db: Session = Depends(get_db)):
    """List all active budget items."""
    from app.finance.services.budget_service import get_budget_items
    return get_budget_items(db)


@router.get("/budget/summary")
async def budget_summary(db: Session = Depends(get_db)):
    """Get full budget breakdown with disposable income."""
    from app.finance.services.budget_service import get_budget_summary
    s = get_budget_summary(db)
    return {
        "items": s.items,
        "total_weekly": round(s.total_weekly, 2),
        "total_monthly": round(s.total_monthly, 2),
        "total_annual": round(s.total_annual, 2),
        "by_category": s.by_category,
        "avg_weekly_income": round(s.avg_weekly_income, 2),
        "tax_reserve_weekly": round(s.tax_reserve_weekly, 2),
        "disposable_weekly": round(s.disposable_weekly, 2),
        "disposable_daily": round(s.disposable_daily, 2),
    }


@router.post("/budget/items")
async def create_budget_item(data: dict, db: Session = Depends(get_db)):
    """Create a new budget item."""
    from app.finance.services.budget_service import create_budget_item as _create
    item = _create(db, data)
    return {"created": True, "id": item.id}


@router.put("/budget/items/{item_id}")
async def update_budget_item_endpoint(item_id: int, data: dict, db: Session = Depends(get_db)):
    """Update a budget item."""
    from app.finance.services.budget_service import update_budget_item
    item = update_budget_item(db, item_id, data)
    return {"updated": True, "id": item.id}


@router.delete("/budget/items/{item_id}")
async def delete_budget_item_endpoint(item_id: int, db: Session = Depends(get_db)):
    """Soft-delete a budget item."""
    from app.finance.services.budget_service import delete_budget_item
    deleted = delete_budget_item(db, item_id)
    return {"deleted": deleted}

@router.get("/credit-cards/{card_id}/transactions")
async def list_card_transactions(
    card_id: int,
    scope: Optional[str] = None,
    page: int = 1,
    per_page: int = 50,
    db: Session = Depends(get_db),
):
    """List transactions for a credit card."""
    from app.finance.services.credit_card_service import get_card_transactions
    return get_card_transactions(db, card_id, scope, page, per_page)


# ─── Van Finance ─────────────────────────────────────────

@router.get("/van")
async def get_van_summary(db: Session = Depends(get_db)):
    """Get van finance summary with HMRC guidance."""
    from app.finance.services.van_finance_service import calculate_van_summary
    summary = get_van_summary_data(db)
    if not summary:
        return {"exists": False}
    return summary.__dict__ | {"exists": True}


def get_van_summary_data(db):
    from app.finance.services.van_finance_service import calculate_van_summary
    return calculate_van_summary(db)


@router.post("/van")
async def create_van(data: dict, db: Session = Depends(get_db)):
    """Create van finance record."""
    from app.finance.services.van_finance_service import create_van_finance
    van = create_van_finance(db, data)
    return {"created": True, "id": van.id}


@router.put("/van/{van_id}")
async def update_van(van_id: int, data: dict, db: Session = Depends(get_db)):
    """Update van finance — refinancing, payment count, MOT dates etc."""
    from app.finance.services.van_finance_service import update_van_finance
    van = update_van_finance(db, van_id, data)
    return {"updated": True, "id": van.id}


# ─── Budget Items ────────────────────────────────────────

@router.get("/budget/items")
async def list_budget_items(db: Session = Depends(get_db)):
    """List all active budget items."""
    from app.finance.services.budget_service import get_budget_items
    return get_budget_items(db)


@router.get("/budget/summary")
async def budget_summary(db: Session = Depends(get_db)):
    """Get full budget breakdown with disposable income."""
    from app.finance.services.budget_service import get_budget_summary
    s = get_budget_summary(db)
    return {
        "items": s.items,
        "total_weekly": round(s.total_weekly, 2),
        "total_monthly": round(s.total_monthly, 2),
        "total_annual": round(s.total_annual, 2),
        "by_category": s.by_category,
        "avg_weekly_income": round(s.avg_weekly_income, 2),
        "tax_reserve_weekly": round(s.tax_reserve_weekly, 2),
        "disposable_weekly": round(s.disposable_weekly, 2),
        "disposable_daily": round(s.disposable_daily, 2),
    }


@router.post("/budget/items")
async def create_budget_item(data: dict, db: Session = Depends(get_db)):
    """Create a new budget item."""
    from app.finance.services.budget_service import create_budget_item as _create
    item = _create(db, data)
    return {"created": True, "id": item.id}


@router.put("/budget/items/{item_id}")
async def update_budget_item_endpoint(item_id: int, data: dict, db: Session = Depends(get_db)):
    """Update a budget item."""
    from app.finance.services.budget_service import update_budget_item
    item = update_budget_item(db, item_id, data)
    return {"updated": True, "id": item.id}


@router.delete("/budget/items/{item_id}")
async def delete_budget_item_endpoint(item_id: int, db: Session = Depends(get_db)):
    """Soft-delete a budget item."""
    from app.finance.services.budget_service import delete_budget_item
    deleted = delete_budget_item(db, item_id)
    return {"deleted": deleted}

@router.get("/credit-cards/{card_id}/summary")
async def card_summary(card_id: int, db: Session = Depends(get_db)):
    """Get spending summary for a credit card."""
    from app.finance.services.credit_card_service import get_card_summary
    return get_card_summary(db, card_id)


@router.post("/credit-cards/{card_id}/transactions/{tx_id}/scope")
async def toggle_card_tx_scope(card_id: int, tx_id: int, data: dict, db: Session = Depends(get_db)):
    """Toggle a credit card transaction between business/personal."""
    from app.finance.models import CreditCardTransaction
    tx = db.query(CreditCardTransaction).filter(
        CreditCardTransaction.id == tx_id,
        CreditCardTransaction.card_id == card_id,
    ).first()
    if not tx:
        raise HTTPException(404, "Transaction not found")
    new_scope = data.get("expense_scope", "personal")
    tx.expense_scope = new_scope
    tx.is_tax_deductible = (new_scope == "business")
    tx.user_confirmed = True
    db.commit()
    return {"updated": True, "new_scope": new_scope}


@router.post("/credit-cards/{card_id}/transactions/batch-scope")
async def batch_card_scope(card_id: int, data: dict, db: Session = Depends(get_db)):
    """Batch update scope for credit card transactions matching criteria."""
    from app.finance.models import CreditCardTransaction
    match_value = data.get("match_value", "")
    new_scope = data.get("expense_scope", "personal")

    matches = db.query(CreditCardTransaction).filter(
        CreditCardTransaction.card_id == card_id,
        CreditCardTransaction.description.ilike(f"%{match_value}%"),
    ).all()

    for tx in matches:
        tx.expense_scope = new_scope
        tx.is_tax_deductible = (new_scope == "business")
        tx.user_confirmed = True
    db.commit()

    return {"updated": len(matches), "match_value": match_value, "new_scope": new_scope}


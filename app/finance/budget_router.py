# FILE: app/finance/budget_router.py
"""
Van finance and personal budget endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/finance",
    tags=["Finance - Budget"],
    dependencies=[Depends(require_auth)],
)

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



@router.get("/van/auto-discover")
async def van_auto_discover(db: Session = Depends(get_db)):
    """Auto-discover van finance details from existing transactions."""
    from app.finance.services.van_finance_service import auto_populate_van_from_transactions
    return auto_populate_van_from_transactions(db)


@router.post("/van/auto-create")
async def van_auto_create(data: dict = None, db: Session = Depends(get_db)):
    """Auto-create van finance from transactions + optional extras from PDF."""
    from app.finance.services.van_finance_service import auto_create_van_from_transactions
    van = auto_create_van_from_transactions(db, data)
    return {"created": True, "id": van.id}


@router.post("/van/import-pdf")
async def van_import_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Import van finance details from a finance agreement PDF using AI vision OCR."""
    from app.finance.services.van_pdf_parser import parse_van_finance_pdf
    from app.finance.services.van_finance_service import (
        auto_populate_van_from_transactions,
        create_van_finance,
    )

    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file"}

    file_bytes = await file.read()
    logger.info("[van_import] Processing PDF: %s (%d bytes)", file.filename, len(file_bytes))

    # 1. Parse the PDF with AI vision
    try:
        pdf_data = await parse_van_finance_pdf(file_bytes, file.filename)
    except Exception as e:
        logger.error("[van_import] PDF parse failed: %s", e)
        return {"error": f"Failed to parse PDF: {str(e)}"}

    # 2. Also check transaction history for payment count
    discovered = auto_populate_van_from_transactions(db)
    mb = discovered.get("moneybarn", {})
    dvla = discovered.get("dvla", {})

    # 3. Merge PDF data with transaction data
    van_data = {
        "vehicle_description": pdf_data.get("vehicle_description") or f"Van ({pdf_data.get('registration', 'Unknown')})",
        "purchase_price": pdf_data.get("purchase_price") or 0,
        "deposit_paid": pdf_data.get("deposit_paid") or 0,
        "finance_amount": pdf_data.get("finance_amount") or 0,
        "apr": pdf_data.get("apr") or 0,
        "monthly_payment": mb.get("monthly_payment") or pdf_data.get("monthly_payment") or 0,
        "total_payments": pdf_data.get("total_payments") or 48,
        "payments_made": mb.get("payments_made", 0),
        "first_payment_date": mb.get("first_payment_date") or pdf_data.get("agreement_date"),
        "finance_provider": pdf_data.get("finance_provider") or "Moneybarn",
        "business_use_percentage": 100,
        "road_tax_amount": dvla.get("annual_amount"),
        "cost_method": "mileage",
    }

    # 4. Create the van finance record
    van = create_van_finance(db, van_data)

    return {
        "created": True,
        "id": van.id,
        "extracted": pdf_data,
        "transactions_found": {
            "moneybarn_payments": mb.get("payments_made", 0),
            "dvla_payments": dvla.get("payments_found", 0),
        },
    }


# ── Tax Method Advisor ──

@router.get("/van/tax-comparison")
async def get_tax_comparison(db: Session = Depends(get_db)):
    """Compare mileage vs actual costs for current tax year."""
    from app.finance.services.van_tax_advisor import compare_tax_methods
    from dataclasses import asdict
    comp = compare_tax_methods(db)
    return asdict(comp)


@router.get("/van/tax-quarterly")
async def get_tax_quarterly(db: Session = Depends(get_db)):
    """Quick quarterly snapshot - am I on the right method?"""
    from app.finance.services.van_tax_advisor import get_quarterly_snapshot
    return get_quarterly_snapshot(db)


@router.get("/van/tax-projection")
async def get_tax_projection(db: Session = Depends(get_db)):
    """Project next year's comparison (no AIA, extrapolated costs)."""
    from app.finance.services.van_tax_advisor import project_next_year
    return project_next_year(db)

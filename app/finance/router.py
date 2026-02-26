# FILE: app/finance/router.py
"""
FastAPI router for finance endpoints.
Follows the same pattern as app/investments/router.py.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.finance import schemas
from app.finance.services import finance_service
from app.finance.models import (
    Transaction, DailyWorkLog, MileageLog,
    RecurringCost, SavingsGoal, ExpenseCategory,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/finance",
    tags=["Finance"],
    dependencies=[Depends(require_auth)],
)


# ─── Dashboard ───────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    """Main dashboard data — earnings, tax, budget, recent activity."""
    try:
        return finance_service.get_dashboard_data(db)
    except Exception as e:
        logger.error("[finance] Dashboard error: %s", e)
        raise HTTPException(500, str(e))


# ─── Transactions ────────────────────────────────────────

@router.post("/transactions", response_model=schemas.TransactionOut)
async def create_transaction(data: schemas.TransactionCreate, db: Session = Depends(get_db)):
    """Record a new transaction."""
    return finance_service.create_transaction(db, data.model_dump())


@router.get("/transactions")
async def list_transactions(
    start_date: Optional[date] = None, end_date: Optional[date] = None,
    transaction_type: Optional[str] = None, expense_scope: Optional[str] = None,
    category_id: Optional[int] = None, page: int = 1, per_page: int = 50,
    db: Session = Depends(get_db),
):
    """List transactions with filters."""
    return finance_service.list_transactions(
        db, start_date, end_date, transaction_type, expense_scope, category_id, page, per_page,
    )


@router.put("/transactions/{tx_id}")
async def update_transaction(tx_id: int, data: schemas.TransactionUpdate, db: Session = Depends(get_db)):
    """Update a transaction."""
    tx = db.query(Transaction).get(tx_id)
    if not tx:
        raise HTTPException(404, "Transaction not found")
    for key, val in data.model_dump(exclude_unset=True).items():
        setattr(tx, key, val)
    db.commit()
    db.refresh(tx)
    return tx


@router.delete("/transactions/{tx_id}")
async def delete_transaction(tx_id: int, db: Session = Depends(get_db)):
    """Soft-delete a transaction."""
    tx = db.query(Transaction).get(tx_id)
    if not tx:
        raise HTTPException(404, "Transaction not found")
    tx.is_deleted = True
    db.commit()
    return {"deleted": True, "id": tx_id}


# ─── Transaction Scope Toggle ────────────────────────────

@router.post("/batch/transactions-scope")
async def batch_update_scope(data: dict, db: Session = Depends(get_db)):
    """Batch update scope for transactions matching criteria.
    Supports matching by merchant_name, description substring, or category.
    Also teaches the categoriser for future auto-matching."""
    match_field = data.get("match_field", "merchant_name")  # merchant_name | description | category_id
    match_value = data.get("match_value", "")
    new_scope = data.get("expense_scope", "personal")
    category_id = data.get("category_id")

    if new_scope not in ("business", "personal", "mixed"):
        raise HTTPException(400, "Scope must be business, personal, or mixed")
    if not match_value:
        raise HTTPException(400, "match_value required")

    # Find matching transactions
    q = db.query(Transaction).filter(Transaction.is_deleted == False)

    if match_field == "merchant_name":
        q = q.filter(Transaction.merchant_name.ilike(f"%{match_value}%"))
    elif match_field == "description":
        q = q.filter(Transaction.description.ilike(f"%{match_value}%"))
    elif match_field == "category_id":
        q = q.filter(Transaction.category_id == int(match_value))
    else:
        raise HTTPException(400, f"Invalid match_field: {match_field}")

    matches = q.all()
    updated = 0
    for tx in matches:
        tx.expense_scope = new_scope
        tx.is_tax_deductible = (new_scope == "business")
        tx.deductible_amount = tx.amount if new_scope == "business" else 0.0
        tx.user_confirmed = True
        if category_id is not None:
            tx.category_id = category_id
        updated += 1

    db.commit()

    # Teach the categoriser
    try:
        from app.finance.services.categoriser_service import confirm_categorisation as do_learn
        do_learn(
            db=db,
            merchant_raw=match_value,
            category_id=category_id or (matches[0].category_id if matches else None),
            expense_scope=new_scope,
        )
    except Exception as e:
        logger.warning("[finance] Categoriser learn failed: %s", e)

    return {
        "updated": updated,
        "match_field": match_field,
        "match_value": match_value,
        "new_scope": new_scope,
        "message": f"Updated {updated} transactions matching '{match_value}' to {new_scope}",
    }


@router.post("/transactions/{tx_id}/scope")
async def toggle_transaction_scope(tx_id: int, data: dict, db: Session = Depends(get_db)):
    """Toggle a transaction between business/personal scope."""
    tx = db.query(Transaction).get(tx_id)
    if not tx:
        raise HTTPException(404, "Transaction not found")
    new_scope = data.get("expense_scope", "personal")
    if new_scope not in ("business", "personal", "mixed"):
        raise HTTPException(400, "Scope must be business, personal, or mixed")
    tx.expense_scope = new_scope
    tx.is_tax_deductible = (new_scope == "business")
    tx.deductible_amount = tx.amount if new_scope == "business" else 0.0
    tx.user_confirmed = True
    db.commit()
    db.refresh(tx)
    return tx

# ─── Daily Work Log ──────────────────────────────────────

@router.post("/daily-log", response_model=schemas.DailyLogOut)
async def create_daily_log(data: schemas.DailyLogCreate, db: Session = Depends(get_db)):
    """Record end-of-day work summary (from Yodel finish tour)."""
    return finance_service.create_daily_log(db, data.model_dump())


@router.get("/daily-log/{work_date}")
async def get_daily_log(work_date: date, db: Session = Depends(get_db)):
    """Get a specific day's work log."""
    log = db.query(DailyWorkLog).filter(DailyWorkLog.work_date == work_date).first()
    if not log:
        raise HTTPException(404, "No log for this date")
    return log


@router.get("/daily-logs")
async def list_daily_logs(
    start_date: Optional[date] = None, end_date: Optional[date] = None,
    limit: int = 30, db: Session = Depends(get_db),
):
    """List daily work logs."""
    q = db.query(DailyWorkLog)
    if start_date:
        q = q.filter(DailyWorkLog.work_date >= start_date)
    if end_date:
        q = q.filter(DailyWorkLog.work_date <= end_date)
    return q.order_by(DailyWorkLog.work_date.desc()).limit(limit).all()


# ─── Mileage ────────────────────────────────────────────

@router.get("/mileage/summary")
async def get_mileage_summary(tax_year: Optional[str] = None, db: Session = Depends(get_db)):
    """Get annual mileage summary with deduction calculation."""
    from app.finance.models import MileageYearSummary
    from app.finance.engines.hmrc_tax_engine import HMRCTaxEngine
    ty = tax_year or finance_service.get_current_tax_year()
    summary = db.query(MileageYearSummary).filter(MileageYearSummary.tax_year == ty).first()
    engine = HMRCTaxEngine()

    if not summary:
        return {
            "tax_year": ty,
            "total_business_miles": 0,
            "miles_at_higher_rate": 0,
            "miles_at_lower_rate": 0,
            "total_claimable": 0.0,
            "claim_method": "simplified",
            "rate_first_10k": 0.45,
            "rate_after_10k": 0.25,
        }

    mileage = engine.calculate_mileage(summary.total_business_miles)
    return {
        "tax_year": ty,
        "total_business_miles": summary.total_business_miles,
        "miles_at_higher_rate": min(summary.total_business_miles, 10000),
        "miles_at_lower_rate": max(0, summary.total_business_miles - 10000),
        "total_claimable": mileage["mileage_deduction"],
        "claim_method": summary.claim_method,
        "rate_first_10k": 0.45,
        "rate_after_10k": 0.25,
    }


@router.post("/mileage/set-annual")
async def set_annual_mileage(data: dict, db: Session = Depends(get_db)):
    """Set total business miles for the tax year (bulk entry).
    Use this when you know your annual total but don't have daily logs."""
    from app.finance.models import MileageYearSummary
    from app.finance.engines.hmrc_tax_engine import HMRCTaxEngine
    ty = data.get("tax_year") or finance_service.get_current_tax_year()
    miles = float(data.get("business_miles", 0))
    engine = HMRCTaxEngine()
    mileage = engine.calculate_mileage(miles)

    summary = db.query(MileageYearSummary).filter(MileageYearSummary.tax_year == ty).first()
    if summary:
        summary.total_business_miles = miles
        summary.miles_at_higher_rate = min(miles, 10000)
        summary.miles_at_lower_rate = max(0, miles - 10000)
        summary.total_claimable = mileage["mileage_deduction"]
    else:
        summary = MileageYearSummary(
            tax_year=ty,
            total_business_miles=miles,
            miles_at_higher_rate=min(miles, 10000),
            miles_at_lower_rate=max(0, miles - 10000),
            total_claimable=mileage["mileage_deduction"],
            claim_method="simplified",
        )
        db.add(summary)
    db.commit()
    db.refresh(summary)
    return {
        "tax_year": ty,
        "total_business_miles": miles,
        "total_claimable": mileage["mileage_deduction"],
        "message": f"Set {miles:.0f} business miles — £{mileage['mileage_deduction']:.2f} deduction",
    }


@router.post("/mileage")
async def log_mileage(data: schemas.MileageCreate, db: Session = Depends(get_db)):
    """Log daily mileage reading."""
    total = data.end_mileage - data.start_mileage
    business = data.business_miles if data.business_miles is not None else total
    tax_year = finance_service.get_current_tax_year()

    log = MileageLog(
        log_date=data.log_date,
        start_mileage=data.start_mileage,
        end_mileage=data.end_mileage,
        total_miles=round(total, 1),
        business_miles=round(business, 1),
        route_description=data.route_description,
        hours_on_road=data.hours_on_road,
        tax_year=tax_year,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


# ─── Tax ─────────────────────────────────────────────────

@router.get("/tax/estimate")
async def get_tax_estimate(tax_year: Optional[str] = None, db: Session = Depends(get_db)):
    """Current tax liability estimate with full breakdown."""
    return finance_service.calculate_tax_estimate(db, tax_year)


# ─── AI Tax Advisor ───────────────────────────────────────

@router.post("/tax/ai-analyse")
async def ai_analyse_transaction(data: dict):
    """AI-powered HMRC deductibility analysis for a transaction."""
    from app.finance.services.tax_advisor_service import TaxAdvisorService
    advisor = TaxAdvisorService()
    result = await advisor.analyse_transaction(
        description=data.get("description", ""),
        amount=data.get("amount", 0.0),
        merchant=data.get("merchant_name"),
        user_context=data.get("user_context"),
        existing_method=data.get("expense_method", "mileage"),
    )
    return result.__dict__


@router.post("/tax/ai-check-conflicts")
async def ai_check_conflicts(data: dict, db: Session = Depends(get_db)):
    """Check for mileage vs actual costs method conflicts."""
    from app.finance.services.tax_advisor_service import TaxAdvisorService
    advisor = TaxAdvisorService()
    # Get all business expenses
    expenses = db.query(Transaction).filter(
        Transaction.expense_scope == "business",
        Transaction.is_deleted == False,
    ).all()
    expense_list = [
        {"description": e.description, "amount": e.amount, "category": e.category.name if e.category else ""}
        for e in expenses
    ]
    conflicts = await advisor.check_method_conflicts(
        expense_list, using_mileage=data.get("using_mileage", True)
    )
    return {"conflicts": conflicts, "total_checked": len(expense_list)}


@router.get("/tax/mtd-status")
async def get_mtd_status(db: Session = Depends(get_db)):
    """Making Tax Digital quarterly status."""
    from app.finance.engines.hmrc_tax_engine import HMRCTaxEngine
    engine = HMRCTaxEngine()
    quarter = engine.get_current_quarter()
    ty = db.query(finance_service.TaxYear).filter(
        finance_service.TaxYear.tax_year == finance_service.get_current_tax_year()
    ).first()
    return {
        "current_quarter": quarter,
        "q1_submitted": ty.q1_submitted if ty else False,
        "q2_submitted": ty.q2_submitted if ty else False,
        "q3_submitted": ty.q3_submitted if ty else False,
        "q4_submitted": ty.q4_submitted if ty else False,
    }


# ─── Categories ──────────────────────────────────────────

@router.get("/categories")
async def list_categories(db: Session = Depends(get_db)):
    """List all expense categories."""
    return db.query(ExpenseCategory).filter(ExpenseCategory.is_active == True).order_by(ExpenseCategory.sort_order).all()


# ─── Recurring Costs ────────────────────────────────────

@router.post("/recurring-costs")
async def create_recurring_cost(data: schemas.RecurringCostCreate, db: Session = Depends(get_db)):
    """Add a recurring cost."""
    freq_multipliers = {"weekly": 1, "fortnightly": 0.5, "monthly": 12/52, "quarterly": 4/52, "annually": 1/52}
    weekly = data.amount * freq_multipliers.get(data.frequency, 12/52)
    monthly = weekly * 52 / 12

    cost = RecurringCost(
        name=data.name, description=data.description, amount=data.amount,
        frequency=data.frequency, monthly_equivalent=round(monthly, 2),
        weekly_equivalent=round(weekly, 2), category_id=data.category_id,
        is_essential=data.is_essential, is_tax_deductible=data.is_tax_deductible,
        deductible_percentage=data.deductible_percentage, payment_day=data.payment_day,
        start_date=date.today(),
    )
    db.add(cost)
    db.commit()
    db.refresh(cost)
    return cost


@router.get("/recurring-costs")
async def list_recurring_costs(active_only: bool = True, db: Session = Depends(get_db)):
    """List recurring costs."""
    q = db.query(RecurringCost)
    if active_only:
        q = q.filter(RecurringCost.is_active == True)
    return q.order_by(RecurringCost.name).all()


# ─── Savings Goals ───────────────────────────────────────

@router.post("/savings/goals")
async def create_savings_goal(data: schemas.SavingsGoalCreate, db: Session = Depends(get_db)):
    """Create a savings goal."""
    goal = SavingsGoal(**data.model_dump())
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return goal


@router.get("/savings/goals")
async def list_savings_goals(status: str = "active", db: Session = Depends(get_db)):
    """List savings goals."""
    q = db.query(SavingsGoal)
    if status != "all":
        q = q.filter(SavingsGoal.status == status)
    return q.all()



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

# ─── CSV Import ──────────────────────────────────────────

@router.post("/import/csv", response_model=schemas.ImportSummaryOut)
async def import_bank_csv(
    file: UploadFile = File(...),
    skip_duplicates: bool = Query(True),
    auto_apply: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Import a NatWest CSV bank statement."""
    from app.finance.services.bank_import_service import import_natwest_csv

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")

    content = await file.read()
    csv_text = content.decode("utf-8-sig")  # BOM-safe

    result = import_natwest_csv(db, csv_text, skip_duplicates, auto_apply)

    return schemas.ImportSummaryOut(
        total_rows=result.total_rows,
        imported=result.imported,
        duplicates=result.duplicates,
        skipped=result.skipped,
        errors=result.errors,
        auto_categorised=result.auto_categorised,
        needs_review=result.needs_review,
        transfers=result.transfers,
        total_income=result.total_income,
        total_expenses=result.total_expenses,
        rows=[
            schemas.ImportRowResult(
                date=r.date,
                description=r.description,
                amount=r.amount,
                balance=r.balance,
                transaction_type=r.transaction_type,
                status=r.status,
                category_name=r.categorisation.category_name if r.categorisation else None,
                confidence=r.categorisation.confidence if r.categorisation else 0,
                needs_review=r.categorisation.needs_user_confirmation if r.categorisation else True,
                error_message=r.error_message,
                transaction_id=r.transaction_id,
            )
            for r in result.rows
        ],
    )


# ─── Screenshot Upload & OCR ─────────────────────────────

@router.post("/upload/screenshot", response_model=schemas.ScreenshotOCRResult)
async def upload_screenshot(
    file: UploadFile = File(...),
):
    """Upload a Yodel Finish Tour screenshot for OCR extraction."""
    from app.finance.services.screenshot_ocr_service import (
        save_screenshot,
        extract_via_llm,
    )

    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Image files only. Got: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large (max 10MB)")

    # Save the screenshot
    save_path = save_screenshot(image_bytes, file.filename)

    # Extract data via LLM
    result = await extract_via_llm(image_bytes, file.content_type)

    return schemas.ScreenshotOCRResult(
        success=result.success,
        work_date=result.work_date,
        tour_id=result.tour_id,
        user_id=result.user_id,
        delivery_count=result.delivery_count,
        collections=result.collections,
        stops=result.stops,
        attempted=result.attempted,
        done=result.done,
        failed_deliveries=result.failed_deliveries,
        gross_earnings=result.gross_earnings,
        route_area=result.route_area,
        raw_text=result.raw_text,
        confidence=result.confidence,
        message=result.message,
    )


# ─── Categorisation Confirmation ─────────────────────────

@router.post("/categorisation/confirm")
async def confirm_categorisation(
    data: schemas.ConfirmCategorisationRequest,
    db: Session = Depends(get_db),
):
    """Confirm or correct a transaction's categorisation (teaches the system)."""
    from app.finance.services.categoriser_service import confirm_categorisation as do_confirm

    pattern = do_confirm(
        db=db,
        merchant_raw=data.merchant_raw,
        category_id=data.category_id,
        expense_scope=data.expense_scope,
        display_name=data.display_name,
    )
    return {
        "confirmed": True,
        "pattern": pattern.merchant_pattern,
        "confidence": pattern.confidence_score,
        "match_count": pattern.match_count,
    }









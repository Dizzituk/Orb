# FILE: app/finance/services/tax_export_service.py
"""
Generates downloadable tax pack files (PDF cover sheet + XLSX workbook).

Pulls live data from the same calculate_tax_estimate() function the
TaxTab uses, so the download always matches what's on screen.

All generation happens in-memory — no temp files on disk.
"""
from __future__ import annotations

import io
import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def generate_tax_pdf(db: Session, tax_year: Optional[str] = None) -> bytes:
    """Generate a professional PDF tax cover sheet.

    Returns raw PDF bytes ready for streaming to the client.
    """
    from app.finance.services.finance_service import calculate_tax_estimate, get_current_tax_year
    from app.finance.services._tax_pdf_builder import build_tax_pdf

    ty = tax_year or get_current_tax_year()
    tax_data = calculate_tax_estimate(db, ty)
    work_summary = _get_work_summary(db, ty)
    van_info = _get_van_info(db)

    buf = io.BytesIO()
    build_tax_pdf(buf, tax_data, work_summary, van_info, ty)
    return buf.getvalue()


def generate_tax_xlsx(db: Session, tax_year: Optional[str] = None) -> bytes:
    """Generate a multi-tab Excel workbook with all accounting data.

    Returns raw XLSX bytes ready for streaming to the client.
    """
    from app.finance.services.finance_service import calculate_tax_estimate, get_current_tax_year
    from app.finance.services._tax_xlsx_builder import build_tax_xlsx

    ty = tax_year or get_current_tax_year()
    tax_data = calculate_tax_estimate(db, ty)
    work_logs = _get_work_logs(db, ty)
    transactions = _get_expense_transactions(db, ty)
    cc_transactions = _get_cc_transactions(db, ty)
    van_info = _get_van_info(db)

    buf = io.BytesIO()
    build_tax_xlsx(buf, tax_data, work_logs, transactions, cc_transactions, van_info, ty)
    return buf.getvalue()


# ── Data helpers ──

def _get_work_summary(db: Session, tax_year: str) -> dict:
    """Aggregate work log stats for the PDF summary."""
    from sqlalchemy import func
    from app.finance.models import DailyWorkLog

    row = db.query(
        func.count(DailyWorkLog.id).label("days"),
        func.coalesce(func.sum(DailyWorkLog.delivery_count), 0).label("deliveries"),
        func.coalesce(func.sum(DailyWorkLog.gross_earnings), 0.0).label("gross"),
        func.coalesce(func.sum(DailyWorkLog.net_hours), 0.0).label("hours"),
        func.coalesce(func.avg(DailyWorkLog.per_hour), 0.0).label("avg_per_hour"),
        func.min(DailyWorkLog.work_date).label("first_date"),
        func.max(DailyWorkLog.work_date).label("last_date"),
    ).filter(DailyWorkLog.tax_year == tax_year).first()

    return {
        "days": row.days or 0,
        "deliveries": int(row.deliveries or 0),
        "gross": float(row.gross or 0),
        "hours": float(row.hours or 0),
        "avg_per_hour": round(float(row.avg_per_hour or 0), 2),
        "first_date": str(row.first_date) if row.first_date else "",
        "last_date": str(row.last_date) if row.last_date else "",
    }


def _get_work_logs(db: Session, tax_year: str) -> list[dict]:
    """Get all daily work logs for the tax year."""
    from app.finance.models import DailyWorkLog

    logs = db.query(DailyWorkLog).filter(
        DailyWorkLog.tax_year == tax_year
    ).order_by(DailyWorkLog.work_date).all()

    return [{
        "date": str(l.work_date),
        "hours": l.net_hours or 0,
        "deliveries": l.delivery_count or 0,
        "gross": l.gross_earnings or 0,
        "per_hour": l.per_hour or 0,
        "route": l.route_area or "",
    } for l in logs]


def _get_expense_transactions(db: Session, tax_year: str) -> list[dict]:
    """Get all expense transactions with category info."""
    from app.finance.models import Transaction, ExpenseCategory

    txs = db.query(Transaction).filter(
        Transaction.tax_year == tax_year,
        Transaction.transaction_type == "expense",
        Transaction.is_deleted == False,
    ).order_by(Transaction.transaction_date).all()

    return [{
        "date": str(t.transaction_date),
        "amount": t.amount,
        "description": t.description or "",
        "category": t.category.display_name if t.category else "Uncategorised",
        "hmrc": t.category.hmrc_category if t.category else "",
        "scope": t.expense_scope or "",
        "deductible": t.deductible_amount or 0,
        "merchant": t.merchant_name or "",
        "source": t.input_source or "",
    } for t in txs]


def _get_cc_transactions(db: Session, tax_year: str) -> list[dict]:
    """Get all credit card transactions with card name."""
    from app.finance.models import CreditCardTransaction, CreditCard, ExpenseCategory

    txs = db.query(CreditCardTransaction).filter(
        CreditCardTransaction.tax_year == tax_year,
    ).order_by(CreditCardTransaction.transaction_date).all()

    return [{
        "date": str(t.transaction_date),
        "amount": t.amount,
        "description": t.description or "",
        "card": t.card.name if t.card else "",
        "category": t.expense_category.display_name if t.expense_category else "Uncategorised",
        "scope": t.expense_scope or "",
        "merchant": t.merchant_name or "",
    } for t in txs]


def _get_van_info(db: Session) -> Optional[dict]:
    """Get active van finance record."""
    from app.finance.models import VanFinance

    van = db.query(VanFinance).filter(VanFinance.is_active == True).first()
    if not van:
        return None

    return {
        "description": van.vehicle_description or "",
        "purchase_price": float(van.purchase_price or 0),
        "deposit": float(van.deposit_paid or 0),
        "finance_amount": float(van.finance_amount or 0),
        "apr": float(van.apr or 0),
        "monthly_payment": float(van.monthly_payment or 0),
        "total_payments": van.total_payments or 0,
        "payments_made": van.payments_made or 0,
        "provider": van.finance_provider or "",
        "business_use_pct": float(van.business_use_percentage or 100),
        "cost_method": van.cost_method or "mileage",
    }

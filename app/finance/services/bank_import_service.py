"""
NatWest CSV bank statement importer.

Parses NatWest CSV format:
  Date, Type, Description, Value, Balance, Account Name, Account Number

Features:
- Auto-categorises using categoriser engine
- Flags transfers (NatWest -> Revolut, etc.)
- Deduplicates against existing transactions
- Returns detailed import summary
"""

import csv
import io
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.finance.models import Transaction, ExpenseCategory
from app.finance.services.categoriser_service import (
    categorise_transaction,
    CategorisationResult,
)


# ── Import result containers ─────────────────────────────────────

@dataclass
class ImportedRow:
    date: str
    description: str
    amount: float
    balance: Optional[float]
    transaction_type: str = "expense"
    categorisation: Optional[CategorisationResult] = None
    status: str = "imported"  # imported | duplicate | skipped | error
    error_message: Optional[str] = None
    transaction_id: Optional[int] = None


@dataclass
class ImportSummary:
    total_rows: int = 0
    imported: int = 0
    duplicates: int = 0
    skipped: int = 0
    errors: int = 0
    auto_categorised: int = 0
    needs_review: int = 0
    transfers: int = 0
    total_income: float = 0.0
    total_expenses: float = 0.0
    rows: list[ImportedRow] = field(default_factory=list)


# ── Deduplication ─────────────────────────────────────────────────

def _make_dedup_key(date_str: str, amount: float, description: str) -> str:
    """Composite key for duplicate detection."""
    desc_normalised = description[:50].lower().strip()
    raw = f"{date_str}|{amount:.2f}|{desc_normalised}"
    return hashlib.md5(raw.encode()).hexdigest()


def _transaction_exists(db: Session, date_val: datetime, amount: float, description: str) -> bool:
    """Check if a matching transaction already exists."""
    desc_prefix = description[:50].lower().strip()
    existing = (
        db.query(Transaction)
        .filter(
            and_(
                Transaction.transaction_date == date_val.date(),
                Transaction.amount == abs(amount),
                Transaction.description.ilike(f"{desc_prefix}%"),
                Transaction.input_source == "import",
            )
        )
        .first()
    )
    return existing is not None


# ── CSV parsing ──────────────────────────────────────────────────

def _parse_natwest_date(date_str: str) -> datetime:
    """Parse NatWest date formats: DD/MM/YYYY or DD-Mon-YYYY."""
    for fmt in ("%d/%m/%Y", "%d-%b-%Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def _determine_tax_year(date_val: datetime) -> str:
    """Get UK tax year string from date."""
    year = date_val.year
    month = date_val.month
    day = date_val.day
    if month > 4 or (month == 4 and day >= 6):
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"


def _determine_tax_quarter(date_val: datetime) -> int:
    """Get HMRC quarter number (1-4) from date."""
    month, day = date_val.month, date_val.day
    if (month == 4 and day >= 6) or month in (5, 6) or (month == 7 and day <= 5):
        return 1
    elif (month == 7 and day >= 6) or month in (8, 9) or (month == 10 and day <= 5):
        return 2
    elif (month == 10 and day >= 6) or month in (11, 12) or (month == 1 and day <= 5):
        return 3
    else:
        return 4


# ── Main import function ─────────────────────────────────────────

def import_natwest_csv(
    db: Session,
    csv_content: str,
    skip_duplicates: bool = True,
    auto_apply_high_confidence: bool = True,
) -> ImportSummary:
    """
    Import a NatWest CSV bank statement.

    Args:
        db: Database session
        csv_content: Raw CSV text content
        skip_duplicates: Skip rows that match existing transactions
        auto_apply_high_confidence: Auto-confirm categorisation >= 0.80

    Returns:
        ImportSummary with counts and per-row details
    """
    summary = ImportSummary()

    # Parse CSV
    reader = csv.DictReader(io.StringIO(csv_content))

    # NatWest headers vary slightly - normalise
    rows_raw = []
    for row in reader:
        # Normalise header keys
        normalised = {}
        for key, val in row.items():
            clean_key = key.strip().lower().replace(" ", "_")
            normalised[clean_key] = val.strip() if val else ""
        rows_raw.append(normalised)

    summary.total_rows = len(rows_raw)

    for raw in rows_raw:
        imported_row = _process_row(
            db, raw, skip_duplicates, auto_apply_high_confidence
        )
        summary.rows.append(imported_row)

        if imported_row.status == "imported":
            summary.imported += 1
            if imported_row.amount > 0:
                summary.total_income += imported_row.amount
            else:
                summary.total_expenses += abs(imported_row.amount)
        elif imported_row.status == "duplicate":
            summary.duplicates += 1
        elif imported_row.status == "skipped":
            summary.skipped += 1
        elif imported_row.status == "error":
            summary.errors += 1

        # Track categorisation stats
        if imported_row.categorisation:
            cat = imported_row.categorisation
            if cat.category_name and cat.category_name.startswith("_"):
                summary.transfers += 1
            elif cat.confidence >= 0.80:
                summary.auto_categorised += 1
            elif cat.needs_user_confirmation:
                summary.needs_review += 1

    return summary


def _process_row(
    db: Session,
    raw: dict,
    skip_duplicates: bool,
    auto_apply: bool,
) -> ImportedRow:
    """Process a single CSV row into a transaction."""

    # Extract fields
    date_str = raw.get("date", "")
    tx_type = raw.get("type", "")
    description = raw.get("description", "")
    value_str = raw.get("value", "0")
    balance_str = raw.get("balance", "")

    # Build result container
    row = ImportedRow(
        date=date_str,
        description=description,
        amount=0.0,
        balance=None,
    )

    try:
        # Parse date
        date_val = _parse_natwest_date(date_str)

        # Parse amount
        amount = float(value_str.replace(",", "").replace("£", "").strip())
        row.amount = amount
        row.transaction_type = "income" if amount > 0 else "expense"

        # Parse balance (optional)
        if balance_str:
            try:
                row.balance = float(balance_str.replace(",", "").replace("£", "").strip())
            except ValueError:
                pass

        # Dedup check
        if skip_duplicates and _transaction_exists(db, date_val, amount, description):
            row.status = "duplicate"
            return row

        # Auto-categorise
        cat_result = categorise_transaction(
            db=db,
            description=description,
            amount=amount,
            merchant_raw=description,
        )
        row.categorisation = cat_result

        # Handle transfers - record but flag differently
        if cat_result.category_name and cat_result.category_name.startswith("_"):
            row.transaction_type = "transfer"

        # Create transaction record
        tax_year = _determine_tax_year(date_val)
        tax_quarter = _determine_tax_quarter(date_val)

        # Determine deductibility from category
        is_deductible = False
        deductible_amount = 0.0
        category_id = cat_result.category_id

        if category_id and auto_apply and cat_result.confidence >= 0.80:
            category = db.query(ExpenseCategory).get(category_id)
            if category and category.is_deductible:
                is_deductible = True
                pct = category.deductible_percentage or 100
                deductible_amount = abs(amount) * (pct / 100.0)

        tx = Transaction(
            transaction_date=date_val.date(),
            amount=abs(amount),
            transaction_type=row.transaction_type,
            description=description,
            category_id=category_id if (auto_apply and cat_result.confidence >= 0.80) else None,
            expense_scope=cat_result.expense_scope if cat_result.confidence >= 0.80 else None,
            is_tax_deductible=is_deductible,
            deductible_amount=deductible_amount,
            auto_categorised=cat_result.confidence >= 0.80,
            categorisation_confidence=cat_result.confidence,
            user_confirmed=False,
            input_source="import",
            merchant_name=cat_result.suggested_merchant_name,
            merchant_raw=description,
            tax_year=tax_year,
            tax_quarter=tax_quarter,
        )
        db.add(tx)
        db.commit()
        db.refresh(tx)

        row.transaction_id = tx.id
        row.status = "imported"

    except Exception as e:
        row.status = "error"
        row.error_message = str(e)
        db.rollback()

    return row


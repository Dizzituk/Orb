# FILE: app/finance/services/pdf_statement_parser.py
"""
PDF credit card statement parser.

Multi-strategy extraction for UK credit card statements:
1. Table extraction (pdfplumber) — works for structured PDFs
2. Text-line parsing with regex — works for text-based PDFs
3. AI vision fallback — for scanned/image PDFs (uses GPT-4o)

Supports common UK card formats: Barclaycard, MBNA, Zable,
Capital One, Tesco Bank, Sainsbury's Bank, HSBC, NatWest, etc.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class ParsedTransaction:
    """Single transaction extracted from a PDF statement."""
    transaction_date: date
    description: str
    amount: float
    is_credit: bool = False  # True = payment/refund, False = purchase


@dataclass
class StatementParseResult:
    """Result of parsing a credit card PDF statement."""
    transactions: list[ParsedTransaction] = field(default_factory=list)
    statement_date: Optional[date] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    minimum_payment: Optional[float] = None
    total_pages: int = 0
    strategy_used: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""


# ── Credit / payment detection ───────────────────────────
# Descriptions that indicate payments TO the card (not spending)

_CREDIT_KEYWORDS = [
    "payment received",
    "payment to",
    "direct debit payment",
    "repayment",
    "refund",
    "credit adjustment",
    "returned payment",
    "cashback",
]


def _is_credit_description(desc: str) -> bool:
    """Check if a transaction description indicates a payment/credit."""
    lower = desc.lower()
    return any(kw in lower for kw in _CREDIT_KEYWORDS)


# ── Date patterns common in UK statements ────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

# Amount pattern: optional £, digits with optional comma, decimal
_AMOUNT_RE = re.compile(r"[\xa3£]?\s*(\d[\d,]*\.\d{2})")

# Transaction line: date + description + amount(s)
_TX_LINE_RE = re.compile(
    r"(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"(?:\s+\d{2,4})?)"  # date part
    r"\s+"
    r"(.+?)"             # description
    r"\s+"
    r"([\xa3£]?\s*\d[\d,]*\.\d{2})"  # amount
    r"(\s+CR)?",          # optional CR for credits
    re.IGNORECASE,
)

# Two-date format: DD Mon  DD Mon  Description  Amount
# Used by Jaja, some Barclaycard statements
_TWO_DATE_TX_RE = re.compile(
    r"(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))"
    r"\s+"
    r"\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+"
    r"(.+?)"
    r"\s+"
    r"(-?[\xa3£]?\s*\d[\d,]*\.\d{2})",
    re.IGNORECASE,
)


def parse_statement_pdf(pdf_path: str | Path) -> StatementParseResult:
    """Parse a credit card PDF statement.
    
    Tries multiple strategies in order of reliability:
    1. Table extraction
    2. Text line parsing
    """
    pdf_path = Path(pdf_path)
    result = StatementParseResult()

    with pdfplumber.open(pdf_path) as pdf:
        result.total_pages = len(pdf.pages)

        # Collect all text for metadata extraction
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text += text + "\n"
        result.raw_text = all_text

        # Extract metadata (statement date, balances)
        _extract_metadata(result, all_text)

        # Strategy 1: Table extraction
        txs = _try_table_extraction(pdf, result.statement_date)
        if txs and len(txs) >= 3:
            result.transactions = txs
            result.strategy_used = "table_extraction"
            logger.info("[pdf_parse] Table extraction: %d transactions", len(txs))
            return result

        # Strategy 2: Text line parsing
        txs = _try_text_parsing(all_text, result.statement_date)
        if txs:
            result.transactions = txs
            result.strategy_used = "text_line_parsing"
            logger.info("[pdf_parse] Text parsing: %d transactions", len(txs))
            return result

        # No transactions found
        result.strategy_used = "none"
        result.warnings.append(
            "Could not extract transactions automatically. "
            "Try the AI vision parser or manual entry."
        )
    return result


def _extract_metadata(result: StatementParseResult, text: str):
    """Pull statement date, balances, minimum payment from text."""
    # Statement date (handles both normal and concatenated pdfplumber text)
    for pattern in [
        r"[Ss]tatement\s*[Dd]ate[:\s]+(\d{1,2}\s*\w+\s*\d{4})",
        r"[Ss]tatement\s+[Dd]ate[:\s]+(\d{2}[/\-]\d{2}[/\-]\d{4})",
        r"[Dd]ate\s+of\s+[Ss]tatement[:\s]+(\d{2}[/\-]\d{2}[/\-]\d{4})",
        # Capital One / generic: "23 May" near "STATEMENT TOTALS"
        r"(\d{1,2}\s+\w+)\s+STATEMENT\s+TOTALS",
    ]:
        m = re.search(pattern, text, re.I)
        if m:
            result.statement_date = _parse_date(m.group(1))
            break

    # Try filename-based date as fallback (e.g. "Jaja 6 August 2025.pdf")
    # This is set by the caller if available

    # Opening/closing balance
    for pattern, attr in [
        (r"[Oo]pening\s+[Bb]alance[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})", "opening_balance"),
        (r"[Pp]revious\s+[Bb]alance[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})", "opening_balance"),
        (r"[Cc]losing\s+[Bb]alance[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})", "closing_balance"),
        (r"[Nn]ew\s+[Cc]losing\s+[Bb]alance[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})", "closing_balance"),
        (r"[Nn]ew\s+[Bb]alance[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})", "closing_balance"),
        (r"[Bb]alance\s+[Dd]ue[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})", "closing_balance"),
    ]:
        m = re.search(pattern, text)
        if m:
            setattr(result, attr, float(m.group(1).replace(",", "")))

    # Minimum payment
    for pattern in [
        r"[Mm]inimum\s+[Pp]ayment[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})",
        r"[Mm]in(?:imum)?\s+[Dd]ue[:\s]+[\xa3£]?\s*([\d,]+\.\d{2})",
    ]:
        m = re.search(pattern, text)
        if m:
            result.minimum_payment = float(m.group(1).replace(",", ""))
            break


def _try_table_extraction(
    pdf, statement_date: Optional[date] = None,
) -> list[ParsedTransaction]:
    """Try to extract transactions from PDF tables."""
    transactions = []

    for page in pdf.pages:
        tables = page.extract_tables()
        if not tables:
            continue

        for table in tables:
            if not table or len(table) < 2:
                continue

            header_idx, col_map = _detect_table_columns(table)
            if not col_map.get("date") and not col_map.get("description"):
                continue

            for row in table[header_idx + 1:]:
                tx = _parse_table_row(row, col_map, statement_date)
                if tx:
                    transactions.append(tx)

    return transactions


def _detect_table_columns(table: list) -> tuple[int, dict]:
    """Detect column layout from table headers."""
    col_map = {}
    header_idx = 0

    for idx, row in enumerate(table[:5]):
        if not row:
            continue
        lower = [str(c).lower().strip() if c else "" for c in row]
        joined = " ".join(lower)

        if any(w in joined for w in ["date", "transaction", "description"]):
            header_idx = idx
            for i, h in enumerate(lower):
                if "date" in h and "date" not in col_map:
                    col_map["date"] = i
                elif any(w in h for w in ["description", "details", "transaction", "reference"]):
                    col_map["description"] = i
                elif any(w in h for w in ["amount", "debit", "money out", "spend", "paid out"]):
                    col_map["amount"] = i
                elif any(w in h for w in ["credit", "money in", "payment", "paid in"]):
                    col_map["credit"] = i
            break

    if not col_map and table and len(table[0]) >= 3:
        col_map = {"date": 0, "description": 1, "amount": 2}

    return header_idx, col_map


def _parse_table_row(
    row: list,
    col_map: dict,
    statement_date: Optional[date] = None,
) -> Optional[ParsedTransaction]:
    """Parse a single table row into a transaction."""
    try:
        date_str = str(row[col_map.get("date", 0)] or "").strip()
        desc = str(row[col_map.get("description", 1)] or "").strip()
        amount_str = str(row[col_map.get("amount", 2)] or "").strip()
        credit_str = str(row[col_map.get("credit", -1)] or "").strip() if "credit" in col_map else ""

        if not date_str or not desc:
            return None

        ref_year = statement_date.year if statement_date else None
        tx_date = _parse_date(date_str, default_year=ref_year)
        if not tx_date:
            return None

        # Fix year using statement date context
        if statement_date:
            tx_date = _fix_transaction_year(tx_date, statement_date)

        # Parse amount — check credit column first
        is_credit = False
        if credit_str and _AMOUNT_RE.search(credit_str):
            amount = _parse_amount(credit_str)
            is_credit = True
        elif amount_str:
            amount = _parse_amount(amount_str)
            is_credit = "CR" in amount_str.upper() or amount < 0
        else:
            return None

        if amount == 0:
            return None

        # Also check description for credit keywords
        if not is_credit:
            is_credit = _is_credit_description(desc)

        return ParsedTransaction(
            transaction_date=tx_date,
            description=desc,
            amount=abs(amount),
            is_credit=is_credit,
        )
    except (IndexError, ValueError, TypeError):
        return None


def _try_text_parsing(
    text: str, statement_date: Optional[date] = None,
) -> list[ParsedTransaction]:
    """Parse transactions from raw text using regex patterns.

    Tries two-date format first (Jaja, some Barclaycard),
    falls back to single-date format.
    """
    txs = _try_two_date_parsing(text, statement_date)
    if txs:
        return txs
    return _try_single_date_parsing(text, statement_date)


def _try_two_date_parsing(
    text: str, statement_date: Optional[date] = None,
) -> list[ParsedTransaction]:
    """Parse lines with two dates: transaction date + posting date."""
    transactions = []
    ref_year = statement_date.year if statement_date else datetime.now().year

    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue

        m = _TWO_DATE_TX_RE.search(line)
        if m:
            date_str = m.group(1).strip()
            desc = m.group(2).strip()
            amount_str = m.group(3).strip()

            tx_date = _parse_date(date_str, default_year=ref_year)
            if not tx_date:
                continue

            # Fix year using statement date context
            if statement_date:
                tx_date = _fix_transaction_year(tx_date, statement_date)

            amount = _parse_amount(amount_str)
            if amount == 0:
                continue

            # Detect credits: negative amount OR credit keywords
            is_credit = amount < 0 or _is_credit_description(desc)

            # Clean description
            desc = re.sub(r"\s{2,}", " ", desc).strip()
            if len(desc) < 2:
                continue

            transactions.append(ParsedTransaction(
                transaction_date=tx_date,
                description=desc,
                amount=abs(amount),
                is_credit=is_credit,
            ))

    return transactions


def _try_single_date_parsing(
    text: str, statement_date: Optional[date] = None,
) -> list[ParsedTransaction]:
    """Parse transactions with single date + description + amount."""
    transactions = []
    ref_year = statement_date.year if statement_date else datetime.now().year

    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Skip header/footer lines
        lower = line.lower()
        if any(w in lower for w in [
            "statement", "page", "account number", "sort code",
            "minimum payment", "balance", "credit limit",
            "interest rate", "apr", "your account",
        ]):
            continue

        m = _TX_LINE_RE.search(line)
        if m:
            date_str = m.group(1).strip()
            desc = m.group(2).strip()
            amount_str = m.group(3).strip()
            is_credit = bool(m.group(4))  # "CR" suffix

            tx_date = _parse_date(date_str, default_year=ref_year)
            if not tx_date:
                continue

            # Fix year using statement date context
            if statement_date:
                tx_date = _fix_transaction_year(tx_date, statement_date)

            amount = _parse_amount(amount_str)
            if amount == 0:
                continue

            # Also check description for credit keywords
            if not is_credit:
                is_credit = _is_credit_description(desc)

            # Clean description
            desc = re.sub(r"\s{2,}", " ", desc).strip()
            if len(desc) < 2:
                continue

            transactions.append(ParsedTransaction(
                transaction_date=tx_date,
                description=desc,
                amount=abs(amount),
                is_credit=is_credit,
            ))

    return transactions


# ── Helpers ──────────────────────────────────────────────

def _fix_transaction_year(tx_date: date, statement_date: date) -> date:
    """Fix a transaction's year — it can never be in the future.

    Simple rule: if the date is after today, subtract a year.
    If today is Feb 2026 and tx says Nov 2026, that's Nov 2025.
    """
    today = date.today()

    # Transaction can't be in the future
    if tx_date > today:
        try:
            return tx_date.replace(year=tx_date.year - 1)
        except ValueError:
            return tx_date

    return tx_date


def _parse_date(s: str, default_year: int = None) -> Optional[date]:
    """Parse a date string in various UK formats."""
    s = s.strip()
    if not default_year:
        default_year = datetime.now().year

    # "02 Jan 2025", "02 Jan 25", "06 August 2025", "08Jan"
    m = re.match(r"(\d{1,2})\s*(\w+)\s*(\d{2,4})?", s, re.I)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        month = _MONTH_MAP.get(month_str, _MONTH_MAP.get(month_str[:3], 0))
        if not month:
            return None
        year_str = m.group(3)
        if year_str:
            year = int(year_str)
            if year < 100:
                year += 2000
        else:
            year = default_year
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # "02/01/2025" or "02-01-2025"
    m = re.match(r"(\d{2})[/\-](\d{2})[/\-](\d{2,4})", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def _parse_amount(s: str) -> float:
    """Parse an amount string like '£1,234.56' or '1234.56 CR'."""
    s = s.replace("£", "").replace("\xa3", "").replace(",", "").strip()
    negative = "CR" in s.upper() or "-" in s
    s = re.sub(r"[^0-9.]", "", s)
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return 0.0

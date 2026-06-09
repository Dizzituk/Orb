# FILE: app/finance/models.py
"""
SQLAlchemy models for the ASTRA finance module.

Tables:
- expense_categories: HMRC-aligned expense classification
- transactions: Every financial transaction (income/expense)
- mileage_logs: Daily odometer readings for HMRC claims
- mileage_year_summaries: Running annual mileage totals
- recurring_costs: Fixed outgoings (rent, insurance, etc.)
- savings_goals: Goal-based savings tracking
- savings_allocations: Individual set-aside events
- tax_years: Per-year tax position and MTD status
- weekly_earnings: Per-week income/performance summaries
- daily_work_logs: Per-day Yodel delivery records
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Date, Text, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Expense Categories ─────────────────────────────────

class ExpenseCategory(Base):
    """Expense categories aligned with HMRC SA103 boxes."""

    __tablename__ = "finance_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(100), nullable=False)
    hmrc_category = Column(String(100), nullable=True)
    default_scope = Column(String(20), default="business")  # business/personal/mixed
    is_deductible = Column(Boolean, default=True)
    deductible_percentage = Column(Float, default=100.0)
    icon = Column(String(50), nullable=True)
    colour = Column(String(7), nullable=True)
    sort_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_now)

    transactions = relationship("Transaction", back_populates="category")


# ─── Transactions ────────────────────────────────────────

class Transaction(Base):
    """Core financial transaction — every penny in or out."""

    __tablename__ = "finance_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_date = Column(Date, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(20), nullable=False)  # income/expense/transfer
    description = Column(String(500), nullable=False)

    # Classification
    category_id = Column(Integer, ForeignKey("finance_categories.id"), nullable=True)
    expense_scope = Column(String(20), nullable=True)
    is_tax_deductible = Column(Boolean, default=False)
    deductible_amount = Column(Float, default=0.0)

    # Categorisation metadata
    auto_categorised = Column(Boolean, default=False)
    categorisation_confidence = Column(String(10), nullable=True)  # high/medium/low/manual
    user_confirmed = Column(Boolean, default=False)

    # Source
    input_source = Column(String(20), default="manual")
    merchant_name = Column(String(200), nullable=True)
    merchant_raw = Column(String(500), nullable=True)
    receipt_image_path = Column(String(500), nullable=True)

    # Yodel income fields
    delivery_count = Column(Integer, nullable=True)
    hours_worked = Column(Float, nullable=True)
    route_info = Column(String(200), nullable=True)

    # Credit card payment linkage
    linked_card_id = Column(Integer, ForeignKey("finance_credit_cards.id"), nullable=True)

    # Tax tracking
    tax_year = Column(String(7), nullable=False)
    tax_quarter = Column(Integer, nullable=True)

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    is_deleted = Column(Boolean, default=False)

    category = relationship("ExpenseCategory", back_populates="transactions")

    @property
    def category_name(self):
        return self.category.name if self.category else None

    __table_args__ = (
        Index("idx_fin_tx_date_type", "transaction_date", "transaction_type"),
        Index("idx_fin_tx_tax_year", "tax_year", "tax_quarter"),
        Index("idx_fin_tx_category", "category_id", "transaction_date"),
    )


# ─── Merchant Patterns ──────────────────────────────────

class RecurringCost(Base):
    """Fixed outgoings for budgeting (rent, insurance, phone, etc.)."""

    __tablename__ = "finance_recurring_costs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    amount = Column(Float, nullable=False)
    frequency = Column(String(20), nullable=False)  # weekly/monthly/annually etc
    monthly_equivalent = Column(Float, nullable=False)
    weekly_equivalent = Column(Float, nullable=False)
    category_id = Column(Integer, ForeignKey("finance_categories.id"), nullable=True)
    is_essential = Column(Boolean, default=True)
    is_tax_deductible = Column(Boolean, default=False)
    deductible_percentage = Column(Float, default=0.0)
    next_due_date = Column(Date, nullable=True)
    payment_day = Column(Integer, nullable=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True)
    auto_track = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


# ─── Savings Goals ───────────────────────────────────────



# Register split-out ledger models so their tables are created on init
from app.finance import models_workday  # noqa: E402,F401

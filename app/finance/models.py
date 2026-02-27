# FILE: app/finance/models.py
"""
SQLAlchemy models for the ASTRA finance module.

Tables:
- expense_categories: HMRC-aligned expense classification
- transactions: Every financial transaction (income/expense)
- merchant_patterns: Learned auto-categorisation patterns
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
    merchant_patterns = relationship("MerchantPattern", back_populates="category")


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

class MerchantPattern(Base):
    """Learned patterns for auto-categorising by merchant name."""

    __tablename__ = "finance_merchant_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    merchant_pattern = Column(String(200), nullable=False, index=True)
    merchant_display_name = Column(String(200), nullable=False)
    category_id = Column(Integer, ForeignKey("finance_categories.id"), nullable=False)
    default_scope = Column(String(20), nullable=False)
    confidence_score = Column(Float, default=0.5)
    match_count = Column(Integer, default=0)
    last_matched = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_now)

    category = relationship("ExpenseCategory", back_populates="merchant_patterns")


# ─── Credit Cards ────────────────────────────────────────

class CreditCard(Base):
    """Registered credit card for tracking statement imports."""

    __tablename__ = "finance_credit_cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)                # "Zable", "Barclaycard"
    provider = Column(String(100), nullable=True)             # Card provider
    last_four = Column(String(4), nullable=True)              # Last 4 digits
    natwest_description = Column(String(200), nullable=True)  # How it appears on NatWest statement
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_now)

    # Relationship to card transactions
    card_transactions = relationship("CreditCardTransaction", back_populates="card")


class CreditCardTransaction(Base):
    """Individual transaction on a credit card statement."""

    __tablename__ = "finance_credit_card_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(Integer, ForeignKey("finance_credit_cards.id"), nullable=False)
    transaction_date = Column(Date, nullable=False, index=True)
    description = Column(String(500), nullable=False)
    amount = Column(Float, nullable=False)
    merchant_name = Column(String(200), nullable=True)
    category_id = Column(Integer, ForeignKey("finance_categories.id"), nullable=True)
    expense_scope = Column(String(20), default="unknown")    # business | personal | mixed
    is_tax_deductible = Column(Boolean, default=False)
    user_confirmed = Column(Boolean, default=False)
    tax_year = Column(String(7), nullable=True)
    dedup_hash = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, default=_now)

    card = relationship("CreditCard", back_populates="card_transactions")
    category = relationship("ExpenseCategory")



class CreditCardStatement(Base):
    """Monthly statement record for a credit card.
    Tracks balances over time to build a running history.
    """

    __tablename__ = "finance_credit_card_statements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(Integer, ForeignKey("finance_credit_cards.id"), nullable=False)
    statement_date = Column(Date, nullable=False)
    period_start = Column(Date, nullable=True)
    period_end = Column(Date, nullable=True)
    opening_balance = Column(Float, default=0.0)
    closing_balance = Column(Float, default=0.0)
    total_charges = Column(Float, default=0.0)
    total_payments = Column(Float, default=0.0)
    interest_charged = Column(Float, default=0.0)
    minimum_payment = Column(Float, nullable=True)
    transactions_imported = Column(Integer, default=0)
    source_filename = Column(String(300), nullable=True)
    drive_file_id = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=_now)

    card = relationship("CreditCard")


class DriveWatchFolder(Base):
    """Google Drive folder mapped to a credit card for auto-import."""

    __tablename__ = "finance_drive_watch_folders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(Integer, ForeignKey("finance_credit_cards.id"), nullable=False)
    drive_folder_id = Column(String(200), nullable=False)
    folder_name = Column(String(200), nullable=True)
    last_checked = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_now)

    card = relationship("CreditCard")


class DriveProcessedFile(Base):
    """Tracks which Drive files have already been imported."""

    __tablename__ = "finance_drive_processed_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drive_file_id = Column(String(200), nullable=False, unique=True)
    drive_filename = Column(String(300), nullable=False)
    card_id = Column(Integer, ForeignKey("finance_credit_cards.id"), nullable=False)
    statement_id = Column(Integer, ForeignKey("finance_credit_card_statements.id"), nullable=True)
    transactions_imported = Column(Integer, default=0)
    processed_at = Column(DateTime, default=_now)
    status = Column(String(20), default="success")  # success | failed | partial
    error_message = Column(String(500), nullable=True)



class VanFinance(Base):
    """Van hire purchase / finance agreement tracking.
    
    HMRC rules (important):
    - If using MILEAGE RATE method: van finance is ALREADY covered by 45p/25p.
      Cannot claim capital allowance OR interest separately.
    - If using ACTUAL COSTS method: can claim AIA (full cost in year 1) 
      plus interest on HP (not capital repayment).
    - Cannot mix methods. Must choose one for the tax year.
    """

    __tablename__ = "finance_van_finance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_description = Column(String(200), nullable=False)  # "2019 Ford Transit Custom"
    purchase_price = Column(Float, nullable=False)              # Total cash price of van
    deposit_paid = Column(Float, default=0.0)
    finance_amount = Column(Float, nullable=False)              # Amount financed (price - deposit)
    apr = Column(Float, nullable=False)                         # Annual percentage rate
    monthly_payment = Column(Float, nullable=False)
    total_payments = Column(Integer, nullable=False)            # Total number of monthly payments
    payments_made = Column(Integer, default=0)
    first_payment_date = Column(Date, nullable=False)
    finance_provider = Column(String(100), nullable=True)       # "Moneybarn"
    agreement_number = Column(String(100), nullable=True)
    business_use_percentage = Column(Float, default=100.0)      # % used for business
    mot_due_date = Column(Date, nullable=True)
    road_tax_due_date = Column(Date, nullable=True)
    road_tax_amount = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    cost_method = Column(String(20), default="mileage")         # mileage | actual_costs
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class BudgetItem(Base):
    """Personal budget line items — rent, food, subscriptions, fuel etc.
    
    These are NOT tax deductible business expenses.
    They represent personal outgoings that reduce disposable income.
    """

    __tablename__ = "finance_budget_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)                  # "Rent", "Food", "Netflix"
    category = Column(String(50), nullable=False)               # housing | food | transport | subscriptions | utilities | other
    amount = Column(Float, nullable=False)
    frequency = Column(String(20), default="monthly")           # weekly | fortnightly | monthly | quarterly | annual
    weekly_equivalent = Column(Float, nullable=True)            # auto-calculated
    is_fixed = Column(Boolean, default=True)                    # Fixed cost vs variable
    is_active = Column(Boolean, default=True)
    notes = Column(String(300), nullable=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

# ─── Mileage ────────────────────────────────────────────

class MileageLog(Base):
    """Daily odometer readings for HMRC mileage claims."""

    __tablename__ = "finance_mileage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_date = Column(Date, nullable=False, unique=True, index=True)
    start_mileage = Column(Float, nullable=False)
    end_mileage = Column(Float, nullable=False)
    total_miles = Column(Float, nullable=False)
    business_miles = Column(Float, nullable=True)
    personal_miles = Column(Float, nullable=True)
    route_description = Column(String(500), nullable=True)
    delivery_area = Column(String(200), nullable=True)
    hours_on_road = Column(Float, nullable=True)
    claim_method = Column(String(20), default="simplified")
    claimable_amount = Column(Float, default=0.0)
    tax_year = Column(String(7), nullable=False)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class MileageYearSummary(Base):
    """Running annual mileage totals — rate changes after 10k miles."""

    __tablename__ = "finance_mileage_year_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tax_year = Column(String(7), nullable=False, unique=True, index=True)
    total_business_miles = Column(Float, default=0.0)
    miles_at_higher_rate = Column(Float, default=0.0)
    miles_at_lower_rate = Column(Float, default=0.0)
    total_claimable = Column(Float, default=0.0)
    claim_method = Column(String(20), nullable=False)
    last_updated = Column(DateTime, default=_now)


# ─── Recurring Costs ────────────────────────────────────

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

class SavingsGoal(Base):
    """Goal-based savings (tax reserve, China trip, investments, etc.)."""

    __tablename__ = "finance_savings_goals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0.0)
    target_date = Column(Date, nullable=True)
    priority = Column(String(20), nullable=False)  # critical/high/medium/low
    weekly_allocation = Column(Float, default=0.0)
    is_percentage_based = Column(Boolean, default=False)
    percentage_of_income = Column(Float, nullable=True)
    is_tax_reserve = Column(Boolean, default=False)
    is_recurring = Column(Boolean, default=False)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    allocations = relationship("SavingsAllocation", back_populates="goal")


class SavingsAllocation(Base):
    """Individual set-aside events — audit trail for savings."""

    __tablename__ = "finance_savings_allocations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    goal_id = Column(Integer, ForeignKey("finance_savings_goals.id"), nullable=False)
    allocation_date = Column(Date, nullable=False)
    amount = Column(Float, nullable=False)
    source_description = Column(String(200), nullable=True)
    running_total = Column(Float, nullable=False)
    created_at = Column(DateTime, default=_now)

    goal = relationship("SavingsGoal", back_populates="allocations")


# ─── Tax Years ───────────────────────────────────────────

class TaxYear(Base):
    """Master record per tax year with running totals and MTD status."""

    __tablename__ = "finance_tax_years"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tax_year = Column(String(7), nullable=False, unique=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    total_income = Column(Float, default=0.0)
    total_business_expenses = Column(Float, default=0.0)
    total_personal_expenses = Column(Float, default=0.0)
    total_deductible = Column(Float, default=0.0)
    taxable_profit = Column(Float, default=0.0)
    estimated_income_tax = Column(Float, default=0.0)
    estimated_ni_class2 = Column(Float, default=0.0)
    estimated_ni_class4 = Column(Float, default=0.0)
    total_estimated_tax = Column(Float, default=0.0)
    tax_reserved = Column(Float, default=0.0)
    q1_submitted = Column(Boolean, default=False)
    q2_submitted = Column(Boolean, default=False)
    q3_submitted = Column(Boolean, default=False)
    q4_submitted = Column(Boolean, default=False)
    final_submitted = Column(Boolean, default=False)
    is_current = Column(Boolean, default=False)
    is_finalised = Column(Boolean, default=False)
    last_calculated = Column(DateTime, default=_now)
    created_at = Column(DateTime, default=_now)


# ─── Weekly Earnings ─────────────────────────────────────

class WeeklyEarnings(Base):
    """Per-week earnings and performance summary."""

    __tablename__ = "finance_weekly_earnings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week_number = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    week_start = Column(Date, nullable=False)
    week_end = Column(Date, nullable=False)
    tax_year = Column(String(7), nullable=False)
    gross_income = Column(Float, default=0.0)
    delivery_count = Column(Integer, default=0)
    total_hours = Column(Float, default=0.0)
    fuel_cost = Column(Float, default=0.0)
    other_business_costs = Column(Float, default=0.0)
    total_costs = Column(Float, default=0.0)
    gross_per_hour = Column(Float, default=0.0)
    net_per_hour = Column(Float, default=0.0)
    per_delivery = Column(Float, default=0.0)
    cost_per_mile = Column(Float, default=0.0)
    total_miles = Column(Float, default=0.0)
    recommended_tax_aside = Column(Float, default=0.0)
    recommended_savings = Column(Float, default=0.0)
    spendable_income = Column(Float, default=0.0)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    __table_args__ = (
        Index("idx_fin_weekly", "year", "week_number"),
    )


# ─── Daily Work Logs ─────────────────────────────────────

class DailyWorkLog(Base):
    """Per-day Yodel delivery record — ties earnings, hours, mileage."""

    __tablename__ = "finance_daily_work_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    work_date = Column(Date, nullable=False, unique=True, index=True)
    start_time = Column(String(5), nullable=True)
    end_time = Column(String(5), nullable=True)
    total_hours = Column(Float, nullable=True)
    break_hours = Column(Float, default=0.0)
    net_hours = Column(Float, nullable=True)
    delivery_count = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    collections = Column(Integer, default=0)
    stops = Column(Integer, default=0)
    attempted = Column(Integer, default=0)
    done = Column(Integer, default=0)
    route_area = Column(String(200), nullable=True)
    tour_id = Column(String(20), nullable=True)
    user_id = Column(String(20), nullable=True)
    rate_per_parcel = Column(Float, default=0.0)
    total_parcels = Column(Integer, default=0)
    gross_earnings = Column(Float, default=0.0)
    per_hour = Column(Float, default=0.0)
    per_delivery = Column(Float, default=0.0)
    qualifies_food_allowance = Column(Boolean, default=False)
    food_allowance_claimed = Column(Float, default=0.0)
    screenshot_path = Column(String(500), nullable=True)
    ocr_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    tax_year = Column(String(7), nullable=False)







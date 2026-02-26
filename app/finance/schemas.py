# FILE: app/finance/schemas.py
"""
Pydantic schemas for the finance API.
Request/response models for all finance endpoints.
"""
from __future__ import annotations

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


# ─── Transaction Schemas ─────────────────────────────────

class TransactionCreate(BaseModel):
    transaction_date: date
    amount: float = Field(gt=0)
    transaction_type: str
    description: str
    category_id: Optional[int] = None
    expense_scope: Optional[str] = None
    merchant_name: Optional[str] = None
    delivery_count: Optional[int] = None
    hours_worked: Optional[float] = None
    notes: Optional[str] = None


class TransactionUpdate(BaseModel):
    category_id: Optional[int] = None
    expense_scope: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None
    user_confirmed: bool = True


class TransactionOut(BaseModel):
    id: int
    transaction_date: date
    amount: float
    transaction_type: str
    description: str
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    expense_scope: Optional[str] = None
    is_tax_deductible: bool = False
    merchant_name: Optional[str] = None
    auto_categorised: bool = False
    user_confirmed: bool = False
    tax_year: str

    class Config:
        from_attributes = True


# ─── Daily Work Log Schemas ──────────────────────────────

class DailyLogCreate(BaseModel):
    work_date: date
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    delivery_count: int = 0
    failed_deliveries: int = 0
    collections: int = 0
    stops: int = 0
    attempted: int = 0
    done: int = 0
    rate_per_parcel: float = 0.0
    gross_earnings: float = 0.0
    route_area: Optional[str] = None
    tour_id: Optional[str] = None
    user_id: Optional[str] = None
    screenshot_path: Optional[str] = None


class DailyLogOut(BaseModel):
    id: int
    work_date: date
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_hours: Optional[float] = None
    net_hours: Optional[float] = None
    delivery_count: int
    failed_deliveries: int
    collections: int
    stops: int
    attempted: int
    done: int
    rate_per_parcel: float = 0.0
    total_parcels: int = 0
    gross_earnings: float
    per_hour: float
    per_delivery: float
    qualifies_food_allowance: bool
    route_area: Optional[str] = None
    tour_id: Optional[str] = None

    class Config:
        from_attributes = True


# ─── Mileage Schemas ─────────────────────────────────────

class MileageCreate(BaseModel):
    log_date: date
    start_mileage: float
    end_mileage: float
    business_miles: Optional[float] = None
    route_description: Optional[str] = None
    hours_on_road: Optional[float] = None


class MileageOut(BaseModel):
    id: int
    log_date: date
    start_mileage: float
    end_mileage: float
    total_miles: float
    claimable_amount: float
    tax_year: str

    class Config:
        from_attributes = True


# ─── Recurring Cost Schemas ──────────────────────────────

class RecurringCostCreate(BaseModel):
    name: str
    amount: float
    frequency: str
    is_essential: bool = True
    is_tax_deductible: bool = False
    deductible_percentage: float = 0.0
    payment_day: Optional[int] = None
    category_id: Optional[int] = None
    description: Optional[str] = None


class RecurringCostOut(BaseModel):
    id: int
    name: str
    amount: float
    frequency: str
    monthly_equivalent: float
    weekly_equivalent: float
    is_essential: bool
    is_tax_deductible: bool
    is_active: bool

    class Config:
        from_attributes = True


# ─── Savings Goal Schemas ────────────────────────────────

class SavingsGoalCreate(BaseModel):
    name: str
    target_amount: float
    target_date: Optional[date] = None
    priority: str = "medium"
    weekly_allocation: Optional[float] = None
    is_percentage_based: bool = False
    percentage_of_income: Optional[float] = None
    description: Optional[str] = None


class SavingsGoalOut(BaseModel):
    id: int
    name: str
    target_amount: float
    current_amount: float
    target_date: Optional[date] = None
    priority: str
    weekly_allocation: float
    status: str
    progress_pct: float = 0.0
    is_tax_reserve: bool = False

    class Config:
        from_attributes = True


# ─── Dashboard / Summary Schemas ─────────────────────────

class TaxEstimate(BaseModel):
    tax_year: str
    gross_income: float
    total_allowable_expenses: float
    taxable_profit: float
    total_income_tax: float
    total_ni: float
    total_tax_liability: float
    effective_tax_rate: float
    weekly_tax_aside: float
    mileage_deduction: float


class WeeklyBudget(BaseModel):
    week_start: date
    gross_income: float
    tax_reserve: float
    essential_costs: float
    savings_total: float
    spendable_income: float
    spendable_per_day: float


class DashboardSummary(BaseModel):
    this_week_earnings: float
    this_week_deliveries: int
    per_hour_gross: float
    per_hour_net: float
    tax_reserved: float
    tax_estimated: float
    tax_reserve_pct: float
    spendable_this_week: float
    spendable_per_day: float
    recent_transactions: list[TransactionOut] = []
    alerts: list[str] = []


# ─── Import / Upload Schemas ─────────────────────────────

class ImportRowResult(BaseModel):
    date: str
    description: str
    amount: float
    balance: Optional[float] = None
    transaction_type: str
    status: str  # imported | duplicate | skipped | error
    category_name: Optional[str] = None
    confidence: float = 0.0
    needs_review: bool = False
    error_message: Optional[str] = None
    transaction_id: Optional[int] = None


class ImportSummaryOut(BaseModel):
    total_rows: int
    imported: int
    duplicates: int
    skipped: int
    errors: int
    auto_categorised: int
    needs_review: int
    transfers: int
    total_income: float
    total_expenses: float
    rows: list[ImportRowResult] = []


class ScreenshotOCRResult(BaseModel):
    success: bool
    work_date: Optional[date] = None
    tour_id: Optional[str] = None
    user_id: Optional[str] = None
    delivery_count: Optional[int] = None
    collections: Optional[int] = None
    stops: Optional[int] = None
    attempted: Optional[int] = None
    done: Optional[int] = None
    failed_deliveries: Optional[int] = None
    gross_earnings: Optional[float] = None
    route_area: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: float = 0.0
    message: str = ""


class ConfirmCategorisationRequest(BaseModel):
    merchant_raw: str
    category_id: int
    expense_scope: str
    display_name: Optional[str] = None


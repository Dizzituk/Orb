# FILE: app/overwatcher/cost_budget.py
"""
Daily and monthly budget enforcement for ASTRA.

Aggregates cost data from the persistent ledger and enforces
spend caps. When thresholds are reached, signals tier downgrading
or blocks further expensive calls.

Env vars:
    ORB_DAILY_BUDGET_GBP=5.00
    ORB_MONTHLY_BUDGET_GBP=50.00
    ORB_BUDGET_WARNING_THRESHOLD=0.8
    ORB_USD_TO_GBP_RATE=0.79

Usage:
    from app.overwatcher.cost_budget import (
        check_daily_budget,
        check_monthly_budget,
        get_spend_summary,
        BudgetState,
    )

    state = check_daily_budget()
    if state.exceeded:
        # Block or downgrade
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from app.overwatcher.cost_ledger import read_all_today, read_all_this_month

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

def _env_float(key: str, default: float) -> float:
    """Read a float from env with fallback."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_daily_budget_gbp() -> float:
    return _env_float("ORB_DAILY_BUDGET_GBP", 5.0)


def get_monthly_budget_gbp() -> float:
    return _env_float("ORB_MONTHLY_BUDGET_GBP", 50.0)


def get_warning_threshold() -> float:
    return _env_float("ORB_BUDGET_WARNING_THRESHOLD", 0.8)


def get_usd_to_gbp_rate() -> float:
    return _env_float("ORB_USD_TO_GBP_RATE", 0.79)


# =========================================================================
# Budget state
# =========================================================================

class BudgetLevel(str, Enum):
    """Budget consumption level."""
    OK = "ok"                       # Under warning threshold
    WARNING = "warning"             # Over warning threshold, under limit
    EXCEEDED = "exceeded"           # Over limit
    CRITICAL = "critical"           # Way over limit (>120%)


@dataclass
class BudgetState:
    """Current state of a budget (daily or monthly)."""
    level: BudgetLevel
    spent_usd: float
    spent_gbp: float
    budget_gbp: float
    remaining_gbp: float
    percentage_used: float
    message: str

    @property
    def ok(self) -> bool:
        return self.level == BudgetLevel.OK

    @property
    def exceeded(self) -> bool:
        return self.level in (BudgetLevel.EXCEEDED, BudgetLevel.CRITICAL)


@dataclass
class SpendSummary:
    """Full spend summary for dashboard display."""
    daily: BudgetState
    monthly: BudgetState
    today_calls: int
    month_calls: int
    by_stage: Dict[str, float]   # stage → GBP
    by_model: Dict[str, float]   # model → GBP


# =========================================================================
# Aggregation helpers
# =========================================================================

def _aggregate_records(records: list) -> tuple:
    """
    Aggregate cost records.

    Returns: (total_usd, by_stage_usd, by_model_usd, call_count)
    """
    total_usd = 0.0
    by_stage: Dict[str, float] = {}
    by_model: Dict[str, float] = {}

    for r in records:
        cost = r.get("cost_usd", 0.0)
        total_usd += cost

        stage = r.get("stage", "unknown")
        by_stage[stage] = by_stage.get(stage, 0.0) + cost

        model = r.get("model", "unknown")
        by_model[model] = by_model.get(model, 0.0) + cost

    return total_usd, by_stage, by_model, len(records)


def _to_gbp(usd: float) -> float:
    """Convert USD to GBP."""
    return round(usd * get_usd_to_gbp_rate(), 4)


def _build_budget_state(
    spent_usd: float,
    budget_gbp: float,
    label: str,
) -> BudgetState:
    """Build a BudgetState from spend data."""
    spent_gbp = _to_gbp(spent_usd)
    remaining = max(0.0, budget_gbp - spent_gbp)
    pct = (spent_gbp / budget_gbp * 100) if budget_gbp > 0 else 0.0
    warning_pct = get_warning_threshold() * 100

    if pct >= 120:
        level = BudgetLevel.CRITICAL
        msg = f"{label} budget critically exceeded: £{spent_gbp:.2f}/£{budget_gbp:.2f} ({pct:.0f}%)"
    elif pct >= 100:
        level = BudgetLevel.EXCEEDED
        msg = f"{label} budget exceeded: £{spent_gbp:.2f}/£{budget_gbp:.2f} ({pct:.0f}%)"
    elif pct >= warning_pct:
        level = BudgetLevel.WARNING
        msg = f"{label} budget warning: £{spent_gbp:.2f}/£{budget_gbp:.2f} ({pct:.0f}%)"
    else:
        level = BudgetLevel.OK
        msg = f"{label}: £{spent_gbp:.2f}/£{budget_gbp:.2f} ({pct:.0f}%)"

    return BudgetState(
        level=level,
        spent_usd=spent_usd,
        spent_gbp=round(spent_gbp, 4),
        budget_gbp=budget_gbp,
        remaining_gbp=round(remaining, 4),
        percentage_used=round(pct, 1),
        message=msg,
    )


# =========================================================================
# Public API
# =========================================================================

def check_daily_budget() -> BudgetState:
    """Check current daily spend against budget."""
    records = read_all_today()
    total_usd, _, _, _ = _aggregate_records(records)
    return _build_budget_state(total_usd, get_daily_budget_gbp(), "Daily")


def check_monthly_budget() -> BudgetState:
    """Check current monthly spend against budget."""
    records = read_all_this_month()
    total_usd, _, _, _ = _aggregate_records(records)
    return _build_budget_state(total_usd, get_monthly_budget_gbp(), "Monthly")


def get_spend_summary() -> SpendSummary:
    """
    Get full spend summary for the dashboard.

    Reads both daily and monthly records.
    """
    today_records = read_all_today()
    month_records = read_all_this_month()

    today_usd, today_by_stage, today_by_model, today_calls = _aggregate_records(today_records)
    month_usd, month_by_stage, month_by_model, month_calls = _aggregate_records(month_records)

    rate = get_usd_to_gbp_rate()

    return SpendSummary(
        daily=_build_budget_state(today_usd, get_daily_budget_gbp(), "Daily"),
        monthly=_build_budget_state(month_usd, get_monthly_budget_gbp(), "Monthly"),
        today_calls=today_calls,
        month_calls=month_calls,
        by_stage={k: round(v * rate, 4) for k, v in month_by_stage.items()},
        by_model={k: round(v * rate, 4) for k, v in month_by_model.items()},
    )


def should_downgrade_tier(stage: str) -> bool:
    """
    Check if a stage should use a cheaper model due to budget pressure.

    Non-critical stages get downgraded when daily budget is in warning zone.
    Critical stages (spec_gate, overwatcher) are never auto-downgraded.
    """
    CRITICAL_STAGES = {"spec_gate", "overwatcher", "architecture"}

    if stage in CRITICAL_STAGES:
        return False

    daily = check_daily_budget()
    return daily.level in (BudgetLevel.WARNING, BudgetLevel.EXCEEDED, BudgetLevel.CRITICAL)


def is_call_allowed(stage: str, break_glass: bool = False) -> tuple:
    """
    Check if an LLM call is allowed given current budget.

    Returns: (allowed: bool, reason: str)
    """
    if break_glass:
        return True, "break-glass active"

    daily = check_daily_budget()

    if daily.level == BudgetLevel.CRITICAL:
        return False, daily.message

    if daily.level == BudgetLevel.EXCEEDED:
        # Only block non-critical stages
        CRITICAL_STAGES = {"spec_gate", "overwatcher"}
        if stage not in CRITICAL_STAGES:
            return False, f"{daily.message} — non-critical stage blocked"

    return True, "within budget"


__all__ = [
    "BudgetLevel",
    "BudgetState",
    "SpendSummary",
    "check_daily_budget",
    "check_monthly_budget",
    "get_spend_summary",
    "should_downgrade_tier",
    "is_call_allowed",
]

# FILE: app/investments/schemas.py
"""
Pydantic schemas for investments API request/response shapes.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


# ═══════════════════════════════════════════
# PORTFOLIO SUMMARY
# ═══════════════════════════════════════════

class PortfolioSummary(BaseModel):
    """Top-level portfolio overview."""

    total_value_gbp: float = 0.0
    cash_balance_gbp: float = 0.0
    invested_value_gbp: float = 0.0
    total_pnl_gbp: float = 0.0
    total_pnl_pct: float = 0.0
    crypto_value_gbp: float = 0.0
    position_count: int = 0
    last_updated: Optional[str] = None
    is_stale: bool = False  # True when showing cached/snapshot data


# ═══════════════════════════════════════════
# POSITIONS
# ═══════════════════════════════════════════

class PositionItem(BaseModel):
    """Single open position."""

    ticker: str
    name: str = ""
    asset_type: str = "stock"  # stock | etf | crypto
    quantity: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0
    value_gbp: float = 0.0
    pnl_gbp: float = 0.0
    pnl_pct: float = 0.0
    currency: str = "GBP"


class PositionsResponse(BaseModel):
    """All open positions."""

    positions: List[PositionItem] = []
    total_count: int = 0
    is_stale: bool = False


# ═══════════════════════════════════════════
# CRYPTO
# ═══════════════════════════════════════════

class CryptoQuote(BaseModel):
    """Single crypto asset quote."""

    symbol: str
    name: str = ""
    price_gbp: float = 0.0
    change_24h_pct: float = 0.0
    change_7d_pct: float = 0.0
    market_cap_gbp: float = 0.0
    holdings: float = 0.0  # 0 = watch only
    holdings_value_gbp: float = 0.0
    is_holding: bool = False


class CryptoResponse(BaseModel):
    """All tracked crypto assets."""

    quotes: List[CryptoQuote] = []
    is_stale: bool = False


# ═══════════════════════════════════════════
# HISTORY (snapshot time-series)
# ═══════════════════════════════════════════

class HistoryPoint(BaseModel):
    """Single data point on the growth chart."""

    timestamp: str
    total_value_gbp: float = 0.0
    invested_value_gbp: float = 0.0
    pnl_gbp: float = 0.0


class HistoryResponse(BaseModel):
    """Portfolio value over time."""

    points: List[HistoryPoint] = []
    range: str = "all"


# ═══════════════════════════════════════════
# DIVIDENDS
# ═══════════════════════════════════════════

class DividendItem(BaseModel):
    """Single dividend payment."""

    ticker: str
    amount: float = 0.0
    currency: str = "GBP"
    paid_on: Optional[str] = None


class DividendsResponse(BaseModel):
    """Dividend history."""

    dividends: List[DividendItem] = []
    total_received: float = 0.0


# ═══════════════════════════════════════════
# SNAPSHOT TRIGGER
# ═══════════════════════════════════════════

class SnapshotResult(BaseModel):
    """Result of a manual snapshot trigger."""

    success: bool = True
    snapshot_id: Optional[int] = None
    message: str = ""
    position_count: int = 0


# ═══════════════════════════════════════════
# ALLOCATION
# ═══════════════════════════════════════════

class AllocationSlice(BaseModel):
    """Single slice of the allocation donut."""

    ticker: str
    name: str = ""
    value_gbp: float = 0.0
    weight_pct: float = 0.0


class AllocationResponse(BaseModel):
    """Portfolio allocation breakdown."""

    slices: List[AllocationSlice] = []


# ═══════════════════════════════════════════
# PERFORMANCE
# ═══════════════════════════════════════════

class PerformerItem(BaseModel):
    """Single performer in best/worst list."""

    ticker: str
    name: str = ""
    pnl_gbp: float = 0.0
    pnl_pct: float = 0.0


class PerformanceResponse(BaseModel):
    """Best and worst performers."""

    best: List[PerformerItem] = []
    worst: List[PerformerItem] = []
    range: str = "3m"

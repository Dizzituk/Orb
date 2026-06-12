# FILE: app/investments/models.py
# Purpose: SQLAlchemy models for portfolio snapshot storage.
# Called-by: app.db, app.investments.service, app.llm.routing.domain_context, app.llm.routing.ui_context_data (+1 more)
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
SQLAlchemy models for portfolio snapshot storage.

Tables:
- portfolio_snapshots: Aggregate portfolio state per snapshot event
- position_snapshots: Per-position data linked to a portfolio snapshot
- dividend_history: Dividend payments synced from Trading 212
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, ForeignKey,
)
from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PortfolioSnapshot(Base):
    """One row per snapshot event (scheduled or manual)."""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    captured_at = Column(DateTime, nullable=False, default=_now)
    total_value_gbp = Column(Float, nullable=True)
    cash_balance_gbp = Column(Float, nullable=True)
    invested_value_gbp = Column(Float, nullable=True)
    total_pnl_gbp = Column(Float, nullable=True)
    total_pnl_pct = Column(Float, nullable=True)
    crypto_value_gbp = Column(Float, nullable=True)
    snapshot_type = Column(String, default="scheduled")  # scheduled | manual


class PositionSnapshot(Base):
    """One row per position per snapshot event."""

    __tablename__ = "position_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(
        Integer,
        ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    ticker = Column(String, nullable=False)
    asset_type = Column(String, nullable=False)  # stock | etf | crypto
    name = Column(Text, nullable=True)
    quantity = Column(Float, nullable=True)
    avg_price = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    value_gbp = Column(Float, nullable=True)
    pnl_gbp = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    currency = Column(String, default="GBP")


class DividendHistory(Base):
    """Dividend payments synced from Trading 212."""

    __tablename__ = "dividend_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    amount = Column(Float, nullable=True)
    currency = Column(String, nullable=True)
    paid_on = Column(DateTime, nullable=True)
    synced_at = Column(DateTime, default=_now)

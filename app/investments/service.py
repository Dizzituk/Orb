# FILE: app/investments/service.py
# Purpose: Core business logic for the investments module.
# Called-by: app.bridge.dashboards, app.investments.chat_service, app.investments.router, app.investments.scheduler (+1 more)
# Depends-on: app.investments.cache, app.investments.coinmarketcap_client, app.investments.models, app.investments.schemas (+2 more)
# Last-renovated: 2026-06-11
"""
Core business logic for the investments module.

Orchestrates:
- Fetching live data from Trading 212 + CoinMarketCap
- Building portfolio summaries
- Storing and querying snapshots
- Allocation and performance calculations
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from app.settings.service import get_key_value
from app.investments.models import (
    PortfolioSnapshot, PositionSnapshot, DividendHistory,
)
from app.investments.schemas import (
    PortfolioSummary, PositionItem, PositionsResponse,
    CryptoQuote, CryptoResponse,
    HistoryPoint, HistoryResponse,
    DividendItem, DividendsResponse,
    SnapshotResult,
    AllocationSlice, AllocationResponse,
)
from app.investments.trading212_client import Trading212Client
from app.investments.coinmarketcap_client import CoinMarketCapClient
from app.investments.cache import get_cache

logger = logging.getLogger(__name__)

# Cache TTLs (seconds)
POSITIONS_TTL = 60
CRYPTO_TTL = 120
SUMMARY_TTL = 60

# Time range mappings
RANGE_DAYS = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "all": None,
}


# ═══════════════════════════════════════════
# CLIENT FACTORIES
# ═══════════════════════════════════════════

def _get_t212_client(db: Session) -> Optional[Trading212Client]:
    """Build a Trading 212 client from stored keys, or None."""
    api_key = get_key_value(db, "trading212_api_key")
    api_secret = get_key_value(db, "trading212_api_secret")
    if not api_key or not api_secret:
        logger.warning("[investments] Trading 212 API keys not configured")
        return None
    return Trading212Client(api_key, api_secret)


def _get_cmc_client(db: Session) -> Optional[CoinMarketCapClient]:
    """Build a CoinMarketCap client from stored key, or None."""
    api_key = get_key_value(db, "coinmarketcap")
    if not api_key:
        logger.warning("[investments] CoinMarketCap API key not configured")
        return None
    return CoinMarketCapClient(api_key)


# ═══════════════════════════════════════════
# LIVE DATA FETCHING
# ═══════════════════════════════════════════

async def fetch_portfolio_summary(db: Session) -> PortfolioSummary:
    """Fetch live portfolio summary from Trading 212 + crypto."""
    cache = get_cache()
    cached = cache.get("summary")
    if cached:
        return cached

    summary = PortfolioSummary()

    # Trading 212 data
    t212 = _get_t212_client(db)
    if t212:
        try:
            acct = await t212.get_summary()

            # New API: nested cash/investments structure
            cash_obj = acct.get("cash", {})
            inv_obj = acct.get("investments", {})

            summary.cash_balance_gbp = float(
                cash_obj.get("availableToTrade", 0)
            )
            summary.invested_value_gbp = float(
                inv_obj.get("totalCost", 0)
            )
            summary.total_pnl_gbp = float(
                inv_obj.get("unrealizedProfitLoss", 0)
            )
            summary.total_value_gbp = float(
                acct.get("totalValue", 0)
            )

            # Count positions from a separate call
            positions = await t212.get_positions()
            summary.position_count = len(positions)

            if summary.invested_value_gbp > 0:
                summary.total_pnl_pct = (
                    summary.total_pnl_gbp / summary.invested_value_gbp
                ) * 100
        except Exception as exc:
            logger.error("[investments] T212 fetch failed: %s", exc)
            summary.is_stale = True

    # Crypto value (TAO holdings only)
    cmc = _get_cmc_client(db)
    if cmc:
        try:
            quotes = await cmc.get_quotes()
            tao_data = quotes.get("TAO", {})
            if isinstance(tao_data, list):
                tao_data = tao_data[0] if tao_data else {}
            gbp_quote = tao_data.get("quote", {}).get("GBP", {})
            tao_price = float(gbp_quote.get("price", 0))
            tao_holdings = CoinMarketCapClient.get_holdings("TAO")
            summary.crypto_value_gbp = tao_price * tao_holdings
            summary.total_value_gbp += summary.crypto_value_gbp
        except Exception as exc:
            logger.error("[investments] CMC fetch failed: %s", exc)

    summary.last_updated = datetime.now(timezone.utc).isoformat()
    cache.set("summary", summary, SUMMARY_TTL)
    return summary


async def fetch_positions(db: Session) -> PositionsResponse:
    """Fetch all open stock/ETF positions from Trading 212."""
    cache = get_cache()
    cached = cache.get("positions")
    if cached:
        return cached

    response = PositionsResponse()
    t212 = _get_t212_client(db)
    if not t212:
        response.is_stale = True
        return response

    try:
        raw_positions = await t212.get_positions()
        for p in raw_positions:
            qty = float(p.get("quantity", 0))
            # Handle both old (averagePrice) and new (averagePricePaid) field names
            avg = float(
                p.get("averagePricePaid", p.get("averagePrice", 0))
            )
            cur = float(p.get("currentPrice", 0))

            # P&L: check walletImpact.unrealizedProfitLoss (new) then ppl (old)
            wallet = p.get("walletImpact", {})
            ppl = float(
                wallet.get("unrealizedProfitLoss", p.get("ppl", 0))
            )

            # Ticker: may be top-level or nested under instrument
            instrument = p.get("instrument", {})
            ticker = p.get("ticker", instrument.get("ticker", ""))
            name = instrument.get("shortName", "") or instrument.get("name", "") or ticker.replace("_US_EQ", "")

            # Use T212's own value if available (accounts for FX, fractional, etc.)
            # Priority: walletImpact.currentValue > top-level currentValue > fallback calc
            value = float(
                wallet.get("currentValue",
                    p.get("currentValue", cur * qty)
                )
            )

            # P&L percentage: prefer T212's own if available
            invested = float(wallet.get("totalCost", avg * qty))
            pnl_pct = (
                float(wallet.get("unrealizedProfitLossPercent", 0))
                or ((ppl / invested * 100) if invested > 0 else 0)
            )

            item = PositionItem(
                ticker=ticker,
                name=name,
                asset_type="stock",
                quantity=qty,
                avg_price=avg,
                current_price=cur,
                value_gbp=value,
                pnl_gbp=ppl,
                pnl_pct=pnl_pct,
            )
            response.positions.append(item)

        response.total_count = len(response.positions)
    except Exception as exc:
        logger.error("[investments] T212 positions failed: %s", exc)
        response.is_stale = True

    cache.set("positions", response, POSITIONS_TTL)
    return response


async def fetch_crypto(db: Session) -> CryptoResponse:
    """Fetch BTC, ETH, TAO quotes from CoinMarketCap."""
    cache = get_cache()
    cached = cache.get("crypto")
    if cached:
        return cached

    response = CryptoResponse()
    cmc = _get_cmc_client(db)
    if not cmc:
        response.is_stale = True
        return response

    try:
        raw_quotes = await cmc.get_quotes()
        for symbol in ["BTC", "ETH", "TAO"]:
            data = raw_quotes.get(symbol, {})
            if isinstance(data, list):
                data = data[0] if data else {}
            gbp = data.get("quote", {}).get("GBP", {})
            price = float(gbp.get("price", 0))
            holdings = CoinMarketCapClient.get_holdings(symbol)

            quote = CryptoQuote(
                symbol=symbol,
                name=data.get("name", symbol),
                price_gbp=price,
                change_24h_pct=float(gbp.get("percent_change_24h", 0)),
                change_7d_pct=float(gbp.get("percent_change_7d", 0)),
                market_cap_gbp=float(gbp.get("market_cap", 0)),
                holdings=holdings,
                holdings_value_gbp=price * holdings,
                is_holding=holdings > 0,
            )
            response.quotes.append(quote)
    except Exception as exc:
        logger.error("[investments] CMC fetch failed: %s", exc)
        response.is_stale = True

    cache.set("crypto", response, CRYPTO_TTL)
    return response


# ═══════════════════════════════════════════
# SNAPSHOT OPERATIONS
# ═══════════════════════════════════════════

async def take_snapshot(
    db: Session,
    snapshot_type: str = "manual",
) -> SnapshotResult:
    """
    Capture current portfolio state into the DB.
    Makes ONE set of API calls and shares the data to avoid
    hitting T212's rate limits (1 req / 5s per endpoint).
    """
    import asyncio
    cache = get_cache()
    cache.clear()  # Force fresh data

    summary = PortfolioSummary()
    positions_resp = PositionsResponse()
    crypto_resp = CryptoResponse()

    # ── T212: fetch summary + positions in sequence with rate-limit gap ──
    t212 = _get_t212_client(db)
    if t212:
        try:
            acct = await t212.get_summary()
            cash_obj = acct.get("cash", {})
            inv_obj = acct.get("investments", {})
            summary.cash_balance_gbp = float(cash_obj.get("availableToTrade", 0))
            summary.invested_value_gbp = float(inv_obj.get("totalCost", 0))
            summary.total_pnl_gbp = float(inv_obj.get("unrealizedProfitLoss", 0))
            summary.total_value_gbp = float(acct.get("totalValue", 0))
            if summary.invested_value_gbp > 0:
                summary.total_pnl_pct = (
                    summary.total_pnl_gbp / summary.invested_value_gbp
                ) * 100
        except Exception as exc:
            logger.error("[investments] T212 summary failed: %s", exc)
            summary.is_stale = True

        # Rate-limit pause before next T212 call
        await asyncio.sleep(1.5)

        try:
            raw_positions = await t212.get_positions()
            summary.position_count = len(raw_positions)
            for p in raw_positions:
                qty = float(p.get("quantity", 0))
                avg = float(p.get("averagePricePaid", p.get("averagePrice", 0)))
                cur = float(p.get("currentPrice", 0))
                wallet = p.get("walletImpact", {})
                ppl = float(wallet.get("unrealizedProfitLoss", p.get("ppl", 0)))
                instrument = p.get("instrument", {})
                ticker = p.get("ticker", instrument.get("ticker", ""))
                name = (
                    instrument.get("shortName", "")
                    or instrument.get("name", "")
                    or ticker.replace("_US_EQ", "")
                )
                # Use T212's own value (accounts for FX, fractional, etc.)
                value = float(
                    wallet.get("currentValue",
                        p.get("currentValue", cur * qty)
                    )
                )
                invested = float(wallet.get("totalCost", avg * qty))
                pnl_pct = (
                    float(wallet.get("unrealizedProfitLossPercent", 0))
                    or ((ppl / invested * 100) if invested > 0 else 0)
                )
                positions_resp.positions.append(PositionItem(
                    ticker=ticker, name=name, asset_type="stock",
                    quantity=qty, avg_price=avg, current_price=cur,
                    value_gbp=value, pnl_gbp=ppl,
                    pnl_pct=pnl_pct,
                ))
            positions_resp.total_count = len(positions_resp.positions)
        except Exception as exc:
            logger.error("[investments] T212 positions failed: %s", exc)
            positions_resp.is_stale = True

    # ── CMC: fetch crypto (independent of T212) ──
    cmc = _get_cmc_client(db)
    if cmc:
        try:
            raw_quotes = await cmc.get_quotes()
            for symbol in ["BTC", "ETH", "TAO"]:
                data = raw_quotes.get(symbol, {})
                if isinstance(data, list):
                    data = data[0] if data else {}
                gbp = data.get("quote", {}).get("GBP", {})
                price = float(gbp.get("price", 0))
                holdings = CoinMarketCapClient.get_holdings(symbol)
                crypto_resp.quotes.append(CryptoQuote(
                    symbol=symbol, name=data.get("name", symbol),
                    price_gbp=price,
                    change_24h_pct=float(gbp.get("percent_change_24h", 0)),
                    change_7d_pct=float(gbp.get("percent_change_7d", 0)),
                    market_cap_gbp=float(gbp.get("market_cap", 0)),
                    holdings=holdings, holdings_value_gbp=price * holdings,
                    is_holding=holdings > 0,
                ))
            # Add TAO value to summary
            tao_q = next((q for q in crypto_resp.quotes if q.symbol == "TAO"), None)
            if tao_q:
                summary.crypto_value_gbp = tao_q.holdings_value_gbp
                summary.total_value_gbp += summary.crypto_value_gbp
        except Exception as exc:
            logger.error("[investments] CMC fetch failed: %s", exc)
            crypto_resp.is_stale = True

    summary.last_updated = datetime.now(timezone.utc).isoformat()

    # ── Populate cache so subsequent tab fetches get warm hits ──
    cache.set("summary", summary, SUMMARY_TTL)
    cache.set("positions", positions_resp, POSITIONS_TTL)
    cache.set("crypto", crypto_resp, CRYPTO_TTL)

    # Store portfolio snapshot
    snap = PortfolioSnapshot(
        total_value_gbp=summary.total_value_gbp,
        cash_balance_gbp=summary.cash_balance_gbp,
        invested_value_gbp=summary.invested_value_gbp,
        total_pnl_gbp=summary.total_pnl_gbp,
        total_pnl_pct=summary.total_pnl_pct,
        crypto_value_gbp=summary.crypto_value_gbp,
        snapshot_type=snapshot_type,
    )
    db.add(snap)
    db.flush()  # Get the ID

    # Store position snapshots
    for pos in positions_resp.positions:
        ps = PositionSnapshot(
            snapshot_id=snap.id,
            ticker=pos.ticker,
            asset_type=pos.asset_type,
            name=pos.name,
            quantity=pos.quantity,
            avg_price=pos.avg_price,
            current_price=pos.current_price,
            value_gbp=pos.value_gbp,
            pnl_gbp=pos.pnl_gbp,
            pnl_pct=pos.pnl_pct,
        )
        db.add(ps)

    # Store crypto positions
    for q in crypto_resp.quotes:
        if q.is_holding:
            ps = PositionSnapshot(
                snapshot_id=snap.id,
                ticker=q.symbol,
                asset_type="crypto",
                name=q.name,
                quantity=q.holdings,
                current_price=q.price_gbp,
                value_gbp=q.holdings_value_gbp,
            )
            db.add(ps)

    db.commit()

    total_positions = len(positions_resp.positions) + sum(
        1 for q in crypto_resp.quotes if q.is_holding
    )
    return SnapshotResult(
        success=True,
        snapshot_id=snap.id,
        message=f"Snapshot #{snap.id} captured ({snapshot_type})",
        position_count=total_positions,
    )


# ═══════════════════════════════════════════
# HISTORY QUERIES
# ═══════════════════════════════════════════

def fetch_history(db: Session, range_key: str = "all") -> HistoryResponse:
    """Query portfolio snapshots for the growth chart."""
    days = RANGE_DAYS.get(range_key)
    query = db.query(PortfolioSnapshot).order_by(
        PortfolioSnapshot.captured_at.asc()
    )

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        query = query.filter(PortfolioSnapshot.captured_at >= cutoff)

    snapshots = query.all()
    points = [
        HistoryPoint(
            timestamp=s.captured_at.isoformat() if s.captured_at else "",
            total_value_gbp=s.total_value_gbp or 0,
            invested_value_gbp=s.invested_value_gbp or 0,
            pnl_gbp=s.total_pnl_gbp or 0,
        )
        for s in snapshots
    ]
    return HistoryResponse(points=points, range=range_key)


def fetch_dividends(db: Session) -> DividendsResponse:
    """Query dividend history from DB."""
    rows = (
        db.query(DividendHistory)
        .order_by(DividendHistory.paid_on.desc())
        .all()
    )
    items = [
        DividendItem(
            ticker=r.ticker,
            amount=r.amount or 0,
            currency=r.currency or "GBP",
            paid_on=r.paid_on.isoformat() if r.paid_on else None,
        )
        for r in rows
    ]
    total = sum(i.amount for i in items)
    return DividendsResponse(dividends=items, total_received=total)


async def fetch_allocation(db: Session) -> AllocationResponse:
    """
    Build allocation breakdown from LIVE positions + crypto.
    Includes all stock/ETF positions from T212 plus crypto holdings.
    """
    # Fetch live stock positions (uses cache if warm)
    positions_resp = await fetch_positions(db)
    # Fetch live crypto (uses cache if warm)
    crypto_resp = await fetch_crypto(db)

    slices: List[AllocationSlice] = []

    # Stock / ETF positions
    for p in positions_resp.positions:
        val = p.value_gbp or 0
        if val > 0:
            slices.append(AllocationSlice(
                ticker=p.ticker,
                name=p.name or p.ticker,
                value_gbp=round(val, 2),
                weight_pct=0,  # calculated below
            ))

    # Crypto holdings (only those with actual holdings, not watch-only)
    for q in crypto_resp.quotes:
        if q.is_holding and q.holdings_value_gbp > 0:
            slices.append(AllocationSlice(
                ticker=q.symbol,
                name=q.name or q.symbol,
                value_gbp=round(q.holdings_value_gbp, 2),
                weight_pct=0,  # calculated below
            ))

    total_value = sum(s.value_gbp for s in slices)
    if total_value == 0:
        return AllocationResponse()

    # Calculate weight percentages
    for s in slices:
        s.weight_pct = round((s.value_gbp / total_value) * 100, 2)

    # Sort by value descending
    slices.sort(key=lambda s: s.value_gbp, reverse=True)

    return AllocationResponse(slices=slices)


def fetch_performance(
    db: Session,
    range_key: str = "3m",
) -> "PerformanceResponse":
    """
    Best 5 / worst 5 performers over a rolling window.
    Uses snapshot data if available (compares earliest vs latest in range).
    Falls back to live P&L data if insufficient snapshots.
    """
    from app.investments.schemas import PerformerItem, PerformanceResponse

    days = RANGE_DAYS.get(range_key, 90)

    # Try to use snapshot-based comparison
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        earliest = (
            db.query(PortfolioSnapshot)
            .filter(PortfolioSnapshot.captured_at >= cutoff)
            .order_by(PortfolioSnapshot.captured_at.asc())
            .first()
        )
        latest = (
            db.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.captured_at.desc())
            .first()
        )

        if earliest and latest and earliest.id != latest.id:
            # Compare positions between earliest and latest snapshots
            early_positions = {
                p.ticker: p
                for p in db.query(PositionSnapshot)
                .filter(PositionSnapshot.snapshot_id == earliest.id)
                .all()
            }
            late_positions = {
                p.ticker: p
                for p in db.query(PositionSnapshot)
                .filter(PositionSnapshot.snapshot_id == latest.id)
                .all()
            }

            performers = []
            for ticker, late in late_positions.items():
                early = early_positions.get(ticker)
                if early and early.value_gbp and early.value_gbp > 0:
                    change_gbp = (late.value_gbp or 0) - (early.value_gbp or 0)
                    change_pct = (change_gbp / early.value_gbp) * 100
                    performers.append(PerformerItem(
                        ticker=ticker,
                        name=late.name or ticker,
                        pnl_gbp=round(change_gbp, 2),
                        pnl_pct=round(change_pct, 2),
                    ))

            if len(performers) >= 2:
                performers.sort(key=lambda p: p.pnl_pct, reverse=True)
                return PerformanceResponse(
                    best=performers[:5],
                    worst=list(reversed(performers[-5:])),
                    range=range_key,
                )

    # Fallback: use latest snapshot's live P&L
    latest = (
        db.query(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.captured_at.desc())
        .first()
    )
    if not latest:
        return PerformanceResponse(range=range_key)

    positions = (
        db.query(PositionSnapshot)
        .filter(PositionSnapshot.snapshot_id == latest.id)
        .all()
    )
    performers = [
        PerformerItem(
            ticker=p.ticker,
            name=p.name or p.ticker,
            pnl_gbp=round(p.pnl_gbp or 0, 2),
            pnl_pct=round(p.pnl_pct or 0, 2),
        )
        for p in positions
        if p.pnl_pct is not None
    ]
    performers.sort(key=lambda p: p.pnl_pct, reverse=True)

    return PerformanceResponse(
        best=performers[:5],
        worst=list(reversed(performers[-5:])),
        range=range_key,
    )


# ═══════════════════════════════════════════
# PER-TICKER HISTORY (TRADES & DIVIDENDS)
# ═══════════════════════════════════════════

async def fetch_ticker_trades(
    db: Session,
    ticker: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch order history for a specific ticker from T212.
    Returns list of trade dicts with normalised fields.
    """
    cache = get_cache()
    cache_key = f"trades:{ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    t212 = _get_t212_client(db)
    if not t212:
        return []

    try:
        raw = await t212.get_order_history(ticker=ticker, limit=limit)
        items = raw.get("items", [])
        trades = []
        for o in items:
            instrument = o.get("instrument", {})
            trades.append({
                "id": o.get("id"),
                "ticker": o.get("ticker", instrument.get("ticker", ticker)),
                "type": o.get("type", "UNKNOWN"),
                "status": o.get("status", "UNKNOWN"),
                "quantity": float(o.get("filledQuantity", o.get("quantity", 0))),
                "price": float(o.get("fillPrice", o.get("limitPrice", 0))),
                "value": float(o.get("filledValue", 0)),
                "currency": o.get("currency", "GBP"),
                "created_at": o.get("dateCreated", ""),
                "executed_at": o.get("dateExecuted", o.get("dateModified", "")),
                "side": "SELL" if float(o.get("quantity", 0)) < 0 else "BUY",
            })
        cache.set(cache_key, trades, 300)  # 5 min cache
        return trades
    except Exception as exc:
        logger.error("[investments] T212 order history failed for %s: %s", ticker, exc)
        return []


async def fetch_ticker_dividends(
    db: Session,
    ticker: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch dividend history for a specific ticker from T212.
    Returns list of dividend dicts with normalised fields.
    """
    cache = get_cache()
    cache_key = f"dividends:{ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    t212 = _get_t212_client(db)
    if not t212:
        return []

    try:
        raw = await t212.get_dividends(ticker=ticker, limit=limit)
        items = raw.get("items", [])
        dividends = []
        for d in items:
            dividends.append({
                "ticker": d.get("ticker", ticker),
                "amount": float(d.get("amount", 0)),
                "quantity": float(d.get("quantity", 0)),
                "amount_per_share": float(d.get("amountPerShare", 0)),
                "currency": d.get("currency", "GBP"),
                "paid_on": d.get("paidOn", d.get("reference", "")),
            })
        cache.set(cache_key, dividends, 300)  # 5 min cache
        return dividends
    except Exception as exc:
        logger.error("[investments] T212 dividends failed for %s: %s", ticker, exc)
        return []

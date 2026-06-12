# FILE: app/investments/router.py
# Purpose: FastAPI router for investments endpoints.
# Called-by: main
# Depends-on: app.auth, app.db, app.investments, app.investments.coinmarketcap_client (+4 more)
# Last-renovated: 2026-06-11
"""
FastAPI router for investments endpoints.

All endpoints are read-only (no trading/order placement).
Auth required via Depends(require_auth).
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.investments.schemas import (
    PortfolioSummary, PositionsResponse, CryptoResponse,
    HistoryResponse, DividendsResponse,
    SnapshotResult, AllocationResponse, PerformanceResponse,
)
from app.investments import service
from app.investments import news_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/investments",
    tags=["Investments"],
    dependencies=[Depends(require_auth)],
)


@router.get("/summary", response_model=PortfolioSummary)
async def get_summary(db: Session = Depends(get_db)):
    """Current portfolio summary (total value, P&L, cash, crypto)."""
    return await service.fetch_portfolio_summary(db)


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(db: Session = Depends(get_db)):
    """All open stock/ETF positions with live P&L."""
    return await service.fetch_positions(db)


@router.get("/crypto", response_model=CryptoResponse)
async def get_crypto(db: Session = Depends(get_db)):
    """BTC, ETH, TAO current quotes + TAO holdings."""
    return await service.fetch_crypto(db)


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    range: str = Query("all", pattern="^(1d|1w|1m|3m|6m|1y|all)$"),
    db: Session = Depends(get_db),
):
    """Portfolio value over time from snapshots."""
    return service.fetch_history(db, range)


@router.get("/dividends", response_model=DividendsResponse)
async def get_dividends(db: Session = Depends(get_db)):
    """Dividend history from synced Trading 212 data."""
    return service.fetch_dividends(db)


@router.get("/allocation", response_model=AllocationResponse)
async def get_allocation(db: Session = Depends(get_db)):
    """Portfolio allocation breakdown — live positions + crypto."""
    return await service.fetch_allocation(db)


@router.get("/performance", response_model=PerformanceResponse)
def get_performance(
    range: str = Query("3m", pattern="^(1m|3m|6m|1y|all)$"),
    db: Session = Depends(get_db),
):
    """Best/worst performers over rolling window."""
    return service.fetch_performance(db, range)


@router.post("/snapshot", response_model=SnapshotResult)
async def trigger_snapshot(db: Session = Depends(get_db)):
    """Manually trigger a portfolio snapshot."""
    return await service.take_snapshot(db, snapshot_type="manual")


@router.get("/ticker/{ticker}/trades")
async def get_ticker_trades(
    ticker: str,
    db: Session = Depends(get_db),
):
    """Order history for a specific ticker."""
    trades = await service.fetch_ticker_trades(db, ticker)
    return {"ticker": ticker, "trades": trades}


@router.get("/ticker/{ticker}/dividends")
async def get_ticker_dividends(
    ticker: str,
    db: Session = Depends(get_db),
):
    """Dividend history for a specific ticker."""
    dividends = await service.fetch_ticker_dividends(db, ticker)
    return {"ticker": ticker, "dividends": dividends}


@router.get("/debug/raw-positions")
async def debug_raw_positions(db: Session = Depends(get_db)):
    """
    Debug: dump raw T212 position data to inspect field names.
    Shows first 3 positions as-is from the API.
    """
    t212 = service._get_t212_client(db)
    if not t212:
        return {"error": "T212 not configured"}
    try:
        raw = await t212.get_positions()
        # Return first 3 to keep it readable
        return {"count": len(raw), "sample": raw[:3]}
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/cmc/credits")
async def get_cmc_credits(db: Session = Depends(get_db)):
    """
    CMC API credit usage — current month stats.
    Calls /v1/key/info to get fresh monthly totals.
    """
    from app.investments.coinmarketcap_client import get_credit_tracker

    cmc = service._get_cmc_client(db)
    if not cmc:
        return {"error": "CoinMarketCap API key not configured"}

    try:
        await cmc.get_key_info()
    except Exception as exc:
        logger.warning("[investments] CMC key info failed: %s", exc)

    return get_credit_tracker().to_dict()


@router.get("/diagnostics")
async def get_diagnostics(db: Session = Depends(get_db)):
    """
    Debug endpoint — reports key config status and API connectivity.
    Never exposes actual key values.
    """
    from app.settings.service import get_key_value, mask_key
    import httpx, base64

    result = {
        "t212_key_set": bool(get_key_value(db, "trading212_api_key")),
        "t212_secret_set": bool(get_key_value(db, "trading212_api_secret")),
        "cmc_key_set": bool(get_key_value(db, "coinmarketcap")),
        "brave_key_set": bool(get_key_value(db, "brave_search")),
        "t212_api_test": None,
        "cmc_api_test": None,
    }

    # Test T212 connectivity
    api_key = get_key_value(db, "trading212_api_key")
    api_secret = get_key_value(db, "trading212_api_secret")
    if api_key and api_secret:
        try:
            creds = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://live.trading212.com/api/v0/equity/account/cash",
                    headers={"Authorization": f"Basic {creds}"},
                )
                result["t212_api_test"] = {
                    "status": resp.status_code,
                    "ok": resp.status_code == 200,
                    "body_preview": resp.text[:200] if resp.status_code != 200 else "OK",
                }
        except Exception as exc:
            result["t212_api_test"] = {"error": str(exc)}

    # Test CMC connectivity
    cmc_key = get_key_value(db, "coinmarketcap")
    if cmc_key:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://pro-api.coinmarketcap.com/v1/key/info",
                    headers={"X-CMC_PRO_API_KEY": cmc_key, "Accepts": "application/json"},
                )
                result["cmc_api_test"] = {
                    "status": resp.status_code,
                    "ok": resp.status_code == 200,
                }
        except Exception as exc:
            result["cmc_api_test"] = {"error": str(exc)}

    return result


@router.get("/news/summary")
async def get_news_summary(db: Session = Depends(get_db)):
    """
    Lightweight news summary for use by other ASTRA components.
    Returns recent headlines across all holdings as plain text.
    Can be called by the main chat pipeline to answer news questions.
    """
    positions_resp = await service.fetch_positions(db)
    holdings = [
        {"ticker": p.ticker, "name": p.name, "asset_type": p.asset_type}
        for p in positions_resp.positions[:8]
    ]
    holdings.append({"ticker": "TAO", "name": "Bittensor", "asset_type": "crypto"})

    articles = await news_service.fetch_news_aggregated(holdings, max_per_holding=2)
    lines = []
    for a in articles[:15]:
        lines.append(
            f"[{a.get('ticker', '')}] {a.get('headline', '')} "
            f"— {a.get('source', '')} ({a.get('published', '')})"
        )
    return {"summary": "\n".join(lines), "article_count": len(articles)}


@router.get("/news")
async def get_news(
    ticker: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    News for holdings. If ticker is provided, returns news for that
    specific holding. Otherwise returns aggregated news across all.
    """
    if ticker:
        # Single-ticker news
        items = await news_service.fetch_news_for_ticker(
            ticker=ticker, max_results=5,
        )
        return {"ticker": ticker, "articles": items}

    # Aggregated — get positions to build holding list
    positions_resp = await service.fetch_positions(db)
    holdings = [
        {
            "ticker": p.ticker,
            "name": p.name,
            "asset_type": p.asset_type,
        }
        for p in positions_resp.positions[:10]  # Top 10 by value
    ]
    # Add crypto holdings
    holdings.append({"ticker": "TAO", "name": "Bittensor", "asset_type": "crypto"})

    articles = await news_service.fetch_news_aggregated(
        holdings, max_per_holding=2,
    )
    return {"ticker": None, "articles": articles}

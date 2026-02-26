# FILE: app/investments/chat_service.py
"""
Investment-aware chat context injection.

Builds a system prompt enriched with live portfolio data
so the LLM can answer questions about the user's holdings.
"""
import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.investments.service import (
    fetch_portfolio_summary,
    fetch_positions,
    fetch_crypto,
)
from app.investments.news_service import fetch_news_aggregated

logger = logging.getLogger(__name__)


async def build_investment_context(
    db: Session,
    user_message: str = "",
) -> str:
    """
    Build a context string with current portfolio data.
    Injected as system context before the user's message.
    """
    parts = [
        "You are ASTRA, an AI assistant with access to the user's live portfolio data.",
        "Answer questions about their investments using the data below.",
        "All values are in GBP. Do not make up numbers — use only the data provided.",
        "If asked about something not in the data, say so.",
        "",
    ]

    # Portfolio summary
    try:
        summary = await fetch_portfolio_summary(db)
        parts.append("=== PORTFOLIO SUMMARY ===")
        parts.append(f"Total value: £{summary.total_value_gbp:,.2f}")
        parts.append(f"Invested: £{summary.invested_value_gbp:,.2f}")
        parts.append(f"Cash balance: £{summary.cash_balance_gbp:,.2f}")
        parts.append(f"Unrealised P&L: £{summary.total_pnl_gbp:,.2f} ({summary.total_pnl_pct:+.2f}%)")
        parts.append(f"Crypto holdings value: £{summary.crypto_value_gbp:,.2f}")
        parts.append(f"Number of positions: {summary.position_count}")
        parts.append("")
    except Exception as exc:
        logger.warning("[investments-chat] Summary fetch failed: %s", exc)
        parts.append("(Portfolio summary unavailable)")
        parts.append("")

    # Positions
    try:
        positions_resp = await fetch_positions(db)
        if positions_resp.positions:
            parts.append("=== OPEN POSITIONS ===")
            for p in positions_resp.positions:
                sign = "+" if p.pnl_gbp >= 0 else ""
                parts.append(
                    f"- {p.name or p.ticker}: "
                    f"{p.quantity:.4f} shares, "
                    f"avg £{p.avg_price:.2f}, "
                    f"now £{p.current_price:.2f}, "
                    f"value £{p.value_gbp:,.2f}, "
                    f"P&L {sign}£{p.pnl_gbp:,.2f} ({p.pnl_pct:+.1f}%)"
                )
            parts.append("")
    except Exception as exc:
        logger.warning("[investments-chat] Positions fetch failed: %s", exc)

    # Crypto
    try:
        crypto_resp = await fetch_crypto(db)
        if crypto_resp.quotes:
            parts.append("=== CRYPTO WATCHLIST ===")
            for q in crypto_resp.quotes:
                label = "HOLDING" if q.is_holding else "WATCHING"
                line = (
                    f"- {q.symbol} ({q.name}): "
                    f"£{q.price_gbp:,.2f}, "
                    f"24h {q.change_24h_pct:+.2f}%, "
                    f"7d {q.change_7d_pct:+.2f}%"
                )
                if q.is_holding:
                    line += f", holdings: {q.holdings} ({q.symbol}), value £{q.holdings_value_gbp:,.2f}"
                line += f" [{label}]"
                parts.append(line)
            parts.append("")
    except Exception as exc:
        logger.warning("[investments-chat] Crypto fetch failed: %s", exc)

    # Recent news headlines (available for any chat question)
    try:
        # Build a lightweight holdings list for news fetch
        news_holdings = []
        try:
            positions_resp_for_news = await fetch_positions(db)
            news_holdings = [
                {
                    "ticker": p.ticker,
                    "name": p.name or p.ticker,
                    "asset_type": p.asset_type,
                }
                for p in positions_resp_for_news.positions[:8]
            ]
        except Exception:
            pass
        news_holdings.append({"ticker": "TAO", "name": "Bittensor", "asset_type": "crypto"})

        news_items = await fetch_news_aggregated(news_holdings, max_per_holding=2)
        if news_items:
            parts.append("=== RECENT NEWS ====")
            for item in news_items[:15]:  # Cap at 15 headlines
                parts.append(
                    f"- [{item.get('ticker', '')}] "
                    f"{item.get('headline', '')} "
                    f"({item.get('source', '')}, {item.get('published', '')})"
                )
            parts.append("")
    except Exception as exc:
        logger.warning("[investments-chat] News fetch failed: %s", exc)

    return "\n".join(parts)

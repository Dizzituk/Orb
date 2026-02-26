# FILE: app/investments/coinmarketcap_client.py
"""
Async HTTP client for the CoinMarketCap API (free tier).

Handles:
- Header-based auth (X-CMC_PRO_API_KEY)
- GBP conversion
- Hard-coded scope: BTC, ETH, TAO
- Credit usage tracking from response headers / status

Free tier limits: 10,000 credits/month, latest data only.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://pro-api.coinmarketcap.com"
TIMEOUT = 15.0

# Hard-coded crypto scope
TRACKED_SYMBOLS = "BTC,ETH,TAO"

# Holdings (manual — no API for wallet balances on free tier)
HOLDINGS: Dict[str, float] = {
    "TAO": 11.48,
    # BTC and ETH are watch-only (0 holdings)
}


class CreditTracker:
    """
    Tracks CMC API credit usage across the session.
    Updated from response status objects after each call.
    """

    def __init__(self) -> None:
        self.credits_used_session: int = 0
        self.credits_used_month: int = 0
        self.credits_limit_month: int = 10_000
        self.plan_name: str = "unknown"
        self.last_updated: Optional[str] = None
        self._call_count: int = 0

    def update_from_status(self, status: Dict[str, Any]) -> None:
        """Extract credit info from CMC response status object."""
        credit_count = status.get("credit_count", 0)
        self.credits_used_session += credit_count
        self._call_count += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def update_from_key_info(self, data: Dict[str, Any]) -> None:
        """Update from /v1/key/info response."""
        usage = data.get("usage", {})
        current_month = usage.get("current_month", {})
        self.credits_used_month = current_month.get("credits_used", 0)
        self.credits_limit_month = current_month.get("credits_left", 10_000) + self.credits_used_month
        plan = data.get("plan", {})
        self.plan_name = plan.get("plan_name", "unknown")
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        remaining = max(0, self.credits_limit_month - self.credits_used_month)
        pct_used = (
            round((self.credits_used_month / self.credits_limit_month) * 100, 1)
            if self.credits_limit_month > 0 else 0
        )
        return {
            "plan": self.plan_name,
            "credits_used_month": self.credits_used_month,
            "credits_limit_month": self.credits_limit_month,
            "credits_remaining": remaining,
            "pct_used": pct_used,
            "session_credits_used": self.credits_used_session,
            "session_call_count": self._call_count,
            "last_updated": self.last_updated,
        }


# Module-level tracker — persists across requests within a server process
_credit_tracker = CreditTracker()


def get_credit_tracker() -> CreditTracker:
    """Access the shared credit tracker."""
    return _credit_tracker


class CoinMarketCapClient:
    """Read-only client for CoinMarketCap crypto data."""

    def __init__(self, api_key: str) -> None:
        self._headers = {
            "X-CMC_PRO_API_KEY": api_key,
            "Accept": "application/json",
        }
        self._tracker = _credit_tracker

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated GET request. Tracks credits from response."""
        url = f"{BASE_URL}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            # Track credits from response status
            status = data.get("status", {})
            if status:
                self._tracker.update_from_status(status)

            return data

    async def get_quotes(self) -> Dict[str, Any]:
        """
        GET /v1/cryptocurrency/quotes/latest
        Returns current prices, market cap, 24h/7d change for
        BTC, ETH, TAO converted to GBP.
        """
        data = await self._get(
            "/v1/cryptocurrency/quotes/latest",
            params={
                "symbol": TRACKED_SYMBOLS,
                "convert": "GBP",
            },
        )
        return data.get("data", {})

    async def get_info(self) -> Dict[str, Any]:
        """
        GET /v1/cryptocurrency/info
        Returns metadata, logos, descriptions.
        """
        data = await self._get(
            "/v1/cryptocurrency/info",
            params={"symbol": TRACKED_SYMBOLS},
        )
        return data.get("data", {})

    async def get_key_info(self) -> Dict[str, Any]:
        """
        GET /v1/key/info
        Returns API key plan details and credit usage for the month.
        Updates the credit tracker with monthly totals.
        """
        data = await self._get("/v1/key/info")
        key_data = data.get("data", {})
        self._tracker.update_from_key_info(key_data)
        return key_data

    @staticmethod
    def get_holdings(symbol: str) -> float:
        """Return known holdings for a symbol (0 = watch only)."""
        return HOLDINGS.get(symbol, 0.0)

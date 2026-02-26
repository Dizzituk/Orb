# FILE: app/investments/trading212_client.py
"""
Async HTTP client for the Trading 212 API.

Handles:
- HTTP Basic auth (API_KEY:API_SECRET → Base64)
- Rate limit header parsing and backoff
- All read-only portfolio endpoints

Docs: https://t212public-api-docs.redoc.ly/
"""
import base64
import logging
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://live.trading212.com/api/v0"
TIMEOUT = 15.0  # seconds


def _build_auth_header(api_key: str, api_secret: str) -> str:
    """Build HTTP Basic auth header value."""
    credentials = f"{api_key}:{api_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def _parse_rate_limit(headers: httpx.Headers) -> Dict[str, Any]:
    """Extract rate limit info from response headers."""
    return {
        "remaining": int(headers.get("x-ratelimit-remaining", -1)),
        "reset": headers.get("x-ratelimit-reset", ""),
    }


class Trading212Client:
    """Read-only client for Trading 212 portfolio data."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._auth_header = _build_auth_header(api_key, api_secret)
        self._headers = {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated GET request. Returns parsed JSON."""
        url = f"{BASE_URL}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=self._headers, params=params)

            rate_info = _parse_rate_limit(resp.headers)
            if rate_info["remaining"] == 0:
                logger.warning(
                    "[T212] Rate limit exhausted, reset at %s",
                    rate_info["reset"],
                )

            resp.raise_for_status()
            return resp.json()

    async def get_summary(self) -> Dict[str, Any]:
        """
        GET /equity/account/summary
        Returns nested structure:
        {
          "cash": {"availableToTrade": 0, "inPies": 0, "reservedForOrders": 0},
          "currency": "GBP",
          "id": 123,
          "investments": {
            "currentValue": 0,
            "realizedProfitLoss": 0,
            "totalCost": 0,
            "unrealizedProfitLoss": 0
          },
          "totalValue": 0
        }
        Rate limit: 1 req / 5s
        """
        return await self._get("/equity/account/summary")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        GET /equity/positions
        Returns list of open positions. Each position has:
          averagePricePaid, currentPrice, quantity, ticker (via instrument),
          ppl (via walletImpact or top-level), etc.

        Handles both old (flat) and new (nested instrument) response shapes.
        """
        data = await self._get("/equity/positions")
        # API returns a list directly
        return data if isinstance(data, list) else []

    async def get_order_history(
        self,
        ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        GET /equity/history/orders (paginated, cursor-based)
        Supports optional ticker filter.
        """
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return await self._get("/equity/history/orders", params=params)

    async def get_dividends(
        self,
        ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        GET /equity/history/dividends
        Supports optional ticker filter.
        """
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return await self._get("/equity/history/dividends", params=params)

    async def get_transactions(
        self,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        GET /equity/history/transactions
        """
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._get("/equity/history/transactions", params=params)

    async def get_instruments(self) -> List[Dict[str, Any]]:
        """
        GET /equity/metadata/instruments
        Returns ticker metadata (name, ISIN, type).
        """
        data = await self._get("/equity/metadata/instruments")
        return data if isinstance(data, list) else []

    async def get_pies(self) -> List[Dict[str, Any]]:
        """
        GET /equity/pies
        Returns pie allocations if used.
        """
        data = await self._get("/equity/pies")
        return data if isinstance(data, list) else []

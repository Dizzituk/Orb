# FILE: app/sentinel/agent_client.py
# Purpose: Async httpx client for the elevated sentinel_agent on 127.0.0.1:8771 —
#          shared-secret auth, graceful agent_offline degradation, never crashes the backend.
# Called-by: app.sentinel.collector, app.sentinel.router
# Depends-on: httpx; data\sentinel\agent_secret (written by agent/installer)
# Last-renovated: 2026-06-12
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_PORT = int(os.getenv("ASTRA_SENTINEL_AGENT_PORT", "8771"))
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_SECRET_FILE = Path(__file__).resolve().parents[2] / "data" / "sentinel" / "agent_secret"

_secret_cache: Optional[str] = None
_secret_mtime: float = 0.0


class AgentOfflineError(RuntimeError):
    """The sentinel_agent is unreachable — degrade, never crash."""


def _read_secret() -> str:
    """Cached read of the shared secret; re-reads if the file changes. Never logged."""
    global _secret_cache, _secret_mtime
    try:
        mtime = _SECRET_FILE.stat().st_mtime
    except OSError:
        return ""
    if _secret_cache is None or mtime != _secret_mtime:
        try:
            _secret_cache = _SECRET_FILE.read_text(encoding="utf-8").strip()
            _secret_mtime = mtime
        except OSError:
            return ""
    return _secret_cache or ""


async def _request(
    method: str,
    path: str,
    json_body: Optional[dict] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    headers = {"X-Sentinel-Secret": _read_secret()}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, f"{_BASE_URL}{path}", json=json_body, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as exc:
        raise AgentOfflineError(f"sentinel_agent unreachable at {_BASE_URL}: {exc}") from exc
    if resp.status_code == 403:
        # Secret mismatch — surface loudly but treat as offline-equivalent for callers.
        logger.error("[sentinel] agent rejected our secret (403) — re-run install_sentinel_agent.ps1?")
        raise AgentOfflineError("sentinel_agent rejected shared secret (403)")
    resp.raise_for_status()
    return resp.json()


async def health() -> Optional[Dict[str, Any]]:
    """Agent /health, or None when offline (never raises)."""
    try:
        return await _request("GET", "/health", timeout=4.0)
    except (AgentOfflineError, httpx.HTTPError) as exc:
        logger.debug("[sentinel] agent health probe failed: %s", exc)
        return None


async def get_connections() -> List[Dict[str, Any]]:
    """Live connection snapshot. Raises AgentOfflineError when the agent is down."""
    data = await _request("GET", "/connections", timeout=15.0)
    return list(data.get("connections") or [])


async def block_ip(ip: str) -> Dict[str, Any]:
    return await _request("POST", "/block", json_body={"ip": ip}, timeout=45.0)


async def unblock_ip(ip: str) -> Dict[str, Any]:
    return await _request("POST", "/unblock", json_body={"ip": ip}, timeout=45.0)


async def list_rules() -> Dict[str, Any]:
    return await _request("GET", "/rules", timeout=45.0)

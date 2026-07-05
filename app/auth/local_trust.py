# FILE: app/auth/local_trust.py
# Purpose: require_local_or_secret — shared-secret/loopback guard for endpoints
#          the Electron renderer, Unity Room, and local tooling hit tokenless.
# Called-by: app.router_registry (include-level dependencies)
# Depends-on: app.security.client_ip
# Last-renovated: 2026-07-02
"""Local-trust dependency (security hardening 2026-07-02).

Some surfaces are LOCAL-TRUSTED BY DESIGN: the Electron renderer's poll
loops, the Unity Room (tokenless renderer), ambient voice WebSockets. They
must keep working with no login, while arbitrary LAN callers are refused.

require_local_or_secret passes when EITHER:
  * the effective client IP is loopback (desktop renderer / Unity / local
    tools talking straight to 127.0.0.1:8000), OR
  * the request carries header X-Astra-Local matching ASTRA_LOCAL_SECRET
    (Electron can inject this for any future non-loopback deployment).

Proxy note: effective_client_ip only honours X-Forwarded-For from trusted
proxy peers, so phone traffic arriving via the co-located Caddy proxy
resolves to the PHONE's IP — not loopback — and is refused here unless it
carries the secret. Phone features use the Bearer-auth'd /bridge surface.

Works on both HTTP and WebSocket routes (HTTPConnection is the shared base;
WebSocket denials close with policy-violation 1008).
"""
from __future__ import annotations

import logging
import os
import secrets as _secrets

from fastapi import HTTPException, WebSocketException, status
from starlette.requests import HTTPConnection

from app.security.client_ip import effective_client_ip, is_loopback_ip

logger = logging.getLogger(__name__)


def _secret_matches(conn: HTTPConnection) -> bool:
    expected = os.getenv("ASTRA_LOCAL_SECRET", "")
    provided = conn.headers.get("x-astra-local", "")
    if not expected or not provided:
        return False
    return _secrets.compare_digest(provided, expected)


async def require_local_or_secret(conn: HTTPConnection) -> bool:
    """Dependency: loopback caller or X-Astra-Local secret — else refuse."""
    ip = effective_client_ip(conn)
    if is_loopback_ip(ip) or _secret_matches(conn):
        return True

    logger.warning("[local_trust] refused %s %s from %s",
                   conn.scope.get("type", "?"),
                   conn.scope.get("path", "?"), ip or "unknown")
    if conn.scope.get("type") == "websocket":
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                 reason="local-trusted endpoint")
    raise HTTPException(status_code=403,
                        detail="local-trusted endpoint: loopback or "
                               "X-Astra-Local required")


__all__ = ["require_local_or_secret"]

# FILE: app/security/client_ip.py
# Purpose: Shared client-IP resolution — proxy-aware, spoof-resistant. The ONE
#          place that decides when X-Forwarded-For may be believed.
# Called-by: app.security.firewall, app.auth.local_trust
# Depends-on: stdlib only
# Last-renovated: 2026-07-02
"""Effective-client-IP resolution (security hardening 2026-07-02).

X-Forwarded-For / X-Real-IP are ATTACKER-CONTROLLED unless the request
demonstrably came from our own reverse proxy. Rule: honour forwarding
headers ONLY when the raw socket peer is a trusted proxy address
(ASTRA_TRUSTED_PROXY_IPS, default loopback — the Caddy proxy is co-located).
Any other peer gets judged by its raw socket address, so a LAN client
sending "X-Forwarded-For: 127.0.0.1" straight at :8000 cannot impersonate
localhost.

"testclient" is accepted as loopback: it is starlette TestClient's synthetic
peer name and can never be produced by a real TCP socket (peers are numeric).
"""
from __future__ import annotations

import os
from ipaddress import ip_address
from typing import Optional

# Synthetic peer names that only in-process test harnesses can produce.
_SYNTHETIC_LOCAL = {"testclient", "localhost"}


def trusted_proxy_ips() -> set[str]:
    """Socket addresses allowed to speak for other clients via XFF.

    Env: ASTRA_TRUSTED_PROXY_IPS (comma-separated, default loopback).
    """
    raw = os.getenv("ASTRA_TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    return {p.strip() for p in raw.split(",") if p.strip()}


def is_loopback_ip(ip_str: Optional[str]) -> bool:
    """True for 127.0.0.0/8, ::1, IPv4-mapped loopback, and synthetic test peers."""
    if not ip_str:
        return False
    if ip_str in _SYNTHETIC_LOCAL:
        return True
    try:
        addr = ip_address(ip_str)
    except ValueError:
        return False
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def effective_client_ip(conn) -> str:
    """Resolve who is REALLY talking, given proxies may sit in front.

    `conn` is any starlette HTTPConnection (Request or WebSocket).
    Returns the forwarded-for client when (and only when) the raw peer is a
    trusted proxy; otherwise the raw socket peer address.
    """
    raw = conn.client.host if conn.client else ""
    if raw in trusted_proxy_ips():
        forwarded = conn.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = conn.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    return raw

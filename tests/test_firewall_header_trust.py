# FILE: tests/test_firewall_header_trust.py
# Purpose: Boot self-check for the firewall — spoofed XFF/X-Real-IP must NOT
#          impersonate localhost; legit proxied traffic passes; main.py wiring
#          + startup assertion actually exist (no silent dead-code regression).
# Called-by: pytest
# Depends-on: app.security.firewall, app.security.client_ip
# Last-renovated: 2026-07-02
"""Firewall header-trust tests (security hardening 2026-07-02).

The 2026-07-02 fix: _get_client_ip only honours X-Forwarded-For / X-Real-IP
when the raw socket peer is a trusted proxy (loopback). These tests pin that
behaviour with real requests through the middleware.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.security.firewall import FirewallMiddleware

SANDBOX_IP = "192.168.250.2"
LAN_IP = "192.168.1.50"


def _client(peer_ip: str) -> TestClient:
    app = FastAPI()

    @app.get("/probe")
    def probe():
        return {"ok": True}

    app.add_middleware(FirewallMiddleware)
    return TestClient(app, client=(peer_ip, 51515))


# ── spoof attempts (the hole this fix closes) ──────────────────────────────

def test_sandbox_peer_with_spoofed_xff_localhost_still_blocked():
    r = _client(SANDBOX_IP).get(
        "/probe", headers={"X-Forwarded-For": "127.0.0.1"})
    assert r.status_code == 403
    assert r.json()["code"] == "FIREWALL_BLOCKED"


def test_sandbox_peer_with_spoofed_x_real_ip_still_blocked():
    r = _client(SANDBOX_IP).get(
        "/probe", headers={"X-Real-IP": "127.0.0.1"})
    assert r.status_code == 403


def test_sandbox_peer_with_xff_chain_still_blocked():
    r = _client(SANDBOX_IP).get(
        "/probe", headers={"X-Forwarded-For": "127.0.0.1, 10.0.0.1"})
    assert r.status_code == 403


def test_lan_peer_spoofing_does_not_gain_localhost_status():
    """LAN isn't in blocked ranges, but the spoof must not be BELIEVED:
    the firewall must judge it by its raw peer IP, not the forged header."""
    app = FastAPI()

    @app.get("/probe")
    def probe():
        return {"ok": True}

    mw_holder = {}

    class _Capture(FirewallMiddleware):
        def _get_client_ip(self, request):
            ip = super()._get_client_ip(request)
            mw_holder["seen_ip"] = ip
            return ip

    app.add_middleware(_Capture)
    client = TestClient(app, client=(LAN_IP, 51515))
    client.get("/probe", headers={"X-Forwarded-For": "127.0.0.1"})
    assert mw_holder["seen_ip"] == LAN_IP


# ── legitimate traffic ─────────────────────────────────────────────────────

def test_direct_localhost_passes():
    assert _client("127.0.0.1").get("/probe").status_code == 200


def test_proxied_lan_client_passes():
    """Phone via the co-located proxy: loopback peer + real XFF → allowed
    (LAN isn't blocked; auth still applies at the endpoint layer)."""
    r = _client("127.0.0.1").get(
        "/probe", headers={"X-Forwarded-For": LAN_IP})
    assert r.status_code == 200


def test_proxied_sandbox_traffic_blocked():
    """Even via the trusted proxy, sandbox-range effective IPs are blocked."""
    r = _client("127.0.0.1").get(
        "/probe", headers={"X-Forwarded-For": SANDBOX_IP})
    assert r.status_code == 403


def test_sandbox_peer_plain_still_blocked():
    assert _client(SANDBOX_IP).get("/probe").status_code == 403


# ── wiring guard: the middleware must stay ACTIVE in main.py ───────────────

def test_main_py_registers_firewall_middleware():
    """2026-07-02: firewall was dead code (defined, never added). Pin the
    add_middleware call so it can't silently vanish again."""
    main_src = (Path(__file__).parent.parent / "main.py").read_text(
        encoding="utf-8", errors="replace")
    assert "app.add_middleware(FirewallMiddleware)" in main_src
    assert "Firewall middleware: ACTIVE" in main_src

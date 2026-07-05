# FILE: tests/test_local_trust.py
# Purpose: Security tests for app/auth/local_trust + app/security/client_ip —
#          loopback pass, foreign-IP refusal, secret header, XFF spoof defence.
# Called-by: pytest
# Depends-on: app.auth.local_trust, app.security.client_ip
# Last-renovated: 2026-07-02
"""require_local_or_secret behaviour (security hardening 2026-07-02).

Foreign peers are simulated with TestClient(client=(ip, port)) so the
dependency sees a real non-loopback socket address.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, WebSocketException
from fastapi.testclient import TestClient

from app.auth.local_trust import require_local_or_secret
from app.security.client_ip import effective_client_ip, is_loopback_ip


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(require_local_or_secret)])
    def probe():
        return {"ok": True}

    return app


def _foreign_client(app: FastAPI, ip: str = "192.168.1.99") -> TestClient:
    return TestClient(app, client=(ip, 51515))


# ── HTTP paths ─────────────────────────────────────────────────────────────

def test_loopback_passes():
    client = TestClient(_app())  # synthetic "testclient" peer = local
    assert client.get("/probe").status_code == 200


def test_real_loopback_ip_passes():
    with _foreign_client(_app(), ip="127.0.0.1") as client:
        assert client.get("/probe").status_code == 200


def test_foreign_ip_refused():
    with _foreign_client(_app()) as client:
        assert client.get("/probe").status_code == 403


def test_foreign_ip_with_secret_passes(monkeypatch):
    monkeypatch.setenv("ASTRA_LOCAL_SECRET", "s3cret-value")
    with _foreign_client(_app()) as client:
        r = client.get("/probe", headers={"X-Astra-Local": "s3cret-value"})
        assert r.status_code == 200


def test_foreign_ip_with_wrong_secret_refused(monkeypatch):
    monkeypatch.setenv("ASTRA_LOCAL_SECRET", "s3cret-value")
    with _foreign_client(_app()) as client:
        r = client.get("/probe", headers={"X-Astra-Local": "guess"})
        assert r.status_code == 403


def test_empty_secret_env_disables_header_path(monkeypatch):
    """Unset/empty ASTRA_LOCAL_SECRET must never make empty headers match."""
    monkeypatch.setenv("ASTRA_LOCAL_SECRET", "")
    with _foreign_client(_app()) as client:
        r = client.get("/probe", headers={"X-Astra-Local": ""})
        assert r.status_code == 403


def test_spoofed_xff_from_foreign_peer_refused():
    """A LAN client claiming to be localhost via XFF stays a LAN client."""
    with _foreign_client(_app()) as client:
        r = client.get("/probe", headers={"X-Forwarded-For": "127.0.0.1"})
        assert r.status_code == 403


def test_proxied_phone_traffic_not_auto_trusted():
    """Via the co-located proxy (loopback peer + XFF) the EFFECTIVE client is
    the phone — local trust must NOT apply."""
    with _foreign_client(_app(), ip="127.0.0.1") as client:
        r = client.get("/probe", headers={"X-Forwarded-For": "192.168.1.42"})
        assert r.status_code == 403


# ── WebSocket denial shape ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_websocket_scope_raises_ws_exception():
    from starlette.requests import HTTPConnection

    scope = {
        "type": "websocket",
        "path": "/astra/ws",
        "headers": [],
        "client": ("192.168.1.99", 51515),
        "query_string": b"",
    }
    with pytest.raises(WebSocketException):
        await require_local_or_secret(HTTPConnection(scope))


# ── client_ip unit behaviour ───────────────────────────────────────────────

class _FakeConn:
    def __init__(self, host, headers=None):
        self.client = type("C", (), {"host": host})() if host else None
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


def test_effective_ip_honours_xff_only_from_trusted_proxy():
    proxied = _FakeConn("127.0.0.1", {"X-Forwarded-For": "10.0.0.7"})
    assert effective_client_ip(proxied) == "10.0.0.7"

    direct = _FakeConn("192.168.1.50", {"X-Forwarded-For": "127.0.0.1"})
    assert effective_client_ip(direct) == "192.168.1.50"


def test_effective_ip_x_real_ip_fallback():
    proxied = _FakeConn("127.0.0.1", {"X-Real-IP": "10.0.0.8"})
    assert effective_client_ip(proxied) == "10.0.0.8"


def test_is_loopback_variants():
    assert is_loopback_ip("127.0.0.1")
    assert is_loopback_ip("::1")
    assert is_loopback_ip("::ffff:127.0.0.1")
    assert is_loopback_ip("testclient")
    assert not is_loopback_ip("192.168.1.10")
    assert not is_loopback_ip("not-an-ip")
    assert not is_loopback_ip("")
    assert not is_loopback_ip(None)

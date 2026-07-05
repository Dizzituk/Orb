# FILE: tests/test_crash_report_guard.py
# Purpose: Security tests for /bridge/crash-report — auth requirement, body
#          size cap (413), per-source rate limit (429).
# Called-by: pytest
# Depends-on: app.bridge.router, app.bridge.crash_guard
# Last-renovated: 2026-07-02
"""Crash-report guard tests (security hardening 2026-07-02).

Every request here is crafted to fail BEFORE the endpoint's email/disk
section (401 pre-handler, 413/429 pre-try) so the suite never writes to
D:\\Orb\\logs or triggers Proton mail.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.bridge import crash_guard
from app.bridge.router import router as bridge_router
from app.bridge.schemas import require_bridge_auth


@pytest.fixture(autouse=True)
def fresh_limiter():
    crash_guard.reset_for_tests()
    yield
    crash_guard.reset_for_tests()


def _client(auth_bypassed: bool) -> TestClient:
    app = FastAPI()
    app.include_router(bridge_router)
    if auth_bypassed:
        app.dependency_overrides[require_bridge_auth] = lambda: True
    return TestClient(app)


def test_unauthenticated_crash_report_rejected():
    r = _client(auth_bypassed=False).post(
        "/bridge/crash-report",
        json={"report": "boom", "app": "AstraBridge", "timestamp": 1},
    )
    assert r.status_code == 401


def test_oversized_body_rejected(monkeypatch):
    monkeypatch.setenv("ASTRA_CRASH_REPORT_MAX_KB", "1")
    big = "x" * (2 * 1024)
    r = _client(auth_bypassed=True).post(
        "/bridge/crash-report",
        json={"report": big, "app": "AstraBridge", "timestamp": 1},
    )
    assert r.status_code == 413


def test_rate_limit_kicks_in(monkeypatch):
    """Slots are consumed before the size check, so oversized bodies both
    exercise the limiter AND stay clear of the email/disk path."""
    monkeypatch.setenv("ASTRA_CRASH_REPORT_MAX_KB", "1")
    monkeypatch.setenv("ASTRA_CRASH_REPORT_PER_HOUR", "2")
    client = _client(auth_bypassed=True)
    big = {"report": "x" * (2 * 1024), "app": "AstraBridge", "timestamp": 1}

    assert client.post("/bridge/crash-report", json=big).status_code == 413
    assert client.post("/bridge/crash-report", json=big).status_code == 413
    assert client.post("/bridge/crash-report", json=big).status_code == 429

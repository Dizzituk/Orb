# FILE: tests/test_sandbox_controller_secret.py
# Purpose: Security tests for sandbox_controller v0.6.0 — bearer-secret gate
#          (fail-closed without env) + /shell/run argv[0] allow-list.
# Called-by: pytest
# Depends-on: sandbox_controller.main
# Last-renovated: 2026-07-02
"""Sandbox controller gate tests (security hardening 2026-07-02, Task 8).

The controller is a standalone FastAPI app (not under app/); imported here
via a path insert. All requests run in-process — nothing touches a real
sandbox. The manager's own flows wrap commands as
["powershell", "-NoProfile", "-Command", ...], which the allow-list admits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "sandbox_controller"))

import main as controller_main  # sandbox_controller/main.py

SECRET = "per-run-secret-for-tests"


@pytest.fixture()
def client():
    return TestClient(controller_main.app)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── fail closed without provisioning ───────────────────────────────────────

def test_unprovisioned_controller_is_locked(client, monkeypatch):
    monkeypatch.delenv("ASTRA_SANDBOX_SECRET", raising=False)
    assert client.get("/health").status_code == 503
    assert client.get("/health", headers=_bearer("anything")).status_code == 503


# ── bearer gate on every endpoint ──────────────────────────────────────────

def test_missing_bearer_rejected(client, monkeypatch):
    monkeypatch.setenv("ASTRA_SANDBOX_SECRET", SECRET)
    assert client.get("/health").status_code == 401
    assert client.post("/shell/run", json={"cmd": ["python", "-c", "1"]}).status_code == 401
    assert client.post("/fs/write", json={"path": "D:\\x.txt", "content": "x"}).status_code == 401


def test_wrong_bearer_rejected(client, monkeypatch):
    monkeypatch.setenv("ASTRA_SANDBOX_SECRET", SECRET)
    assert client.get("/health", headers=_bearer("wrong")).status_code == 401


def test_correct_bearer_passes(client, monkeypatch):
    monkeypatch.setenv("ASTRA_SANDBOX_SECRET", SECRET)
    r = client.get("/health", headers=_bearer(SECRET))
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /shell/run allow-list ──────────────────────────────────────────────────

def test_disallowed_binary_refused(client, monkeypatch):
    monkeypatch.setenv("ASTRA_SANDBOX_SECRET", SECRET)
    for evil in (["curl.exe", "http://x"], ["regedit"], ["C:\\evil\\payload.exe"]):
        r = client.post("/shell/run", json={"cmd": evil}, headers=_bearer(SECRET))
        assert r.status_code == 403, f"{evil[0]} should be refused"


def test_allowed_python_runs(client, monkeypatch):
    monkeypatch.setenv("ASTRA_SANDBOX_SECRET", SECRET)
    r = client.post(
        "/shell/run",
        json={"cmd": [sys.executable.split("\\")[-1].split("/")[-1], "-c", "print('gate-ok')"],
              "cwd": str(Path(__file__).parent)},
        headers=_bearer(SECRET),
    )
    assert r.status_code == 200
    assert "gate-ok" in r.json()["stdout"]


def test_powershell_wrapper_shape_is_admitted(client, monkeypatch):
    """The manager always sends ["powershell", "-NoProfile", "-Command", ...] —
    that shape must stay allow-listed or start/stop plumbing breaks."""
    monkeypatch.setenv("ASTRA_SANDBOX_SECRET", SECRET)
    r = client.post(
        "/shell/run",
        json={"cmd": ["powershell", "-NoProfile", "-Command", "Write-Output 'ps-ok'"],
              "timeout_sec": 30},
        headers=_bearer(SECRET),
    )
    assert r.status_code == 200
    assert "ps-ok" in r.json()["stdout"]


# ── no hardcoded master key anywhere live ──────────────────────────────────

def test_no_hardcoded_master_key_constant():
    """Task 9: the known-weak base64 constant must not exist in live code
    (QUARANTINE snapshots excluded — that's the undo trail)."""
    needle = "MDEy" + "MzQ1"  # split so this test file never matches itself
    repo = Path(__file__).parent.parent
    offenders = []
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        parts = {q.lower() for q in p.parts}
        if {".git", ".venv", "__pycache__", "node_modules", ".architecture",
                "_sandbox_cache", "data"} & parts:
            continue
        if p.suffix.lower() in {".db", ".sqlite", ".bin", ".pyc", ".exe", ".dll",
                                ".png", ".jpg", ".mp3", ".mp4", ".zip", ".onnx"}:
            continue
        try:
            if needle in p.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(p))
        except (OSError, PermissionError):
            continue
    assert offenders == [], f"hardcoded master key found in: {offenders}"

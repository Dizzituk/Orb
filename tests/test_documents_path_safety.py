# FILE: tests/test_documents_path_safety.py
# Purpose: Security tests for /documents — allowed-roots enforcement on
#          /open + /save (traversal, out-of-root, existence non-leak) and
#          the include-level auth gate.
# Called-by: pytest
# Depends-on: app.documents.router, app.drive.file_utils, app.auth
# Last-renovated: 2026-07-02
"""Path-safety + auth tests for the documents router (hardening 2026-07-02).

The router is mounted exactly as router_registry mounts it (include-level
require_auth). Path tests override require_auth so they exercise the
allowed-roots guard in isolation; the auth test leaves the gate in place.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.documents.router import router as documents_router
from app.drive import file_utils as _file_utils


def _build_app(auth_bypassed: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(documents_router, dependencies=[Depends(require_auth)])
    if auth_bypassed:
        app.dependency_overrides[require_auth] = lambda: AuthResult(authenticated=True)
    return app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Auth-bypassed client whose ONLY allowed root is tmp_path."""
    monkeypatch.setattr(_file_utils, "get_allowed_roots", lambda: [tmp_path])
    return TestClient(_build_app(auth_bypassed=True))


# ── allowed-roots enforcement ──────────────────────────────────────────────

def test_open_rejects_absolute_out_of_root(client):
    r = client.post("/documents/open", json={"path": r"C:\Windows\win.ini"})
    assert r.status_code == 403


def test_open_rejects_traversal(client, tmp_path):
    sneaky = str(tmp_path / ".." / ".." / "secret.docx")
    r = client.post("/documents/open", json={"path": sneaky})
    assert r.status_code == 403


def test_save_rejects_absolute_out_of_root(client):
    r = client.post("/documents/save", json={
        "path": r"C:\Windows\evil.md", "kind": "doc",
        "snapshot": {"body": {"dataStream": "x\n"}},
    })
    assert r.status_code == 403


def test_save_rejects_traversal(client, tmp_path):
    sneaky = str(tmp_path / ".." / "escape.md")
    r = client.post("/documents/save", json={
        "path": sneaky, "kind": "doc",
        "snapshot": {"body": {"dataStream": "x\n"}},
    })
    assert r.status_code == 403


def test_out_of_root_open_does_not_leak_existence(client):
    """403 (not 404) even for files that don't exist outside the roots."""
    r = client.post("/documents/open",
                    json={"path": r"C:\definitely\not\here.md"})
    assert r.status_code == 403


def test_in_root_round_trip_still_works(client, tmp_path):
    doc = tmp_path / "note.md"
    doc.write_text("# hello\n", encoding="utf-8")

    opened = client.post("/documents/open", json={"path": str(doc)})
    assert opened.status_code == 200
    body = opened.json()
    assert body["ok"] is True and body["kind"] == "doc"

    saved = client.post("/documents/save", json={
        "path": str(doc), "kind": "doc", "snapshot": body["snapshot"],
    })
    assert saved.status_code == 200
    assert saved.json()["ok"] is True
    assert doc.read_text(encoding="utf-8").strip() != ""


# ── include-level auth gate ────────────────────────────────────────────────

def test_documents_requires_auth_when_gated(tmp_path, monkeypatch):
    """Without a token the gate rejects before any path logic runs.

    401 with credentials missing, or 503 when no password is configured yet —
    either way the request never reaches the converters.
    """
    monkeypatch.setattr(_file_utils, "get_allowed_roots", lambda: [tmp_path])
    doc = tmp_path / "note.md"
    doc.write_text("# hello\n", encoding="utf-8")

    ungated = TestClient(_build_app(auth_bypassed=False))
    r = ungated.post("/documents/open", json={"path": str(doc)})
    assert r.status_code in (401, 503)

# FILE: tests/test_auth_hardening.py
# Purpose: Security tests for app/auth/config hardening — bcrypt mandatory
#          (fail closed), session TTL + pruning, min length 8, sha256
#          migrate-on-login.
# Called-by: pytest
# Depends-on: app.auth.config
# Last-renovated: 2026-07-02
"""Auth strength tests (security hardening 2026-07-02).

Every test redirects AUTH_CONFIG_PATH to tmp_path — the live data/auth.json
must never be touched by the suite.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta

import pytest

from app.auth import config as auth_config


@pytest.fixture(autouse=True)
def isolated_auth_store(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_config, "AUTH_CONFIG_PATH", tmp_path / "auth.json")
    yield


# ── bcrypt mandatory / fail closed ─────────────────────────────────────────

def test_hash_password_uses_bcrypt():
    h = auth_config._hash_password("a-strong-password")
    assert h.startswith("$2")


def test_hash_password_fails_closed_without_bcrypt(monkeypatch):
    monkeypatch.setattr(auth_config, "HAS_BCRYPT", False)
    with pytest.raises(RuntimeError):
        auth_config._hash_password("whatever-password")


def test_startup_check_fails_closed_without_bcrypt(monkeypatch):
    monkeypatch.setattr(auth_config, "HAS_BCRYPT", False)
    with pytest.raises(SystemExit):
        auth_config.assert_strong_hash_available()


def test_startup_check_passes_with_bcrypt():
    auth_config.assert_strong_hash_available()  # must not raise


# ── min password length ────────────────────────────────────────────────────

def test_setup_rejects_short_password():
    with pytest.raises(ValueError):
        auth_config.setup_password("seven77")  # 7 chars


def test_setup_accepts_eight_chars():
    result = auth_config.setup_password("eight888")
    assert result["session_token"].startswith("orb_session_")


def test_change_rejects_short_new_password():
    auth_config.setup_password("initial-password")
    with pytest.raises(ValueError):
        auth_config.change_password("initial-password", "short77")


# ── session TTL ────────────────────────────────────────────────────────────

def _age_all_sessions(hours: float) -> None:
    """Rewrite every stored session's created_at to `hours` ago."""
    cfg = json.loads(auth_config.AUTH_CONFIG_PATH.read_text())
    old = (datetime.now() - timedelta(hours=hours)).isoformat()
    for s in cfg.get("active_sessions", []):
        s["created_at"] = old
    if cfg.get("current_session"):
        cfg["current_session"]["created_at"] = old
    auth_config.AUTH_CONFIG_PATH.write_text(json.dumps(cfg))


def test_fresh_session_validates():
    token = auth_config.setup_password("initial-password")["session_token"]
    assert auth_config.validate_session(token) is True


def test_expired_session_rejected_and_pruned():
    token = auth_config.setup_password("initial-password")["session_token"]
    _age_all_sessions(hours=auth_config.DEFAULT_SESSION_TTL_HOURS + 24)

    assert auth_config.validate_session(token) is False
    # pruned on access — the stale rows are gone from disk
    cfg = json.loads(auth_config.AUTH_CONFIG_PATH.read_text())
    assert cfg.get("active_sessions") == []
    assert not cfg.get("current_session", {}).get("token")


def test_ttl_env_override(monkeypatch):
    monkeypatch.setenv(auth_config.SESSION_TTL_HOURS_ENV, "1")
    token = auth_config.setup_password("initial-password")["session_token"]
    _age_all_sessions(hours=2)
    assert auth_config.validate_session(token) is False


def test_session_without_created_at_is_expired():
    token = auth_config.setup_password("initial-password")["session_token"]
    cfg = json.loads(auth_config.AUTH_CONFIG_PATH.read_text())
    for s in cfg.get("active_sessions", []):
        s.pop("created_at", None)
    cfg.get("current_session", {}).pop("created_at", None)
    auth_config.AUTH_CONFIG_PATH.write_text(json.dumps(cfg))
    assert auth_config.validate_session(token) is False


# ── sha256 legacy verify + migrate-on-login ────────────────────────────────

def _write_sha256_store(password: str) -> None:
    salt = secrets.token_hex(16)
    hash_val = hashlib.sha256((salt + password).encode()).hexdigest()
    auth_config.AUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    auth_config.AUTH_CONFIG_PATH.write_text(json.dumps({
        "password_hash": f"sha256:{salt}:{hash_val}",
        "auth_type": "password",
    }))


def test_sha256_hash_migrates_to_bcrypt_on_login():
    _write_sha256_store("legacy-password")

    result = auth_config.login("legacy-password")
    assert result is not None

    cfg = json.loads(auth_config.AUTH_CONFIG_PATH.read_text())
    assert cfg["password_hash"].startswith("$2")
    assert cfg.get("hash_migrated_at")

    # and the migrated hash still verifies on a second login
    assert auth_config.login("legacy-password") is not None


def test_sha256_wrong_password_still_rejected():
    _write_sha256_store("legacy-password")
    assert auth_config.login("wrong-password") is None

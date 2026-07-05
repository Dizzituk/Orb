# FILE: tests/test_stage_roles.py
# Purpose: Derek phase 2 — stage-role registry resolution, provenance, fail-loud, endpoint truth.
# Called-by: pytest
# Depends-on: app.llm.stage_roles, app.settings.stage_roles_api
# Last-renovated: 2026-07-04
"""Gate 2 tests: canonical keys win, aliases warn, missing keys fail loudly,
and the introspection endpoint reports live truth."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llm import stage_roles
from app.llm.stage_roles import (
    STAGE_ROLES,
    StageRoleResolutionError,
    resolve_all_stage_roles,
    resolve_stage_role,
)


@pytest.fixture(autouse=True)
def _reset_warn_cache():
    stage_roles._deprecation_warned.clear()
    yield
    stage_roles._deprecation_warned.clear()


def test_all_roles_defined():
    assert set(STAGE_ROLES) == {
        "WEAVER_MAIN", "SPECGATE_MAIN", "SPECGATE_AGENT",
        "BUILDER_MAIN", "BUILDER_WORKER", "CHECKOUT_JUDGE", "CHECKOUT_EYES",
        "CHAT_MAIN", "DEBUG_MAIN",
    }


def test_canonical_key_wins_with_provenance(monkeypatch):
    monkeypatch.setenv("ASTRA_STAGE_BUILDER_MAIN_MODEL", "test-model-canonical")
    monkeypatch.setenv("ASTRA_STAGE_BUILDER_MAIN_PROVIDER", "openai")
    monkeypatch.setenv("ASTRA_V2_BUILDER_MODEL", "test-model-legacy")
    r = resolve_stage_role("BUILDER_MAIN")
    assert r.model == "test-model-canonical"
    assert r.model_source == "ASTRA_STAGE_BUILDER_MAIN_MODEL"
    assert r.provider_source == "ASTRA_STAGE_BUILDER_MAIN_PROVIDER"
    assert not r.deprecated_source


def test_legacy_alias_feeds_role_with_deprecation(monkeypatch, caplog):
    monkeypatch.delenv("ASTRA_STAGE_BUILDER_MAIN_MODEL", raising=False)
    monkeypatch.delenv("ASTRA_STAGE_BUILDER_MAIN_PROVIDER", raising=False)
    monkeypatch.setenv("ASTRA_V2_BUILDER_MODEL", "gpt-5.4")
    monkeypatch.setenv("ASTRA_V2_BUILDER_PROVIDER", "openai")
    with caplog.at_level("WARNING"):
        r = resolve_stage_role("BUILDER_MAIN")
    assert r.model == "gpt-5.4"
    assert r.model_source == "ASTRA_V2_BUILDER_MODEL"
    assert r.deprecated_source
    assert "DEPRECATED" in caplog.text


def test_missing_key_fails_loudly_naming_the_key(monkeypatch):
    monkeypatch.delenv("ASTRA_STAGE_BUILDER_WORKER_MODEL", raising=False)
    monkeypatch.delenv("ASTRA_STAGE_BUILDER_WORKER_PROVIDER", raising=False)
    with pytest.raises(StageRoleResolutionError) as exc:
        resolve_stage_role("BUILDER_WORKER")
    assert "ASTRA_STAGE_BUILDER_WORKER_MODEL" in str(exc.value)


def test_blank_key_is_missing_not_silent(monkeypatch):
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_MODEL", "   ")
    monkeypatch.delenv("ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER", raising=False)
    with pytest.raises(StageRoleResolutionError):
        resolve_stage_role("CHECKOUT_JUDGE")


def test_frontier_alias_resolves(monkeypatch):
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_MODEL", "anthropic:frontier-opus-thinking")
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("FRONTIER_ANTHROPIC_OPUS_THINKING_MODEL", "claude-opus-4-8")
    r = resolve_stage_role("CHECKOUT_JUDGE")
    assert r.model == "claude-opus-4-8"
    assert r.raw_model == "anthropic:frontier-opus-thinking"


def test_provider_inferred_when_absent(monkeypatch):
    monkeypatch.setenv("ASTRA_STAGE_SPECGATE_AGENT_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.delenv("ASTRA_STAGE_SPECGATE_AGENT_PROVIDER", raising=False)
    r = resolve_stage_role("SPECGATE_AGENT")
    assert r.provider == "anthropic"
    assert r.provider_source == "inferred from model"


def test_unknown_role_rejected():
    with pytest.raises(StageRoleResolutionError):
        resolve_stage_role("NOT_A_ROLE")


def test_resolve_all_never_raises(monkeypatch):
    monkeypatch.delenv("ASTRA_STAGE_CHECKOUT_EYES_MODEL", raising=False)
    monkeypatch.delenv("ASTRA_V2_VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("ASTRA_STAGE_CHECKOUT_EYES_PROVIDER", raising=False)
    results = resolve_all_stage_roles()
    assert len(results) == len(STAGE_ROLES)
    eyes = next(r for r in results if r.role == "CHECKOUT_EYES")
    assert "ASTRA_STAGE_CHECKOUT_EYES_MODEL" in eyes.error


def test_live_env_resolves_all_roles():
    """With the seeded .env loaded, every role must resolve cleanly."""
    from dotenv import load_dotenv
    load_dotenv("D:/Orb/.env", override=False)
    failures = [r for r in resolve_all_stage_roles() if r.error]
    assert not failures, f"unresolved roles against live .env: {[(r.role, r.error) for r in failures]}"


# ---------------------------------------------------------------------------
# Introspection endpoint — mounted standalone so the test needs no auth/boot
# ---------------------------------------------------------------------------

def _client() -> TestClient:
    from app.settings.stage_roles_api import stage_roles_router
    api = FastAPI()
    api.include_router(stage_roles_router, prefix="/settings")
    return TestClient(api)


def test_endpoint_returns_live_truth(monkeypatch):
    monkeypatch.setenv("ASTRA_STAGE_BUILDER_WORKER_MODEL", "endpoint-truth-model")
    monkeypatch.setenv("ASTRA_STAGE_BUILDER_WORKER_PROVIDER", "openai")
    body = _client().get("/settings/stage-models").json()
    assert body["count"] == len(STAGE_ROLES)
    worker = next(r for r in body["roles"] if r["role"] == "BUILDER_WORKER")
    assert worker["model"] == "endpoint-truth-model"
    assert worker["model_source"] == "ASTRA_STAGE_BUILDER_WORKER_MODEL"
    assert worker["tier"] == "HANDS"


# ---------------------------------------------------------------------------
# 2026-07-04 live fixes: mirror-on-save + CHAT/DEBUG utility roles
# ---------------------------------------------------------------------------

def test_save_mirrors_to_legacy_consumer_key(monkeypatch, tmp_path):
    """THE BUG: the Weaver reads WEAVER_MODEL, not the canonical key. A card
    save must mirror the value to the legacy consumer key so the running
    stage actually honours it."""
    import app.settings.env_model_store as store
    monkeypatch.setattr(store, "ENV_PATH", tmp_path / ".env")
    # setenv so reload_env's os.environ mutation is reverted at teardown
    monkeypatch.setenv("WEAVER_MODEL", "gpt-5.5")
    monkeypatch.setenv("WEAVER_PROVIDER", "openai")
    monkeypatch.setenv("ASTRA_STAGE_WEAVER_MAIN_MODEL", "gpt-5.5")
    (tmp_path / ".env").write_text(
        "WEAVER_MODEL=gpt-5.5\nWEAVER_PROVIDER=openai\n", encoding="utf-8"
    )
    resp = _client().put("/settings/stage-models/WEAVER_MAIN",
                         json={"model": "gpt-5.4", "provider": "openai"})
    assert resp.status_code == 200
    written = resp.json()["written"]
    # Both the canonical AND the legacy consumer key were written
    assert "ASTRA_STAGE_WEAVER_MAIN_MODEL" in written
    assert "WEAVER_MODEL" in written and written["WEAVER_MODEL"]["new"] == "gpt-5.4"
    # The Weaver's real read now returns gpt-5.4
    from app.llm._weaver_stream_utils_12 import _get_weaver_config
    assert _get_weaver_config() == ("openai", "gpt-5.4")


def test_chat_role_home_is_legacy_no_deprecation(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("CHAT_PROVIDER", "openai")
    r = resolve_stage_role("CHAT_MAIN")
    assert r.model == "gpt-5.4-mini"
    assert r.model_source == "CHAT_MODEL"       # its real home
    assert not r.deprecated_source              # NOT a deprecated alias
    assert r.tier == "UTILITY"


def test_chat_save_writes_chat_model(monkeypatch, tmp_path):
    import app.settings.env_model_store as store
    monkeypatch.setattr(store, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("CHAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("CHAT_PROVIDER", "openai")
    (tmp_path / ".env").write_text("CHAT_MODEL=gpt-5.4-mini\nCHAT_PROVIDER=openai\n", encoding="utf-8")
    resp = _client().put("/settings/stage-models/CHAT_MAIN",
                         json={"model": "claude-sonnet-5", "provider": "anthropic"})
    assert resp.status_code == 200
    written = resp.json()["written"]
    assert written["CHAT_MODEL"]["new"] == "claude-sonnet-5"
    assert "ASTRA_STAGE_CHAT_MAIN_MODEL" not in written  # no dead canonical key


def test_debug_role_honours_provider_switch(monkeypatch):
    """DEBUG_MAIN must show the ACTIVE column, never lie about the switch."""
    monkeypatch.setenv("DEBUG_PROVIDER", "anthropic")
    monkeypatch.setenv("DEBUG_CHAT_MODEL", "gpt-5.5")               # openai column
    monkeypatch.setenv("DEBUG_CHAT_MODEL_ANTHROPIC", "claude-sonnet-5")  # active
    r = resolve_stage_role("DEBUG_MAIN")
    assert r.provider == "anthropic"
    assert r.model == "claude-sonnet-5"          # the anthropic column, not gpt-5.5
    assert r.model_source == "DEBUG_CHAT_MODEL_ANTHROPIC"


def test_debug_save_sets_column_and_toggle(monkeypatch, tmp_path):
    import app.settings.env_model_store as store
    monkeypatch.setattr(store, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("DEBUG_PROVIDER", "openai")
    monkeypatch.setenv("DEBUG_CHAT_MODEL", "gpt-5.5")
    monkeypatch.setenv("DEBUG_CHAT_MODEL_ANTHROPIC", "claude-sonnet-5")
    (tmp_path / ".env").write_text(
        "DEBUG_PROVIDER=openai\nDEBUG_CHAT_MODEL=gpt-5.5\nDEBUG_CHAT_MODEL_ANTHROPIC=claude-sonnet-5\n",
        encoding="utf-8",
    )
    resp = _client().put("/settings/stage-models/DEBUG_MAIN",
                         json={"model": "claude-haiku-4-5-20251001", "provider": "anthropic"})
    assert resp.status_code == 200
    written = resp.json()["written"]
    assert written["DEBUG_PROVIDER"]["new"] == "anthropic"
    assert written["DEBUG_CHAT_MODEL_ANTHROPIC"]["new"] == "claude-haiku-4-5-20251001"


def test_endpoint_reports_error_instead_of_lying(monkeypatch):
    monkeypatch.delenv("ASTRA_STAGE_CHECKOUT_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER", raising=False)
    body = _client().get("/settings/stage-models").json()
    judge = next(r for r in body["roles"] if r["role"] == "CHECKOUT_JUDGE")
    assert judge["error"] and "ASTRA_STAGE_CHECKOUT_JUDGE_MODEL" in judge["error"]
    assert body["errors"] >= 1


def test_endpoint_put_rejects_unknown_role():
    resp = _client().put("/settings/stage-models/NOT_A_ROLE", json={"model": "x"})
    assert resp.status_code == 404


def test_allowed_models_lists(monkeypatch):
    monkeypatch.setenv("ASTRA_STAGE_ALLOWED_MODELS_OPENAI", "gpt-5.5, gpt-5.4-mini")
    body = _client().get("/settings/stage-models/allowed").json()
    assert body["allowed"]["openai"] == ["gpt-5.5", "gpt-5.4-mini"]
    assert "vllm_local" in body["allowed"]

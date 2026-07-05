# Purpose: Lane C tests — env-driven RAG model selection (no hardcoded tiers).
# Called-by: pytest
# Depends-on: app.rag._answerer_model_selection
# Last-renovated: 2026-07-01
# tests/test_rag_model_selection_env.py
"""
Tests for the de-hardcoded RAG model selection (Lane C, 2026-07-01).

select_rag_model resolution order:
    1. ORB_RAG_MODEL (manual override)
    2. RAG_<TIER>_PROVIDER/MODEL env vars per complexity tier
    3. DEFAULT_PROVIDER/DEFAULT_MODEL env vars
    4. ("", "") + error log — never a hardcoded literal

The complexity classifier is stubbed via sys.modules so these tests
exercise only the env resolution logic.
"""

import sys
import types
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from app.rag._answerer_model_selection import (
    RAG_TIER_ENV_VARS,
    select_rag_model,
)

_ALL_VARS = [
    "ORB_RAG_MODEL",
    "RAG_LOOKUP_PROVIDER", "RAG_LOOKUP_MODEL",
    "RAG_REASONING_PROVIDER", "RAG_REASONING_MODEL",
    "RAG_DEEP_PROVIDER", "RAG_DEEP_MODEL",
    "DEFAULT_PROVIDER", "DEFAULT_MODEL",
]


@pytest.fixture()
def clean_env(monkeypatch):
    """Strip every var this module reads, so each test starts blank."""
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture()
def stub_complexity(monkeypatch):
    """Install a stub app.memory.complexity whose tier the test controls."""
    stub = types.ModuleType("app.memory.complexity")
    state = {"tier": "lookup"}

    def classify_complexity(query):
        return types.SimpleNamespace(tier=state["tier"], confidence=0.9)

    stub.classify_complexity = classify_complexity
    monkeypatch.setitem(sys.modules, "app.memory.complexity", stub)
    return state


def test_override_wins_and_infers_provider(clean_env, stub_complexity):
    clean_env.setenv("ORB_RAG_MODEL", "claude-test-model")
    provider, model, tier = select_rag_model("anything")
    assert (provider, model, tier) == ("anthropic", "claude-test-model", "override")


def test_tier_env_vars_resolve(clean_env, stub_complexity):
    clean_env.setenv("RAG_LOOKUP_PROVIDER", "openai")
    clean_env.setenv("RAG_LOOKUP_MODEL", "test-lookup-model")
    stub_complexity["tier"] = "lookup"
    provider, model, tier = select_rag_model("where is the tts queue?")
    assert (provider, model, tier) == ("openai", "test-lookup-model", "lookup")


def test_ping_pong_shares_lookup_vars(clean_env, stub_complexity):
    clean_env.setenv("RAG_LOOKUP_MODEL", "test-lookup-model")
    stub_complexity["tier"] = "ping_pong"
    _, model, tier = select_rag_model("hi")
    assert model == "test-lookup-model" and tier == "ping_pong"


def test_multimodal_shares_deep_vars(clean_env, stub_complexity):
    clean_env.setenv("RAG_DEEP_PROVIDER", "anthropic")
    clean_env.setenv("RAG_DEEP_MODEL", "test-deep-model")
    stub_complexity["tier"] = "multimodal"
    provider, model, _ = select_rag_model("analyse this screenshot of the router")
    assert (provider, model) == ("anthropic", "test-deep-model")


def test_missing_provider_var_infers_from_model(clean_env, stub_complexity):
    clean_env.setenv("RAG_REASONING_MODEL", "gemini-test-model")
    stub_complexity["tier"] = "reasoning"
    provider, model, _ = select_rag_model("why does the router escalate?")
    assert (provider, model) == ("google", "gemini-test-model")


def test_missing_tier_model_falls_back_to_default_env(clean_env, stub_complexity):
    clean_env.setenv("DEFAULT_PROVIDER", "openai")
    clean_env.setenv("DEFAULT_MODEL", "test-default-model")
    stub_complexity["tier"] = "deep"
    provider, model, tier = select_rag_model("audit the whole memory pipeline")
    assert (provider, model, tier) == ("openai", "test-default-model", "deep")


def test_unknown_tier_falls_back_to_default_env(clean_env, stub_complexity):
    clean_env.setenv("DEFAULT_MODEL", "claude-test-default")
    stub_complexity["tier"] = "some_new_tier"
    provider, model, _ = select_rag_model("question")
    assert (provider, model) == ("anthropic", "claude-test-default")


def test_nothing_configured_returns_empty_not_literal(clean_env, stub_complexity):
    stub_complexity["tier"] = "lookup"
    provider, model, tier = select_rag_model("question")
    assert (provider, model) == ("", "")  # loud downstream failure > rotted literal
    assert tier == "lookup"


def test_classifier_failure_uses_default_env(clean_env, monkeypatch):
    stub = types.ModuleType("app.memory.complexity")

    def classify_complexity(query):
        raise RuntimeError("classifier down")

    stub.classify_complexity = classify_complexity
    monkeypatch.setitem(sys.modules, "app.memory.complexity", stub)
    clean_env.setenv("DEFAULT_PROVIDER", "openai")
    clean_env.setenv("DEFAULT_MODEL", "test-default-model")
    provider, model, tier = select_rag_model("question")
    assert (provider, model, tier) == ("openai", "test-default-model", "fallback")


def test_tier_map_covers_all_classifier_tiers():
    assert set(RAG_TIER_ENV_VARS) == {
        "ping_pong", "lookup", "reasoning", "deep", "multimodal",
    }

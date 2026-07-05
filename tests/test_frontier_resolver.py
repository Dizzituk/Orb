# FILE: tests/test_frontier_resolver.py
# Purpose: Lane A (2026-07-01) — env-only model resolution: get_role_model
#          fallback chains, family-prefix inference, alias map compatibility.
# Called-by: pytest
# Depends-on: app.llm.frontier_models, app.llm.routing.cognitive_escalation,
#             app.llm.routing.tier_momentum
# Last-renovated: 2026-07-01
from __future__ import annotations

import pytest

from app.llm.frontier_models import (
    FRONTIER_ALIASES,
    all_aliases,
    get_provider_default_model,
    get_role_model,
    is_alias,
    resolve_model_alias,
)
from app.llm.routing.cognitive_escalation import is_small_model


_ALL_VARS = (
    "CHAT_PROVIDER", "CHAT_MODEL", "REASONING_PROVIDER", "REASONING_MODEL",
    "FRONTIER_PROVIDER", "FRONTIER_MODEL", "DEFAULT_PROVIDER", "DEFAULT_MODEL",
    "ARCHITECT_PROVIDER", "ARCHITECT_MODEL",
    "FRONTIER_OPENAI_FAST_MODEL", "ASTRA_VERIFIER_FAMILY_MODEL",
    "OPENAI_DEFAULT_MODEL", "ANTHROPIC_DEFAULT_MODEL", "GOOGLE_DEFAULT_MODEL",
    "ASTRA_SMALL_MODELS",
)


@pytest.fixture(autouse=True)
def _bare_env(monkeypatch):
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


# =========================================================================
# get_role_model
# =========================================================================

def test_role_pair_wins(monkeypatch):
    monkeypatch.setenv("REASONING_PROVIDER", "anthropic")
    monkeypatch.setenv("REASONING_MODEL", "claude-test")
    assert get_role_model("REASONING") == ("anthropic", "claude-test")


def test_fallback_role_chain(monkeypatch):
    monkeypatch.setenv("FRONTIER_PROVIDER", "openai")
    monkeypatch.setenv("FRONTIER_MODEL", "gpt-test")
    assert get_role_model("COGNITIVE_ESCALATION", "FRONTIER") == ("openai", "gpt-test")


def test_default_pair_is_last_resort(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-test-mini")
    assert get_role_model("ARCHITECT") == ("openai", "gpt-test-mini")


def test_family_inference_beats_default_provider(monkeypatch):
    """2026-07-01 review fix: a model-only role var must get its FAMILY's
    provider, not the global DEFAULT_PROVIDER — otherwise uncommenting only
    AGENTIC_CONTEXT_MODEL=claude-... in .env would call OpenAI with an
    Anthropic model ID."""
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("REASONING_MODEL", "claude-test-model")
    assert get_role_model("REASONING") == ("anthropic", "claude-test-model")


def test_role_model_may_be_an_alias(monkeypatch):
    monkeypatch.setenv("FRONTIER_OPENAI_FAST_MODEL", "gpt-test-fast")
    monkeypatch.setenv("CHAT_MODEL", "openai:frontier-fast")
    assert get_role_model("CHAT") == ("openai", "gpt-test-fast")


def test_missing_everything_fails_loud():
    with pytest.raises(RuntimeError) as exc:
        get_role_model("CHAT")
    msg = str(exc.value)
    assert "CHAT" in msg and ".env" in msg


# =========================================================================
# get_provider_default_model
# =========================================================================

def test_provider_default_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_DEFAULT_MODEL", "gemini-test")
    assert get_provider_default_model("google") == "gemini-test"


def test_provider_default_falls_back_to_default_model(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-test-mini")
    assert get_provider_default_model("anthropic") == "gpt-test-mini"


def test_provider_default_fails_loud():
    with pytest.raises(RuntimeError):
        get_provider_default_model("openai")


# =========================================================================
# Alias map compatibility (stage_models / capability_manifest contract)
# =========================================================================

def test_alias_map_mapping_semantics(monkeypatch):
    monkeypatch.setenv("ASTRA_VERIFIER_FAMILY_MODEL", "claude-test-verifier")
    assert "anthropic:frontier-verifier" in FRONTIER_ALIASES
    assert FRONTIER_ALIASES["anthropic:frontier-verifier"] == "claude-test-verifier"
    assert is_alias("anthropic:frontier-verifier") is True
    assert is_alias("gpt-test") is False
    assert len(dict(FRONTIER_ALIASES)) == len(FRONTIER_ALIASES) > 0
    assert set(all_aliases()) == set(FRONTIER_ALIASES)


def test_resolve_model_alias_env_backed(monkeypatch):
    monkeypatch.setenv("ASTRA_VERIFIER_FAMILY_MODEL", "claude-test-verifier")
    assert resolve_model_alias("anthropic:frontier-verifier") == "claude-test-verifier"
    # Concrete IDs and unknown strings pass through unchanged.
    assert resolve_model_alias("gpt-test") == "gpt-test"
    assert resolve_model_alias("weird:alias") == "weird:alias"


def test_unresolved_alias_passes_through_unchanged():
    # No env var, no DEFAULT_MODEL: loud error log, alias returned unchanged
    # (documented pass-through; provider errors then carry the alias name).
    assert resolve_model_alias("openai:frontier-fast") == "openai:frontier-fast"


def test_unresolved_alias_uses_default_model(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-test-mini")
    assert resolve_model_alias("openai:frontier-fast") == "gpt-test-mini"


# =========================================================================
# Small-model classification (env + tier-marker heuristic)
# =========================================================================

def test_small_models_env_list(monkeypatch):
    monkeypatch.setenv("ASTRA_SMALL_MODELS", "my-tiny-model, another-small")
    assert is_small_model("my-tiny-model") is True
    assert is_small_model("another-small") is True


@pytest.mark.parametrize("model,expected", [
    ("gpt-5.4-mini", True),
    ("gpt-5-nano", True),
    ("gemini-2.5-flash", True),
    ("gemini-2.5-flash-lite", True),
    ("claude-haiku-4-5", True),
    ("gemini-3.1-pro-preview", False),   # 'mini' inside 'gemini' must NOT match
    ("gemini-2.5-pro", False),
    ("claude-opus-4-8", False),
    ("claude-sonnet-5", False),
    ("gpt-5.4", False),
    ("", False),
])
def test_small_model_tier_markers(model, expected):
    assert is_small_model(model) is expected

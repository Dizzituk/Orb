# FILE: tests/test_debug_model_config.py
# Purpose: Debug provider toggle (2026-07-02) — provider-aware resolution in
#          debug_model_config (columns, intra-column fallbacks, never-cross rule,
#          missing-key soft-fail) + model_router call-time resolution.
# Called-by: pytest
# Depends-on: app.debug.debug_model_config, app.debug.model_router
# Last-renovated: 2026-07-02
from __future__ import annotations

import pytest

from app.debug import debug_model_config as cfg
from app.debug.model_router import (
    DebugTier,
    TIER_REASONING,
    classify_query,
    tier_model_config,
)

# Every env var the resolver reads — cleared before each test so the suite is
# hermetic regardless of what conftest / a developer shell loaded from .env.
_ALL_VARS = [
    "DEBUG_PROVIDER",
    "OPENAI_DEFAULT_MODEL", "ANTHROPIC_DEFAULT_MODEL", "ANTHROPIC_API_KEY",
    "DEBUG_CHAT_MODEL", "DEBUG_ANALYSIS_MODEL", "DEBUG_AGENTIC_MODEL",
    "DEBUG_DECOMPOSER_MODEL", "DEBUG_PLANNER_MODEL", "DEBUG_SUBAGENT_MODEL",
    "DEBUG_EXECUTOR_MODEL", "DEBUG_CODE_VERIFIER_MODEL", "DEBUG_COMPILER_MODEL",
    "DEBUG_CHAT_MODEL_ANTHROPIC", "DEBUG_ANALYSIS_MODEL_ANTHROPIC",
    "DEBUG_AGENTIC_MODEL_ANTHROPIC", "DEBUG_DECOMPOSER_MODEL_ANTHROPIC",
    "DEBUG_PLANNER_MODEL_ANTHROPIC", "DEBUG_SUBAGENT_MODEL_ANTHROPIC",
    "DEBUG_EXECUTOR_MODEL_ANTHROPIC", "DEBUG_CODE_VERIFIER_MODEL_ANTHROPIC",
    "DEBUG_COMPILER_MODEL_ANTHROPIC",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


def _select_anthropic(monkeypatch, key: str = "test-key"):
    monkeypatch.setenv("DEBUG_PROVIDER", "anthropic")
    if key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)


# ─────────────────────────────────────────────────────────────────────────────
# Provider switch semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_provider_defaults_to_openai_when_unset():
    assert cfg.get_debug_provider() == "openai"
    assert cfg.get_active_provider() == "openai"
    assert cfg.provider_fallback_active() is False


def test_provider_openai_explicit(monkeypatch):
    monkeypatch.setenv("DEBUG_PROVIDER", "openai")
    assert cfg.get_debug_provider() == "openai"


def test_unknown_provider_falls_to_openai(monkeypatch):
    monkeypatch.setenv("DEBUG_PROVIDER", "banana")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert cfg.get_debug_provider() == "openai"
    assert cfg.get_active_provider() == "openai"
    assert cfg.provider_fallback_active() is False  # openai selected, not a fallback


def test_provider_value_normalised(monkeypatch):
    monkeypatch.setenv("DEBUG_PROVIDER", "  Anthropic ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert cfg.get_debug_provider() == "anthropic"
    assert cfg.get_active_provider() == "anthropic"


def test_missing_key_soft_fail(monkeypatch):
    """anthropic selected + no key -> active provider openai, flag raised."""
    monkeypatch.setenv("DEBUG_PROVIDER", "anthropic")
    assert cfg.get_debug_provider() == "anthropic"
    assert cfg.get_active_provider() == "openai"
    assert cfg.provider_fallback_active() is True


def test_missing_key_resolution_uses_openai_column(monkeypatch):
    monkeypatch.setenv("DEBUG_PROVIDER", "anthropic")  # no key set
    monkeypatch.setenv("DEBUG_CHAT_MODEL", "oai-brain")
    monkeypatch.setenv("DEBUG_CHAT_MODEL_ANTHROPIC", "anthro-brain")
    assert cfg.resolve_debug_model("chat") == ("openai", "oai-brain")


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI column — the legacy chains, unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_openai_role_var_wins(monkeypatch):
    monkeypatch.setenv("DEBUG_CHAT_MODEL", "oai-brain")
    assert cfg.resolve_debug_model("chat") == ("openai", "oai-brain")


def test_openai_executor_chains_to_subagent_then_default(monkeypatch):
    monkeypatch.setenv("DEBUG_SUBAGENT_MODEL", "oai-sub")
    assert cfg.resolve_debug_model("executor") == ("openai", "oai-sub")
    monkeypatch.setenv("DEBUG_EXECUTOR_MODEL", "oai-exec")
    assert cfg.resolve_debug_model("executor") == ("openai", "oai-exec")


def test_openai_chain_ends_at_default_then_last_resort(monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "oai-default")
    assert cfg.resolve_debug_model("subagent") == ("openai", "oai-default")
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    assert cfg.resolve_debug_model("subagent") == ("openai", cfg._LAST_RESORT)


def test_openai_analysis_falls_to_chat(monkeypatch):
    monkeypatch.setenv("DEBUG_CHAT_MODEL", "oai-brain")
    assert cfg.resolve_debug_model("analysis") == ("openai", "oai-brain")
    monkeypatch.setenv("DEBUG_ANALYSIS_MODEL", "oai-analysis")
    assert cfg.resolve_debug_model("analysis") == ("openai", "oai-analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic column
# ─────────────────────────────────────────────────────────────────────────────

def test_anthropic_role_var_wins(monkeypatch):
    _select_anthropic(monkeypatch)
    monkeypatch.setenv("DEBUG_CHAT_MODEL_ANTHROPIC", "anthro-brain")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "anthro-default")
    assert cfg.resolve_debug_model("chat") == ("anthropic", "anthro-brain")


def test_anthropic_intra_column_fallbacks(monkeypatch):
    _select_anthropic(monkeypatch)
    monkeypatch.setenv("DEBUG_SUBAGENT_MODEL_ANTHROPIC", "anthro-sub")
    # executor + verifier fall to the anthropic subagent var
    assert cfg.resolve_debug_model("executor") == ("anthropic", "anthro-sub")
    assert cfg.resolve_debug_model("verifier") == ("anthropic", "anthro-sub")
    # compiler falls executor -> subagent
    assert cfg.resolve_debug_model("compiler") == ("anthropic", "anthro-sub")
    monkeypatch.setenv("DEBUG_EXECUTOR_MODEL_ANTHROPIC", "anthro-exec")
    assert cfg.resolve_debug_model("compiler") == ("anthropic", "anthro-exec")
    # analysis/agentic fall to the anthropic chat var
    monkeypatch.setenv("DEBUG_CHAT_MODEL_ANTHROPIC", "anthro-brain")
    assert cfg.resolve_debug_model("analysis") == ("anthropic", "anthro-brain")
    assert cfg.resolve_debug_model("agentic") == ("anthropic", "anthro-brain")


def test_anthropic_column_backstop_is_anthropic_default(monkeypatch):
    _select_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "anthro-default")
    for kind in ("chat", "decomposer", "planner", "subagent", "executor",
                 "verifier", "compiler", "default"):
        assert cfg.resolve_debug_model(kind) == ("anthropic", "anthro-default")


def test_no_cross_provider_fallback(monkeypatch):
    """A populated openai column must NEVER leak into an anthropic chain —
    the anthropic side resolves only _ANTHROPIC vars + ANTHROPIC_DEFAULT_MODEL."""
    _select_anthropic(monkeypatch)
    monkeypatch.setenv("DEBUG_EXECUTOR_MODEL", "oai-exec")
    monkeypatch.setenv("DEBUG_SUBAGENT_MODEL", "oai-sub")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "oai-default")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "anthro-default")
    provider, model = cfg.resolve_debug_model("executor")
    assert provider == "anthropic"
    assert model == "anthro-default"          # never oai-exec / oai-sub / oai-default


def test_whole_column_empty_falls_back_to_openai_column(monkeypatch):
    """Key present but the entire anthropic column (incl. the default) is empty:
    the WHOLE resolution falls to the openai column — provider and model stay
    consistent, never a half-crossed pair."""
    _select_anthropic(monkeypatch)
    monkeypatch.setenv("DEBUG_EXECUTOR_MODEL", "oai-exec")
    assert cfg.resolve_debug_model("executor") == ("openai", "oai-exec")


def test_get_model_for_role_returns_provider_tuples(monkeypatch):
    monkeypatch.setenv("DEBUG_SUBAGENT_MODEL", "oai-sub")
    monkeypatch.setenv("DEBUG_EXECUTOR_MODEL", "oai-exec")
    monkeypatch.setenv("DEBUG_CODE_VERIFIER_MODEL", "oai-verify")
    assert cfg.get_model_for_role("investigator") == ("openai", "oai-sub")
    assert cfg.get_model_for_role("pattern_matcher") == ("openai", "oai-sub")
    assert cfg.get_model_for_role("executor") == ("openai", "oai-exec")
    assert cfg.get_model_for_role("code_verifier") == ("openai", "oai-verify")
    assert cfg.get_model_for_role("behaviour_verifier") == ("openai", "oai-verify")
    assert cfg.get_model_for_role("???") == ("openai", "oai-sub")  # unknown -> subagent tier


def test_debug_role_env_var_tracks_selected_column(monkeypatch):
    assert cfg.debug_role_env_var("chat") == "DEBUG_CHAT_MODEL"
    monkeypatch.setenv("DEBUG_PROVIDER", "anthropic")  # no key — still the SELECTED column
    assert cfg.debug_role_env_var("chat") == "DEBUG_CHAT_MODEL_ANTHROPIC"
    assert cfg.debug_role_env_var("verifier") == "DEBUG_CODE_VERIFIER_MODEL_ANTHROPIC"


# ─────────────────────────────────────────────────────────────────────────────
# model_router — call-time resolution (the import-time freeze is dead)
# ─────────────────────────────────────────────────────────────────────────────

def test_router_resolves_at_call_time(monkeypatch):
    """Flipping env mid-process changes the very next decision — no restart."""
    monkeypatch.setenv("DEBUG_ANALYSIS_MODEL", "oai-analysis")
    d1 = classify_query("why does the boot hang?")
    assert d1.tier == DebugTier.ANALYSIS
    assert (d1.provider, d1.model) == ("openai", "oai-analysis")

    _select_anthropic(monkeypatch)
    monkeypatch.setenv("DEBUG_ANALYSIS_MODEL_ANTHROPIC", "anthro-analysis")
    d2 = classify_query("why does the boot hang?")
    assert (d2.provider, d2.model) == ("anthropic", "anthro-analysis")

    monkeypatch.setenv("DEBUG_PROVIDER", "openai")
    d3 = classify_query("why does the boot hang?")
    assert (d3.provider, d3.model) == ("openai", "oai-analysis")


def test_router_openai_model_edit_applies_without_restart(monkeypatch):
    """The pre-existing bug this fixes: an OpenAI-side model change used to
    freeze at import. Now it applies on the next classification."""
    monkeypatch.setenv("DEBUG_AGENTIC_MODEL", "oai-agentic-v1")
    d1 = classify_query("fix it please")
    assert d1.tier == DebugTier.AGENTIC and d1.model == "oai-agentic-v1"
    monkeypatch.setenv("DEBUG_AGENTIC_MODEL", "oai-agentic-v2")
    d2 = classify_query("fix it please")
    assert d2.model == "oai-agentic-v2"


def test_router_triage_uses_default_tier(monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "oai-default")
    d = classify_query("hello")
    assert d.tier == DebugTier.TRIAGE
    assert (d.provider, d.model) == ("openai", "oai-default")
    assert d.reasoning is None
    _select_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "anthro-default")
    d2 = classify_query("hello")
    assert (d2.provider, d2.model) == ("anthropic", "anthro-default")


def test_tier_reasoning_stays_provider_agnostic():
    assert TIER_REASONING[DebugTier.TRIAGE] is None
    assert TIER_REASONING[DebugTier.ANALYSIS] == {"effort": "high"}
    assert TIER_REASONING[DebugTier.AGENTIC] == {"effort": "high"}


def test_tier_model_config_shape(monkeypatch):
    monkeypatch.setenv("DEBUG_CHAT_MODEL", "oai-brain")
    cfg_dict = tier_model_config(DebugTier.ANALYSIS)
    assert set(cfg_dict) == {"provider", "model"}
    assert cfg_dict == {"provider": "openai", "model": "oai-brain"}

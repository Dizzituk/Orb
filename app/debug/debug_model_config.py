# FILE: app/debug/debug_model_config.py
# Purpose: Single source of truth for Debug-surface model selection. Every Debug
#          model (orchestrator brain, analysis/agentic tiers, decomposer/planner,
#          spawned sub-agents, compiler) resolves here from ENV. DEBUG_PROVIDER
#          selects a column: 'openai' = the legacy DEBUG_* vars (unchanged);
#          'anthropic' = the parallel DEBUG_*_ANTHROPIC vars. Resolvers return
#          (provider, model) so call sites never assume a provider.
# Called-by: app.debug.debug_chat, app.llm.stream_router, app.debug.model_router,
#            app.debug.orchestrator.{spawn_tool,loop_controller,report_compiler},
#            app.debug.debug_lock_stream, app.llm.model_roles (settings rows)
# Depends-on: stdlib only
# Last-renovated: 2026-07-02
"""Central model resolution for the Debug tab + sub-agent orchestration.

HARD RULE (spec section 3): no hardcoded model IDs at selection sites. The only
literal model string lives in `_LAST_RESORT` below (openai column only), and even
that is reached only when OPENAI_DEFAULT_MODEL is unset. The openai column chains:

    DEBUG_<thing>_MODEL  ->  (sometimes a broader DEBUG_* default)
                         ->  OPENAI_DEFAULT_MODEL  ->  _LAST_RESORT

The anthropic column (DEBUG_PROVIDER=anthropic) chains the same intra-column
shape and ends at ANTHROPIC_DEFAULT_MODEL:

    DEBUG_<thing>_MODEL_ANTHROPIC  ->  (same role fallbacks, _ANTHROPIC suffixed)
                                   ->  ANTHROPIC_DEFAULT_MODEL

Fallback chains NEVER cross providers mid-chain. The only provider crossover is
whole-column: if DEBUG_PROVIDER=anthropic but the Anthropic key is missing (or
the whole anthropic column is empty), resolution falls back to the complete
openai column with provider "openai" — a loud warning is logged and callers can
surface it via provider_fallback_active().

Resolution happens at call time (functions, not import-time constants) so an ENV
change — including the settings-UI DEBUG_PROVIDER toggle — takes effect on the
next debug message without a restart.

Fan-out tier (Taz, 2026-06-17): thinking-heavy roles run the subagent tier;
executors run the cheaper executor tier; verifiers run the verifier tier. On the
anthropic column the same shape maps to Sonnet (thinking) + Haiku (doing).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# The ONE place a bare model literal is allowed. Matches the historical Debug
# triage default. Reached only if OPENAI_DEFAULT_MODEL itself is unset.
# (openai column only — anthropic models come exclusively from env.)
_LAST_RESORT = "gpt-5.4-mini"

_KNOWN_PROVIDERS = ("openai", "anthropic")


def _env(name: str) -> str:
    """Return a stripped ENV value, or '' if unset/blank."""
    return (os.getenv(name) or "").strip()


# --- Provider switch ---------------------------------------------------------

def get_debug_provider() -> str:
    """The SELECTED provider column: DEBUG_PROVIDER, default 'openai'.

    Unknown values resolve to 'openai' with a warning (the debug tab must never
    brick because of a toggle). This is the raw toggle — use
    get_active_provider() for the key-gated provider a call should actually use.
    """
    raw = _env("DEBUG_PROVIDER").lower()
    if not raw or raw == "openai":
        return "openai"
    if raw == "anthropic":
        return "anthropic"
    logger.warning(
        "[debug_model_config] Unknown DEBUG_PROVIDER=%r (expected one of %s) — using 'openai'",
        raw, "/".join(_KNOWN_PROVIDERS),
    )
    return "openai"


def _anthropic_key_present() -> bool:
    return bool(_env("ANTHROPIC_API_KEY"))


def provider_fallback_active() -> bool:
    """True when the anthropic column is selected but unusable (no API key) —
    debug calls are falling back to the openai column. Surfaces the missing-key
    soft-fail to SSE/narration streams."""
    return get_debug_provider() == "anthropic" and not _anthropic_key_present()


def get_active_provider() -> str:
    """The provider debug calls should ACTUALLY use right now.

    'anthropic' only when both selected AND the key is available at call time;
    otherwise 'openai' (loud warning — the debug tab must keep working).
    """
    if get_debug_provider() == "anthropic":
        if _anthropic_key_present():
            return "anthropic"
        logger.warning(
            "[debug_model_config] DEBUG_PROVIDER=anthropic but ANTHROPIC_API_KEY "
            "is unavailable — debug falling back to the OpenAI column for this call"
        )
    return "openai"


# --- OpenAI column (the legacy chains — behaviour unchanged) -----------------

def default_model() -> str:
    """Global last-resort model: OPENAI_DEFAULT_MODEL -> _LAST_RESORT."""
    return _env("OPENAI_DEFAULT_MODEL") or _LAST_RESORT


def get_debug_chat_model() -> str:
    """The Debug-tab orchestrator brain. DEBUG_CHAT_MODEL -> OPENAI_DEFAULT_MODEL.

    Intended: gpt-5.5 (set DEBUG_CHAT_MODEL=gpt-5.5 in .env).
    """
    return _env("DEBUG_CHAT_MODEL") or default_model()


def get_debug_analysis_model() -> str:
    """Rigid-pipeline ANALYSIS tier. DEBUG_ANALYSIS_MODEL -> DEBUG_CHAT_MODEL -> default."""
    return _env("DEBUG_ANALYSIS_MODEL") or get_debug_chat_model()


def get_debug_agentic_model() -> str:
    """Rigid-pipeline AGENTIC tier. DEBUG_AGENTIC_MODEL -> DEBUG_CHAT_MODEL -> default."""
    return _env("DEBUG_AGENTIC_MODEL") or get_debug_chat_model()


def get_decomposer_model() -> str:
    """Orchestration decomposer. DEBUG_DECOMPOSER_MODEL -> OPENAI_DEFAULT_MODEL.

    (Moved here from loop_controller 2026-07-02 — was frozen at import time
    there, with a stray literal fallback.)
    """
    return _env("DEBUG_DECOMPOSER_MODEL") or default_model()


def get_planner_model() -> str:
    """Orchestration planner. DEBUG_PLANNER_MODEL -> OPENAI_DEFAULT_MODEL."""
    return _env("DEBUG_PLANNER_MODEL") or default_model()


def get_subagent_model_default() -> str:
    """Investigators / pattern-matchers. DEBUG_SUBAGENT_MODEL -> OPENAI_DEFAULT_MODEL."""
    return _env("DEBUG_SUBAGENT_MODEL") or default_model()


def get_executor_model() -> str:
    """Executors. DEBUG_EXECUTOR_MODEL -> DEBUG_SUBAGENT_MODEL -> default."""
    return _env("DEBUG_EXECUTOR_MODEL") or get_subagent_model_default()


def get_code_verifier_model() -> str:
    """Verifiers. DEBUG_CODE_VERIFIER_MODEL -> DEBUG_SUBAGENT_MODEL -> default."""
    return _env("DEBUG_CODE_VERIFIER_MODEL") or get_subagent_model_default()


def get_compiler_model() -> str:
    """Report compiler that digests sub-agent fan-out reports into one short brief
    before the orchestrator brain reflects on them (phase-narration spec).

    DEBUG_COMPILER_MODEL -> DEBUG_EXECUTOR_MODEL -> DEBUG_SUBAGENT_MODEL -> default.
    Intended: a cheap model (the executor tier) — the digest is small and
    frequent, so it must not cost brain-tier money. Point DEBUG_COMPILER_MODEL at
    a local model (Ollama) instead once one is installed.
    """
    return _env("DEBUG_COMPILER_MODEL") or get_executor_model()


# --- Provider-aware resolution (the new public surface) ----------------------
# A "kind" names a debug role slot. Each kind has: an openai-column resolver
# (the legacy functions above, untouched), an anthropic-column env chain
# (own var first, then the same intra-column role fallbacks), and its own
# writable env var per column (settings rows edit the active column).

_OPENAI_RESOLVERS = {
    "chat": get_debug_chat_model,
    "analysis": get_debug_analysis_model,
    "agentic": get_debug_agentic_model,
    "decomposer": get_decomposer_model,
    "planner": get_planner_model,
    "subagent": get_subagent_model_default,
    "executor": get_executor_model,
    "verifier": get_code_verifier_model,
    "compiler": get_compiler_model,
    "default": default_model,
}

# Intra-column fallback chains for the anthropic side (mirrors the openai
# chains; every chain implicitly ends at ANTHROPIC_DEFAULT_MODEL).
_ANTHROPIC_CHAINS: Dict[str, Tuple[str, ...]] = {
    "chat": ("DEBUG_CHAT_MODEL_ANTHROPIC",),
    "analysis": ("DEBUG_ANALYSIS_MODEL_ANTHROPIC", "DEBUG_CHAT_MODEL_ANTHROPIC"),
    "agentic": ("DEBUG_AGENTIC_MODEL_ANTHROPIC", "DEBUG_CHAT_MODEL_ANTHROPIC"),
    "decomposer": ("DEBUG_DECOMPOSER_MODEL_ANTHROPIC",),
    "planner": ("DEBUG_PLANNER_MODEL_ANTHROPIC",),
    "subagent": ("DEBUG_SUBAGENT_MODEL_ANTHROPIC",),
    "executor": ("DEBUG_EXECUTOR_MODEL_ANTHROPIC", "DEBUG_SUBAGENT_MODEL_ANTHROPIC"),
    "verifier": ("DEBUG_CODE_VERIFIER_MODEL_ANTHROPIC", "DEBUG_SUBAGENT_MODEL_ANTHROPIC"),
    "compiler": ("DEBUG_COMPILER_MODEL_ANTHROPIC", "DEBUG_EXECUTOR_MODEL_ANTHROPIC",
                 "DEBUG_SUBAGENT_MODEL_ANTHROPIC"),
    "default": (),
}

# The role slot's OWN var per kind (openai spelling; anthropic = + "_ANTHROPIC").
# Used by the settings inventory so edits write the active column.
_OWN_VARS: Dict[str, str] = {
    "chat": "DEBUG_CHAT_MODEL",
    "analysis": "DEBUG_ANALYSIS_MODEL",
    "agentic": "DEBUG_AGENTIC_MODEL",
    "decomposer": "DEBUG_DECOMPOSER_MODEL",
    "planner": "DEBUG_PLANNER_MODEL",
    "subagent": "DEBUG_SUBAGENT_MODEL",
    "executor": "DEBUG_EXECUTOR_MODEL",
    "verifier": "DEBUG_CODE_VERIFIER_MODEL",
    "compiler": "DEBUG_COMPILER_MODEL",
}


def _anthropic_model_for(kind: str) -> str:
    for var in _ANTHROPIC_CHAINS.get(kind, ()):
        val = _env(var)
        if val:
            return val
    return _env("ANTHROPIC_DEFAULT_MODEL")


def resolve_debug_model(kind: str) -> Tuple[str, str]:
    """(provider, model) for a debug role kind, honouring DEBUG_PROVIDER.

    Kinds: chat, analysis, agentic, decomposer, planner, subagent, executor,
    verifier, compiler, default. Unknown kinds resolve as 'subagent'.
    """
    kind = (kind or "").strip().lower()
    if kind not in _OPENAI_RESOLVERS:
        kind = "subagent"
    if get_active_provider() == "anthropic":
        model = _anthropic_model_for(kind)
        if model:
            return "anthropic", model
        logger.warning(
            "[debug_model_config] anthropic column empty for kind=%s "
            "(set DEBUG_*_MODEL_ANTHROPIC or ANTHROPIC_DEFAULT_MODEL) — "
            "using the openai column",
            kind,
        )
    return "openai", _OPENAI_RESOLVERS[kind]()


def debug_role_env_var(kind: str) -> str:
    """The env var a settings edit for this kind should write, in the SELECTED
    column (raw DEBUG_PROVIDER — even when the key-gated fallback is active,
    edits target the column the user chose)."""
    base = _OWN_VARS.get((kind or "").strip().lower(), _OWN_VARS["subagent"])
    return base + "_ANTHROPIC" if get_debug_provider() == "anthropic" else base


# Role.value -> kind. Role strings match SubagentRole.value in
# app/debug/orchestrator/schemas.py, kept as plain strings here so this module
# stays dependency-free (stdlib only). Unknown/blank roles fall back to the
# investigator (subagent) tier.
_ROLE_KINDS = {
    "investigator": "subagent",
    "pattern_matcher": "subagent",
    "executor": "executor",
    "code_verifier": "verifier",
    "behaviour_verifier": "verifier",
}


def get_model_for_role(role: str) -> Tuple[str, str]:
    """Resolve a spawned sub-agent's (provider, model) from its role string."""
    kind = _ROLE_KINDS.get((role or "").strip().lower(), "subagent")
    return resolve_debug_model(kind)

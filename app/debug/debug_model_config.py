# FILE: app/debug/debug_model_config.py
# Purpose: Single source of truth for Debug-surface model selection. Every Debug
#          model (orchestrator brain, analysis/agentic tiers, spawned sub-agents)
#          resolves here from ENV with fallback chains to OPENAI_DEFAULT_MODEL, so
#          there are no hardcoded model IDs at call sites and an ENV change
#          re-points the model with no code edit.
# Called-by: app.debug.debug_chat, app.llm.stream_router, app.debug.model_router,
#            app.debug.orchestrator.spawn_tool (WI-2)
# Depends-on: stdlib only
# Last-renovated: 2026-06-17
"""Central model resolution for the Debug tab + sub-agent orchestration.

HARD RULE (spec section 3): no hardcoded model IDs at selection sites. The only
literal model string lives in `_LAST_RESORT` below, and even that is reached only
when OPENAI_DEFAULT_MODEL is unset. Everything else chains:

    DEBUG_<thing>_MODEL  ->  (sometimes a broader DEBUG_* default)
                         ->  OPENAI_DEFAULT_MODEL  ->  _LAST_RESORT

Resolution happens at call time (functions, not import-time constants) so an ENV
change -- or a per-request override -- takes effect without code edits.

Fan-out tier (Taz, 2026-06-17): thinking-heavy roles (investigator,
pattern_matcher) run DEBUG_SUBAGENT_MODEL (intended gpt-5.5); executors run the
cheaper DEBUG_EXECUTOR_MODEL (intended gpt-5.4-mini); verifiers run
DEBUG_CODE_VERIFIER_MODEL.
"""

from __future__ import annotations

import os

# The ONE place a bare model literal is allowed. Matches the historical Debug
# triage default. Reached only if OPENAI_DEFAULT_MODEL itself is unset.
_LAST_RESORT = "gpt-5.4-mini"


def _env(name: str) -> str:
    """Return a stripped ENV value, or '' if unset/blank."""
    return (os.getenv(name) or "").strip()


def default_model() -> str:
    """Global last-resort model: OPENAI_DEFAULT_MODEL -> _LAST_RESORT."""
    return _env("OPENAI_DEFAULT_MODEL") or _LAST_RESORT


# --- Orchestrator brain + rigid-pipeline tiers ------------------------------

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


# --- Spawned sub-agent models (per role) ------------------------------------
# Role strings match SubagentRole.value in app/debug/orchestrator/schemas.py,
# kept as plain strings here so this module stays dependency-free (stdlib only).

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
    Intended: a cheap model (the executor tier, gpt-5.4-mini) -- the digest is small
    and frequent, so it must not cost brain-tier money. Point DEBUG_COMPILER_MODEL at
    a local model (Ollama) instead once one is installed.
    """
    return _env("DEBUG_COMPILER_MODEL") or get_executor_model()


# Role.value -> resolver. Unknown/blank roles fall back to the investigator tier.
_ROLE_RESOLVERS = {
    "investigator": get_subagent_model_default,
    "pattern_matcher": get_subagent_model_default,
    "executor": get_executor_model,
    "code_verifier": get_code_verifier_model,
    "behaviour_verifier": get_code_verifier_model,
}


def get_model_for_role(role: str) -> str:
    """Resolve a spawned sub-agent's model from its role string."""
    resolver = _ROLE_RESOLVERS.get((role or "").strip().lower(), get_subagent_model_default)
    return resolver()

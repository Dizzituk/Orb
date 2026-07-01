# FILE: app/llm/frontier_models.py
# Purpose: Frontier Model Resolution — THE single env-driven resolver: reads .env,
#          resolves aliases/roles, returns concrete models. Zero hardcoded model IDs.
# Called-by: app.llm.stage_models, app.pipeline_v2.config, app.llm.routing.*,
#            app.scene_director, app.sentinel.triage, app.self_model.capability_manifest
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-07-01
"""
Frontier Model Resolution — single source of truth for "which model".

Purpose
-------
Lane A de-hardcode (2026-07-01): NO model string is hardcoded anywhere in
this module (or the chat-routing modules that call it). Every model choice
resolves from `.env` (loaded by main.py at startup, before anything reads
os.getenv). Changing any model = editing only `.env`.

Two resolution surfaces:

1. Capability aliases (pipeline stages) — ``"{provider}:frontier-<cap>"``
   strings resolve through an env var per alias (see _ALIAS_ENV below).
   `resolve_model_alias()` keeps its historical contract: concrete model
   IDs pass through unchanged; unknown strings pass through unchanged.
   A KNOWN alias whose env var is unset falls back to DEFAULT_MODEL; if
   that is also unset the alias is returned unchanged with a loud error
   log (consumers like stage_models will then surface the alias name in
   their provider errors — visible, attributable, non-crashing).

2. Chat-routing roles — `get_role_model("ARCHITECT")` reads
   ``ARCHITECT_PROVIDER`` / ``ARCHITECT_MODEL``, then any fallback roles,
   then ``DEFAULT_PROVIDER`` / ``DEFAULT_MODEL``, then fails LOUD with a
   RuntimeError naming the exact vars to set. Never a literal.

Hard constraint — NO Pro-tier GPT models
-----------------------------------------
Taz's explicit rule: never route any ASTRA stage to a GPT Pro tier. That
rule now lives in `.env` (the values assigned there) — when updating
`.env`, preserve it.

Reasoning / thinking — enabled by default on load-bearing stages
----------------------------------------------------------------
Per-stage reasoning defaults live here (STAGE_REASONING). Light
classifier / summariser stages stay cheap (no reasoning).

v2.0 (2026-07-01): Lane A de-hardcode. FRONTIER_ALIASES became an
    env-backed Mapping; added get_role_model()/get_provider_default_model()
    as the one resolver for chat routing.
v1.0 (2026-04-18): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Frontier alias table — alias -> .env var (EDIT MODELS IN .env, NOT HERE)
# ─────────────────────────────────────────────────────────────────────────────
# Format: "{provider}:{capability}" -> env var holding the concrete model id.
#
# Rules when updating:
#   1. Keep aliases stable; models change in .env only.
#   2. NEVER point a frontier alias at a Pro-tier OpenAI model (in .env).
#   3. Resolution order per alias: its env var -> DEFAULT_MODEL -> alias
#      returned unchanged + error log.
#
_ALIAS_ENV: Dict[str, str] = {
    # ── OpenAI (non-Pro) ──────────────────────────────────────────────
    # Primary agentic builder model (reasoning enabled at call time).
    "openai:frontier-reasoning":        "FRONTIER_OPENAI_REASONING_MODEL",
    # Cheaper/faster pass (simple classification, summarisation).
    "openai:frontier-fast":             "FRONTIER_OPENAI_FAST_MODEL",

    # ── Anthropic ─────────────────────────────────────────────────────
    # Primary thinking model for SpecGate, Overwatcher, Verifier.
    "anthropic:frontier-opus-thinking": "FRONTIER_ANTHROPIC_OPUS_THINKING_MODEL",
    # Same tier without thinking routed in.
    "anthropic:frontier-opus":          "FRONTIER_ANTHROPIC_OPUS_MODEL",
    # JOB 12 (2026-06-10): THE verifier-family slot — one .env line
    # (ASTRA_VERIFIER_FAMILY_MODEL) overrides the whole family.
    "anthropic:frontier-verifier":      "ASTRA_VERIFIER_FAMILY_MODEL",
    # Mid-tier Claude for lighter work (job-checker, compaction).
    "anthropic:frontier-sonnet":        "FRONTIER_ANTHROPIC_SONNET_MODEL",

    # ── Google ────────────────────────────────────────────────────────
    # Vision / fast classifier / summariser tier.
    "google:frontier-flash":            "FRONTIER_GOOGLE_FLASH_MODEL",
    # Tiniest tier for high-volume classification.
    "google:frontier-flash-lite":       "FRONTIER_GOOGLE_FLASH_LITE_MODEL",
}


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _resolve_alias_value(alias: str) -> Optional[str]:
    """Resolve a KNOWN alias via its env var, then DEFAULT_MODEL. None if unset."""
    var = _ALIAS_ENV.get(alias)
    if var is None:
        return None
    value = _env(var)
    if value:
        return value
    default = _env("DEFAULT_MODEL")
    if default:
        logger.warning(
            "[frontier_models] %s unset — alias %r falling back to DEFAULT_MODEL=%s",
            var, alias, default,
        )
        return default
    logger.error(
        "[frontier_models] UNRESOLVED alias %r: set %s (or DEFAULT_MODEL) in .env",
        alias, var,
    )
    return None


class _EnvAliasMap(Mapping):
    """Mapping-compatible view of the alias table, resolved from .env on access.

    Keeps the historical FRONTIER_ALIASES contract for consumers that do
    ``alias in FRONTIER_ALIASES`` / ``FRONTIER_ALIASES[alias]`` /
    ``dict(FRONTIER_ALIASES)``. A known alias whose env vars are unset
    resolves to itself (documented pass-through, error already logged).
    """

    def __getitem__(self, alias: str) -> str:
        if alias not in _ALIAS_ENV:
            raise KeyError(alias)
        return _resolve_alias_value(alias) or alias

    def __iter__(self) -> Iterator[str]:
        return iter(_ALIAS_ENV)

    def __len__(self) -> int:
        return len(_ALIAS_ENV)


FRONTIER_ALIASES: Mapping = _EnvAliasMap()


# ─────────────────────────────────────────────────────────────────────────────
# Role resolver — THE entry point for chat routing (Lane A, 2026-07-01)
# ─────────────────────────────────────────────────────────────────────────────

# Provider inference from a model id's family prefix. These are provider
# names (allowed), not model IDs — used only when a role has a model but
# no provider var and no DEFAULT_PROVIDER.
_PROVIDER_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
)


def _infer_provider(model: str) -> Optional[str]:
    lower = (model or "").strip().lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return provider
    return None


def get_role_model(role: str, *fallback_roles: str) -> Tuple[str, str]:
    """Resolve (provider, model) for a routing role, env-only, fail-loud.

    Reads ``{ROLE}_PROVIDER`` / ``{ROLE}_MODEL`` for the role, then each
    fallback role in order, then ``DEFAULT_PROVIDER`` / ``DEFAULT_MODEL``.
    Model values may be frontier aliases — they resolve through
    resolve_model_alias first. A model without a provider var resolves via
    family-prefix inference (the model's own family is a stronger signal
    than the global default), then DEFAULT_PROVIDER. If nothing resolves,
    raises RuntimeError naming the exact .env vars to set — never a
    hardcoded literal.
    """
    for r in (role, *fallback_roles):
        key = r.upper()
        model = _env(f"{key}_MODEL")
        if not model:
            continue
        model = resolve_model_alias(model)
        provider = (
            _env(f"{key}_PROVIDER")
            or _infer_provider(model)
            or _env("DEFAULT_PROVIDER")
        )
        if provider:
            return provider, model
        raise RuntimeError(
            f"[frontier_models] {key}_MODEL={model!r} is set but no provider "
            f"resolves: set {key}_PROVIDER or DEFAULT_PROVIDER in .env"
        )
    default_model = _env("DEFAULT_MODEL")
    if default_model:
        default_model = resolve_model_alias(default_model)
        provider = _infer_provider(default_model) or _env("DEFAULT_PROVIDER")
        if provider:
            return provider, default_model
    tried = " -> ".join([role.upper(), *[r.upper() for r in fallback_roles]])
    raise RuntimeError(
        f"[frontier_models] No model configured for role {tried}: set "
        f"{role.upper()}_PROVIDER/{role.upper()}_MODEL (or DEFAULT_PROVIDER/"
        f"DEFAULT_MODEL) in .env — models are env-only, never hardcoded."
    )


def get_provider_default_model(provider: str) -> str:
    """Default model for a provider, env-only (``{PROVIDER}_DEFAULT_MODEL``).

    Falls back to DEFAULT_MODEL; raises RuntimeError if neither is set.
    Used by availability fallback when swapping to a different provider.
    """
    key = (provider or "").strip().upper()
    model = _env(f"{key}_DEFAULT_MODEL") or _env("DEFAULT_MODEL")
    if model:
        return model
    raise RuntimeError(
        f"[frontier_models] No default model for provider {provider!r}: set "
        f"{key}_DEFAULT_MODEL or DEFAULT_MODEL in .env"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage reasoning defaults
# ─────────────────────────────────────────────────────────────────────────────
# Maps STAGE_NAME -> reasoning config dict, or None to disable reasoning
# for that stage. Config shape matches what providers.registry expects:
#   {"effort": "low" | "medium" | "high" | "xhigh" | "max"}
#
# Stages absent from this table fall back to _DEFAULT_REASONING (None —
# i.e. no reasoning) so that new stages added without thought don't
# silently start burning extra tokens.
#
# Taz's stance: load-bearing stages (Weaver, SpecGate, Builder, Verifier)
# ALL get reasoning=high because a wrong answer there corrupts the whole
# downstream pipeline. Cheap stages that classify / summarise stay off.
#
STAGE_REASONING: Dict[str, Optional[Dict[str, str]]] = {
    # ── Load-bearing stages: reasoning ON ─────────────────────────────
    "WEAVER":              {"effort": "high"},
    "SPEC_GATE":           {"effort": "high"},
    "CRITICAL_PIPELINE":   {"effort": "high"},   # Agentic Builder
    "IMPLEMENTER":         {"effort": "high"},
    "CRITICAL_SUPERVISOR": {"effort": "high"},
    "PLANNER":             {"effort": "high"},
    "ARCHITECTURE":        {"effort": "high"},
    "CRITIQUE":            {"effort": "high"},
    "REVISION":            {"effort": "high"},
    "OVERWATCHER":         {"effort": "high"},   # Verifier second-opinion
    "SPEC_REVIEW":         {"effort": "high"},   # Always-on spec reviewer
    "VERIFIER_AGENT":      {"effort": "high"},   # JOB 11: agentic checkout

    # Needle-based architecture tiers: medium for the cheaper route,
    # high for the premium route.
    "ARCH_TIER_LOW":       {"effort": "medium"},
    "ARCH_TIER_HIGH":      {"effort": "high"},

    # ── Light stages: reasoning OFF ───────────────────────────────────
    # These stages run at high volume or on trivial inputs. Enabling
    # reasoning here wastes tokens without moving the quality needle.
    "CLASSIFIER":          None,
    "SUMMARIZER":          None,
    "NEEDLE_CLASSIFIER":   None,
    "CHAT":                None,
    "OVERWATCH":           None,
    "WEAVER_COMPACTION":   None,
    "COHESION_CHECK":      None,
    "COHESION_MICRO_PATCH": None,
    "SMART_SEGMENTATION":  None,
    "JOB_CHECKER":         None,
    "ARCHMAP":             None,
}

# Fallback when a stage has no explicit entry above.
_DEFAULT_REASONING: Optional[Dict[str, str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resolve_model_alias(model: str) -> str:
    """Resolve a frontier alias to a concrete model ID (from .env).

    Aliases look like ``"provider:frontier-<capability>"``. Anything
    that doesn't match a known alias is returned unchanged, so concrete
    model IDs from env vars keep working without modification. A known
    alias with no env value (and no DEFAULT_MODEL) is returned unchanged
    after an error log — see module docstring.

    Args:
        model: Either an alias (e.g. ``"openai:frontier-reasoning"``)
               or a concrete model ID.

    Returns:
        Concrete model ID (or the input unchanged). Never None, never
        empty unless input was.
    """
    if not model:
        return model
    stripped = model.strip()
    if stripped in _ALIAS_ENV:
        resolved = _resolve_alias_value(stripped)
        if resolved:
            logger.debug("[frontier_models] Resolved alias %s -> %s", stripped, resolved)
            return resolved
        return stripped
    return stripped


def get_reasoning_for_stage(stage: str) -> Optional[Dict[str, str]]:
    """Return the reasoning config for a stage, or None if disabled.

    Args:
        stage: Stage name (e.g. ``"WEAVER"``, ``"SPEC_GATE"``).

    Returns:
        ``{"effort": "<level>"}`` dict, or None if reasoning is
        explicitly disabled / the stage is unknown.
    """
    if not stage:
        return _DEFAULT_REASONING
    stage_upper = stage.upper().replace("-", "_").replace(" ", "_")
    # v1.3 (2026-06-11): per-stage env override, e.g. SPEC_GATE_REASONING_EFFORT=xhigh.
    # Valid: low | medium | high | xhigh | max. Use none/off to disable reasoning
    # for the stage. Invalid values are ignored with a warning (table default wins).
    _env_effort = os.getenv(f"{stage_upper}_REASONING_EFFORT", "").strip().lower()
    if _env_effort:
        if _env_effort in ("none", "off", "0", "disabled"):
            return None
        if _env_effort in ("low", "medium", "high", "xhigh", "max"):
            return {"effort": _env_effort}
        logger.warning(
            "[frontier_models] Ignoring invalid %s_REASONING_EFFORT=%r",
            stage_upper, _env_effort,
        )
    if stage_upper in STAGE_REASONING:
        return STAGE_REASONING[stage_upper]
    return _DEFAULT_REASONING


def is_alias(model: str) -> bool:
    """True if ``model`` is a known frontier alias."""
    return bool(model) and model.strip() in _ALIAS_ENV


def all_aliases() -> Dict[str, str]:
    """Return the fully-resolved alias table (for audit/debug).

    Unresolved aliases appear as themselves (error already logged).
    """
    return dict(FRONTIER_ALIASES)


def all_stage_reasoning() -> Dict[str, Optional[Dict[str, str]]]:
    """Return a copy of the stage-reasoning table (for audit/debug)."""
    return dict(STAGE_REASONING)


# ─────────────────────────────────────────────────────────────────────────────
# Provider-list audit helper (optional, NOT called at runtime)
# ─────────────────────────────────────────────────────────────────────────────
# Kept here so there's a single script you can run to check whether the
# frontier has moved. It queries each provider's model-list endpoint and
# prints suggestions. Never called from the pipeline — running it on
# every pipeline start would add fragility.
#
# Usage:
#     python -m app.llm.frontier_models
#

def _audit_openai() -> None:
    """Print current OpenAI model list and flag likely newer frontier."""
    import os
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai SDK not installed")
        return
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("  OPENAI_API_KEY not set")
        return
    try:
        client = OpenAI(api_key=key, timeout=15)
        models = client.models.list()
        ids = sorted(
            (m.id for m in models.data if m.id.startswith("gpt-")),
            reverse=True,
        )
        print("  Latest gpt-* models (newest first):")
        for mid in ids[:10]:
            is_pro = "pro" in mid.lower()
            tag = " [PRO — skip]" if is_pro else ""
            print(f"    - {mid}{tag}")
    except Exception as exc:
        print(f"  OpenAI model list failed: {exc}")


def _audit_anthropic() -> None:
    """Print current Anthropic model list and flag likely newer frontier."""
    import os
    try:
        import anthropic
    except ImportError:
        print("  anthropic SDK not installed")
        return
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        print("  ANTHROPIC_API_KEY not set")
        return
    try:
        client = anthropic.Anthropic(api_key=key, timeout=15)
        models = client.models.list(limit=20)
        ids = sorted((m.id for m in models.data), reverse=True)
        print("  Latest Claude models (newest first):")
        for mid in ids[:10]:
            print(f"    - {mid}")
    except Exception as exc:
        print(f"  Anthropic model list failed: {exc}")


def audit_frontier() -> None:
    """Print current configured aliases and probe providers for newer models.

    Run as:  python -m app.llm.frontier_models
    """
    print("=" * 70)
    print("FRONTIER MODEL AUDIT")
    print("=" * 70)
    print("\nCurrent aliases (resolved from .env):")
    for alias in sorted(_ALIAS_ENV):
        print(f"  {alias:<42} -> {FRONTIER_ALIASES[alias]}  [{_ALIAS_ENV[alias]}]")
    print("\n--- OpenAI available models ---")
    _audit_openai()
    print("\n--- Anthropic available models ---")
    _audit_anthropic()
    print("\n" + "=" * 70)
    print("If a newer non-Pro GPT or newer Opus appears above, update the")
    print("corresponding env var in .env (shown in brackets), then re-run")
    print("to confirm the new mapping is picked up. No code edits needed.")
    print("=" * 70)


if __name__ == "__main__":
    audit_frontier()


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "FRONTIER_ALIASES",
    "STAGE_REASONING",
    "resolve_model_alias",
    "get_role_model",
    "get_provider_default_model",
    "get_reasoning_for_stage",
    "is_alias",
    "all_aliases",
    "all_stage_reasoning",
    "audit_frontier",
]

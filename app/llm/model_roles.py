# FILE: app/llm/model_roles.py
# Purpose: THE role inventory for model settings — every routing role the
#          resolver knows: subsystem group, env vars, verifier/restart-gated
#          flags, and the LIVE resolution (what would actually run right now).
# Called-by: app.settings.model_settings (GET/PUT /settings/models)
# Depends-on: app.llm.frontier_models, app.llm.stage_models
# Last-renovated: 2026-07-02
"""
LANE D (2026-07-02) — Task 4. Single source for "which roles exist".

Three role shapes:
  * standard  — {ROLE}_PROVIDER / {ROLE}_MODEL env vars, resolved live via
                frontier_models.get_role_model (exactly what routing runs).
  * stage     — pipeline stages enumerated from stage_models.STAGE_DEFAULTS,
                resolved via stage_models.get_stage_config (alias-aware).
  * env       — a single var-shaped env slot (e.g. GEMINI_VISION_MODEL_FAST)
                with a locked provider; resolved by reading the var.
  * alias     — frontier alias slots from frontier_models._ALIAS_ENV.

The CHAT pipeline stage and the CHAT chat-tier role share the same
CHAT_PROVIDER/CHAT_MODEL env slot, so CHAT is listed ONCE (Chat Tiers) and
excluded from the stage enumeration.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from app.llm.frontier_models import (
    FRONTIER_ALIASES,
    _ALIAS_ENV,
    get_role_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Groups (display order for the UI)
# ─────────────────────────────────────────────────────────────────────────────

GROUP_ORDER: Tuple[str, ...] = (
    "Chat Tiers",
    "Pipeline",
    "Debug Assistant",
    "RAG",
    "Memory",
    "Scene",
    "Image",
    "Nat-Local",
    "Core Defaults",
    "Frontier Aliases",
)

# ─────────────────────────────────────────────────────────────────────────────
# Verifier roles — model changes here alter pass/fail behaviour (UI locks them
# behind a confirm dialog; jobspec Task 5b).
# ─────────────────────────────────────────────────────────────────────────────

VERIFIER_ROLES = {
    "SPEC_GATE", "OVERWATCHER", "OVERWATCH", "SPEC_REVIEW", "VERIFIER_AGENT",
    "CRITIQUE", "JOB_CHECKER", "DEBUG_CODE_VERIFIER", "SCENE_CRITIC",
    "anthropic:frontier-verifier",
}

# ─────────────────────────────────────────────────────────────────────────────
# Restart-gated roles — at least one consumer freezes the value at import time
# (Task 2c audit). Everything not listed applies on the NEXT REQUEST after a
# .env reload because the resolver reads env on access.
# ─────────────────────────────────────────────────────────────────────────────

RESTART_GATED: Dict[str, str] = {
    "GEMINI_FRONTIER": "import-time constant in job_classifier (OVERRIDE routing)",
    "ANTHROPIC_FRONTIER": "import-time constant in job_classifier (OVERRIDE routing)",
    "OPENAI_FRONTIER": "import-time constant in job_classifier (OVERRIDE routing)",
    "ARCHMAP_TRIGGER": "import-time class attribute in legacy_triggers/translation_routing",
    "ORB_ARCHMAP": "import-time constant in local_tools/archmap_helpers",
    "ORB_ARCHMAP_FALLBACK": "import-time constant in local_tools/archmap_helpers",
    "GEMINI_CRITIC": "module constant in pipeline/critique_parts/model_config",
    "ASTRA_V2_BUILDER": "pipeline_v2/config resolves at import",
    "ASTRA_V2_FALLBACK": "pipeline_v2/config resolves at import",
    "IMAGE_GEN": "partial: dispatch re-reads env per call, but routing/_prompt_blocks "
                 "bakes the model name into prompt text at import",
}

# Alias slots: stage_models re-reads per call, but pipeline_v2/config.py froze
# copies at import (Lane A note) — flag the whole alias group.
_ALIAS_RESTART_NOTE = (
    "partial: stage_models resolves per call; pipeline_v2/config froze alias "
    "values at import"
)

# ─────────────────────────────────────────────────────────────────────────────
# Hand-listed role specs (standard + env shapes). Stages and aliases are
# enumerated programmatically below.
# ─────────────────────────────────────────────────────────────────────────────

_STANDARD: List[Dict[str, Any]] = [
    # Chat Tiers (Lane A resolver roles)
    {"role": "CHAT", "group": "Chat Tiers", "desc": "TIER 1 everyday chat model (also the CHAT pipeline stage)"},
    {"role": "ARCHITECT", "group": "Chat Tiers", "desc": "TIER 4 architecture/deep-design chat"},
    {"role": "REASONING", "group": "Chat Tiers", "desc": "TIER 2 pure-reasoning chat", "fallbacks": ("FRONTIER",)},
    {"role": "IMAGE_VISION", "group": "Chat Tiers", "desc": "TIER 3 image/vision chat"},
    {"role": "MULTIMODAL", "group": "Chat Tiers", "desc": "TIER 3 multimodal/tool-driven chat"},
    {"role": "FRONTIER", "group": "Chat Tiers", "desc": "frontier escalation slot"},
    {"role": "COGNITIVE_ESCALATION", "group": "Chat Tiers", "desc": "explicit think-hard escalation", "fallbacks": ("FRONTIER",)},
    {"role": "JOB_CONTINUATION", "group": "Chat Tiers", "desc": "spec-clarification job continuation stream", "fallbacks": ("ARCHITECT",)},
    {"role": "CHAT_DEEP", "group": "Chat Tiers", "desc": "legacy deep-chat slot (unset: sites fall to role models)"},
    {"role": "OPENAI_MODEL_CODE", "group": "Chat Tiers", "env_var": "OPENAI_MODEL_CODE", "provider_locked": "openai", "desc": "code-creation job slot"},
    {"role": "OPENAI_MODEL_LIGHT_CHAT", "group": "Chat Tiers", "env_var": "OPENAI_MODEL_LIGHT_CHAT", "provider_locked": "openai", "desc": "CHAT_LIGHT job route"},
    {"role": "OPENAI_MODEL_HEAVY_TEXT", "group": "Chat Tiers", "env_var": "OPENAI_MODEL_HEAVY_TEXT", "provider_locked": "openai", "desc": "TEXT_HEAVY job route"},
    {"role": "TOOL_CHAT", "group": "Chat Tiers", "env_var": "TOOL_CHAT_MODEL", "provider_locked": "google", "desc": "tool-prompt chat slot"},

    # Pipeline extras not in STAGE_DEFAULTS
    {"role": "ASTRA_V2_BUILDER", "group": "Pipeline", "env_var": "ASTRA_V2_BUILDER_MODEL", "provider_locked": "openai", "desc": "pipeline v2 builder"},
    {"role": "ASTRA_V2_FALLBACK", "group": "Pipeline", "env_var": "ASTRA_V2_FALLBACK_MODEL", "provider_locked": "anthropic", "desc": "pipeline v2 fallback"},

    # Debug Assistant (provider-toggled surface — resolved via
    # app.debug.debug_model_config, NOT the {ROLE}_PROVIDER/_MODEL resolver:
    # DEBUG_PROVIDER picks the openai/anthropic column and model edits write
    # the ACTIVE column's var, so the rows always mirror what actually runs).
    {"role": "DEBUG_PROVIDER", "group": "Debug Assistant", "kind": "provider_switch",
     "values": ["openai", "anthropic"],
     "desc": "Debug surface provider switch (GPT ↔ Claude) — flips every debug role "
             "between the openai/anthropic model columns, live"},
    {"role": "DEBUG_CHAT", "group": "Debug Assistant", "debug_kind": "chat", "desc": "Debug-tab brain"},
    {"role": "DEBUG_DECOMPOSER", "group": "Debug Assistant", "debug_kind": "decomposer", "desc": "debug decomposer"},
    {"role": "DEBUG_PLANNER", "group": "Debug Assistant", "debug_kind": "planner", "desc": "debug planner"},
    {"role": "DEBUG_SUBAGENT", "group": "Debug Assistant", "debug_kind": "subagent", "desc": "debug sub-agents"},
    {"role": "DEBUG_EXECUTOR", "group": "Debug Assistant", "debug_kind": "executor", "desc": "debug executor"},
    {"role": "DEBUG_CODE_VERIFIER", "group": "Debug Assistant", "debug_kind": "verifier", "desc": "debug code verifier"},

    # RAG (Lane C)
    {"role": "RAG_LOOKUP", "group": "RAG", "desc": "RAG ping-pong/lookup tier"},
    {"role": "RAG_REASONING", "group": "RAG", "desc": "RAG reasoning tier"},
    {"role": "RAG_DEEP", "group": "RAG", "desc": "RAG deep/multimodal tier"},
    {"role": "ORB_EMBEDDING", "group": "RAG", "desc": "embedding model (Lane C truth)"},

    # Memory
    {"role": "SUMMARY", "group": "Memory", "desc": "cheap summariser tier (Lane B model_env)"},
    {"role": "EXTRACTION", "group": "Memory", "desc": "knowledge extractor"},

    # Scene
    {"role": "ASTRA_SCENE_DIRECTOR", "group": "Scene", "desc": "scene director"},
    {"role": "SCENE_CRITIC", "group": "Scene", "desc": "scene critic"},

    # Image
    {"role": "IMAGE_GEN", "group": "Image", "desc": "image generation model"},
    {"role": "IMAGE_GEN_FALLBACK", "group": "Image", "desc": "image generation fallback"},
    {"role": "IMAGE_PROMPT_SYNTH", "group": "Image", "desc": "image prompt synthesiser"},
    {"role": "OPENAI_VISION", "group": "Image", "desc": "OpenAI vision fallback"},
    {"role": "GEMINI_VISION_FAST", "group": "Image", "env_var": "GEMINI_VISION_MODEL_FAST", "provider_locked": "google", "desc": "fast vision tier"},
    {"role": "GEMINI_VISION_COMPLEX", "group": "Image", "env_var": "GEMINI_VISION_MODEL_COMPLEX", "provider_locked": "google", "desc": "complex vision tier"},
    {"role": "GEMINI_VISION_PRO", "group": "Image", "env_var": "GEMINI_VISION_MODEL_PRO", "provider_locked": "google", "desc": "PDF/image pro tier"},
    {"role": "GEMINI_VIDEO_HEAVY", "group": "Image", "env_var": "GEMINI_VIDEO_HEAVY_MODEL", "provider_locked": "google", "desc": "heavy video tier"},
    {"role": "GEMINI_VIDEO", "group": "Image", "env_var": "GEMINI_VIDEO_MODEL", "provider_locked": "google", "desc": "video job route"},
    {"role": "GEMINI_OPUS_CRITIC", "group": "Image", "env_var": "GEMINI_OPUS_CRITIC_MODEL", "provider_locked": "google", "desc": "opus-critic vision tier"},
    {"role": "GEMINI_CRITIC", "group": "Image", "env_var": "GEMINI_CRITIC_MODEL", "provider_locked": "google", "desc": "legacy critique model var"},

    # Nat-Local
    {"role": "NAT", "group": "Nat-Local", "desc": "local vLLM worker (Nat, port 8003)"},

    # Core Defaults
    {"role": "DEFAULT", "group": "Core Defaults", "desc": "global default provider/model"},
    {"role": "OPENAI_DEFAULT", "group": "Core Defaults", "desc": "OpenAI provider default"},
    {"role": "ANTHROPIC_DEFAULT", "group": "Core Defaults", "desc": "Anthropic provider default"},
    {"role": "GOOGLE_DEFAULT", "group": "Core Defaults", "desc": "Google provider default"},
    {"role": "GEMINI_DEFAULT", "group": "Core Defaults", "desc": "legacy Gemini provider default"},
    {"role": "ANTHROPIC_SONNET", "group": "Core Defaults", "desc": "Anthropic mid-tier slot"},
    {"role": "ANTHROPIC_OPUS", "group": "Core Defaults", "desc": "Anthropic opus slot"},
    {"role": "ARCHMAP_TRIGGER", "group": "Core Defaults", "env_var": "ARCHMAP_TRIGGER_MODEL", "provider_locked": "openai", "desc": "legacy archmap trigger model"},
    {"role": "ORB_ARCHMAP", "group": "Core Defaults", "env_var": "ORB_ARCHMAP_MODEL", "provider_locked": "anthropic", "desc": "architecture-map builder"},
    {"role": "ORB_ARCHMAP_FALLBACK", "group": "Core Defaults", "env_var": "ORB_ARCHMAP_FALLBACK_MODEL", "provider_locked": "openai", "desc": "architecture-map fallback"},
    {"role": "GEMINI_FRONTIER", "group": "Core Defaults", "env_var": "GEMINI_FRONTIER_MODEL_ID", "provider_locked": "google", "desc": "OVERRIDE frontier target (google)"},
    {"role": "ANTHROPIC_FRONTIER", "group": "Core Defaults", "env_var": "ANTHROPIC_FRONTIER_MODEL_ID", "provider_locked": "anthropic", "desc": "OVERRIDE frontier target (anthropic)"},
    {"role": "OPENAI_FRONTIER", "group": "Core Defaults", "env_var": "OPENAI_FRONTIER_MODEL_ID", "provider_locked": "openai", "desc": "OVERRIDE frontier target (openai)"},
]

_STAGE_DESCRIPTIONS = {
    "WEAVER": "conversation weaver", "SPEC_GATE": "spec gate (pass/fail)",
    "PLANNER": "planner", "ARCHITECTURE": "architecture drafts",
    "CRITIQUE": "adversarial critique", "REVISION": "revision pass",
    "IMPLEMENTER": "implementer", "OVERWATCHER": "verifier second-opinion",
    "CRITICAL_PIPELINE": "agentic builder", "CRITICAL_SUPERVISOR": "segment supervisor",
    "COHESION_CHECK": "cohesion check", "NEEDLE_CLASSIFIER": "needle classifier",
    "SMART_SEGMENTATION": "smart segmentation", "ARCH_TIER_LOW": "arch tier (cheap)",
    "ARCH_TIER_HIGH": "arch tier (premium)", "COHESION_MICRO_PATCH": "cohesion micro-patch",
    "JOB_CHECKER": "post-write job checker", "WEAVER_COMPACTION": "weaver compaction",
    "SPEC_REVIEW": "always-on spec reviewer", "VERIFIER_AGENT": "agentic checkout verifier",
    "ARCHMAP": "architecture map stage", "SUMMARIZER": "pipeline summariser",
    "CLASSIFIER": "pipeline classifier", "OVERWATCH": "legacy overwatch",
}


def _resolve_standard(role: str, fallbacks: Tuple[str, ...]) -> Tuple[str, str, str]:
    """(provider, model, source) via the live resolver; never raises."""
    try:
        provider, model = get_role_model(role, *fallbacks)
    except RuntimeError:
        return "", "", "unresolved"
    if (os.getenv(f"{role}_MODEL") or "").strip():
        source = "role_env"
    elif any((os.getenv(f"{fb}_MODEL") or "").strip() for fb in fallbacks):
        source = "fallback_role"
    else:
        source = "default_chain"
    return provider, model, source


def _entry(role: str, group: str, desc: str, provider: str, model: str,
           source: str, provider_env: Optional[str], model_env: str) -> Dict[str, Any]:
    restart_note = RESTART_GATED.get(role)
    return {
        "role": role,
        "group": group,
        "description": desc,
        "provider": provider,
        "model": model,
        "source": source,
        "provider_env": provider_env,
        "model_env": model_env,
        "verifier": role in VERIFIER_ROLES,
        "restart_gated": bool(restart_note),
        "restart_note": restart_note,
    }


def list_roles() -> List[Dict[str, Any]]:
    """Every known role with its LIVE resolution. Ordered by GROUP_ORDER."""
    out: List[Dict[str, Any]] = []

    # Hand-listed roles (standard + env-shaped + debug provider-toggled)
    for spec in _STANDARD:
        role = spec["role"]
        group = spec["group"]
        desc = spec.get("desc", "")
        if spec.get("kind") == "provider_switch":
            # The DEBUG_PROVIDER toggle: a dropdown-only row (no model field).
            # PUT writes the raw value into the model_env var (validated
            # against `values` in model_settings).
            from app.debug.debug_model_config import get_debug_provider
            val = get_debug_provider()
            e = _entry(role, group, desc, val, val, "env", None, "DEBUG_PROVIDER")
            e["kind"] = "provider_switch"
            e["values"] = list(spec["values"])
            out.append(e)
        elif "debug_kind" in spec:
            # Debug roles resolve through the DEBUG_PROVIDER-aware column
            # resolver; model_env is the ACTIVE column's own var so PUT edits
            # write the right suffix.
            from app.debug.debug_model_config import (
                debug_role_env_var,
                provider_fallback_active,
                resolve_debug_model,
            )
            provider, model = resolve_debug_model(spec["debug_kind"])
            source = "provider_column"
            if provider_fallback_active():
                source = "fallback_missing_key"
            out.append(_entry(role, group, desc, provider, model, source,
                              None, debug_role_env_var(spec["debug_kind"])))
        elif "env_var" in spec:  # env-shaped: one var, locked provider
            model = (os.getenv(spec["env_var"]) or "").strip()
            out.append(_entry(role, group, desc, spec["provider_locked"], model,
                              "env" if model else "unset", None, spec["env_var"]))
        else:
            fallbacks = tuple(spec.get("fallbacks", ()))
            provider, model, source = _resolve_standard(role, fallbacks)
            out.append(_entry(role, group, desc, provider, model, source,
                              f"{role}_PROVIDER", f"{role}_MODEL"))

    # Pipeline stages (programmatic; CHAT is listed under Chat Tiers already)
    try:
        from app.llm.stage_models import STAGE_DEFAULTS, get_stage_config
        for stage in sorted(STAGE_DEFAULTS.keys()):
            if stage == "CHAT":
                continue
            try:
                cfg = get_stage_config(stage)
                provider, model = cfg.provider, cfg.model
            except Exception:
                provider, model = "", ""
            out.append(_entry(stage, "Pipeline",
                              _STAGE_DESCRIPTIONS.get(stage, "pipeline stage"),
                              provider, model,
                              "role_env" if (os.getenv(f"{stage}_MODEL") or "").strip() else "stage_default",
                              f"{stage}_PROVIDER", f"{stage}_MODEL"))
    except Exception:
        pass  # stage table unavailable: hand-listed roles still serve

    # Frontier alias slots (programmatic from _ALIAS_ENV)
    for alias in sorted(_ALIAS_ENV.keys()):
        var = _ALIAS_ENV[alias]
        try:
            resolved = FRONTIER_ALIASES[alias]
        except Exception:
            resolved = ""
        e = _entry(alias, "Frontier Aliases",
                   f"pipeline capability alias ({var})",
                   alias.split(":", 1)[0], resolved, "alias", None, var)
        e["restart_gated"] = True
        e["restart_note"] = _ALIAS_RESTART_NOTE
        out.append(e)

    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    out.sort(key=lambda e: (order.get(e["group"], 99), e["role"]))
    return out


def get_role_spec(role: str) -> Optional[Dict[str, Any]]:
    """Writable env vars for one role (for PUT /settings/models/{role})."""
    for entry in list_roles():
        if entry["role"].upper() == role.upper() or entry["role"] == role:
            writable = [entry["model_env"]]
            if entry["provider_env"]:
                writable.insert(0, entry["provider_env"])
            return {**entry, "writable_vars": writable}
    return None


__all__ = ["GROUP_ORDER", "VERIFIER_ROLES", "RESTART_GATED",
           "list_roles", "get_role_spec"]

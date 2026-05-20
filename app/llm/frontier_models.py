# FILE: app/llm/frontier_models.py
"""
Frontier Model Aliases — single source of truth for "latest good model".

Purpose
-------
When a new Anthropic Opus or OpenAI GPT drops, ASTRA should pick it up
without hunting through scattered model strings. This module is the ONE
place to change when the frontier moves.

Every pipeline stage references a capability alias like
``openai:frontier-reasoning`` rather than a concrete model ID. The alias
table below maps each alias to the current best model for that role.

Hard constraint — NO Pro-tier GPT models
-----------------------------------------
Taz's explicit rule: never route any ASTRA stage to GPT-5 Pro
(``gpt-5.2-pro``, future ``gpt-5.5-pro`` etc.). They are far more
expensive than the standard frontier model for a marginal quality
bump that isn't worth it at pipeline volumes. The aliases defined here
are guaranteed non-Pro. When updating the table, preserve this rule.

Reasoning / thinking — enabled by default on load-bearing stages
----------------------------------------------------------------
Taz's stance: the pipeline is a high-stakes job, so the extra few quid
for reasoning/thinking is worth it. Per-stage reasoning defaults live
here too. Light classifier / summariser stages stay cheap (no
reasoning) because their correctness doesn't move the dial.

Resolution
----------
`resolve_model_alias(s)` turns either an alias or a concrete model ID
into a concrete model ID. Concrete IDs pass through unchanged, so
env-var overrides that use real model names keep working.

v1.0 (2026-04-18): Initial implementation. Replaces scattered hard-coded
    model strings in stage_models.py / pipeline_v2/config.py.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Frontier alias table — UPDATE THIS WHEN NEW MODELS DROP
# ─────────────────────────────────────────────────────────────────────────────
# Format: "{provider}:{capability}" -> concrete model id
#
# Rules when updating:
#   1. Keep aliases stable; only change the right-hand-side model ID.
#   2. NEVER point a frontier alias at a Pro-tier OpenAI model.
#   3. For Anthropic, prefer the latest Opus that supports adaptive
#      thinking + output_config.effort (currently Opus 4.6 and 4.7).
#   4. For OpenAI, use the latest non-Pro GPT that accepts the
#      `reasoning={"effort": "..."}` parameter.
#
# Current frontier (2026-04-18):
#   - OpenAI latest non-Pro GPT = gpt-5.4 (reasoning param supported)
#   - Anthropic latest Opus = claude-opus-4-7 (adaptive thinking only)
#   - Anthropic latest Sonnet = claude-sonnet-4-6
#   - Google frontier flash = gemini-2.5-flash
#
FRONTIER_ALIASES: Dict[str, str] = {
    # ── OpenAI (non-Pro) ──────────────────────────────────────────────
    # The primary agentic builder model. Reasoning is enabled via
    # the `reasoning={"effort": "high"}` parameter at call time.
    "openai:frontier-reasoning":      "gpt-5.4",
    # Same underlying model, used when we want a cheaper/faster pass
    # (e.g. simple classification, summarisation). Still non-Pro.
    "openai:frontier-fast":           "gpt-5.4-mini",

    # ── Anthropic ─────────────────────────────────────────────────────
    # Primary thinking model for SpecGate, Overwatcher, Verifier.
    # Thinking is enabled via adaptive thinking + output_config.effort.
    "anthropic:frontier-opus-thinking": "claude-opus-4-7",
    # Same model without thinking routed in (for cheaper diagnostic
    # calls that still benefit from Opus's reasoning).
    "anthropic:frontier-opus":          "claude-opus-4-7",
    # Mid-tier Claude for lighter work (e.g. job-checker, compaction).
    "anthropic:frontier-sonnet":        "claude-sonnet-4-6",

    # ── Google ────────────────────────────────────────────────────────
    # Vision / fast classifier / summariser tier.
    "google:frontier-flash":           "gemini-2.5-flash",
    # Tiniest tier for high-volume classification.
    "google:frontier-flash-lite":      "gemini-2.5-flash-lite",
}


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
    """Resolve a frontier alias to a concrete model ID.

    Aliases look like ``"provider:frontier-<capability>"``. Anything
    that doesn't match a known alias is returned unchanged, so concrete
    model IDs from env vars keep working without modification.

    Args:
        model: Either an alias (e.g. ``"openai:frontier-reasoning"``)
               or a concrete model ID (e.g. ``"gpt-5.4"``).

    Returns:
        Concrete model ID. Never None, never empty unless input was.
    """
    if not model:
        return model
    stripped = model.strip()
    if stripped in FRONTIER_ALIASES:
        resolved = FRONTIER_ALIASES[stripped]
        logger.debug("[frontier_models] Resolved alias %s -> %s", stripped, resolved)
        return resolved
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
    if stage_upper in STAGE_REASONING:
        return STAGE_REASONING[stage_upper]
    return _DEFAULT_REASONING


def is_alias(model: str) -> bool:
    """True if ``model`` is a known frontier alias."""
    return bool(model) and model.strip() in FRONTIER_ALIASES


def all_aliases() -> Dict[str, str]:
    """Return a copy of the full alias table (for audit/debug)."""
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
    print("\nCurrent aliases:")
    for alias, model in sorted(FRONTIER_ALIASES.items()):
        print(f"  {alias:<42} -> {model}")
    print("\n--- OpenAI available models ---")
    _audit_openai()
    print("\n--- Anthropic available models ---")
    _audit_anthropic()
    print("\n" + "=" * 70)
    print("If a newer non-Pro GPT or newer Opus appears above, update")
    print("FRONTIER_ALIASES in this file to point at it, then re-run to")
    print("confirm the new mapping is picked up.")
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
    "get_reasoning_for_stage",
    "is_alias",
    "all_aliases",
    "all_stage_reasoning",
    "audit_frontier",
]

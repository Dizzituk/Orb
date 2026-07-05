# FILE: app/llm/model_families.py
# Purpose: Single source of truth for "which model is the current one in family X".
# Called-by: app.debug.gemini_tool_loop, app.llm.routing.image_chat_routing
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Single source of truth for "which model is the current one in family X".

The problem this solves: across the codebase we had model strings like
`gpt-5.4` hardcoded in ~15 places, sometimes as function defaults,
sometimes in ENV fallbacks, sometimes inconsistent (one file had
`gpt-5.2` as a default while everything else was on `gpt-5.4`). When
a new model lands, upgrading meant grepping the codebase and praying
you caught everything.

The principle: every caller that wants "the current flagship OpenAI
model" should ask `resolve('openai_flagship')` and get back whatever
string is currently considered best. Updates happen in ONE place —
the ASTRA_MODEL_<FAMILY> vars in .env (LANE D 2026-07-02: the baked-in
defaults are gone; .env is the single source, read on access).

Families are identified by (provider_tier) pairs:
    <provider>_flagship    most capable, slowest, highest cost
    <provider>_balanced    good quality, fair cost
    <provider>_fast        cheapest / fastest, lower quality

Call sites should use families, not raw model strings, unless they
have a concrete reason to pin a version (e.g. reproducing a specific
evaluation).

Migration status: this module just landed. Existing call sites still
use hardcoded strings or their own ENV vars. See TODO list in the
docstring at the bottom of this file for places that still need
migrating. Migrate opportunistically, don't rush.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


# ─── Current families ────────────────────────────────────────────────
#
# LANE D (2026-07-02): the baked-in default model strings are GONE. Every
# family's value lives in .env as ASTRA_MODEL_<FAMILY> (seeded with the old
# defaults). resolve() reads that env var on access, then falls through to
# the frontier_models provider-default chain keyed by the family's provider
# below (embedding additionally honours Lane C's ORB_EMBEDDING_MODEL).
# Update models in .env or the Models settings UI, never here.
#
_FAMILY_FALLBACK_PROVIDER: dict[str, str] = {
    # OpenAI — GPT family
    "openai_flagship": "openai",     # reasoning-capable, highest quality
    "openai_balanced": "openai",
    "openai_fast":     "openai",     # lower cost
    "openai_turbo":    "openai",     # cost-optimised variant

    # Anthropic — Claude family
    "anthropic_flagship": "anthropic",
    "anthropic_balanced": "anthropic",
    "anthropic_fast":     "anthropic",

    # Google — Gemini family
    "google_flagship": "google",
    "google_balanced": "google",
    "google_fast":     "google",

    # Role-specific families — abstract "what's this FOR" from
    # "which model". Lets you change providers per role without
    # touching call sites.
    "chat_flagship":    "openai",    # voice / conversational chat
    "chat_tools":       "anthropic", # tool-heavy agentic chat
    "embedding":        "google",
    "vision_fast":      "google",

    # -- ASTRA role models: primary = cheaper everyday model;
    #    fallback = pricier model, used only if the primary fails. --
    "role_chat":                  "google",
    "role_chat_fallback":         "openai",
    "role_vision_tools":          "google",
    "role_vision_tools_fallback": "openai",
    "role_cheap":                 "google",
}


# ─── Resolution ──────────────────────────────────────────────────────

def _env_key(family: str) -> str:
    """
    ASTRA_MODEL_<FAMILY_UPPER>. For pinning a model version that differs
    from the code default without editing this file.

    Example: `ASTRA_MODEL_OPENAI_FLAGSHIP=gpt-5.5`
    """
    return f"ASTRA_MODEL_{family.upper()}"


def resolve(family: str) -> str:
    """
    Return the current model string for a family, env-only (LANE D).
    ASTRA_MODEL_<FAMILY> wins (seeded in .env); otherwise the family's
    provider-default chain. Raises ValueError on unknown families; an
    entirely unconfigured .env yields "" and fails loudly downstream.

    Don't call this on every token, it's not THAT cheap — cache the
    result at module-load or function-entry where appropriate.
    """
    override = os.getenv(_env_key(family))
    if override:
        logger.debug("[model_families] %s = %s (env override)", family, override)
        return override

    provider = _FAMILY_FALLBACK_PROVIDER.get(family)
    if provider is None:
        raise ValueError(
            f"Unknown model family: {family!r}. "
            f"Known families: {sorted(_FAMILY_FALLBACK_PROVIDER.keys())}"
        )
    if family == "embedding":
        # Lane C: the embedding-model truth is ORB_EMBEDDING_MODEL.
        emb = (os.getenv("ORB_EMBEDDING_MODEL") or "").strip()
        if emb:
            return emb
    from app.llm.frontier_models import get_provider_default_model
    model = get_provider_default_model(provider, strict=False)
    if not model:
        logger.error(
            "[model_families] %s unresolved — set %s in .env",
            family, _env_key(family),
        )
    return model


def known_families() -> list[str]:
    """List the known family identifiers. Useful for diagnostics."""
    return sorted(_FAMILY_FALLBACK_PROVIDER.keys())


def snapshot() -> dict[str, dict]:
    """
    Return {family: {default, override, effective}} for every known
    family. Handy for the settings UI / a diagnostic endpoint.
    "default" is now the provider-chain fallback (env-only; LANE D).
    """
    from app.llm.frontier_models import get_provider_default_model
    out: dict[str, dict] = {}
    for fam, provider in _FAMILY_FALLBACK_PROVIDER.items():
        override = os.getenv(_env_key(fam))
        default = get_provider_default_model(provider, strict=False)
        out[fam] = {
            "default":   default,
            "override":  override,
            "effective": override or default,
            "env_key":   _env_key(fam),
        }
    return out


# ─── Migration TODO ──────────────────────────────────────────────────
#
# Files that still use hardcoded model strings or one-off ENV vars,
# grouped by priority. Migrate these as the relevant code is touched.
#
# HIGH (chat path — directly affects voice / Coursera use):
#   - app/llm/llm_helpers.py          CHAT_MODEL / CHAT_DEEP_MODEL env
#   - app/capability_layer.py         TOOL_CHAT_MODEL env
#   - app/chat_attachments.py         OPENAI_DEFAULT_MODEL default
#                                     inconsistency (gpt-5.2 vs 5.4)
#
# MEDIUM (pipeline — affects build system):
#   - app/pipeline/pipeline.py        arch_model / checkout_model defaults
#   - app/pipeline/loop_controller.py model_id default
#   - app/video/script_analyzer.py    DEFAULT_MODEL constant
#
# LOW (internal tooling — less user-visible):
#   - app/config.py                   AGENTIC_LOOP_MODEL / CHECKOUT_MODEL
#   - app/pipeline/segment_printer.py arch_model default
#
# When migrating, the replacement pattern is:
#   # before
#   _mod = os.getenv("CHAT_MODEL", "gpt-5-mini")
#   # after
#   from app.llm.model_families import resolve
#   _mod = os.getenv("CHAT_MODEL") or resolve("chat_flagship")
#
# Keeping the legacy ENV var as the first check preserves anyone's
# existing overrides; the family resolver is the new default path.


def chain(role: str) -> list:
    """Return [primary, fallback] model strings for a role.

    Primary is resolve(role). If a "<role>_fallback" family (or its
    ASTRA_MODEL_<ROLE>_FALLBACK env override) exists and differs, it is
    appended. Lets callers run a cheap-primary / pricier-fallback chain
    where the fallback only fires if the primary errors.
    """
    primary = resolve(role)
    out = [primary]
    fb_key = role + "_fallback"
    fb = os.getenv(_env_key(fb_key)) or (
        resolve(fb_key) if fb_key in _FAMILY_FALLBACK_PROVIDER else None
    )
    if fb and fb != primary:
        out.append(fb)
    return out

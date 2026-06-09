# FILE: app/llm/model_families.py
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
string is currently considered best. Updates happen in ONE place
(the _DEFAULTS dict below), and environments that want to pin a
specific version can override via ASTRA_MODEL_<FAMILY> env vars.

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


# ─── Current defaults ────────────────────────────────────────────────
#
# Update these when a new model lands. One-liner per family.
# Everything else in the codebase that uses resolve() inherits
# automatically.
#
_DEFAULTS: dict[str, str] = {
    # OpenAI — GPT-5 family
    "openai_flagship": "gpt-5.5",         # reasoning-capable, highest quality
    "openai_balanced": "gpt-5.5",         # same as flagship for now
    "openai_fast":     "gpt-5.4-mini",    # lower cost, still GPT-5 family
    "openai_turbo":    "gpt-5.4-turbo",   # cost-optimised variant

    # Anthropic — Claude 4 family
    "anthropic_flagship": "claude-opus-4-6",
    "anthropic_balanced": "claude-sonnet-4-6",
    "anthropic_fast":     "claude-haiku-4-5-20251001",

    # Google — Gemini 3 family
    "google_flagship": "gemini-3.1-pro-preview",
    "google_balanced": "gemini-3.5-flash",
    "google_fast":     "gemini-3.1-flash-lite",

    # Role-specific families — abstract "what's this FOR" from
    # "which model". Lets you change providers per role without
    # touching call sites.
    "chat_flagship":    "gpt-5.5",          # voice / conversational chat
    "chat_tools":       "claude-opus-4-6",  # tool-heavy agentic chat
    "embedding":        "gemini-embedding-2-preview",
    "vision_fast":      "gemini-3.5-flash",

    # -- ASTRA role models: primary = cheaper everyday model;
    #    fallback = pricier model, used only if the primary fails. --
    "role_chat":                  "gemini-3.5-flash",
    "role_chat_fallback":         "gpt-5.5",
    "role_vision_tools":          "gemini-3.5-flash",
    "role_vision_tools_fallback": "gpt-5.5",
    "role_cheap":                 "gemini-3.1-flash-lite",
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
    Return the current model string for a family. ENV override wins
    over the baked-in default. Raises ValueError on unknown families.

    Usage:
        from app.llm.model_families import resolve
        model = resolve("openai_flagship")   # "gpt-5.4"

    Don't call this on every token, it's not THAT cheap — cache the
    result at module-load or function-entry where appropriate.
    """
    override = os.getenv(_env_key(family))
    if override:
        logger.debug("[model_families] %s = %s (env override)", family, override)
        return override

    default = _DEFAULTS.get(family)
    if default is None:
        raise ValueError(
            f"Unknown model family: {family!r}. "
            f"Known families: {sorted(_DEFAULTS.keys())}"
        )
    return default


def known_families() -> list[str]:
    """List the known family identifiers. Useful for diagnostics."""
    return sorted(_DEFAULTS.keys())


def snapshot() -> dict[str, dict]:
    """
    Return {family: {default, override, effective}} for every known
    family. Handy for the settings UI / a diagnostic endpoint.
    """
    out: dict[str, dict] = {}
    for fam, default in _DEFAULTS.items():
        override = os.getenv(_env_key(fam))
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
    fb = os.getenv(_env_key(fb_key)) or _DEFAULTS.get(fb_key)
    if fb and fb != primary:
        out.append(fb)
    return out

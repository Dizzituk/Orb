# FILE: app/providers/_registry_anthropic_shapes.py
# Purpose: Anthropic per-model-family request shaping + refusal handling.
# Called-by: app.providers.registry
# Depends-on: stdlib only
# Last-renovated: 2026-06-11
"""
Anthropic request shaping for model families with different parameter rules.

Why this exists (2026-06-11):
Claude Fable 5 / Mythos 5 changed the request contract vs Opus 4.x:
  - Adaptive thinking is ALWAYS ON. No `thinking` config should be sent
    (thinking: {"type": "disabled"} is rejected; setting it is unnecessary).
  - Sampling params (temperature / top_p / top_k) are NOT supported and
    return HTTP 400. They must be stripped unconditionally.
  - Effort is the single depth knob, sent as output_config={"effort": ...}.
    Valid levels: low | medium | high | xhigh | max.
  - max_tokens is a hard cap on thinking + visible output combined, so
    higher effort levels need generous headroom (docs recommend 64k).
  - Blocking safety classifiers return stop_reason "refusal" (HTTP 200)
    at materially higher rates than prior models. Refusal is a normal
    code path: detect it and retry once on a fallback model.

Opus 4.6 / 4.7 / 4.8 / Sonnet 4.6 keep the v2.3 shape:
  thinking={"type": "adaptive"} + output_config.effort, temperature omitted
  when thinking is enabled.

NOTE: anthropic-python 0.75.x doesn't expose output_config as a top-level
kwarg, so it's routed via extra_body, which the SDK forwards verbatim.
Remove the indirection once the SDK ships native support.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_EFFORT = ("low", "medium", "high", "xhigh", "max")

# Generous floors so thinking + visible output both fit under max_tokens.
_EFFORT_MIN_MAX_TOKENS = {
    "low": 8192,
    "medium": 16000,
    "high": 32000,
    "xhigh": 64000,
    "max": 64000,
}

_ANTHROPIC_MAX_TOKENS_CEILING = 128000


def is_always_thinking_family(model_id: str) -> bool:
    """True for models where adaptive thinking is always on and sampling
    params are rejected (Claude Fable 5 / Mythos 5 family)."""
    m = (model_id or "").strip().lower()
    return m.startswith("claude-fable") or m.startswith("claude-mythos")


def shape_anthropic_create_kwargs(
    create_kwargs: dict,
    model_id: str,
    reasoning: Optional[dict],
) -> None:
    """Mutate `create_kwargs` in place so the request is valid for the
    target Anthropic model family.

    - Fable/Mythos: strip sampling params always; never send `thinking`;
      route effort via extra_body.output_config; enforce max_tokens floor.
    - Opus 4.6+/Sonnet 4.6 with reasoning: v2.3 behaviour (adaptive
      thinking + output_config.effort, temperature omitted, floors).
    - Anything else without reasoning: left untouched.
    """
    fable_family = is_always_thinking_family(model_id)

    effort = ""
    if reasoning and isinstance(reasoning, dict):
        effort = str(reasoning.get("effort", "")).strip().lower()
        if effort and effort not in _VALID_EFFORT:
            logger.warning(
                "[registry] Ignoring invalid Anthropic effort %r for %s",
                effort, model_id,
            )
            effort = ""

    if fable_family:
        # Sampling params are rejected outright on this family.
        for k in ("temperature", "top_p", "top_k"):
            create_kwargs.pop(k, None)
        # Thinking is always on; sending a thinking config is unnecessary.
        create_kwargs.pop("thinking", None)
        if effort:
            create_kwargs["extra_body"] = {"output_config": {"effort": effort}}
        # Effort defaults to high server-side when unset, and max_tokens
        # caps thinking + response combined — keep a sane floor either way.
        _floor = _EFFORT_MIN_MAX_TOKENS.get(effort or "high", 16000)
        if create_kwargs.get("max_tokens", 0) < _floor:
            create_kwargs["max_tokens"] = min(_floor, _ANTHROPIC_MAX_TOKENS_CEILING)
        logger.info(
            "[registry] Anthropic always-thinking family: model=%s effort=%s max_tokens=%d",
            model_id, effort or "high(default)", create_kwargs["max_tokens"],
        )
        return

    # Pre-Fable models: v2.3 adaptive-thinking shape, only when effort is set.
    if effort:
        create_kwargs["thinking"] = {"type": "adaptive"}
        create_kwargs["extra_body"] = {"output_config": {"effort": effort}}
        # Temperature MUST be omitted (not set to 1.0) when thinking is on.
        create_kwargs.pop("temperature", None)
        _floor = _EFFORT_MIN_MAX_TOKENS.get(effort, 16000)
        if create_kwargs.get("max_tokens", 0) < _floor:
            create_kwargs["max_tokens"] = min(_floor, _ANTHROPIC_MAX_TOKENS_CEILING)
        logger.info(
            "[registry] Anthropic adaptive thinking: model=%s effort=%s max_tokens=%d",
            model_id, effort, create_kwargs["max_tokens"],
        )


def is_refusal(resp) -> bool:
    """True when the response was blocked by a safety classifier."""
    return getattr(resp, "stop_reason", None) == "refusal"


def extract_refusal_detail(resp) -> str:
    """Best-effort human-readable detail from stop_details on a refusal."""
    details = getattr(resp, "stop_details", None)
    if details is None:
        return "refusal (no stop_details)"
    try:
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        return f"refusal: {details}"
    except Exception:
        return f"refusal: {details!r}"


def get_refusal_fallback_model(current_model_id: str) -> Optional[str]:
    """Return the configured refusal-fallback model, or None when no
    fallback applies (unset, blank, or same as the current model)."""
    fb = os.getenv("ANTHROPIC_REFUSAL_FALLBACK_MODEL", "").strip()
    if not fb or fb == (current_model_id or "").strip():
        return None
    return fb


__all__ = [
    "is_always_thinking_family",
    "shape_anthropic_create_kwargs",
    "is_refusal",
    "extract_refusal_detail",
    "get_refusal_fallback_model",
]

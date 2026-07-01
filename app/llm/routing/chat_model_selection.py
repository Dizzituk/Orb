# FILE: app/llm/routing/chat_model_selection.py
# Purpose: Model selection logic for chat routing.
# Called-by: app.bridge.capability_layer, app.llm.routing.chat_routing
# Depends-on: app.llm.frontier_models, app.llm.routing.chat_intent_detection, app.llm.routing.cognitive_escalation, app.llm.routing.conversational_gate (+4 more)
# Last-renovated: 2026-07-01
"""
Model selection logic for chat routing.

Extracted from chat_routing.py during modularisation (2026-04-06).

This module provides:
- Session-level model stickiness (in-memory + history fallback)
- Complexity-based four-tier model selection
- Provider availability fallback
- Explicit model request detection pass-through

Lane A de-hardcode (2026-07-01): NO model string is hardcoded here. Every
tier resolves through app.llm.frontier_models.get_role_model() — env-only
({ROLE}_PROVIDER/{ROLE}_MODEL, DEFAULT_PROVIDER/DEFAULT_MODEL), fail-loud.
A conversational turn (conversational_gate) never escalates on the
keyword-counted complexity tiers (TIER 4 architecture / TIER 2 reasoning):
talking ABOUT the repo/memory/architecture is not a request to DO work on
it. Cognitive cues ("think hard", "no rush") deliberately still escalate —
they are an explicit request for more compute, even mid-musing.

All functions are synchronous.  The main entry point is
`select_chat_model()` which returns (provider, model, extras_dict).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.memory import service as memory_service
from app.memory.complexity import classify_complexity
from app.llm.streaming import (
    get_available_streaming_provider,
    get_available_streaming_providers,
)

from .chat_intent_detection import (
    detect_file_creation_intent,
    detect_image_gen_intent,
    detect_image_refinement,
    last_assistant_was_image,
)
from .cognitive_escalation import (
    detect_cognitive_escalation,
    is_small_model,
)
from .conversational_gate import is_conversational_turn
from ._capability_guard import is_capability_question
from app.llm.frontier_models import get_role_model, get_provider_default_model

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION-LEVEL MODEL STICKINESS  (v10.0)
# =============================================================================

_session_model_cache: dict[int, tuple[str, str]] = {}

# v2026-06-24: WHY each sticky was set, kept in a parallel dict so the 2-tuple
# return shape of get_sticky_model stays unchanged (external readers unpack it).
# Only PINNED sources short-circuit routing on later turns / survive a restart;
# everything else is governed by tier_momentum and decays.
_sticky_source: dict[int, str] = {}
_PINNED_SOURCES = frozenset({"frontend_override", "explicit_user", "agentic_tab"})


def get_sticky_model(project_id: int) -> tuple[str, str] | None:
    return _session_model_cache.get(project_id)


def set_sticky_model(
    project_id: int, provider: str, model: str, source: str = "auto",
) -> None:
    """Pin a model for the session.

    `source` records WHY. The default "auto" is NON-pinned: automatic
    escalations (and external callers that don't pass a source) no longer latch
    the session — only an explicit user/policy choice (a value in
    _PINNED_SOURCES) short-circuits routing on later turns and survives a
    restart. This is what lets the tier de-escalate again.
    """
    _session_model_cache[project_id] = (provider, model)
    _sticky_source[project_id] = source


def is_explicit_sticky(project_id: int) -> bool:
    """True only when the session's sticky was an explicit user/policy choice
    (frontend dropdown, "use opus", agentic tab) — not an automatic escalation."""
    return _sticky_source.get(project_id) in _PINNED_SOURCES


# =============================================================================
# AGENTIC TAB CONTEXT DETECTION  (v14.1, 2026-04-26)
# =============================================================================
# Tabs whose work warrants a frontier model regardless of message intent.
# Tab presence alone is enough — the user's explicit policy: as soon as I'm
# in this tab, treat every command as agentic and route to the frontier
# model. No intent-detection required, no "only if message looks like a
# build request" gating. These flows often run unattended (driving), so
# accuracy beats speed/cost.
#
# Covered:
#   - Browser-driven flows (website, courses, social composer)
#   - Builds / Debug (where ASTRA's own code or apps are being modified)
#
# Detection signal: req.ui_context.job OR req.ui_context.job_type. The
# frontend uses different field names in different paths, so we check both.
_AGENTIC_TAB_JOBS = {
    # Browser-driven flows
    "website",
    "courses",
    "coursera",
    "social",
    "social_media",
    # Builds / debug
    "builds",
    "project_builds",
    "debug",
    "debug_assistant",
}


def _is_agentic_context(req: Any) -> bool:
    """True when this chat turn happens in a browser/builds/debug tab.

    Reads req.ui_context.job AND req.ui_context.job_type (different code
    paths use different field names for the same concept) and matches
    against the agentic job set. Tab-only — no message intent gating.
    """
    ui = getattr(req, "ui_context", None)
    if ui is None:
        return False
    for field in ("job", "job_type"):
        if isinstance(ui, dict):
            v = ui.get(field)
        else:
            v = getattr(ui, field, None)
        if isinstance(v, str) and v.lower() in _AGENTIC_TAB_JOBS:
            return True
    return False


def infer_sticky_from_history(project_id: int, db) -> tuple[str, str] | None:
    """Check last assistant message in history for an elevated explicit pin.
    Falls back when the in-memory cache is empty (e.g. after app restart)."""
    try:
        # v2026-06-10 FIX: memory_service.get_messages() never existed --
        # the AttributeError was swallowed and sticky inference silently
        # never worked after a restart. list_messages now returns the
        # most recent N in chronological order.
        msgs = memory_service.list_messages(db, project_id, limit=10)
        for msg in reversed(msgs):
            # v2026-06-24: only restore a model that was an EXPLICIT user/policy
            # pin (model_source in _PINNED_SOURCES). An automatic escalation left
            # in history (model_source NULL or '*_auto') must NOT re-latch — that
            # was the bug that held the elevated model across a whole session.
            if (msg.role == 'assistant' and msg.model
                    and getattr(msg, 'model_source', None) in _PINNED_SOURCES):
                # v2026-07-01: the NEWEST pinned row decides — full stop. If the
                # user's latest explicit choice was a small model, restore
                # NOTHING (normal tiering picks the cheap tier anyway). Scanning
                # past it would resurrect an OLDER elevated pin over the user's
                # most recent downgrade. "Elevated" = not small-tier
                # (env-driven is_small_model, replaces _ELEVATED_MODELS).
                if is_small_model(msg.model):
                    return None
                prov = msg.provider or os.getenv("DEFAULT_PROVIDER", "openai")
                return (prov, msg.model)
    except Exception:
        pass
    return None


# =============================================================================
# PROVIDER AVAILABILITY FALLBACK
# =============================================================================

def ensure_provider_available(provider: str, model: str) -> tuple[str, str]:
    """Check provider availability and fall back if needed.

    Returns (provider, model) — possibly swapped to a working alternative.
    CRITICAL: when swapping provider, also swaps the model to match.
    """
    providers_available = get_available_streaming_providers()
    print(f"[MODEL_SELECT] Provider availability: {providers_available}")

    if providers_available.get(provider, False):
        return provider, model

    available = get_available_streaming_provider()
    if not available:
        print(f"[MODEL_SELECT] WARNING: No providers available at all")
        return provider, model  # caller will get an error downstream

    print(f"[MODEL_SELECT] Provider '{provider}' not available, falling back to {available} WITH its default model")
    # v2026-07-01: swap model from {PROVIDER}_DEFAULT_MODEL / DEFAULT_MODEL in
    # .env (was hardcoded per provider — and the google branch wrongly read
    # CHAT_MODEL, which no longer has to be a Google model at all).
    try:
        return available, get_provider_default_model(available)
    except RuntimeError as e:
        print(f"[MODEL_SELECT] WARNING: {e} — keeping requested model on {available}")
        return available, model


# =============================================================================
# MAIN MODEL SELECTION  (v13.0 four-tier)
# =============================================================================

# Return value keys in the extras dict:
#   "image_route"      -> True if image gen detected (caller should route to image pipeline)
#   "set_sticky"       -> True if the selected model should be made sticky
#   "skip_confirm"     -> True if confirmation gate was already handled or skipped
#   "confirmation_sse" -> async generator if a confirmation gate should be returned to the user
#   "synthetic_attachments" -> list of attachment dicts built from file_upload_* fields

def select_chat_model(
    req: Any,
    db: Session,
    synthetic_attachments: list | None = None,
    is_image_upload: bool = False,
    is_video_upload: bool = False,
) -> tuple[str, str, dict]:
    """Decide which provider/model to use for a chat message.

    Returns (provider, model, extras) where extras is a dict of
    routing metadata the caller may need (see keys above).
    """
    extras: dict = {}
    _all_attachments = synthetic_attachments or []
    # tier_momentum imported here (not at module top) to avoid a routing-package
    # import cycle; it is stdlib-only so the call-time import is cheap.
    from app.llm.routing import tier_momentum

    # ── Frontend model override (from model switcher dropdown) ──
    if req.provider and req.model:
        provider = req.provider
        model = req.model
        print(f"[MODEL_SELECT] Using frontend override: {provider}/{model}")
        set_sticky_model(req.project_id, provider, model, source="frontend_override")
        extras["model_source"] = "frontend_override"
        return *ensure_provider_available(provider, model), extras

    # ── Cognitive escalation / explicit model request (highest priority after frontend) ──
    # Runs BEFORE intent-based routing (builds, file-creation, image-gen) because
    # those detectors are regex-based and produce false positives on long messages
    # that describe apps/concepts. If the user has explicitly said "think about
    # this" or named a provider, that signal beats any implicit intent guess.
    from app.memory.complexity import _detect_explicit_model_request as _detect_explicit
    _first_line_lower = req.message.lower().strip().split("\n")[0]
    _explicit = (
        _detect_explicit(_first_line_lower)
        or _detect_explicit(req.message.lower())
    )
    if _explicit:
        provider = _explicit["provider"]
        model = _explicit["model"]
        print(f"[MODEL_SELECT] EXPLICIT model request (pre-intent): {provider}/{model}")
        set_sticky_model(req.project_id, provider, model, source="explicit_user")
        extras["model_source"] = "explicit_user"
        return *ensure_provider_available(provider, model), extras

    _cog_match = detect_cognitive_escalation(req.message)
    if _cog_match:
        # If the user is already on an EXPLICIT capable pin, keep it.
        _current = get_sticky_model(req.project_id)
        if (_current and not is_small_model(_current[1])
                and is_explicit_sticky(req.project_id)):
            provider, model = _current
            print(
                f"[MODEL_SELECT] COGNITIVE cue ({_cog_match!r}); "
                f"keeping explicit capable pin: {provider}/{model}"
            )
            extras["model_source"] = "explicit_user"
            return *ensure_provider_available(provider, model), extras
        # Otherwise escalate to frontier for THIS turn and record momentum so the
        # boost holds briefly then decays — NO permanent pin (lets it drop back).
        provider, model = get_role_model("COGNITIVE_ESCALATION", "FRONTIER")
        print(
            f"[MODEL_SELECT] COGNITIVE escalation ({_cog_match!r}) -> "
            f"{provider}/{model}"
        )
        tier_momentum.record_turn(
            req.project_id, "reasoning", provider, model,
            reason=f"cognitive cue {_cog_match!r}",
        )
        extras["model_source"] = "cognitive_auto"
        return *ensure_provider_available(provider, model), extras

    # ── File creation → frontier model ──
    if detect_file_creation_intent(req.message):
        provider, model = get_role_model("FILE_CREATION", "FRONTIER")
        print(f"[MODEL_SELECT] File creation detected -> {provider}/{model}")
        tier_momentum.record_turn(
            req.project_id, "reasoning", provider, model,
            reason="file_creation_intent",
        )
        extras["model_source"] = "file_auto"
        return *ensure_provider_available(provider, model), extras

    # ── Image generation / refinement → hint to chat layer (v16.0) ──
    # Previously this returned empty strings to signal "bypass chat, go straight
    # to image_router". That bypass routed to a fresh Gemini synth that lost
    # all conversation context. Now we still flag image_route=True (so the
    # chat layer wraps the stream with the [IMAGE_PROMPT] extractor and
    # injects the marker instructions into the system prompt), but we ALSO
    # return a real frontier model — the chat LLM is now responsible for
    # writing the gpt-image-2 prompt itself, with full context in scope.
    if (
        detect_image_gen_intent(req.message)
        or (
            detect_image_refinement(req.message)
            and last_assistant_was_image(req.project_id, db)
        )
    ) and not is_capability_question(req.message):
        provider, model = get_role_model("IMAGE_CHAT", "FRONTIER")
        print(f"[MODEL_SELECT] Image gen detected -> chat LLM writes prompt: {provider}/{model}")
        extras["image_route"] = True
        # 'explore' (not 'reasoning') so the model drops back to mini on the
        # very next turn — an image is a one-off; a follow-up caption is light.
        tier_momentum.record_turn(
            req.project_id, "explore", provider, model,
            reason="image_gen_intent",
        )
        extras["model_source"] = "image_auto"
        return *ensure_provider_available(provider, model), extras

    # ── Agentic tab context → frontier model ──
    # Tab-only check (website / builds / debug / etc.) — no message intent
    # gating. Subsumes the old separate builds-context and web-automation
    # paths into one. Overrides any prior sticky-mini choice from earlier
    # turns; if the user just opened the tab, this turn IS agentic
    # regardless of what the chat was about before.
    if _is_agentic_context(req):
        provider, model = get_role_model("AGENTIC_CONTEXT", "FRONTIER")
        print(f"[MODEL_SELECT] Agentic tab context detected -> {provider}/{model}")
        set_sticky_model(req.project_id, provider, model, source="agentic_tab")
        extras["model_source"] = "agentic_tab"
        return *ensure_provider_available(provider, model), extras

    # ── Session model stickiness (with explicit override check) ──
    # v2026-06-24: only an EXPLICIT pin (dropdown / "use opus" / agentic tab)
    # short-circuits here. An automatic escalation no longer latches, so the
    # turn falls through to complexity + tier_momentum and can de-escalate.
    sticky = get_sticky_model(req.project_id)
    if sticky and is_explicit_sticky(req.project_id):
        provider, model = _apply_sticky_with_override(req, sticky)
        extras["model_source"] = "explicit_user"
        return *ensure_provider_available(provider, model), extras

    # ── History-based stickiness ──
    # v2026-06-24: infer_sticky_from_history now returns ONLY a model that was an
    # explicit user/policy pin (gated on model_source), so this restores a
    # genuine choice after a restart without re-latching an auto escalation.
    history_sticky = infer_sticky_from_history(req.project_id, db)
    if history_sticky:
        provider, model = _apply_sticky_with_override(req, history_sticky, from_history=True)
        set_sticky_model(req.project_id, provider, model, source="explicit_user")
        extras["model_source"] = "explicit_user"
        return *ensure_provider_available(provider, model), extras

    # ── Complexity-based tier selection ──
    return _select_by_complexity(req, db, _all_attachments, is_image_upload, is_video_upload, extras)


def _apply_sticky_with_override(
    req: Any,
    sticky: tuple[str, str],
    from_history: bool = False,
) -> tuple[str, str]:
    """Use sticky model unless the user explicitly asked for a different one."""
    from app.memory.complexity import _detect_explicit_model_request

    _first_line = req.message.lower().strip().split("\n")[0]
    _explicit = (
        _detect_explicit_model_request(_first_line)
        or _detect_explicit_model_request(req.message.lower())
    )
    if _explicit:
        provider = _explicit["provider"]
        model = _explicit["model"]
        set_sticky_model(req.project_id, provider, model, source="explicit_user")
        label = "history sticky" if from_history else "sticky"
        print(f"[MODEL_SELECT] EXPLICIT override of {label}: {provider}/{model}")
        return provider, model

    provider, model = sticky
    label = "history" if from_history else "session"
    print(f"[MODEL_SELECT] Using sticky model from {label}: {provider}/{model}")
    return provider, model


def _select_by_complexity(
    req: Any,
    db: Session,
    all_attachments: list,
    is_image_upload: bool,
    is_video_upload: bool,
    extras: dict,
) -> tuple[str, str, dict]:
    """Run complexity classifier and pick model tier."""
    complexity = classify_complexity(
        query=req.message,
        intent=None,
        attachments=all_attachments if all_attachments else None,
    )
    print(
        f"[MODEL_SELECT] Complexity: tier={complexity.tier}, target={complexity.model_target}, "
        f"confidence={complexity.confidence}, signals={complexity.signals}"
    )

    _chat_provider, _chat_model = get_role_model("CHAT")
    _skip_confirm = getattr(req, 'ui_context', None) is not None

    _signals = complexity.signals
    _arch_hits = _signals.get("arch_hits", 0)
    _explore_hits = _signals.get("explore_hits", 0)
    _reasoning_hits = _signals.get("reasoning_hits", 0)

    # ── Conversational gate (2026-07-01) ──
    # Talking ABOUT a topic (repo, memory, architecture) is not a request to
    # DO work on it: "can we talk about mapping out the repo" must stay on
    # the chat model no matter how many architecture keywords it contains.
    # Genuine imperatives ("map the repo", "refactor X") pass the gate.
    _conversational = is_conversational_turn(req.message)
    if _conversational and (_arch_hits or _reasoning_hits):
        print(
            f"[MODEL_SELECT] CONVERSATIONAL GATE: suppressing escalation "
            f"(arch={_arch_hits}, reasoning={_reasoning_hits}) — talk, not work"
        )

    # ── TIER 3: Multimodal — media attached, needs vision ──
    if complexity.tier == "multimodal":
        if is_image_upload:
            provider, model = get_role_model("IMAGE_VISION")
            print(f"[MODEL_SELECT] TIER 3 (image vision): {len(all_attachments)} attachments -> {provider}/{model}")
        else:
            provider, model = get_role_model("MULTIMODAL")
            print(f"[MODEL_SELECT] TIER 3 (video/multimodal): {len(all_attachments)} attachments -> {provider}/{model}")
        # v2026-06-24: route vision THIS turn but do not pin — recording as
        # 'explore' keeps it out of the held-model slot so a follow-up text turn
        # isn't stuck on the vision model.
        from app.llm.routing import tier_momentum
        tier_momentum.record_turn(
            req.project_id, "explore", provider, model, reason="multimodal vision",
        )
        extras["model_source"] = "multimodal_auto"
        return *ensure_provider_available(provider, model), extras

    # Momentum (Job 6, 2026-06-12): complexity-driven escalations no longer
    # latch set_sticky_model() — that pinned the session at the top tier
    # after ONE architectural question. Instead each turn records a decaying
    # momentum score; casual turns during high momentum hold the elevated
    # model for a bounded lag, then the session settles back to cheap.
    from app.llm.routing import tier_momentum

    # ── TIER 4: Architecture — large-scale design ──
    # Gate first (2026-07-01): keyword presence alone is never sufficient —
    # a conversational turn falls through to the cheap tiers below.
    if (not _conversational) and (_arch_hits >= 2 or (_arch_hits >= 1 and _reasoning_hits >= 2)):
        provider, model = get_role_model("ARCHITECT")
        print(f"[MODEL_SELECT] TIER 4 (architecture): arch={_arch_hits}, reasoning={_reasoning_hits} -> {provider}/{model}")
        tier_momentum.record_turn(
            req.project_id, "architect", provider, model,
            reason=f"arch_hits={_arch_hits}, reasoning_hits={_reasoning_hits}",
        )

        if not _skip_confirm:
            confirm_gen = _try_confirmation_gate(complexity, req)
            if confirm_gen is not None:
                extras["confirmation_sse"] = confirm_gen
                return provider, model, extras
        else:
            print(f"[MODEL_SELECT] Confirmation gate skipped (chat panel request)")

        return *ensure_provider_available(provider, model), extras

    # ── TIER 2: Reasoning + exploration ──
    if (not _conversational) and complexity.tier == "deep" and _reasoning_hits >= 1:
        provider, model = get_role_model("REASONING", "FRONTIER")
        print(f"[MODEL_SELECT] TIER 2 (reasoning+tools): explore={_explore_hits}, reasoning={_reasoning_hits} -> {provider}/{model}")
        tier_momentum.record_turn(
            req.project_id, "reasoning", provider, model,
            reason=f"deep tier, reasoning_hits={_reasoning_hits}",
        )
        return *ensure_provider_available(provider, model), extras

    # ── TIER 1: Exploration only ──
    if complexity.tier == "deep" and _explore_hits >= 1:
        provider, model = _chat_provider, _chat_model
        print(f"[MODEL_SELECT] TIER 1 (exploration): explore={_explore_hits}, arch={_arch_hits} -> {provider}/{model}")
        tier_momentum.record_turn(
            req.project_id, "explore", provider, model,
            reason=f"explore_hits={_explore_hits}",
        )
        return *ensure_provider_available(provider, model), extras

    # ── TIER 2: Pure reasoning (no exploration keywords) ──
    if (not _conversational) and complexity.tier == "reasoning":
        provider, model = get_role_model("REASONING", "FRONTIER")
        print(f"[MODEL_SELECT] TIER 2 (reasoning): reasoning={_reasoning_hits} -> {provider}/{model}")
        tier_momentum.record_turn(
            req.project_id, "reasoning", provider, model,
            reason=f"reasoning tier, reasoning_hits={_reasoning_hits}",
        )
        return *ensure_provider_available(provider, model), extras

    # ── Explicit model requests ──
    if complexity.tier.startswith("explicit_"):
        from app.memory.complexity import EXPLICIT_MODEL_PATTERNS
        for _mk, _mc in EXPLICIT_MODEL_PATTERNS.items():
            if _mc["tier"] == complexity.tier:
                provider = _mc["provider"]
                model = _mc["model"]
                break
        else:
            provider = _chat_provider
            model = _chat_model
        print(f"[MODEL_SELECT] EXPLICIT model request: {provider}/{model}")
        set_sticky_model(req.project_id, provider, model, source="explicit_user")
        extras["model_source"] = "explicit_user"
        return *ensure_provider_available(provider, model), extras

    # ── TIER 1: Default — lightweight chat (with momentum hold) ──
    held = tier_momentum.momentum_hold(req.project_id)
    if held:
        provider, model, _momentum = held
        print(
            f"[MODEL_SELECT] TIER HOLD (momentum {_momentum:.2f} >= "
            f"{tier_momentum.HOLD_THRESHOLD}): casual turn keeps {provider}/{model}"
        )
        tier_momentum.record_turn(
            req.project_id, "casual", provider, model,
            reason=f"momentum hold ({_momentum:.2f}), decaying",
        )
        return *ensure_provider_available(provider, model), extras

    provider = _chat_provider
    model = _chat_model
    print(f"[MODEL_SELECT] TIER 1 (default {complexity.tier}): {provider}/{model}")
    tier_momentum.record_turn(
        req.project_id, "casual", provider, model,
        reason=f"default tier ({complexity.tier})",
    )
    return *ensure_provider_available(provider, model), extras


def _try_confirmation_gate(complexity, req) -> Any | None:
    """Attempt to fire the confirmation gate for architect-tier escalation.

    Returns an async generator to yield as an SSE response, or None.
    """
    try:
        from app.llm.routing.confirmation_gate import (
            should_confirm_model_escalation,
            format_confirmation_sse,
        )
        confirm_req = should_confirm_model_escalation(
            from_tier="lookup",
            to_tier="architect",
            confidence=complexity.confidence,
            message=req.message,
        )
        if confirm_req:
            import json as _json

            async def _confirm_stream():
                yield format_confirmation_sse(confirm_req)
                yield f"data: {_json.dumps({'type': 'done', 'provider': 'local', 'model': 'confirmation_gate', 'total_length': 0})}\n\n"

            return _confirm_stream()
    except ImportError:
        pass
    return None

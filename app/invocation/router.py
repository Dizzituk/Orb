# FILE: app/invocation/router.py
"""
Invocation Router - thin orchestrator.

Takes an InvocationRequest, classifies the intent, runs it through the
tier gate, dispatches to the appropriate LLM path, and persists the
turn into the ambient conversation memory so subsequent invocations in
the same session have context.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import List, Optional, Tuple

from app.invocation.classifier import classify
from app.invocation.tier_gate import decide
from app.invocation.models import (
    Intent,
    Tier,
    InvocationRequest,
    InvocationResponse,
)
from app.voice_ambient.conversation_memory import get_conversation_memory
from app.voice_ambient.persistence import record_turn

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are Astra, the user's personal AI assistant, answering through a "
    "voice interface. Keep replies concise and conversational - ideally one "
    "to three short sentences. Use the prior conversation to resolve "
    "pronouns and follow-up questions. Don't read lists, headings, or "
    "markdown out loud; phrase everything as natural spoken prose."
)


_LOOK_PROMPTS = {
    Intent.LOOK_READ: (
        "Extract the main readable content from this screenshot. Ignore "
        "navigation, ads and sidebars. Return only the clean readable text "
        "- no commentary."
    ),
    Intent.LOOK_DESCRIBE: (
        "Briefly describe what is visible on this screen in one or two "
        "sentences. Be factual. No speculation. Keep it short enough to speak aloud."
    ),
    Intent.LOOK_EXPLAIN: (
        "The user is asking about what they are looking at. Explain what is "
        "shown, what they might be trying to do, and any obvious next step. "
        "Direct and practical. Under 80 words."
    ),
    Intent.LOOK_DIAGNOSE: (
        "The user thinks something on this screen is not working correctly. "
        "Look for errors, unexpected states, broken layouts. Give the most "
        "likely diagnosis plus a concrete next step. Under 100 words."
    ),
}


def _decode_screenshot(b64: Optional[str]) -> Optional[bytes]:
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception as e:
        logger.warning("Failed to decode screenshot base64: %s", e)
        return None


def _format_history_as_prose(history: List[dict]) -> str:
    """Render prior turns as plain prose for LOOK prompts (Gemini Vision doesn't take chat history directly in our wrapper)."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Astra"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def route(req: InvocationRequest) -> InvocationResponse:
    """Dispatch an invocation and return the result. Records the turn."""
    start = time.time()

    intent = classify(req.transcript)
    tier_decision = decide(req, intent)

    logger.info(
        "[invocation] session=%s intent=%s rule=%s tier=%s reasons=%s",
        (req.session_id or "-")[:8],
        intent.intent.value,
        intent.matched_rule,
        tier_decision.tier.value,
        [r.value for r in tier_decision.reasons],
    )

    memory = get_conversation_memory()
    history = memory.as_llm_messages(req.session_id) if req.session_id else []

    answer = ""
    provider = None
    model = None
    error = None

    try:
        if intent.needs_screenshot and req.screenshot_b64:
            answer, provider, model = await _handle_look(
                req, intent, tier_decision, history
            )
        else:
            answer, provider, model = await _handle_converse(
                req, intent, tier_decision, history
            )
    except Exception as e:
        logger.exception("[invocation] dispatch failed")
        error = str(e)
        answer = f"Sorry, I hit an error: {e}"

    # Persist the turn on success (skip on empty/error)
    if req.session_id and not error and answer:
        try:
            memory.append_user(req.session_id, req.transcript)
            memory.append_assistant(req.session_id, answer)
            logger.info("[invocation] session=%s memory updated, total=%d turns",
                        req.session_id[:8], len(memory.get_history(req.session_id)))
        except Exception:
            logger.exception("[invocation] failed to persist turn")

        # Write to the chat messages DB so it appears in the user's chat history
        # and survives backend restart. Best-effort; in-memory memory keeps the
        # voice session functional even if DB write fails.
        try:
            from app.db import SessionLocal
            _db = SessionLocal()
            try:
                record_turn(
                    db=_db,
                    session_id=req.session_id,
                    user_text=req.transcript,
                    assistant_text=answer,
                    provider=provider,
                    model=model,
                )
            finally:
                _db.close()
        except Exception:
            logger.exception("[invocation] voice persistence write failed")

    elapsed_ms = int((time.time() - start) * 1000)

    return InvocationResponse(
        intent=intent,
        tier_decision=tier_decision,
        answer=answer,
        provider=provider,
        model=model,
        error=error,
        elapsed_ms=elapsed_ms,
    )


async def _handle_look(req, intent, tier_decision, history):
    """LOOK-family: existing ask_about_image, with history as prose preamble."""
    from app.llm.gemini_vision import ask_about_image
    import asyncio

    image_bytes = _decode_screenshot(req.screenshot_b64)
    if not image_bytes:
        return ("Couldn't process the screenshot.", None, None)

    base_prompt = _LOOK_PROMPTS.get(
        intent.intent,
        "Describe what is on this screen and help the user with whatever they need. Keep it short enough to speak aloud."
    )
    parts = [
        "(Voice mode: reply in 1-2 short sentences as spoken prose. No lists, no markdown, no headings.)",
        base_prompt,
    ]
    history_prose = _format_history_as_prose(history)
    if history_prose:
        parts.append("Prior conversation for context:\n" + history_prose)
    parts.append(f"User just said: {req.transcript!r}")
    if req.active_window_title:
        parts.append(f"Active window: {req.active_window_title}")
    user_prompt = "\n\n".join(parts)

    # For ambient voice, route single-screenshot LOOK tasks to the FAST tier
    # (gemini-flash-latest). The COMPLEX tier (gemini-3.1-pro-preview) is
    # 5-10x slower than flash, and single-screen "describe this" does not need
    # its reasoning depth. Only escalate to complex on T4_OPUS promotion.
    tier_str = "complex" if tier_decision.tier == Tier.T4_OPUS else "fast"
    result = await asyncio.to_thread(
        ask_about_image,
        image_source=image_bytes,
        user_question=user_prompt,
        mime_type=req.screenshot_mime,
        tier=tier_str,
    )

    return (
        result.get("answer", "") or "(empty response)",
        result.get("provider"),
        result.get("model"),
    )


async def _handle_converse(req, intent, tier_decision, history):
    """CONVERSE: multi-turn chat with history."""
    from app.llm.schemas import LLMTask, JobType
    from app.llm.routing.core import call_llm_async

    # Build messages: history + current user turn.
    # We wrap the user turn with an explicit brevity hint so the model replies
    # in a voice-friendly short form even if the system prompt gets diluted by
    # downstream envelope processing (memory injection, project context, etc).
    messages = list(history) if history else []
    wrapped_user = (
        f"(Voice mode: reply in 1-2 short sentences as spoken prose. "
        f"No lists, no markdown.)\n\n{req.transcript}"
    )
    messages.append({"role": "user", "content": wrapped_user})

    # Pick job type based on tier
    job_type = JobType.CHAT_LIGHT
    if tier_decision.tier in (Tier.T4_OPUS, Tier.T3_MULTI):
        job_type = JobType.ORCHESTRATOR
    elif tier_decision.tier == Tier.T2_STD:
        job_type = JobType.CHAT_LIGHT

    task = LLMTask(
        messages=messages,
        job_type=job_type,
        system_prompt=_SYSTEM_PROMPT,
    )
    try:
        result = await call_llm_async(task)
        return (
            (result.content or "").strip() or "(empty response)",
            result.provider,
            result.model,
        )
    except Exception as e:
        logger.exception("[invocation] converse failed")
        return (f"(converse failed: {e})", None, None)
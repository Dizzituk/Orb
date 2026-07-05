# FILE: app/llm/image_core.py
# Purpose: Non-streaming image-generation path for the bridge (generate_image_core).
# Called-by: app.llm.image_router (shim re-export -> bridge.capability_layer)
# Depends-on: app.llm.image_providers, app.llm.image_research
# Last-renovated: 2026-06-21
"""Non-streaming image generation for the bridge route.

Split out of image_router.py (batch 3, 2026-06-21). Mirrors image_stream.py;
the ~80% overlap is PRESERVED verbatim (no dedup) per the manifest warning.
"""
import logging
import os
from typing import Optional

from sqlalchemy.orm import Session

from app.llm.image_providers import (
    _get_provider,
    _get_fallback_provider,
    _generate_with_provider,
)
from app.llm.image_research import _run_research

logger = logging.getLogger(__name__)


async def generate_image_core(
    project_id: int,
    message: str,
    db: Session,
) -> Optional[dict]:
    """Non-streaming image generation for the bridge route.

    Mirrors the same logic as generate_image_stream() — classification,
    refinement detection, prompt synthesis, primary provider + fallback —
    but returns a single result dict instead of yielding SSE events.
    Status updates are dropped; the bridge has its own user-facing
    progress UX (the phone's "thinking" indicator covers the latency).

    Does NOT save messages to memory — the bridge router persists its own
    user+assistant messages, including any artifact markers, via its
    existing create_message calls. We only save the synth prompt for
    refinement chaining (same as the stream variant).

    Returns the generator result dict (path, filename, size_bytes,
    mime_type, base64_data, provider, model) on success, or None on
    total failure. The "provider" and "model" keys are added by this
    function (the underlying generators don't include them) so the caller
    has the info it needs without re-deriving.

    v1.0 (2026-05-25): Initial implementation for bridge image support.
    """
    primary = _get_provider()
    fallback = _get_fallback_provider()
    used_provider = primary
    model_name = os.getenv("IMAGE_GEN_MODEL", "")
    result: Optional[dict] = None
    synth_prompt: Optional[str] = None
    aspect_ratio: Optional[str] = None

    # Stage 0 — classify
    try:
        from app.llm.image_type_classifier import classify_image_request, ImageType
        classification = classify_image_request(message)
        image_type = classification.image_type
        logger.info(
            "[image_core] Classification: %s (confidence=%.2f, reason=%s)",
            image_type.value, classification.confidence, classification.reason,
        )
    except Exception as e:
        logger.warning("[image_core] Classification failed: %s, defaulting to creative", e)
        from app.llm.image_type_classifier import ImageType  # for the comparison below
        image_type = ImageType.CREATIVE

    # Branch A: DATA_CHART — inline data extraction + Plotly
    if image_type == ImageType.DATA_CHART:
        try:
            from app.llm.chart_inline_extractor import extract_inline_chart_data
            from app.llm.chart_renderer import render_chart
            chart_data = await extract_inline_chart_data(message)
            if chart_data:
                result = render_chart(chart_data)
                if result:
                    used_provider = "plotly"
                    model_name = "plotly/kaleido"
                    logger.info("[image_core] Plotly chart rendered: %s", result["filename"])
        except Exception as e:
            logger.warning("[image_core] Inline chart pipeline failed: %s", e)

    # Branch B: DATA_RESEARCH — web search + Plotly (with AI fallback)
    elif image_type == ImageType.DATA_RESEARCH:
        try:
            research_context = await _run_research(message)
            if research_context:
                from app.llm.chart_data_extractor import extract_chart_data
                from app.llm.chart_renderer import render_chart
                chart_data = await extract_chart_data(
                    user_message=message,
                    research_text=research_context,
                )
                if chart_data:
                    result = render_chart(chart_data)
                    if result:
                        used_provider = "plotly"
                        model_name = "plotly/kaleido"
                        logger.info("[image_core] Plotly chart rendered: %s", result["filename"])
        except Exception as e:
            logger.warning("[image_core] Research chart pipeline failed: %s", e)

    # Branch C: CREATIVE — or fallback from failed chart paths
    if not result:
        # DATA_CHART failed and we won't hallucinate data via AI on the bridge route either
        if image_type == ImageType.DATA_CHART:
            logger.warning("[image_core] DATA_CHART failed; refusing AI fallback")
            return None

        # Refinement detection — thread previous prompt through synth
        previous_prompt: Optional[str] = None
        try:
            from app.llm.routing.chat_intent_detection import detect_image_refinement
            from app.llm.image_prompt_log import get_last_prompt_for_project
            if detect_image_refinement(message):
                previous_prompt = get_last_prompt_for_project(project_id)
                if previous_prompt:
                    logger.info(
                        "[image_core] Refinement detected, threading prev prompt (%d chars)",
                        len(previous_prompt),
                    )
        except Exception as e:
            logger.warning("[image_core] Refinement lookup failed: %s", e)

        # Prompt synthesis
        try:
            from app.llm.image_prompt_synth import synthesise_image_prompt
            try:
                from app.memory import service as mem_svc
                all_msgs = mem_svc.list_messages(db, project_id, limit=100)
                recent = all_msgs[-8:] if len(all_msgs) > 8 else all_msgs
                history = [{"role": m.role, "content": m.content} for m in recent]
            except Exception as hist_err:
                logger.warning("[image_core] Failed to load history: %s", hist_err)
                history = None

            synth_prompt, aspect_ratio = await synthesise_image_prompt(
                user_message=message,
                conversation_history=history,
                previous_image_prompt=previous_prompt,
            )
            logger.info("[image_core] Synthesised prompt: %s (ar=%s)",
                        synth_prompt[:120], aspect_ratio or "default")
        except Exception as e:
            logger.warning("[image_core] Prompt synthesis failed, using raw: %s", e)
            synth_prompt = message

        # Primary then fallback
        result = await _generate_with_provider(primary, prompt=synth_prompt, aspect_ratio=aspect_ratio)
        if not result and fallback and fallback != primary:
            logger.info("[image_core] Primary %s failed, trying %s", primary, fallback)
            result = await _generate_with_provider(fallback, prompt=synth_prompt, aspect_ratio=aspect_ratio)
            used_provider = fallback
            model_name = os.getenv("IMAGE_GEN_FALLBACK_MODEL", "")

    if not result:
        logger.error("[image_core] All generation methods exhausted")
        return None

    # Save the synth prompt so future refinements can thread it through.
    # Plotly charts skip this — refinements there go through chart_data flow.
    if used_provider != "plotly" and synth_prompt:
        try:
            from app.llm.image_prompt_log import save_prompt
            save_prompt(project_id, result["filename"], synth_prompt)
        except Exception as e:
            logger.warning("[image_core] Failed to save prompt log: %s", e)

    # Enrich the result with provider/model so the bridge can attribute it
    result["provider"] = used_provider
    result["model"] = model_name
    return result

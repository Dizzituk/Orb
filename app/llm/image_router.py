# FILE: app/llm/image_router.py
# NOTE: Plotly chart rendering requires: pip install plotly kaleido
"""
Image generation endpoint + SSE stream handler.

Provider routing read from .env:
  IMAGE_GEN_PROVIDER / IMAGE_GEN_MODEL — primary (default: openai / gpt-image-1.5)
  IMAGE_GEN_FALLBACK_PROVIDER / IMAGE_GEN_FALLBACK_MODEL — fallback (default: google / gemini-2.5-flash-image)

v3.0 (2026-03-20): Added generate_image_stream for end-to-end chat integration.
v2.0 (2026-03-20): Env-driven provider routing with automatic fallback.
v1.0 (2026-03-09): Initial implementation (Nano Banana only).
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth.middleware import require_auth, AuthResult
from app.memory import service as memory_service, schemas as memory_schemas

logger = logging.getLogger(__name__)

router = APIRouter(tags=["image_generation"])


def _get_provider() -> str:
    return os.getenv("IMAGE_GEN_PROVIDER", "openai").lower()


def _get_fallback_provider() -> str:
    return os.getenv("IMAGE_GEN_FALLBACK_PROVIDER", "google").lower()


class ImageGenRequest(BaseModel):
    project_id: int
    prompt: str


async def _generate_with_provider(
    provider: str,
    prompt: str,
    aspect_ratio: Optional[str] = None,
) -> Optional[dict]:
    """Call the appropriate backend based on provider string."""
    if provider == "openai":
        from app.llm.image_gen import generate_image
        return await generate_image(prompt=prompt, aspect_ratio=aspect_ratio)
    elif provider in ("google", "gemini"):
        from app.llm.nano_banana import generate_image
        return await generate_image(prompt=prompt, aspect_ratio=aspect_ratio)
    else:
        logger.error("[image_router] Unknown provider: %s", provider)
        return None


@router.post("/generate-image")
async def generate_image_endpoint(
    req: ImageGenRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """Generate an image using configured provider, with automatic fallback."""
    primary = _get_provider()
    fallback = _get_fallback_provider()
    used_provider = primary

    logger.info("[image_router] Generating image: primary=%s, fallback=%s", primary, fallback)

    # Try primary
    result = await _generate_with_provider(primary, prompt=req.prompt)

    # Fallback if primary failed and fallback is different
    if not result and fallback and fallback != primary:
        logger.warning("[image_router] Primary (%s) failed, trying fallback (%s)", primary, fallback)
        result = await _generate_with_provider(fallback, prompt=req.prompt)
        used_provider = fallback

    if not result:
        raise HTTPException(500, "Image generation failed (primary and fallback)")

    # Resolve model name for memory log
    if used_provider == "openai":
        model_name = os.getenv("IMAGE_GEN_MODEL", "gpt-image-1.5")
    else:
        model_name = os.getenv("IMAGE_GEN_FALLBACK_MODEL", "gemini-2.5-flash-image")

    # Save to memory
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="user",
        content=f"[Image request] {req.prompt}",
        provider="local",
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="assistant",
        content=f"Generated image: {result['filename']}",
        provider=used_provider,
        model=model_name,
    ))

    return {
        "success": True,
        "provider": used_provider,
        "model": model_name,
        "file": {
            "path": result["path"],
            "filename": result["filename"],
            "type": "image",
            "size": result["size_bytes"],
            "description": f"Generated: {req.prompt[:80]}",
        },
        "data_uri": f"data:{result['mime_type']};base64,{result['base64_data']}",
    }


# =============================================================================
# SSE STREAM HANDLER (for chat integration via dispatch table)
# =============================================================================

def _sse(event_type: str, **kwargs) -> str:
    """Build an SSE data line."""
    payload = {"type": event_type, **kwargs}
    return f"data: {json.dumps(payload)}\n\n"


def _sse_token(text: str) -> str:
    """Emit a token event (rendered as chat text by frontend)."""
    return _sse("token", content=text)


def _sse_status(text: str) -> str:
    """Emit a status message as a token (frontend only renders 'token' type)."""
    return _sse("token", content=f"*{text}*\n")


# Keywords that suggest the user wants data gathered before image creation
_RESEARCH_KEYWORDS = [
    "latest", "recent", "current", "data", "benchmark", "benchmarks",
    "statistics", "stats", "compare", "comparison", "research",
    "findings", "results", "performance", "ranking", "rankings",
    "trends", "trend", "chart", "graph", "infographic",
    "numbers", "figures", "metrics", "scores",
]


def _needs_research(message: str) -> bool:
    """Detect if the image request implies data gathering first."""
    lower = message.lower()
    return any(kw in lower for kw in _RESEARCH_KEYWORDS)


async def _run_research(message: str) -> Optional[str]:
    """Run targeted multi-search research for chart data.

    Uses LLM to plan focused queries, runs multiple Brave searches,
    and combines results for richer data extraction.
    """
    try:
        from app.llm.chart_research import run_multi_search
        return await run_multi_search(message)
    except ImportError:
        logger.warning("[image_stream] chart_research not available, falling back to single search")
        # Fallback: single search with cleaned query
        try:
            from app.llm.web_search import WebSearchRequest, search_and_answer
            from app.llm.chart_research import _clean_query
            query = _clean_query(message)
            req = WebSearchRequest(query=query, max_results=5)
            result = await search_and_answer(req)
            if result.ok and result.answer:
                return result.answer
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning("[image_stream] Research step failed: %s", e)
        return None


async def generate_image_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[object] = None,
) -> AsyncIterator[str]:
    """Stream image generation results as SSE events.

    Called by the dispatch table when GENERATE_IMAGE intent is matched.

    v4.0 (2026-04-02): Image type classification + inline data extraction.

    Pipeline branches:
      DATA_CHART    → inline extract → Plotly render (no AI, exact data)
      DATA_RESEARCH → web search → LLM extract → Plotly render
      CREATIVE      → prompt synthesis → GPT Image 1.5 / Nano Banana
    """
    primary = _get_provider()
    fallback = _get_fallback_provider()
    used_provider = primary
    model_name = os.getenv("IMAGE_GEN_MODEL", "gpt-image-1.5")
    result = None
    synth_prompt: Optional[str] = None  # set in creative branch; saved on success

    # Emit metadata so frontend shows the correct provider badge
    yield _sse("metadata", provider=primary, model=model_name)

    # =========================================================================
    # Stage 0: CLASSIFY — what kind of image is this?
    # =========================================================================
    try:
        from app.llm.image_type_classifier import classify_image_request, ImageType
        classification = classify_image_request(message)
        image_type = classification.image_type
        logger.info(
            "[image_stream] Classification: %s (confidence=%.2f, reason=%s)",
            image_type.value, classification.confidence, classification.reason,
        )
        yield _sse_status(f"Detected: {image_type.value} request...")
    except ImportError:
        logger.warning("[image_stream] Classifier not available, defaulting to creative")
        image_type = "creative"
    except Exception as e:
        logger.warning("[image_stream] Classification failed: %s, defaulting to creative", e)
        image_type = "creative"

    # =========================================================================
    # Branch A: DATA_CHART — user provided inline data → Plotly (no AI)
    # =========================================================================
    if image_type == ImageType.DATA_CHART:
        yield _sse_status("Extracting your data for chart rendering...")
        try:
            from app.llm.chart_inline_extractor import extract_inline_chart_data
            from app.llm.chart_renderer import render_chart

            chart_data = await extract_inline_chart_data(message)

            if chart_data:
                chart_type = chart_data.get("chart_type", "bar")
                yield _sse_status(f"Rendering {chart_type} chart with Plotly...")
                result = render_chart(chart_data)

                if result:
                    used_provider = "plotly"
                    model_name = "plotly/kaleido"
                    logger.info("[image_stream] Plotly chart rendered: %s", result["filename"])
                else:
                    logger.warning("[image_stream] Plotly render failed")
                    yield _sse_status("Chart render failed. Please check data format.")
            else:
                logger.warning("[image_stream] Inline data extraction returned nothing")
                yield _sse_status("Couldn't extract chart data from your message.")

        except ImportError as e:
            logger.error("[image_stream] Chart modules not available: %s", e)
            yield _sse_status("Chart rendering not available (missing plotly/kaleido).")
        except Exception as e:
            logger.error("[image_stream] Inline chart pipeline failed: %s", e)
            yield _sse_status(f"Chart pipeline error: {e}")

    # =========================================================================
    # Branch B: DATA_RESEARCH — needs web search first → then Plotly
    # =========================================================================
    elif image_type == ImageType.DATA_RESEARCH:
        yield _sse_status("Researching data for your chart...")
        research_context = await _run_research(message)

        if research_context:
            logger.info("[image_stream] Research gathered: %d chars", len(research_context))
            yield _sse_status("Data gathered. Extracting chart data...")

            try:
                from app.llm.chart_data_extractor import extract_chart_data
                from app.llm.chart_renderer import render_chart

                chart_data = await extract_chart_data(
                    user_message=message,
                    research_text=research_context,
                )

                if chart_data:
                    chart_type = chart_data.get("chart_type", "bar")
                    yield _sse_status(f"Rendering {chart_type} chart with Plotly...")
                    result = render_chart(chart_data)

                    if result:
                        used_provider = "plotly"
                        model_name = "plotly/kaleido"
                        logger.info("[image_stream] Plotly chart rendered: %s", result["filename"])
                    else:
                        yield _sse_status("Chart render failed. Falling back to AI...")
                else:
                    yield _sse_status("Couldn't extract chart data. Falling back to AI...")

            except ImportError:
                yield _sse_status("Plotly not available. Falling back to AI...")
            except Exception as e:
                logger.warning("[image_stream] Chart pipeline failed: %s", e)
                yield _sse_status("Chart pipeline error. Falling back to AI...")
        else:
            yield _sse_status("Research returned no data. Falling back to AI...")

    # =========================================================================
    # Branch C: CREATIVE — or fallback from failed chart paths
    # =========================================================================
    if not result:
        # Only do creative AI if the type was CREATIVE or chart paths failed
        if image_type == ImageType.DATA_CHART:
            # DATA_CHART failed — do NOT fall back to AI (it will hallucinate)
            yield _sse("error", error="Chart rendering failed. AI image models cannot "
                       "render accurate data charts. Please check your data format "
                       "or try simplifying the request.")
            yield _sse("done", provider="none", model="none", total_length=0)
            return

        yield _sse_status("Enriching image prompt...")

        # If this looks like a refinement of a previous image, thread the
        # previous synth prompt through so the synth's refinement path fires.
        # Without this, "recreate that image" loses its subject anchor and
        # the image-gen model autocompletes to canonical training-data examples
        # (e.g. asks for a steep-decline chart -> gets WWF biodiversity index).
        previous_prompt: Optional[str] = None
        try:
            from app.llm.routing.chat_intent_detection import detect_image_refinement
            from app.llm.image_prompt_log import get_last_prompt_for_project
            if detect_image_refinement(message):
                previous_prompt = get_last_prompt_for_project(project_id)
                if previous_prompt:
                    logger.info(
                        "[image_stream] Refinement detected \u2014 threading previous prompt (%d chars)",
                        len(previous_prompt),
                    )
        except Exception as e:
            logger.warning("[image_stream] Refinement lookup failed: %s", e)

        # Prompt synthesis for creative images
        try:
            from app.llm.image_prompt_synth import synthesise_image_prompt

            try:
                from app.memory import service as mem_svc
                all_msgs = mem_svc.list_messages(db, project_id, limit=100)
                recent = all_msgs[-8:] if len(all_msgs) > 8 else all_msgs
                history = [{"role": m.role, "content": m.content} for m in recent]
                logger.info("[image_stream] Loaded %d history messages for prompt synth", len(history))
            except Exception as hist_err:
                logger.warning("[image_stream] Failed to load history: %s", hist_err)
                history = None

            synth_prompt, aspect_ratio = await synthesise_image_prompt(
                user_message=message,
                conversation_history=history,
                previous_image_prompt=previous_prompt,
            )
            logger.info("[image_stream] Synthesised prompt: %s (ar=%s)",
                         synth_prompt[:120], aspect_ratio or "default")
        except Exception as e:
            logger.warning("[image_stream] Prompt synthesis failed, using raw: %s", e)
            synth_prompt = message
            aspect_ratio = None

        yield _sse_status(f"Generating image with {primary}...")

        # Primary provider
        result = await _generate_with_provider(primary, prompt=synth_prompt, aspect_ratio=aspect_ratio)

        # Fallback provider
        if not result and fallback and fallback != primary:
            yield _sse_status(f"Primary failed, trying {fallback}...")
            result = await _generate_with_provider(fallback, prompt=synth_prompt, aspect_ratio=aspect_ratio)
            used_provider = fallback
            model_name = os.getenv("IMAGE_GEN_FALLBACK_MODEL", "gemini-2.5-flash-image")

    # =========================================================================
    # Stage 3: Emit results
    # =========================================================================
    if not result:
        yield _sse("error", error="Image generation failed (all methods exhausted).")
        yield _sse("done", provider=used_provider, model=model_name, total_length=0)
        return

    # Build inline image as URL served by the backend static mount
    # (data URIs are too large for the frontend markdown renderer)
    image_url = f"http://localhost:8000/output/images/{result['filename']}"
    yield _sse_token(f"![Generated Image]({image_url})\n\n")

    # Text summary below the image — file path as a clickable link
    file_link = f"[{result['path']}](file://{result['path'].replace(chr(92), '/')})"
    if used_provider == "plotly":
        response_text = f"Data chart rendered with Plotly: **{result['filename']}**\n{file_link}\n"
    else:
        response_text = f"Generated with {used_provider}/{model_name}: **{result['filename']}**\n{file_link}\n"

    yield _sse_token(response_text)

    # Save to memory
    try:
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id,
            role="user",
            content=f"[Image request] {message}",
            provider="local",
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id,
            role="assistant",
            content=response_text,
            provider=used_provider,
            model=model_name,
        ))
    except Exception as e:
        logger.warning("[image_stream] Failed to save history: %s", e)

    # Save the synth prompt so future "recreate that image" / "make it darker"
    # requests can thread it back through the synth's refinement path.
    # Plotly charts skip this \u2014 refinements there go through chart_data flow.
    if used_provider != "plotly" and synth_prompt:
        try:
            from app.llm.image_prompt_log import save_prompt
            save_prompt(project_id, result['filename'], synth_prompt)
        except Exception as e:
            logger.warning("[image_stream] Failed to save prompt log: %s", e)

    yield _sse("done", provider=used_provider, model=model_name,
              total_length=len(response_text))

# FILE: app/content/style_analyser.py
# Purpose: Style analysis pipeline — sends style references to Gemini for analysis.
# Called-by: app.content.style_router
# Depends-on: app.content, app.content.project_models, app.content.style_service, app.llm.clients (+1 more)
# Last-renovated: 2026-06-11
"""
Style analysis pipeline — sends style references to Gemini for analysis.
Uses existing app/llm/gemini_vision.py and app/llm/clients.py.
"""

import asyncio
import logging
import base64
import os
import mimetypes

from typing import Optional
from sqlalchemy.orm import Session

from app.content.project_models import StyleReference, StyleCategory
from app.content import style_service

logger = logging.getLogger(__name__)

PROMPTS = {
    StyleCategory.video: (
        "Analyse this video/frame for visual style. Describe: colour palette, "
        "lighting, composition, text overlays, transitions, mood, energy, target audience. "
        "Be specific and actionable for reproducing this style."
    ),
    StyleCategory.image: (
        "Analyse this image for design style. Describe: colour palette (hex codes), "
        "typography, layout, whitespace, graphic elements, mood, aesthetic. "
        "Be specific and actionable."
    ),
    StyleCategory.blog: (
        "Analyse this text for writing style. Describe: tone, sentence structure, "
        "vocabulary level, header/list usage, paragraph length, narrative style, "
        "target audience. Be specific and actionable."
    ),
    StyleCategory.brand: (
        "Analyse this brand asset. Describe: primary/secondary colours (hex), "
        "logo style, typography, design language, brand personality. "
        "Be specific and actionable."
    ),
}


async def analyse_reference(db: Session, reference_id: str, sse_callback=None) -> Optional[StyleReference]:
    ref = style_service.get_reference(db, reference_id)
    if not ref:
        return None

    style_service.mark_analysing(db, reference_id)
    if sse_callback:
        await sse_callback({"type": "job_started", "data": {"reference_id": reference_id, "status": "analysing"}})

    try:
        prompt = PROMPTS.get(ref.category, PROMPTS[StyleCategory.image])

        if ref.category in (StyleCategory.video, StyleCategory.image, StyleCategory.brand):
            notes = await _analyse_visual(ref, prompt)
        else:
            notes = await _analyse_text(ref, prompt)

        result = style_service.save_analysis_result(db, reference_id, notes=notes)
        if sse_callback:
            await sse_callback({"type": "analysis_result", "data": {"reference_id": reference_id, "status": "done", "notes": notes}})
        return result

    except Exception as e:
        logger.error(f"Analysis failed for {reference_id}: {e}")
        style_service.mark_failed(db, reference_id, str(e))
        if sse_callback:
            await sse_callback({"type": "analysis_result", "data": {"reference_id": reference_id, "status": "failed", "error": str(e)}})
        return None


async def _analyse_visual(ref: StyleReference, prompt: str) -> str:
    try:
        from app.llm.gemini_vision import analyse_image
        if not os.path.exists(ref.upload_path):
            raise FileNotFoundError(ref.upload_path)
        with open(ref.upload_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        mime = ref.mime_type or mimetypes.guess_type(ref.upload_path)[0] or "image/jpeg"
        return await analyse_image(image_data=data, mime_type=mime, prompt=prompt)
    except ImportError:
        logger.warning("gemini_vision not available, using fallback")
        return f"[Pending] Style analysis for {ref.filename}. Connect Gemini Vision to complete."


async def _analyse_text(ref: StyleReference, prompt: str) -> str:
    try:
        if not os.path.exists(ref.upload_path):
            raise FileNotFoundError(ref.upload_path)
        with open(ref.upload_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(10000)
        full_prompt = f"{prompt}\n\nContent:\n---\n{text}\n---"
        from app.llm.clients import get_client
        client = get_client("gemini")
        resp = await client.chat(messages=[{"role": "user", "content": full_prompt}], model="gemini-2.0-flash")
        return resp.get("content", "Analysis complete.")
    except ImportError:
        return f"[Pending] Text analysis for {ref.filename}. Connect LLM clients to complete."


async def analyse_all(db: Session, project_id: str, category=None, sse_callback=None):
    refs = style_service.list_references(db, project_id, category)
    pending = [r for r in refs if r.analysis_status == "pending"]
    for ref in pending:
        await analyse_reference(db, ref.id, sse_callback)
        await asyncio.sleep(0.5)

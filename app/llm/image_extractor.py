# FILE: app/llm/image_extractor.py
# Purpose: Image prompt extractor + SSE stream wrapper.
# Called-by: app.llm.routing.chat_routing
# Depends-on: app.llm.image_gen, app.llm.image_prompt_log
# Last-renovated: 2026-06-11
"""
Image prompt extractor + SSE stream wrapper.

The new image-gen architecture (v16, 2026-05-01):
    - Chat LLM (gpt-5.4) handles the whole turn with full context.
    - It emits a single line:  [IMAGE_PROMPT]: <complete prompt>
    - This wrapper passes chat tokens through unchanged, accumulates the
      text, and after the inner stream finishes, pulls out the prompt
      and dispatches it directly to gpt-image-2.
    - No Gemini synth in the middle. No classifier branching. The chat
      LLM had every byte of conversation in scope when it wrote the
      prompt — it does not need a downstream "translator" mangling it.

Why a marker rather than tool calls:
    Tool calls work but require provider-specific SDK plumbing (OpenAI
    function calling, Anthropic tools API, Gemini function declarations)
    and add a round-trip per image. A text marker is provider-agnostic,
    streams naturally, and keeps the chat LLM's narrative response intact
    above the rendered image.

v1.0 (2026-05-01): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


# Marker matches lines like:
#     [IMAGE_PROMPT]: A square Instagram graphic ...
#     **[IMAGE_PROMPT]:** A square Instagram graphic ...    (LLM sometimes bolds it)
#     `[IMAGE_PROMPT]:` A square Instagram graphic ...      (LLM sometimes code-fences)
# Captures the rest of the line as the prompt. Multi-line prompts are
# discouraged in the system prompt — the marker should be a single line.
_MARKER_RE = re.compile(
    r"(?:\*\*|`)?\s*\[IMAGE_PROMPT\]\s*:?\s*(?:\*\*|`)?\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def extract_image_prompt(text: str) -> Optional[str]:
    """Pull the first [IMAGE_PROMPT] payload out of an assistant message.

    Returns the prompt body (stripped of marker, surrounding whitespace,
    and any wrapping quotes the LLM might have added) or None if the
    message has no marker.
    """
    if not text:
        return None
    match = _MARKER_RE.search(text)
    if not match:
        return None
    payload = match.group(1).strip()
    # Trim wrapping quotes / backticks the LLM sometimes adds around the prompt.
    payload = payload.strip("\"'`")
    if not payload or len(payload) < 10:
        return None
    return payload


def strip_image_prompt_marker(text: str) -> str:
    """Remove the [IMAGE_PROMPT] line from a message — used for chat persistence
    cleanup if we want to keep the saved transcript free of routing markers."""
    if not text:
        return text
    return _MARKER_RE.sub("", text).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Stream wrapper
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Format an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


async def wrap_stream_with_image_dispatch(
    inner_gen: AsyncIterator[str],
    project_id: int,
    db,
) -> AsyncIterator[str]:
    """Wrap a chat SSE stream — pass tokens through, then render any image.

    Pipeline:
        1. Yield every chunk from `inner_gen` to the client AS IT ARRIVES,
           except 'done' events which we hold until the image render finishes.
        2. Accumulate token content for marker scanning.
        3. After the inner stream ends, pull out the [IMAGE_PROMPT] block.
        4. If found, call image_gen.generate_image (gpt-image-2) directly
           and yield image SSE events.
        5. Yield the final 'done' event last so the frontend stops the
           thinking indicator only after the image is in.

    If the inner stream produces no marker, this is a no-op pass-through.
    """
    accumulated = ""
    held_done: Optional[str] = None

    async for chunk in inner_gen:
        # Try to peek into the SSE event type. Non-SSE chunks (e.g. heartbeats)
        # are passed through untouched.
        event = _safe_parse_sse(chunk)
        event_type = event.get("type") if event else None

        if event_type == "done":
            # Hold — we may need to render an image before this fires.
            held_done = chunk
            continue

        if event_type == "token":
            content = event.get("content", "")
            if isinstance(content, str):
                accumulated += content

        yield chunk

    # ----- inner stream complete -----
    image_prompt = extract_image_prompt(accumulated)
    if image_prompt:
        try:
            async for event in _dispatch_image(image_prompt, project_id, db):
                yield event
        except Exception as e:
            logger.error("[image_extractor] Dispatch failed: %s", e, exc_info=True)
            yield _sse({
                "type": "token",
                "content": f"\n*Image generation failed: {e}*\n",
            })

    if held_done is not None:
        yield held_done
    else:
        # Inner stream forgot to send 'done' — emit one so the frontend ends.
        yield _sse({"type": "done", "provider": "openai", "model": os.getenv("IMAGE_GEN_MODEL", ""), "total_length": 0})


# ---------------------------------------------------------------------------
# Image dispatch (the actual gpt-image-2 call)
# ---------------------------------------------------------------------------

async def _dispatch_image(
    prompt: str,
    project_id: int,
    db,
) -> AsyncIterator[str]:
    """Call gpt-image-2 with the extracted prompt and yield SSE events.

    Bypasses image_router.generate_image_stream entirely — no classifier,
    no synth, no fallback to Gemini. Just: prompt in, image out.
    """
    model_label = os.getenv("IMAGE_GEN_MODEL", "") or "the image model"
    yield _sse({"type": "token", "content": f"\n*Rendering image with {model_label}...*\n"})

    aspect_ratio = _detect_aspect_ratio(prompt)

    from app.llm.image_gen import generate_image
    result = await generate_image(prompt=prompt, aspect_ratio=aspect_ratio)

    if not result:
        yield _sse({
            "type": "token",
            "content": "\n*Image generation returned no result.*\n",
        })
        return

    # Inline image as a URL served by the backend's static mount.
    image_url = f"http://localhost:8000/output/images/{result['filename']}"
    yield _sse({"type": "token", "content": f"\n![Generated Image]({image_url})\n\n"})

    file_path = result["path"]
    file_link = f"[{file_path}](file://{file_path.replace(chr(92), '/')})"
    yield _sse({
        "type": "token",
        "content": f"Generated with openai/{model_label}: **{result['filename']}**\n{file_link}\n",
    })

    # Persist the prompt for refinement loops ("make it darker", "recreate it")
    try:
        from app.llm.image_prompt_log import save_prompt
        save_prompt(project_id, result["filename"], prompt)
    except Exception as e:
        logger.warning("[image_extractor] Prompt log save failed: %s", e)

    logger.info(
        "[image_extractor] Rendered %s for project=%d (prompt=%d chars)",
        result["filename"], project_id, len(prompt),
    )


def _detect_aspect_ratio(prompt: str) -> Optional[str]:
    """Pick the most likely aspect ratio from the prompt body itself.

    The chat LLM is instructed to mention the format explicitly in the
    prompt (Instagram square, banner, etc.), so we just scan for keywords.
    """
    p = prompt.lower()
    if any(kw in p for kw in ("16:9", "banner", "youtube banner", "landscape banner")):
        return "16:9"
    if any(kw in p for kw in ("9:16", "vertical", "story", "reel", "phone wallpaper")):
        return "9:16"
    if any(kw in p for kw in ("1:1", "instagram square", "square", "profile picture", "avatar")):
        return "1:1"
    if "21:9" in p or "ultrawide" in p:
        return "21:9"
    return None


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------

def _safe_parse_sse(chunk: str) -> Optional[dict]:
    """Parse an SSE 'data: {...}' chunk into a dict, or None on any failure."""
    if not chunk or not chunk.startswith("data: "):
        return None
    body = chunk[len("data: "):].strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


__all__ = [
    "extract_image_prompt",
    "strip_image_prompt_marker",
    "wrap_stream_with_image_dispatch",
]

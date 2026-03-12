# FILE: app/llm/image_prompt_synth.py
"""
Context-aware image prompt synthesis for Nano Banana.

Stage 1 of the two-stage image generation pipeline:
  1. Gemini Flash Lite reads conversation context + user request
  2. Outputs a rich, detailed image generation prompt
  3. That prompt is sent to Nano Banana (Stage 2)

v1.0 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SYNTHESIS_MODEL = "gemini-2.5-flash-lite"

_SYSTEM_PROMPT = """You are an image prompt engineer for an AI image generator.

Your job: read the conversation context and the user's request, then write a single, 
detailed image generation prompt that will produce exactly what the user wants.

Rules:
- Output ONLY the image prompt. No explanation, no markdown, no quotes, no preamble.
- Be specific about: subject, composition, style, colours, mood, lighting.
- If the user mentions text that should appear in the image, include the EXACT wording.
- If the user references "this" or "that", infer what they mean from conversation context.
- If the user wants to modify a previous image, incorporate their changes into a fresh complete prompt.
- If format/aspect ratio is mentioned (banner, square, portrait, thumbnail), note it in the prompt.
- Default style: modern, clean, professional. Unless the user specifies otherwise.
- Keep the prompt under 200 words."""

_REFINEMENT_PROMPT_ADDITION = """
The user is asking to MODIFY a previously generated image.
Previous image prompt was: {previous_prompt}
Incorporate their requested changes into a new complete image prompt."""


async def synthesise_image_prompt(
    user_message: str,
    conversation_history: list[dict] | None = None,
    previous_image_prompt: str | None = None,
) -> tuple[str, str | None]:
    """Synthesise a rich image prompt from conversation context.

    Args:
        user_message: The user's current message
        conversation_history: Last N messages as [{"role": ..., "content": ...}]
        previous_image_prompt: If refining, the prompt used for the last image

    Returns:
        Tuple of (synthesised_prompt, detected_aspect_ratio or None)
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("[image_prompt_synth] No API key, falling back to raw prompt")
            return user_message, None

        client = genai.Client(api_key=api_key)

        # Build the context block
        context_parts = []

        if conversation_history:
            context_parts.append("CONVERSATION CONTEXT (most recent messages):")
            for msg in conversation_history[-8:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Truncate long messages (e.g. base64 image data)
                if len(content) > 300:
                    content = content[:300] + "..."
                context_parts.append(f"  [{role}]: {content}")
            context_parts.append("")

        context_parts.append(f"USER'S CURRENT REQUEST:\n  {user_message}")

        # Add refinement context if modifying a previous image
        system = _SYSTEM_PROMPT
        if previous_image_prompt:
            system += _REFINEMENT_PROMPT_ADDITION.format(
                previous_prompt=previous_image_prompt
            )

        full_prompt = "\n".join(context_parts)

        logger.info("[image_prompt_synth] Synthesising prompt from %d context messages",
                     len(conversation_history) if conversation_history else 0)

        response = client.models.generate_content(
            model=_SYNTHESIS_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                max_output_tokens=300,
            ),
        )

        synthesised = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    synthesised += part.text

        synthesised = synthesised.strip().strip('"').strip("'").strip("`")

        if not synthesised:
            logger.warning("[image_prompt_synth] Empty synthesis, falling back to raw")
            return user_message, None

        # Detect aspect ratio from the synthesised prompt or user message
        aspect_ratio = _detect_aspect_ratio(user_message, synthesised)

        logger.info("[image_prompt_synth] Synthesised: %s (ar=%s)",
                     synthesised[:120], aspect_ratio or "default")

        return synthesised, aspect_ratio

    except Exception as e:
        logger.error("[image_prompt_synth] Synthesis failed: %s", e)
        return user_message, None


def _detect_aspect_ratio(user_msg: str, synth_prompt: str) -> str | None:
    """Detect intended aspect ratio from user message or synthesised prompt."""
    combined = (user_msg + " " + synth_prompt).lower()

    if any(kw in combined for kw in ['banner', '16:9', '2048', 'youtube banner', 'landscape banner']):
        return "16:9"
    if any(kw in combined for kw in ['9:16', 'vertical', 'phone wallpaper', 'story', 'reels']):
        return "9:16"
    if any(kw in combined for kw in ['square', '1:1', 'instagram square', 'profile picture', 'avatar', 'icon']):
        return "1:1"
    if any(kw in combined for kw in ['thumbnail', '4:3']):
        return "4:3"
    if any(kw in combined for kw in ['widescreen', '21:9', 'ultrawide']):
        return "21:9"
    return None


__all__ = ["synthesise_image_prompt"]

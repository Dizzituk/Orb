# FILE: app/llm/image_prompt_synth.py
"""
Context-aware image prompt synthesis.

Stage 1 of the two-stage image generation pipeline:
  1. Synthesis model reads conversation context + user request
  2. Outputs a rich, detailed image generation prompt
  3. That prompt is sent to the image backend (Stage 2)

The synth model acts as a "visual translator" — converting abstract concepts,
long essays, and vague requests into concrete, drawable image prompts that
GPT Image can actually render well.

Model read from .env (IMAGE_PROMPT_SYNTH_MODEL) at runtime.

v3.0 (2026-04-05): Hardened system prompt — visual translation layer,
                    no-abstract-concepts rule, split/contrast composition
                    guidance. Removed 300-char truncation on context.
v2.0 (2026-03-20): Env-driven model config.
v1.0 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_synth_model() -> str:
    return os.getenv("IMAGE_PROMPT_SYNTH_MODEL", "gemini-2.5-flash")


# ============================================================================
# SYSTEM PROMPT — the core of the visual translation layer
# ============================================================================

_SYSTEM_PROMPT = """You are a visual translator for an AI image generator (GPT Image).

Your job: read the conversation context and the user's request, then write a single,
detailed image generation prompt that will produce exactly what the user wants.

=== CRITICAL RULES ===

1. OUTPUT FORMAT:
   - Output ONLY the image prompt. No explanation, no markdown, no quotes, no preamble.
   - Keep the prompt under 200 words.

2. CONCRETE OBJECTS ONLY — NO ABSTRACT CONCEPTS:
   - Image generators cannot draw "harmony", "oppression", "freedom", or "hope".
   - You MUST translate every abstract concept into SPECIFIC DRAWABLE OBJECTS.
   - WRONG: "a utopian society showing harmony and prosperity"
   - RIGHT: "a sunlit city with solar panels on every roof, people walking through
     green parks, children playing near a clean river, modern glass buildings with
     vertical gardens, warm golden-hour lighting"
   - WRONG: "a dystopian world showing oppression"
   - RIGHT: "a grey concrete city under smog, surveillance cameras on every corner,
     barbed wire fences, empty streets with litter, crumbling buildings with broken
     windows, cold blue-grey colour palette, harsh fluorescent lighting"

3. VISUAL TRANSLATION TABLE — use these mappings:
   - Prosperity/abundance → overflowing market stalls, fruit trees, well-dressed people
   - Poverty/scarcity → empty shelves, boarded-up shops, torn clothing, food queues
   - Freedom → open doors, wide horizons, birds in flight, people dancing/creating
   - Control/surveillance → cameras, drones, guards, ID checkpoints, high walls
   - Technology (positive) → clean energy, helpful robots, holographic displays
   - Technology (negative) → cold server racks, faceless screens, automated barriers
   - Community → people eating together, shared gardens, children playing
   - Isolation → single figures in large spaces, empty rooms, closed curtains
   - Nature thriving → wildflowers, clear water, birdsong-implying greenery, sunlight
   - Nature dying → dead trees, polluted water, grey skies, cracked earth
   - Wealth inequality → glass towers next to shanties, luxury cars passing homeless people
   - Democracy/agency → town halls, raised hands voting, open microphones, diverse crowds
   - Authoritarianism → uniformed figures, propaganda posters, locked gates

4. SPLIT/CONTRAST COMPOSITIONS:
   When the user asks for "one side X / other side Y" or contrasting halves:
   - Describe the LEFT side with 3-5 specific objects/elements
   - Describe the RIGHT side with 3-5 contrasting specific objects/elements
   - Describe the CENTRE TRANSITION explicitly (how the two sides blend)
   - Specify different COLOUR PALETTES for each side (e.g. warm gold vs cold blue-grey)
   - Specify different LIGHTING for each side (e.g. sunlight vs overcast/fluorescent)

5. LONG CONTENT DISTILLATION:
   When the conversation context contains a long essay, article, or script:
   - Identify the 2-3 CORE VISUAL THEMES (not arguments, not philosophy — visuals)
   - Find the central TENSION or CONTRAST in the content
   - Build the image around that tension using concrete objects
   - Do NOT try to illustrate every point — pick the most visually powerful moment

6. COLOUR AND LIGHTING:
   - Always specify a colour palette (e.g. "warm earth tones", "neon cyberpunk palette")
   - Always specify lighting direction and quality (e.g. "golden hour side-lighting",
     "harsh overhead fluorescent", "soft diffused overcast")
   - For contrast images, each side MUST have a different colour temperature

7. COMPOSITION:
   - Specify viewpoint (wide establishing shot, medium shot, close-up, aerial, etc.)
   - Specify foreground/midground/background elements where relevant
   - For text that should appear in the image, include the EXACT wording in quotes

8. STYLE:
   - Default: modern, cinematic, photorealistic. Unless the user specifies otherwise.
   - If format/aspect ratio is mentioned (banner, square, portrait, thumbnail), note it.

=== DATA-DRIVEN IMAGES ===
- If [RESEARCH DATA] is provided, the user wants an image that incorporates real data.
- For data/benchmarks/comparisons: describe a clean infographic or chart-style visual.
- Include the SPECIFIC numbers, labels, and data points from the research in the prompt.
- Specify chart type (bar chart, line graph, comparison table, etc.) that best fits the data.
- Use clear readable text labels.
- Style: modern data visualisation aesthetic, clean layout, professional colour palette."""


_REFINEMENT_PROMPT_ADDITION = """
The user is asking to MODIFY a previously generated image.
Previous image prompt was: {previous_prompt}
Incorporate their requested changes into a new complete image prompt."""


# ============================================================================
# CONTEXT BUDGET — how much of each message to include
# ============================================================================

# The most recent assistant message (likely the content being illustrated)
# gets the highest budget. Older messages get less.
_LATEST_ASSISTANT_BUDGET = 8000   # ~2000 tokens — enough for a full essay
_OTHER_MESSAGE_BUDGET = 1500      # ~375 tokens — enough for meaningful context
_MAX_HISTORY_MESSAGES = 8         # How far back to look


def _build_context_block(
    user_message: str,
    conversation_history: list[dict] | None,
) -> str:
    """Build the context block for the synthesis model.

    The most recent assistant message gets the full budget (it's likely
    the content being illustrated). Other messages get a smaller budget
    to provide conversational context without overwhelming the synth model.
    """
    context_parts = []

    if conversation_history:
        messages = conversation_history[-_MAX_HISTORY_MESSAGES:]

        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        context_parts.append("CONVERSATION CONTEXT (most recent messages):")
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Apply budget: latest assistant message gets full content,
            # everything else gets a shorter window
            if i == last_assistant_idx:
                budget = _LATEST_ASSISTANT_BUDGET
            else:
                budget = _OTHER_MESSAGE_BUDGET

            if len(content) > budget:
                content = content[:budget] + "\n... [truncated]"

            context_parts.append(f"  [{role}]: {content}")
        context_parts.append("")

    context_parts.append(f"USER'S CURRENT REQUEST:\n  {user_message}")
    return "\n".join(context_parts)


# ============================================================================
# MAIN SYNTHESIS FUNCTION
# ============================================================================

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

        # Build the context with proper budgets
        full_prompt = _build_context_block(user_message, conversation_history)

        # Add refinement context if modifying a previous image
        system = _SYSTEM_PROMPT
        if previous_image_prompt:
            system += _REFINEMENT_PROMPT_ADDITION.format(
                previous_prompt=previous_image_prompt
            )

        logger.info(
            "[image_prompt_synth] Synthesising prompt from %d context messages "
            "(%d chars of context)",
            len(conversation_history) if conversation_history else 0,
            len(full_prompt),
        )

        synth_model = _get_synth_model()
        logger.info("[image_prompt_synth] Using model: %s", synth_model)

        response = client.models.generate_content(
            model=synth_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                max_output_tokens=400,
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

        logger.info(
            "[image_prompt_synth] Synthesised: %s (ar=%s)",
            synthesised[:120], aspect_ratio or "default",
        )

        return synthesised, aspect_ratio

    except Exception as e:
        logger.error("[image_prompt_synth] Synthesis failed: %s", e)
        return user_message, None


def _detect_aspect_ratio(user_msg: str, synth_prompt: str) -> str | None:
    """Detect intended aspect ratio from user message or synthesised prompt."""
    combined = (user_msg + " " + synth_prompt).lower()

    if any(kw in combined for kw in [
        'banner', '16:9', '2048', 'youtube banner', 'landscape banner',
    ]):
        return "16:9"
    if any(kw in combined for kw in [
        '9:16', 'vertical', 'phone wallpaper', 'story', 'reels',
    ]):
        return "9:16"
    if any(kw in combined for kw in [
        'square', '1:1', 'instagram square', 'profile picture', 'avatar', 'icon',
    ]):
        return "1:1"
    if any(kw in combined for kw in ['thumbnail', '4:3']):
        return "4:3"
    if any(kw in combined for kw in ['widescreen', '21:9', 'ultrawide']):
        return "21:9"
    return None


__all__ = ["synthesise_image_prompt"]
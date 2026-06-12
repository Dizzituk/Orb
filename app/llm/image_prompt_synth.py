# FILE: app/llm/image_prompt_synth.py
# Purpose: Context-aware image prompt synthesis.
# Called-by: app.llm.image_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
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
import re
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
# SYNTH CALL + TRUNCATION GUARD (v3.1, 2026-06-10)
# ============================================================================

_MIN_PROMPT_CHARS = 60          # anything shorter is certainly broken
_SHORT_PROMPT_CHARS = 200       # short AND unterminated => treat as cut off
_TERMINAL_CHARS = '.!?")'


def _generate_once(client, types, model, system, contents, max_tokens):
    """Single synth call. Returns (text, finish_reason_str)."""
    cfg_kwargs = dict(
        system_instruction=system,
        temperature=0.3,
        max_output_tokens=max_tokens,
    )
    # gemini-2.5+ are thinking models: without a zero thinking budget the
    # internal reasoning consumes max_output_tokens and the visible text
    # gets silently truncated (root cause of the 2026-06-10 85-char
    # fragment that reached gpt-image-2). Older SDKs lack ThinkingConfig,
    # so fall back gracefully.
    try:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(**cfg_kwargs)
    except (AttributeError, TypeError):
        cfg_kwargs.pop("thinking_config", None)
        config = types.GenerateContentConfig(**cfg_kwargs)

    response = client.models.generate_content(
        model=model, contents=contents, config=config,
    )

    text = ""
    finish = ""
    if response.candidates:
        cand = response.candidates[0]
        finish = str(getattr(cand, "finish_reason", "") or "")
        if cand.content:
            for part in cand.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
    return text.strip().strip('"').strip("'").strip("`"), finish


def _looks_truncated(text: str, finish_reason: str) -> bool:
    """True if the synth output is unusable as an image prompt."""
    if not text:
        return True
    fr = (finish_reason or "").upper()
    if "MAX_TOKEN" in fr or "LENGTH" in fr:
        return True
    if len(text) < _MIN_PROMPT_CHARS:
        return True
    if len(text) < _SHORT_PROMPT_CHARS and text[-1] not in _TERMINAL_CHARS:
        return True
    return False


def _build_fallback_prompt(user_message, conversation_history):
    """Deterministic fallback when synthesis fails twice: the user's request
    plus the most recent assistant message (which usually contains the
    content being illustrated, e.g. the agreed quote text). Far better than
    the bare user message, which is often just 'yep make the image please'.
    """
    parts = [user_message.strip()]
    if conversation_history:
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                snippet = msg["content"][:1200]
                parts.append(
                    "Use the following conversation content as the subject "
                    "of the image (include any quoted text verbatim):\n"
                    + snippet
                )
                break
    return "\n\n".join(parts)


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

        # v3.1 (2026-06-10): truncation guard + one retry. A truncated
        # fragment must never reach the image backend -- that is how an
        # 85-char prompt with no quote text produced a junk image.
        synthesised, finish = _generate_once(
            client, types, synth_model, system, full_prompt, max_tokens=1024,
        )
        if _looks_truncated(synthesised, finish):
            logger.warning(
                "[image_prompt_synth] Output looks truncated "
                "(finish=%s, %d chars) -- retrying with larger budget",
                finish, len(synthesised),
            )
            synthesised, finish = _generate_once(
                client, types, synth_model, system, full_prompt,
                max_tokens=2048,
            )

        if _looks_truncated(synthesised, finish):
            fallback = _build_fallback_prompt(user_message, conversation_history)
            logger.error(
                "[image_prompt_synth] Synthesis unusable after retry "
                "(finish=%s, %d chars) -- using deterministic fallback "
                "(%d chars)",
                finish, len(synthesised), len(fallback),
            )
            return fallback, _detect_aspect_ratio(user_message, "")

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


def _word_in(text: str, *words: str) -> bool:
    """Whole-word match so 'history' never matches 'story', 'iconic'
    never matches 'icon', etc."""
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", text):
            return True
    return False


def _ar_from_text(text: str) -> str | None:
    """Ordered aspect-ratio detection within ONE piece of text.

    v3.1 (2026-06-10): explicit ratios first, then named formats, then
    bare adjectives LAST. Previously 'vertical' was checked before
    'square', so a synth prompt containing 'split vertically' forced an
    Instagram SQUARE request into 9:16 portrait.
    """
    t = (text or "").lower()
    if not t:
        return None
    # 1. Explicit ratios beat everything
    for token, ar in (("16:9", "16:9"), ("9:16", "9:16"), ("1:1", "1:1"),
                      ("4:3", "4:3"), ("21:9", "21:9")):
        if token in t:
            return ar
    # 2. Named formats
    if ("instagram square" in t or "insta square" in t
            or "profile picture" in t
            or _word_in(t, "avatar", "icon", "icons")):
        return "1:1"
    if _word_in(t, "banner", "banners"):
        return "16:9"
    if "phone wallpaper" in t or _word_in(t, "story", "stories", "reel", "reels"):
        return "9:16"
    if _word_in(t, "thumbnail", "thumbnails"):
        return "4:3"
    if _word_in(t, "ultrawide"):
        return "21:9"
    # 3. Bare adjectives last (weakest signal)
    if _word_in(t, "square"):
        return "1:1"
    if _word_in(t, "vertical", "portrait"):
        return "9:16"
    if _word_in(t, "widescreen"):
        return "21:9"
    return None


def _detect_aspect_ratio(user_msg: str, synth_prompt: str) -> str | None:
    """Detect intended aspect ratio. The USER'S message wins outright;
    the synthesised prompt is only consulted when the user gave no
    signal. (Previously both were concatenated, letting synth wording
    override the user's explicit format request.)"""
    return _ar_from_text(user_msg) or _ar_from_text(synth_prompt)


__all__ = ["synthesise_image_prompt"]
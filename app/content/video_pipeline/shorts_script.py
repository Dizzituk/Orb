# FILE: app/content/video_pipeline/shorts_script.py
# Purpose: Hook-first shorts script + platform caption generation (env-role model).
# Called-by: app.content.video_pipeline.shorts_orchestrator, tests.test_shorts_script
# Depends-on: app.providers.registry (injectable), app.llm.model_families
# Last-renovated: 2026-07-02
"""
Shorts script generation (jobspec Job 6, script stage).

Produces, from a topic, a hook-first spoken script capped at ~45 seconds
(<=120 words, enforced HERE — trimmed/regenerated, never discovered at
render) plus the platform caption + hashtags + a short title, all in one
model call. Model comes from an env role (no hardcoded model strings).

The LLM call is injectable so tests run without any provider.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

WORD_CAP = 120          # ~45s at ~150 wpm; hard ceiling
WORD_TARGET = 110       # what we ask the model for

# llm_fn(messages, system, model, provider) -> answer text
LlmFn = Callable[[list, str, str, Optional[str]], Awaitable[str]]

_SYSTEM = (
    "You write punchy vertical short-form video scripts for an AI persona. "
    "One idea per short. Open with a hook in the first sentence. Written to be "
    "spoken aloud by an avatar — natural, direct, no stage directions, no emojis "
    "in the spoken script. Hard limit: 110 words."
)

_PROMPT = (
    "Topic: {topic}\n"
    "{notes}"
    "Write a short (max ~45 seconds, MAX 110 words) hook-first spoken script on the topic.\n\n"
    "Return STRICT JSON with keys:\n"
    '  "script": the spoken words only (<=110 words),\n'
    '  "title": a 3-6 word title,\n'
    '  "caption": a 1-2 sentence platform caption (may include emojis),\n'
    '  "hashtags": array of 3-6 relevant hashtags without the # sign.\n'
    "Return ONLY the JSON object."
)


def _provider_for(model: str) -> Optional[str]:
    m = (model or "").lower()
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    return None  # let the registry choose (e.g. local)


def _script_model() -> str:
    """Env role -> model string. ASTRA_SHORTS_SCRIPT_MODEL wins, else role_chat."""
    explicit = os.getenv("ASTRA_SHORTS_SCRIPT_MODEL")
    if explicit:
        return explicit
    from app.llm.model_families import resolve
    return resolve("role_chat")


async def _default_llm(messages: list, system: str, model: str, provider: Optional[str]) -> str:
    from app.providers.registry import llm_call
    result = await llm_call(
        provider_id=provider,
        model_id=model,
        messages=messages,
        system_prompt=system,
        temperature=0.7,
        max_tokens=800,
        stage="shorts_script",
    )
    if hasattr(result, "content"):
        return result.content or ""
    return str(result or "")


def _word_count(text: str) -> int:
    return len((text or "").split())


def _trim_words(text: str, cap: int) -> str:
    words = (text or "").split()
    if len(words) <= cap:
        return text
    trimmed = " ".join(words[:cap])
    # end on sentence punctuation if one is near the tail
    m = re.search(r"[.!?][^.!?]*$", trimmed)
    return trimmed


def _parse(answer: str) -> Optional[Dict[str, Any]]:
    if not answer:
        return None
    text = answer.strip()
    # strip code fences if present
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict) or not obj.get("script"):
        return None
    tags = obj.get("hashtags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,\s]+", tags) if t.strip()]
    return {
        "script": str(obj.get("script", "")).strip(),
        "title": str(obj.get("title", "")).strip(),
        "caption": str(obj.get("caption", "")).strip(),
        "hashtags": [str(t).lstrip("#") for t in tags][:6],
    }


async def generate_script(
    topic: str, notes: str = "", *, llm: Optional[LlmFn] = None
) -> Dict[str, Any]:
    """Return {script, title, caption, hashtags}. Word cap enforced here."""
    llm = llm or _default_llm
    model = _script_model()
    provider = _provider_for(model)
    notes_line = f"Creator notes: {notes}\n" if notes else ""
    prompt = _PROMPT.format(topic=topic, notes=notes_line)
    messages = [{"role": "user", "content": prompt}]

    answer = await llm(messages, _SYSTEM, model, provider)
    parsed = _parse(answer)

    # One stricter regen if the model ignored JSON or blew the word cap.
    if parsed is None or _word_count(parsed["script"]) > WORD_CAP:
        strict = prompt + f"\n\nIMPORTANT: the script MUST be {WORD_TARGET} words or fewer. Return ONLY JSON."
        answer2 = await llm([{"role": "user", "content": strict}], _SYSTEM, model, provider)
        parsed2 = _parse(answer2)
        if parsed2 is not None:
            parsed = parsed2

    if parsed is None:
        # Last resort: use the raw text as the script (trimmed), empty caption.
        parsed = {"script": _trim_words((answer or topic), WORD_CAP), "title": topic[:60],
                  "caption": "", "hashtags": []}

    # Hard ceiling enforced at the script stage, per jobspec.
    if _word_count(parsed["script"]) > WORD_CAP:
        parsed["script"] = _trim_words(parsed["script"], WORD_CAP)
        logger.info("[shorts_script] trimmed script to %d-word cap", WORD_CAP)

    if not parsed.get("title"):
        parsed["title"] = (topic or "short")[:60]
    parsed["word_count"] = _word_count(parsed["script"])
    return parsed

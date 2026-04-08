# FILE: app/bridge/llm_helpers.py
"""
LLM model selection, system prompt, and direct LLM call helpers for the Bridge API.

Extracted from router.py during modularisation (2026-04-06).

NOTE: The `_call_llm` function in this module is a plain text completion with
NO tool access.  The main bridge refactor (job brief Phase 1) will replace
calls to this with the shared capability layer from `chat_model_selection.py`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DESKTOP NAVIGATION  (v4.0)
# =============================================================================

_pending_navigation: Optional[dict] = None


def push_desktop_navigation(domain: str) -> None:
    """Queue a navigation event for the desktop to pick up."""
    global _pending_navigation
    _DOMAIN_TO_JOB = {
        "finance": "accounts",
        "investments": "investments",
        "content": "content",
        "social": "social_media",
        "lifestyle": "health_fitness",
        "debug": "debug",
        "education": "education",
        "builds": "project_builds",
    }
    _pending_navigation = {
        "domain": domain,
        "job_type": _DOMAIN_TO_JOB.get(domain, domain),
        "timestamp": datetime.now().isoformat(),
    }
    logger.info("[bridge] Queued desktop navigation: %s -> %s", domain, _DOMAIN_TO_JOB.get(domain, domain))


def pop_pending_navigation() -> Optional[dict]:
    """Return and clear the pending navigation event (polled by desktop)."""
    global _pending_navigation
    if _pending_navigation:
        event = _pending_navigation
        _pending_navigation = None
        return event
    return None


# =============================================================================
# MODEL SELECTION  (v5.0)
# =============================================================================

_ESCALATED_DOMAINS = {"builds", "debug"}


def select_bridge_model(
    message: str,
    translation_result,
    domain_context: str,
) -> tuple:
    """Pick provider/model based on domain and message complexity.

    Escalation rules:
      1. BUILDS or DEBUG domain -> GPT-5.4
      2. Complexity 'deep' -> env-driven deep model
      3. Complexity 'reasoning' -> env-driven chat model
      4. Default -> gpt-5-mini
    """
    _domain = None
    if (translation_result
            and translation_result.resolved_intent
            and translation_result.resolved_intent.value.startswith("DOMAIN_")):
        try:
            from app.llm.translation_routing import intent_to_routing_info
            info = intent_to_routing_info(translation_result.resolved_intent)
            _domain = info.get("domain") if info else None
        except Exception:
            pass

    if _domain in _ESCALATED_DOMAINS:
        _prov = os.getenv("BUILD_CHAT_PROVIDER", "openai")
        _mod = os.getenv("BUILD_CHAT_MODEL", "gpt-5.4")
        logger.info("[bridge] Domain '%s' -> escalated to %s/%s", _domain, _prov, _mod)
        return (_prov, _mod)

    try:
        from app.memory.complexity import classify_complexity
        complexity = classify_complexity(query=message, intent=None)

        if complexity.tier == "deep":
            _prov = os.getenv("CHAT_DEEP_PROVIDER", "openai")
            _mod = os.getenv("CHAT_DEEP_MODEL", "gpt-5.4")
            logger.info("[bridge] Complexity 'deep' -> %s/%s", _prov, _mod)
            return (_prov, _mod)

        if complexity.tier == "reasoning":
            _prov = os.getenv("CHAT_PROVIDER", "openai")
            _mod = os.getenv("CHAT_MODEL", "gpt-5-mini")
            logger.info("[bridge] Complexity 'reasoning' -> %s/%s", _prov, _mod)
            return (_prov, _mod)
    except Exception as e:
        logger.debug("[bridge] Complexity classifier unavailable: %s", e)

    _prov = "openai"
    _mod = os.getenv("CHAT_QUICK_MODEL", "gpt-5-mini")
    logger.info("[bridge] Default model -> %s/%s", _prov, _mod)
    return (_prov, _mod)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_bridge_system_prompt(domain_context: str) -> str:
    """Build the system prompt for bridge chat.

    Keeps it concise (phone context) but includes domain data when available.
    """
    prompt = (
        "You are Astra, a personal AI assistant connected to a full backend system. "
        "The user is speaking from their phone via the Astra Bridge app — they may be driving. "
        "Keep responses concise but informative.\n\n"
        "You have access to these domains via ASTRA's backend:\n"
        "- Finance/Accounts: earnings, expenses, tax, deliveries, mileage\n"
        "- Investments: Trading 212 portfolio, crypto via CoinMarketCap\n"
        "- Content Creation: YouTube channel, video pipeline, scripts, publishing\n"
        "- Social Media: Facebook, Instagram, TikTok engagement\n"
        "- Health & Fitness: workout plans, nutrition, surfing, bodyboarding\n"
        "- Education: philosophy, psychology, economics, AI & society curriculum\n"
        "- Debug/Builds: ASTRA codebase, project builds, pipeline status\n\n"
        "When you have domain data below, use it to give real answers with actual numbers. "
        "If your training data might be stale for the question asked, tell the user you can "
        "search the web. Suggest they say 'search for X' or 'get the latest on X' "
        "and you will fetch current information."
    )
    if domain_context:
        prompt += (
            "\n\nYou have access to the following real-time data from ASTRA's systems. "
            "Use it to answer the user's question accurately:\n\n"
            + domain_context
        )
    return prompt


# =============================================================================
# DIRECT LLM CALL  (no tools — will be replaced by shared capability layer)
# =============================================================================

async def call_llm(provider: str, model: str, messages: list) -> str:
    """Call the LLM via the appropriate provider.

    Supports openai, anthropic, and google providers.
    WARNING: This is a plain text completion — no tool loop.
    """
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or "No response generated."

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                chat_messages.append(m)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_msg.strip(),
            messages=chat_messages,
        )
        return response.content[0].text if response.content else "No response generated."

    if provider == "google":
        import google.generativeai as genai
        gmodel = genai.GenerativeModel(model)
        history = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            elif m["role"] == "user":
                history.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})
        if system_text and history:
            history[0]["parts"].insert(0, system_text.strip() + "\n\n")
        chat = gmodel.start_chat(history=history[:-1] if history else [])
        last_msg = history[-1]["parts"][0] if history else "Hello"
        response = chat.send_message(last_msg)
        return response.text or "No response generated."

    raise ValueError(f"Unknown provider: {provider}")

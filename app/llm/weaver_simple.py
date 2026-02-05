# FILE: app/llm/weaver_simple.py
r"""
Simple Weaver - Text organizer for ASTRA.

This module implements the LOCKED Weaver behaviour specification:
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- No system access, no DB, no file inspection
- Stateless, tool-free, isolated

v4.0.0 (2026-02-04): LLM-GENERATED QUESTIONS - Remove hardcoded game-design questions
- Removed hardcoded SHALLOW_QUESTIONS ("Dark mode?", "Arcade-style?", "Keyboard or controller?")
- LLM now generates contextual questions based on actual gaps in requirements
- System prompt rewritten: domain-agnostic, no game-specific examples
- Added "Key requirements" section to preserve all user-stated requirements
- Added rule: comprehensive user requests may have ZERO questions (correct behavior)
- Prompt synced with weaver_stream.py v4.0.0

v3.5.0 (2026-01-22): WEAVER HARDENING + SCOPE BOUNDARY FIX
- Bug 3: Added deduplication rules (What/Outcome must be different)
- Bug 5: Added scope boundary enforcement (shallow questions only)
- Prompt updated to match weaver_stream.py v3.5.0
- No framework/algorithm/architecture questions allowed

v1.2 (2026-01-19): Fixed LLM call signature (system_prompt is first arg).
v1.1 (2026-01-19): Fixed LLM routing to use ENV-based stage_models.
v1.0 (2026-01-19): Initial implementation per locked behaviour spec.
"""
from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# WEAVER SYSTEM PROMPT (LOCKED BEHAVIOUR)
# =============================================================================

# =============================================================================
# v4.0.0 (2026-02-04): LLM-GENERATED QUESTIONS
# - Removed hardcoded game-design question menu
# - LLM generates contextual questions based on actual gaps
# - Domain-agnostic prompt (no Tetris/game-specific examples)
# - Added "Key requirements" section to output format
# =============================================================================

WEAVER_SYSTEM_PROMPT = """You are Weaver, a SHALLOW text organizer.

Your ONLY job: Take the human's rambling and restructure it into a minimal, stable job outline.

## What You DO:
- Extract the core goal as a SHORT NOUN PHRASE (not a full sentence)
- Summarize intent into "What is being built" and "Intended outcome" (DIFFERENT wording)
- Faithfully list ALL requirements, constraints, and specifications the user provided
- List unresolved ambiguities at high level
- Generate up to 3-5 contextual clarifying questions about GENUINE GAPS (see rules below)

## What You DO NOT DO (CRITICAL - SCOPE BOUNDARY):
- NO framework/library choices (don't suggest specific libraries or tools)
- NO file structure discussion
- NO algorithm or data structure talk
- NO architecture proposals
- NO implementation plans
- NO technical questions (those belong to later pipeline stages)
- NO resolving ambiguities yourself
- NO inventing requirements the user didn't state

## RENAME/REBRAND HANDLING (v3.6 - CRITICAL):
When the user mentions renaming, rebranding, or changing a name:
- ALWAYS extract and output BOTH the source name AND target name
- If user uses pronouns ("it", "that", "this"), infer source from context:
  - Project/folder name mentioned (e.g., "orb-desktop" → "Orb")
  - Earlier mention in conversation ("the Orb system" → "Orb")
  - Screenshot/image context if mentioned
- Format: "rename SOURCE to TARGET" or "change SOURCE to TARGET"

EXAMPLES:
- User: "change it to Astra" (context: "Orb Desktop" folder)
  → Output: "rename Orb to Astra" NOT "rename to Astra"
- User: "rebrand the UI to Astra" (context: Orb system)
  → Output: "rebrand Orb UI to Astra" NOT "rebrand UI to Astra"

NEVER output just "rename to X" without the source term.

## DEDUPLICATION RULES (CRITICAL):
NEVER repeat the same sentence or near-identical phrasing across sections.
"What is being built" and "Intended outcome" must use DIFFERENT words.

BAD: What: "Voice input feature" / Outcome: "Voice input feature"
GOOD: What: "Voice-to-text input system" / Outcome: "Local speech transcription for desktop app"

## QUESTION GENERATION RULES (v4.1 - CRITICAL):
Zero questions is the PREFERRED and DEFAULT outcome. You generate questions ONLY when there
is a genuine gap that would make the requirement AMBIGUOUS TO BUILD.

Do NOT manufacture questions to appear thorough. Do NOT ask questions to fill a quota.
If the user gave clear, comprehensive requirements: output "Questions: none" and move on.

Rules:
1. DEFAULT TO ZERO QUESTIONS. Only ask if you genuinely cannot determine what to build.
2. READ the user's requirements carefully first. Do NOT ask about things they already specified.
3. Questions must be HIGH-LEVEL framing questions, never technical implementation questions.
4. Absolute maximum: 3 questions. But 0 is almost always correct for detailed requests.
5. Each question must address a GENUINE GAP - something the user didn't cover that would affect
   what gets built (not how it gets built).
6. Before writing ANY question, ask yourself: "Would the downstream pipeline be blocked without
   this answer?" If no, don't ask it.
7. NEVER ask these if the user already specified them (check carefully!):
   - Platform (if they said "desktop app" or "Windows" - that's answered)
   - Controls (if they described input methods - that's answered)
   - Scope (if they defined phases or boundaries - that's answered)
   - Architecture (if they described backend/frontend structure - that's answered)
   - Technology choices (if they named specific tools/libraries - that's answered)
8. If the user provided a detailed, well-structured request with explicit requirements,
   constraints, and phase boundaries, you MUST output "Questions: none".

ANTI-PATTERNS (never do these):
- Asking 3-5 questions on every request regardless of completeness
- Rephrasing stated requirements as questions ("You mentioned X, did you mean X?")
- Asking about preferences the user clearly stated
- Asking about things the downstream pipeline will handle (file paths, exact APIs, etc.)

BAD questions (generic, context-blind):
- "Dark mode or light mode?" (when user is asking for a backend service)
- "Keyboard or touch?" (when user specified keyboard shortcuts)
- "Bare minimum or extras?" (when user defined explicit Phase 1 boundaries)

GOOD questions (contextual, gap-filling — but ONLY if genuinely needed):
- "What latency target for transcription?" (voice feature, not specified)
- "Should wake word detection run continuously or only when app is focused?" (genuine ambiguity)
- "Target OS(es) beyond Windows?" (user said desktop but didn't clarify OS scope)

## Output Format:
Produce a structured job description document. Structure adapts to the content.
Possible sections (use only what's relevant):
- What is being built or changed (SHORT NOUN PHRASE)
- Intended outcome (DIFFERENT wording from above)
- Execution mode (only if user specified discussion-only, no code, etc.)
- Key requirements (bullet list of what user explicitly asked for)
- Platform/environment (only if mentioned)
- Design preferences (visual/UI only - color, layout, style)
- Constraints (only if explicitly stated)
- Unresolved ambiguities (preserved, not resolved)
- Questions (usually "none" — only include if a genuine gap would block building)

## Critical Rules:
1. If the human didn't say it, it doesn't appear in your output.
2. If the human DID say it, it MUST appear in your output (don't drop requirements).
3. You are a TEXT ORGANIZER, not a solution designer.
4. Preserve the user's terminology and domain language.
"""


def _get_weaver_config() -> Tuple[str, str]:
    """
    Get provider and model for Weaver using ENV-based stage_models.
    
    Uses centralized configuration from app.llm.stage_models which reads:
      - WEAVER_PROVIDER (e.g., "openai", "anthropic", "google")
      - WEAVER_MODEL (e.g., "gpt-4.1-mini", "gemini-2.0-flash")
    
    Falls back to defaults if not set.
    """
    try:
        from app.llm.stage_models import get_weaver_config as _get_cfg
        cfg = _get_cfg()
        logger.debug("[weaver_simple] Using stage_models config: provider=%s, model=%s", cfg.provider, cfg.model)
        return cfg.provider, cfg.model
    except ImportError:
        # Fallback if stage_models not available
        import os
        provider = os.getenv("WEAVER_PROVIDER", "openai")
        model = os.getenv("WEAVER_MODEL", "gpt-4.1-mini")
        logger.warning("[weaver_simple] stage_models not available, using env directly: provider=%s, model=%s", provider, model)
        return provider, model


def _call_llm(system_prompt: str, user_message: str, provider: str, model: str) -> str:
    """
    Make an LLM call using the appropriate provider client.
    
    Uses the unified clients module which routes through the provider registry.
    This ensures consistent behaviour regardless of provider.
    
    IMPORTANT: All client functions have signature:
        call_xxx(system_prompt, messages, temperature=0.7, model=None)
    """
    messages = [{"role": "user", "content": user_message}]
    
    provider_lower = provider.lower()
    
    try:
        if provider_lower in ("openai", "openai-compatible"):
            from app.llm.clients import call_openai
            # Signature: call_openai(system_prompt, messages, temperature=0.7, model=None)
            content, usage = call_openai(system_prompt, messages, 0.3, model)
            logger.info("[weaver_simple] OpenAI call complete: %d tokens", usage.get("total_tokens", 0))
            return content
            
        elif provider_lower in ("anthropic", "claude"):
            from app.llm.clients import call_anthropic
            # Signature: call_anthropic(system_prompt, messages, temperature=0.7, model=None)
            content, usage = call_anthropic(system_prompt, messages, 0.3, model)
            logger.info("[weaver_simple] Anthropic call complete: %d tokens", usage.get("total_tokens", 0))
            return content
            
        elif provider_lower in ("google", "gemini"):
            from app.llm.clients import call_google
            # Signature: call_google(system_prompt, messages, temperature=0.7, model=None)
            content, usage = call_google(system_prompt, messages, 0.3, model)
            logger.info("[weaver_simple] Google call complete: %d tokens", usage.get("total_tokens", 0))
            return content
            
        else:
            # Unknown provider - try OpenAI as fallback
            logger.warning("[weaver_simple] Unknown provider '%s', trying OpenAI fallback", provider)
            from app.llm.clients import call_openai
            content, usage = call_openai(system_prompt, messages, 0.3, model)
            return content
            
    except Exception as e:
        logger.exception("[weaver_simple] LLM call failed")
        raise


def _is_goal_missing(ramble_text: str) -> bool:
    """
    Check if the ramble has no discernible goal at all.
    This is the ONLY case where Weaver may ask a question.
    """
    # Very short or empty
    if len(ramble_text.strip()) < 20:
        return True
    
    # Contains only vague phrases with no concrete nouns
    vague_only_patterns = [
        "make it better",
        "fix it",
        "improve things",
        "do something",
        "help me",
        "you know",
        "just do it",
    ]
    lower = ramble_text.lower()
    
    # If the ENTIRE content is just vague phrases
    stripped = lower
    for pattern in vague_only_patterns:
        stripped = stripped.replace(pattern, "")
    
    # If after removing vague phrases, nothing substantial remains
    remaining = stripped.strip()
    if len(remaining) < 10:
        return True
    
    return False


def weave(ramble_text: str) -> str:
    """
    Transform rambling text into a structured job description.
    
    This is a SYNCHRONOUS function that calls an LLM to organize text.
    Uses ENV-based provider/model configuration.
    
    Args:
        ramble_text: The raw, unstructured human input to organize
        
    Returns:
        A structured job description document (plain text)
    """
    if not ramble_text or not ramble_text.strip():
        return "❓ **Clarification needed:**\n\nWhat would you like me to help you with?"
    
    # Check for completely missing goal
    if _is_goal_missing(ramble_text):
        return "❓ **Clarification needed:**\n\nI couldn't identify a clear goal. What specifically would you like to accomplish?"
    
    provider, model = _get_weaver_config()
    logger.info("[weaver_simple] Starting weave with provider=%s, model=%s", provider, model)
    
    try:
        user_message = f"Organize this ramble into a job description:\n\n{ramble_text}\n\nRemember: Only include what was actually said. Preserve any ambiguities or contradictions."
        
        result = _call_llm(
            system_prompt=WEAVER_SYSTEM_PROMPT,
            user_message=user_message,
            provider=provider,
            model=model,
        )
        
        return result
            
    except Exception as e:
        logger.exception("[weaver_simple] LLM call failed")
        return f"❌ **Weaver Error:**\n\nFailed to organize the text: {str(e)}"


async def weave_async(ramble_text: str) -> str:
    """
    Async version of weave() for streaming contexts.
    Wraps the synchronous call in an executor.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, weave, ramble_text)


def _format_messages_as_ramble(messages: list[dict]) -> str:
    """Format conversation messages as a ramble text block."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        # Skip system messages and Orb control messages
        if role == "system":
            continue
        # Skip Weaver/command outputs
        lower_content = content.lower()
        if any(marker in lower_content for marker in [
            "🧵 weaving", "📋 spec", "astra, command:", "astra command:",
            "shall i send", "say yes to proceed", "⚠️ weak spots",
        ]):
            continue
        
        speaker = "Human" if role == "user" else "Assistant"
        lines.append(f"[{speaker}]: {content}")
    
    return "\n\n".join(lines)


def weave_from_messages(messages: list[dict]) -> str:
    """
    Convenience function to weave from a list of conversation messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        A structured job description document
    """
    ramble_text = _format_messages_as_ramble(messages)
    return weave(ramble_text)


# =============================================================================
# TEST HARNESS
# =============================================================================

def _test_weave():
    """Simple test harness for Weaver."""
    
    test_cases = [
        # Test 1: Normal ramble
        {
            "name": "Normal ramble",
            "input": "I want an app that runs on Android, it needs to be secure, and it should track my deliveries...",
            "expected_contains": ["Android", "secure"],
            "expected_not_contains": ["React", "Firebase"],  # Should not invent tech
        },
        # Test 2: Contradiction preserved
        {
            "name": "Contradiction preserved",
            "input": "It must be offline only... but also it should sync live to the cloud...",
            "expected_contains": ["offline", "cloud"],  # Both should appear
            "expected_not_contains": [],
        },
        # Test 3: No goal - question allowed
        {
            "name": "No goal - question allowed",
            "input": "Just make it better",
            "expected_contains": ["Clarification", "?"],
            "expected_not_contains": [],
        },
    ]
    
    print("=" * 60)
    print("WEAVER SIMPLE - TEST HARNESS")
    print("=" * 60)
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {tc['name']} ---")
        print(f"Input: {tc['input'][:50]}...")
        
        result = weave(tc["input"])
        print(f"Output:\n{result}\n")
        
        # Check expected content
        for expected in tc["expected_contains"]:
            if expected.lower() in result.lower():
                print(f"  ✅ Contains '{expected}'")
            else:
                print(f"  ❌ Missing '{expected}'")
        
        for not_expected in tc["expected_not_contains"]:
            if not_expected.lower() not in result.lower():
                print(f"  ✅ Does not contain '{not_expected}'")
            else:
                print(f"  ⚠️  Unexpectedly contains '{not_expected}'")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    _test_weave()

# app/llm/streaming.py
"""
Streaming LLM responses using Server-Sent Events.
All system prompts include current date/time context.

v0.16.0: Added reasoning support and metadata emission.
"""

import os
import json
from typing import AsyncGenerator, Optional, List, Dict
from datetime import datetime


# ============ DATETIME CONTEXT ============

def get_datetime_context() -> str:
    """Get current date/time for system prompts."""
    now = datetime.now()
    formatted = now.strftime("%A, %B %d, %Y at %I:%M %p")
    try:
        import time
        tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        formatted += f" ({tz_name})"
    except:
        pass
    return f"Current date and time: {formatted}"


def enhance_system_prompt(prompt: str) -> str:
    """Add datetime context to system prompt."""
    context = get_datetime_context()
    if prompt:
        return f"{context}\n\n{prompt}"
    return context


# ============ REASONING INSTRUCTION ============

REASONING_INSTRUCTION = """
When you respond, structure your answer using these tags:

<THINKING>
Your internal reasoning, step-by-step analysis, and thought process goes here.
</THINKING>
<ANSWER>
Your final response to the user goes here.
</ANSWER>

Always include both tags in every response.
"""


def enhance_system_prompt_with_reasoning(prompt: str, enable: bool = True) -> str:
    """Add reasoning instruction to system prompt if enabled."""
    enhanced = enhance_system_prompt(prompt)
    if enable:
        return f"{enhanced}\n\n{REASONING_INSTRUCTION}"
    return enhanced


# ============ PROVIDER CHECKS ============

HAS_OPENAI = False
HAS_ANTHROPIC = False
HAS_GEMINI = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    pass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
}


def get_available_streaming_providers() -> Dict[str, bool]:
    """Check which providers support streaming."""
    return {
        "openai": HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY")),
        "gemini": HAS_GEMINI and bool(os.getenv("GOOGLE_API_KEY")),
    }


def get_available_streaming_provider() -> Optional[str]:
    """Get the first available provider name."""
    providers = get_available_streaming_providers()
    for name, available in providers.items():
        if available:
            return name
    return None


def get_default_provider() -> Optional[str]:
    """Get the first available provider."""
    return get_available_streaming_provider()


# ============ STREAMING GENERATORS ============

async def stream_openai(
    messages: List[Dict],
    system_prompt: str = "",
    model: str = "gpt-4o-mini",
    enable_reasoning: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream from OpenAI using async client."""
    if not HAS_OPENAI:
        yield json.dumps({"type": "error", "error": "openai package not installed"})
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield json.dumps({"type": "error", "error": "OPENAI_API_KEY not set"})
        return

    client = AsyncOpenAI(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    full_messages = [{"role": "system", "content": enhanced_prompt}]
    full_messages.extend(messages)

    # Emit metadata first
    yield json.dumps({"type": "metadata", "provider": "openai", "model": model})

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=full_messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield json.dumps({"type": "error", "error": str(e)})


async def stream_anthropic(
    messages: List[Dict],
    system_prompt: str = "",
    model: str = "claude-sonnet-4-20250514",
    enable_reasoning: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream from Anthropic using async client."""
    if not HAS_ANTHROPIC:
        yield json.dumps({"type": "error", "error": "anthropic package not installed"})
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield json.dumps({"type": "error", "error": "ANTHROPIC_API_KEY not set"})
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # Emit metadata first
    yield json.dumps({"type": "metadata", "provider": "anthropic", "model": model})

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=4096,
            system=enhanced_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    except Exception as e:
        yield json.dumps({"type": "error", "error": str(e)})


async def stream_gemini(
    messages: List[Dict],
    system_prompt: str = "",
    model: str = "gemini-2.0-flash",
    enable_reasoning: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream from Gemini."""
    if not HAS_GEMINI:
        yield json.dumps({"type": "error", "error": "google-generativeai package not installed"})
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield json.dumps({"type": "error", "error": "GOOGLE_API_KEY not set"})
        return

    genai.configure(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # Emit metadata first
    yield json.dumps({"type": "metadata", "provider": "gemini", "model": model})

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=enhanced_prompt,
        )

        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=history)

        last_msg = messages[-1]["content"] if messages else ""
        
        response = chat.send_message(last_msg, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield json.dumps({"type": "error", "error": str(e)})


# ============ MAIN STREAMING FUNCTION ============

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response.

    Args:
        messages: List of {role, content} dicts
        system_prompt: System prompt (datetime will be added automatically)
        provider: openai, anthropic, or gemini (auto-selects if None)
        model: Model name (uses default if None)
        enable_reasoning: If True, instruct LLM to use THINKING/ANSWER tags

    Yields:
        First yield: JSON metadata {"type": "metadata", "provider": "...", "model": "..."}
        Subsequent yields: Raw tokens as strings
    """
    # Auto-select provider if not specified
    if not provider:
        provider = get_default_provider()
        if not provider:
            yield json.dumps({"type": "error", "error": "No LLM providers available"})
            return

    # Route to appropriate provider
    if provider == "openai":
        model = model or DEFAULT_MODELS["openai"]
        async for token in stream_openai(messages, system_prompt, model, enable_reasoning):
            yield token

    elif provider == "anthropic":
        model = model or DEFAULT_MODELS["anthropic"]
        async for token in stream_anthropic(messages, system_prompt, model, enable_reasoning):
            yield token

    elif provider == "gemini":
        model = model or DEFAULT_MODELS["gemini"]
        async for token in stream_gemini(messages, system_prompt, model, enable_reasoning):
            yield token

    else:
        yield json.dumps({"type": "error", "error": f"Unknown provider '{provider}'"})
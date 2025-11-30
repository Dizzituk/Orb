# app/llm/streaming.py
"""
Streaming LLM responses using Server-Sent Events.
All system prompts include current date/time context.
"""

import os
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


# ============ PROVIDER CHECKS ============

# Check which packages are installed
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


# Alias for backwards compatibility
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
) -> AsyncGenerator[str, None]:
    """Stream from OpenAI using async client."""
    if not HAS_OPENAI:
        yield "[Error: openai package not installed]"
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield "[Error: OPENAI_API_KEY not set]"
        return

    # Use async client for proper streaming
    client = AsyncOpenAI(api_key=api_key)

    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)

    full_messages = [{"role": "system", "content": enhanced_prompt}]
    full_messages.extend(messages)

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
        yield f"[Error: {str(e)}]"


async def stream_anthropic(
    messages: List[Dict],
    system_prompt: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> AsyncGenerator[str, None]:
    """Stream from Anthropic using async client."""
    if not HAS_ANTHROPIC:
        yield "[Error: anthropic package not installed]"
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield "[Error: ANTHROPIC_API_KEY not set]"
        return

    # Use async client
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)

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
        yield f"[Error: {str(e)}]"


async def stream_gemini(
    messages: List[Dict],
    system_prompt: str = "",
    model: str = "gemini-2.0-flash",
) -> AsyncGenerator[str, None]:
    """Stream from Gemini."""
    if not HAS_GEMINI:
        yield "[Error: google-generativeai package not installed]"
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield "[Error: GOOGLE_API_KEY not set]"
        return

    genai.configure(api_key=api_key)

    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)

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
        
        # Gemini streaming is synchronous, wrap in async
        response = chat.send_message(last_msg, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"[Error: {str(e)}]"


# ============ MAIN STREAMING FUNCTION ============

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response.

    Args:
        messages: List of {role, content} dicts
        system_prompt: System prompt (datetime will be added automatically)
        provider: openai, anthropic, or gemini (auto-selects if None)
        model: Model name (uses default if None)

    Yields:
        Tokens as they arrive
    """
    # Auto-select provider if not specified
    if not provider:
        provider = get_default_provider()
        if not provider:
            yield "[Error: No LLM providers available]"
            return

    # Route to appropriate provider
    if provider == "openai":
        model = model or DEFAULT_MODELS["openai"]
        async for token in stream_openai(messages, system_prompt, model):
            yield token

    elif provider == "anthropic":
        model = model or DEFAULT_MODELS["anthropic"]
        async for token in stream_anthropic(messages, system_prompt, model):
            yield token

    elif provider == "gemini":
        model = model or DEFAULT_MODELS["gemini"]
        async for token in stream_gemini(messages, system_prompt, model):
            yield token

    else:
        yield f"[Error: Unknown provider '{provider}']"
# FILE: app/llm/streaming.py
"""
Streaming LLM responses using Server-Sent Events.
All system prompts include current date/time context.

v0.13.9 - Streaming Event Contract Fix:
- CANONICAL EVENT SCHEMA: All providers now yield dict events consistently
- Removed json.dumps() - events are dicts, not JSON strings
- Fixed: OpenAI/Anthropic/Gemini now yield {"type": "token", "text": "..."}
- Fixed: Metadata/error events remain dicts
- Contract: {"type": "metadata|token|reasoning|error|done", ...}

v0.13.0 - Phase 4 Routing Fix:
- Updated DEFAULT_MODELS to use 8-route env vars
- Added chat.light vs text.heavy distinction
- Added video.heavy and opus.critic model support

v0.16.1: Updated default models to match Phase-4 routing.
v0.16.0: Added reasoning support and metadata emission.

CANONICAL STREAMING EVENT SCHEMA
================================
All streaming functions must yield dict events with this structure:

{"type": "metadata", "provider": "...", "model": "..."}
    - First event from each provider
    - Contains provider and model information

{"type": "token", "text": "<chunk of answer>"}
    - Incremental text chunks for the response
    - Accumulated to build the complete answer

{"type": "reasoning", "text": "<hidden chain-of-thought>"}
    - Optional reasoning/thinking content
    - Extracted from THINKING tags or similar

{"type": "error", "message": "<error message>"}
    - Error occurred during streaming
    - Stream should terminate after this

{"type": "done"}
    - Optional final event indicating completion
    - Not currently used by providers but part of spec

IMPORTANT: All events are Python dicts, NOT JSON strings.
The SSE layer (stream_router.py) handles JSON serialization.
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


# ============ 8-ROUTE DEFAULT MODELS ============

DEFAULT_MODELS = {
    # OpenAI routes
    "openai": {
        "chat.light": os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
        "text.heavy": os.getenv("OPENAI_MODEL_HEAVY_TEXT", "gpt-4.1"),
        "default": os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
    },
    # Anthropic routes
    "anthropic": {
        "code.medium": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
        "orchestrator": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20250514"),
        "default": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    },
    # Google routes
    "google": {
        "image.simple": os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash"),
        "image.complex": os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
        "video.heavy": os.getenv("GEMINI_VIDEO_HEAVY_MODEL", "gemini-3.0-pro-preview"),
        "opus.critic": os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3.0-pro-preview"),
        "default": os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash"),
    },
}

# Legacy flat defaults (backward compat)
LEGACY_DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
    "anthropic": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    "gemini": os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash"),
}


def get_model_for_route(provider: str, route: str) -> str:
    """Get model for a provider/route combination."""
    provider_models = DEFAULT_MODELS.get(provider, {})
    if isinstance(provider_models, dict):
        return provider_models.get(route, provider_models.get("default", ""))
    return provider_models


def get_default_model(provider: str) -> str:
    """Get default model for a provider."""
    provider_models = DEFAULT_MODELS.get(provider, {})
    if isinstance(provider_models, dict):
        return provider_models.get("default", "")
    return provider_models


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
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from OpenAI using async client.
    
    Yields dict events following canonical schema:
    - {"type": "metadata", "provider": "openai", "model": "..."}
    - {"type": "token", "text": "..."}
    - {"type": "error", "message": "..."}
    """
    if not HAS_OPENAI:
        yield {"type": "error", "message": "openai package not installed"}
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    # Use provided model, route-based model, or default
    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("openai", route)
    else:
        use_model = get_default_model("openai")

    client = AsyncOpenAI(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    full_messages = [{"role": "system", "content": enhanced_prompt}]
    full_messages.extend(messages)

    # Emit metadata first
    yield {"type": "metadata", "provider": "openai", "model": use_model}

    try:
        stream = await client.chat.completions.create(
            model=use_model,
            messages=full_messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                # v0.13.9: Changed from raw string to dict event
                yield {"type": "token", "text": chunk.choices[0].delta.content}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


async def stream_anthropic(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from Anthropic using async client.
    
    Yields dict events following canonical schema:
    - {"type": "metadata", "provider": "anthropic", "model": "..."}
    - {"type": "token", "text": "..."}
    - {"type": "error", "message": "..."}
    """
    if not HAS_ANTHROPIC:
        yield {"type": "error", "message": "anthropic package not installed"}
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY not set"}
        return

    # Use provided model, route-based model, or default
    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("anthropic", route)
    else:
        use_model = get_default_model("anthropic")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # Emit metadata first
    yield {"type": "metadata", "provider": "anthropic", "model": use_model}

    try:
        async with client.messages.stream(
            model=use_model,
            max_tokens=4096,
            system=enhanced_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                # v0.13.9: Changed from raw string to dict event
                yield {"type": "token", "text": text}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


async def stream_gemini(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from Gemini.
    
    Yields dict events following canonical schema:
    - {"type": "metadata", "provider": "gemini", "model": "..."}
    - {"type": "token", "text": "..."}
    - {"type": "error", "message": "..."}
    """
    if not HAS_GEMINI:
        yield {"type": "error", "message": "google-generativeai package not installed"}
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "GOOGLE_API_KEY not set"}
        return

    # Use provided model, route-based model, or default
    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("google", route)
    else:
        use_model = get_default_model("google")

    genai.configure(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # Emit metadata first
    yield {"type": "metadata", "provider": "gemini", "model": use_model}

    try:
        gemini_model = genai.GenerativeModel(
            model_name=use_model,
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
                # v0.13.9: Changed from raw string to dict event
                yield {"type": "token", "text": chunk.text}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


# ============ MAIN STREAMING FUNCTION ============

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream LLM response with canonical event schema.

    Args:
        messages: List of {role, content} dicts
        system_prompt: System prompt (datetime will be added automatically)
        provider: openai, anthropic, or gemini (auto-selects if None)
        model: Model name (uses route or default if None)
        enable_reasoning: If True, instruct LLM to use THINKING/ANSWER tags
        route: Route name for model selection (e.g., "chat.light", "orchestrator")

    Yields:
        Dict events following canonical schema:
        - {"type": "metadata", "provider": "...", "model": "..."}
        - {"type": "token", "text": "..."}
        - {"type": "error", "message": "..."}
        
    v0.13.9: All events are now dicts (not JSON strings).
    """
    # Auto-select provider if not specified
    if not provider:
        provider = get_default_provider()
        if not provider:
            yield {"type": "error", "message": "No LLM providers available"}
            return

    # Route to appropriate provider
    if provider == "openai":
        async for event in stream_openai(messages, system_prompt, model, enable_reasoning, route):
            yield event

    elif provider == "anthropic":
        async for event in stream_anthropic(messages, system_prompt, model, enable_reasoning, route):
            yield event

    elif provider == "gemini" or provider == "google":
        async for event in stream_gemini(messages, system_prompt, model, enable_reasoning, route):
            yield event

    else:
        yield {"type": "error", "message": f"Unknown provider '{provider}'"}
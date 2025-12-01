# FILE: app/llm/clients.py
"""
LLM Provider API Clients.
Wraps OpenAI, Anthropic, and Google Gemini APIs.
All system prompts include current date/time context.

BACKWARD COMPATIBLE: Original function signatures preserved.
New optional parameters added for policy-based routing.
"""

import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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

def check_provider_availability() -> Dict[str, bool]:
    """Check which LLM providers have API keys configured."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
    }


def list_available_providers() -> list[str]:
    """List providers with configured API keys."""
    available = []
    status = check_provider_availability()
    for provider, is_available in status.items():
        if is_available:
            available.append(provider)
    return available


# ============ OPENAI ============

def call_openai(
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",  # Original default preserved
    temperature: float = 0.7,
    max_tokens: int = 4096,  # NEW: optional param
) -> Tuple[str, Dict]:
    """
    Call OpenAI API.
    Returns (content, usage_dict).
    
    Args:
        system_prompt: System instructions
        messages: Conversation messages
        model: Model to use (default: gpt-4o-mini)
        temperature: Sampling temperature
        max_tokens: Max response tokens (NEW)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)
    
    full_messages = [{"role": "system", "content": enhanced_prompt}]
    full_messages.extend(messages)
    
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message.content or ""
    
    # Build usage dict
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    
    return content, usage


# ============ ANTHROPIC ============

def call_anthropic(
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 4096,  # NEW: optional param
) -> Tuple[str, Dict]:
    """
    Call Anthropic API.
    Returns (content, usage_dict).
    
    Args:
        system_prompt: System instructions
        messages: Conversation messages
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Max response tokens (NEW)
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=enhanced_prompt,
        messages=messages,
        temperature=temperature,
    )
    
    content = response.content[0].text if response.content else ""
    
    # Build usage dict
    usage = {}
    if hasattr(response, 'usage') and response.usage:
        usage["input_tokens"] = response.usage.input_tokens
        usage["output_tokens"] = response.usage.output_tokens
    
    return content, usage


# ============ GOOGLE GEMINI ============

def call_gemini(
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash",
    temperature: float = 0.7,
    max_tokens: int = 8192,  # NEW: optional param
    enable_web_search: bool = False,  # NEW: optional param
    attachments: Optional[List[Dict]] = None,  # NEW: optional param for vision
) -> Tuple[str, Dict]:
    """
    Call Google Gemini API.
    Returns (content, usage_dict).
    
    Args:
        system_prompt: System instructions
        messages: Conversation messages
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Max response tokens (NEW)
        enable_web_search: Enable web grounding (NEW)
        attachments: Image/doc attachments for vision (NEW)
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    
    # Try new SDK first for web search support
    if enable_web_search:
        try:
            return _call_gemini_new_sdk(
                api_key, system_prompt, messages, model,
                temperature, max_tokens, enable_web_search, attachments
            )
        except ImportError:
            logger.warning("google-genai not installed, web search disabled")
    
    # Fall back to old SDK (or use if no web search needed)
    return _call_gemini_old_sdk(
        api_key, system_prompt, messages, model,
        temperature, attachments
    )


def _call_gemini_old_sdk(
    api_key: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    attachments: Optional[List[Dict]] = None,
) -> Tuple[str, Dict]:
    """Call Gemini using google-generativeai SDK (original implementation)."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("google-generativeai package not installed")
    
    genai.configure(api_key=api_key)
    
    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)
    
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=enhanced_prompt,
        generation_config={"temperature": temperature},
    )
    
    # Convert messages to Gemini format
    history = []
    for msg in messages[:-1]:  # All but last
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    
    chat = gemini_model.start_chat(history=history)
    
    # Build last message parts
    last_content = messages[-1]["content"] if messages else ""
    parts = [last_content]
    
    # Add attachments if present (vision)
    if attachments:
        import base64
        for att in attachments:
            mime_type = att.get("mime_type", "image/jpeg")
            data = att.get("data", "")
            if data:
                if isinstance(data, str):
                    data = base64.b64decode(data)
                parts.append({"mime_type": mime_type, "data": data})
    
    response = chat.send_message(parts if len(parts) > 1 else last_content)
    
    content = response.text if response.text else ""
    usage = {}
    
    return content, usage


def _call_gemini_new_sdk(
    api_key: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    enable_web_search: bool,
    attachments: Optional[List[Dict]],
) -> Tuple[str, Dict]:
    """Call Gemini using google-genai SDK (new, supports web search)."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    
    # Enhanced system prompt
    enhanced_prompt = enhance_system_prompt(system_prompt)
    
    # Build contents
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "assistant":
            role = "model"
        elif role == "system":
            continue
        else:
            role = "user"
        
        parts = [types.Part.from_text(content)]
        
        # Add attachments to last user message
        if role == "user" and attachments and msg == messages[-1]:
            import base64
            for att in attachments:
                mime_type = att.get("mime_type", "image/jpeg")
                data = att.get("data", "")
                if data:
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        
        contents.append(types.Content(role=role, parts=parts))
    
    # Configure tools
    tools = []
    if enable_web_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    
    # Generate
    config = types.GenerateContentConfig(
        system_instruction=enhanced_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
        tools=tools if tools else None,
    )
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    
    content = response.text if response.text else ""
    
    # Extract sources if web search was used
    if enable_web_search:
        sources = _extract_grounding_sources(response)
        if sources:
            content += f"\n\n---\nSources:\n{sources}"
    
    usage = {}
    if hasattr(response, 'usage_metadata'):
        usage = {
            "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
            "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
        }
    
    return content, usage


def _extract_grounding_sources(response) -> str:
    """Extract grounding sources from Gemini response."""
    sources = []
    try:
        for candidate in response.candidates:
            if hasattr(candidate, 'grounding_metadata'):
                metadata = candidate.grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            sources.append(f"- [{chunk.web.title}]({chunk.web.uri})")
    except Exception:
        pass
    return "\n".join(sources) if sources else ""


# ============ EMBEDDINGS ============

def get_embeddings(
    text: str,
    model: str = "text-embedding-3-small",
) -> Optional[List[float]]:
    """
    Generate embeddings using OpenAI.
    
    Args:
        text: Text to embed
        model: Embedding model
    
    Returns:
        Embedding vector or None on error
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI package not installed")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return None
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


# ============ ALIASES ============

# Original alias preserved
call_google = call_gemini
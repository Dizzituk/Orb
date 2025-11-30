# app/llm/clients.py
"""
LLM Provider API Clients.
Wraps OpenAI, Anthropic, and Google Gemini APIs.
All system prompts include current date/time context.
"""

import os
from typing import List, Dict, Tuple, Optional
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

def check_provider_availability() -> Dict[str, bool]:
    """Check which LLM providers have API keys configured."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
    }


# ============ OPENAI ============

def call_openai(
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[str, Dict]:
    """
    Call OpenAI API.
    Returns (content, usage_dict).
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
) -> Tuple[str, Dict]:
    """
    Call Anthropic API.
    Returns (content, usage_dict).
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
        max_tokens=4096,
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
) -> Tuple[str, Dict]:
    """
    Call Google Gemini API.
    Returns (content, usage_dict).
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("google-generativeai package not installed")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    
    genai.configure(api_key=api_key)
    
    # Enhance system prompt with datetime
    enhanced_prompt = enhance_system_prompt(system_prompt)
    
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=enhanced_prompt,
    )
    
    # Convert messages to Gemini format
    history = []
    for msg in messages[:-1]:  # All but last
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    
    chat = gemini_model.start_chat(history=history)
    
    # Send last message
    last_msg = messages[-1]["content"] if messages else ""
    response = chat.send_message(last_msg)
    
    content = response.text if response.text else ""
    
    # Build usage dict (Gemini doesn't provide token counts in same way)
    usage = {}
    
    return content, usage


# Alias for backwards compatibility
call_google = call_gemini
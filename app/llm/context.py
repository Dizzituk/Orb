# app/llm/context.py
"""
Context utilities for LLM calls.
Provides current datetime and other contextual information.
"""

from datetime import datetime
import locale


def get_current_datetime_context() -> str:
    """
    Get current date and time formatted for LLM context.
    Returns a string like: "Saturday, November 30, 2025 at 5:45 PM (GMT)"
    """
    now = datetime.now()
    
    # Format: Day, Month DD, YYYY at HH:MM AM/PM
    formatted = now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    # Add timezone info if available
    try:
        import time
        tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        formatted += f" ({tz_name})"
    except:
        pass
    
    return formatted


def get_system_context() -> str:
    """
    Get full system context string to prepend to system prompts.
    """
    datetime_str = get_current_datetime_context()
    return f"Current date and time: {datetime_str}"


def enhance_system_prompt(base_prompt: str) -> str:
    """
    Enhance a system prompt with current context (datetime, etc).
    """
    context = get_system_context()
    
    if base_prompt:
        return f"{context}\n\n{base_prompt}"
    else:
        return context

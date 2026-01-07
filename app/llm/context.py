# app/llm/context.py
"""
Context utilities for LLM calls.
Provides current datetime, capability awareness, and other contextual information.

v1.1 - Added ASTRA capability layer injection
"""

from datetime import datetime
import locale
import logging

logger = logging.getLogger(__name__)


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
    Enhance a system prompt with current context (datetime, capabilities, etc).
    
    Order of injection (top to bottom in final prompt):
    1. ASTRA Capability Layer (what the system can/cannot do)
    2. DateTime context
    3. Base prompt
    """
    # 1. Inject ASTRA capability layer FIRST
    try:
        from app.capabilities import inject_capabilities
        prompt_with_caps = inject_capabilities(base_prompt)
        logger.debug("Capability layer injected successfully")
    except ImportError as e:
        logger.warning(f"Capability layer not available: {e}")
        prompt_with_caps = base_prompt
    except Exception as e:
        logger.error(f"Error injecting capabilities: {e}")
        prompt_with_caps = base_prompt
    
    # 2. Add datetime context
    context = get_system_context()
    
    if prompt_with_caps:
        return f"{context}\n\n{prompt_with_caps}"
    else:
        return context
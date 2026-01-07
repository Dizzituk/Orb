# FILE: app/capabilities/injector.py
"""
ASTRA Capability Layer Injector

Provides functions to inject capability context into system prompts.
This ensures every LLM call in ASTRA has awareness of system capabilities.

The capability layer is injected at the VERY TOP of system prompts,
before any other context (datetime, memory, project context, etc.).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .loader import get_capability_context, get_capability_summary

logger = logging.getLogger(__name__)

# Environment variable to control capability injection
CAPABILITIES_ENABLED = os.getenv("ASTRA_CAPABILITIES_ENABLED", "1") == "1"

# Use summary vs full context based on token budget concerns
USE_SUMMARY = os.getenv("ASTRA_CAPABILITIES_SUMMARY_ONLY", "0") == "1"


def inject_capabilities(system_prompt: Optional[str] = None) -> str:
    """
    Inject capability context at the beginning of a system prompt.
    
    This is the PRIMARY injection function. It should be called on
    every system prompt before sending to any LLM.
    
    Args:
        system_prompt: The original system prompt (can be None or empty)
        
    Returns:
        System prompt with capability context prepended.
    """
    if not CAPABILITIES_ENABLED:
        logger.debug("Capability injection disabled via ASTRA_CAPABILITIES_ENABLED")
        return system_prompt or ""
    
    # Get appropriate capability context
    if USE_SUMMARY:
        capability_context = get_capability_summary()
    else:
        capability_context = get_capability_context()
    
    # Handle None or empty prompt
    if not system_prompt:
        return capability_context
    
    # Prepend capabilities to existing prompt
    return f"{capability_context}\n{system_prompt}"


def enhance_system_prompt_with_capabilities(
    base_prompt: str,
    include_full: bool = True,
    additional_context: Optional[str] = None
) -> str:
    """
    Enhanced system prompt builder with capability layer.
    
    This function is designed to replace or wrap the existing
    enhance_system_prompt() in app/llm/context.py.
    
    Args:
        base_prompt: The base system prompt
        include_full: If True, include full capability context; if False, use summary
        additional_context: Optional additional context to include
        
    Returns:
        Fully enhanced system prompt with capabilities at the top.
    """
    if not CAPABILITIES_ENABLED:
        result = base_prompt
        if additional_context:
            result = f"{additional_context}\n\n{result}"
        return result
    
    # Build the enhanced prompt with capabilities FIRST
    parts = []
    
    # 1. Capability layer (ALWAYS first)
    if include_full:
        parts.append(get_capability_context())
    else:
        parts.append(get_capability_summary())
    
    # 2. Additional context (if any)
    if additional_context:
        parts.append(additional_context)
    
    # 3. Base prompt
    parts.append(base_prompt)
    
    return "\n\n".join(parts)


def get_stage_specific_context(stage: str) -> str:
    """
    Get capability context tailored for a specific pipeline stage.
    
    Different stages may need different emphasis on capabilities.
    
    Args:
        stage: The pipeline stage (chat, weaver, spec_gate, critical, overwatcher, implementer)
        
    Returns:
        Stage-appropriate capability context.
    """
    base_context = get_capability_summary()
    
    stage_additions = {
        "chat": """
Remember: When users ask you to perform file operations, code execution,
or repository modifications, you CAN do these things via the ASTRA pipeline.
Offer to route their request through the appropriate mechanism.
""",
        "weaver": """
Specifications you generate must respect ASTRA's capability boundaries:
- All file operations target sandbox user-space only
- Host PC is never a valid target
- Deletions must specify confirmation requirement
""",
        "spec_gate": """
When validating specifications, verify they operate within allowed zones:
- Sandbox user-space: ✅ ALLOWED
- Sandbox system: ❌ FORBIDDEN  
- Host PC: ❌ FORBIDDEN
Flag any specs that would violate hard safety rules.
""",
        "critical": """
Architectures must be implementable within ASTRA's constraints:
- All code changes happen in sandbox repo clone
- File operations respect zone boundaries
- Delete operations include confirmation step
""",
        "overwatcher": """
You are supervising implementation within these boundaries:
- Sandbox user-space operations: ✅ PROCEED
- System-level changes: ❌ HALT AND REPORT
- Unauthorized deletions: ❌ REQUIRE CONFIRMATION
Apply conduct policy strictly.
""",
        "implementer": """
Execute implementation chunks within sandbox only:
- Target: Sandbox repo clone at D:\\Orb
- Allowed: File create, modify, read
- Forbidden: System paths, host access, unconfirmed deletes
"""
    }
    
    addition = stage_additions.get(stage.lower(), "")
    
    if addition:
        return f"{base_context}\n{addition}"
    return base_context


def should_block_action(action: str, target: str) -> tuple[bool, str]:
    """
    Quick check if an action should be blocked.
    
    Used by enforcement layers to quickly validate requests.
    
    Args:
        action: The action being attempted
        target: The target path or environment
        
    Returns:
        Tuple of (should_block: bool, reason: str)
    """
    # Hard blocks for host PC access
    host_indicators = [
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Users\\",  # Note: sandbox user is WDAGUtilityAccount
        "D:\\Orb",  # This is the HOST repo path
    ]
    
    # Check for sandbox user path (allowed, but delete needs confirmation)
    if "WDAGUtilityAccount" in target:
        # Delete actions require confirmation even in sandbox
        if action.lower() in ["delete", "remove", "unlink", "rmdir"]:
            return (False, "Sandbox delete action - allowed but requires confirmation")
        return (False, "Sandbox user path - allowed")
    
    # Check for host paths
    for indicator in host_indicators:
        if indicator.lower() in target.lower():
            # Special case: D:\Orb inside sandbox is OK
            if "sandbox" in str(target).lower() or "192.168.250" in str(target):
                return (False, "Sandbox path - allowed")
            return (True, f"Host PC path detected ({indicator}) - forbidden by HSR-001")
    
    # Check for system paths in sandbox
    system_paths = [
        "\\Windows\\",
        "\\System32\\",
        "\\SysWOW64\\",
        "\\Program Files\\",
        "\\ProgramData\\",
    ]
    
    for sys_path in system_paths:
        if sys_path.lower() in target.lower():
            return (True, f"System path detected ({sys_path}) - forbidden by HSR-002")
    
    # Delete actions require confirmation
    if action.lower() in ["delete", "remove", "unlink", "rmdir"]:
        return (False, "Delete action - allowed but requires confirmation")
    
    return (False, "Action appears to be within allowed boundaries")


# Convenience function matching existing context.py pattern
def enhance_system_prompt(base_prompt: str) -> str:
    """
    Drop-in replacement for app/llm/context.py enhance_system_prompt.
    
    This version adds capability context BEFORE the datetime context.
    
    Args:
        base_prompt: Original system prompt
        
    Returns:
        Enhanced prompt with capabilities and datetime context.
    """
    # Import here to avoid circular imports
    try:
        from app.llm.context import get_system_context
        datetime_context = get_system_context()
    except ImportError:
        import datetime
        datetime_context = f"Current time: {datetime.datetime.now().isoformat()}"
    
    return enhance_system_prompt_with_capabilities(
        base_prompt,
        include_full=True,
        additional_context=datetime_context
    )
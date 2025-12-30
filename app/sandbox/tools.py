# FILE: app/sandbox/tools.py
"""Sandbox Tools: LLM-callable tools for controlling the zombie Orb.

These tools allow Main Orb to control its sandbox clone through
natural language commands processed by the LLM.

Trigger phrases:
- "start your zombie" / "start your clone" / "boot sandbox orb"
- "stop your zombie" / "stop the clone" / "shutdown sandbox"
- "check zombie status" / "is your clone running?"
- "wake up your clone" / "spin up the sandbox"
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.sandbox.manager import (
    get_sandbox_manager,
    start_zombie_orb,
    stop_zombie_orb,
    check_zombie_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions (for LLM tool calling)
# =============================================================================

SANDBOX_TOOLS = [
    {
        "name": "start_sandbox_clone",
        "description": (
            "Start the zombie/sandbox clone of Orb. This boots up a copy of Orb "
            "running in Windows Sandbox for testing and self-improvement purposes. "
            "Use when the user says things like: 'start your zombie', 'boot your clone', "
            "'spin up the sandbox', 'wake up your clone', 'start sandbox orb'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_mode": {
                    "type": "boolean",
                    "description": "If true, start both backend and frontend. If false, backend only.",
                    "default": True,
                }
            },
            "required": [],
        },
    },
    {
        "name": "stop_sandbox_clone",
        "description": (
            "Stop the zombie/sandbox clone of Orb. This shuts down the Orb instance "
            "running in Windows Sandbox. Use when the user says things like: "
            "'stop your zombie', 'shutdown the clone', 'kill the sandbox', 'stop sandbox orb'."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_sandbox_status",
        "description": (
            "Check the status of the zombie/sandbox clone. Returns whether the sandbox "
            "is connected and if the Orb clone is running. Use when the user asks: "
            "'is your zombie running?', 'check clone status', 'is the sandbox up?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# =============================================================================
# Tool Implementations
# =============================================================================

def execute_sandbox_tool(tool_name: str, parameters: Dict[str, Any] = None) -> str:
    """Execute a sandbox tool and return the result message.
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters (optional)
    
    Returns:
        Human-readable result message
    """
    parameters = parameters or {}
    
    if tool_name == "start_sandbox_clone":
        full_mode = parameters.get("full_mode", True)
        success, message = start_zombie_orb(full=full_mode)
        return message
    
    elif tool_name == "stop_sandbox_clone":
        success, message = stop_zombie_orb()
        return message
    
    elif tool_name == "check_sandbox_status":
        return check_zombie_status()
    
    else:
        return f"Unknown sandbox tool: {tool_name}"


# =============================================================================
# Intent Detection (for routing prompts to tools)
# =============================================================================

# Keywords that suggest sandbox/zombie operations
START_KEYWORDS = [
    "start your zombie", "start the zombie", "start zombie",
    "start your clone", "start the clone", "start clone",
    "boot your clone", "boot the clone", "boot clone",
    "boot sandbox", "start sandbox", "boot the sandbox",
    "wake up your clone", "wake your clone", "wake clone",
    "spin up sandbox", "spin up your clone", "spin up the sandbox",
    "start sandbox orb", "boot sandbox orb", "launch sandbox",
    "bring up your clone", "bring up the sandbox",
]

STOP_KEYWORDS = [
    "stop your zombie", "stop the zombie", "stop zombie",
    "stop your clone", "stop the clone", "stop clone",
    "shutdown sandbox", "shut down sandbox", "shutdown your clone",
    "kill the sandbox", "kill your clone", "kill zombie",
    "stop sandbox orb", "shutdown zombie",
    "turn off your clone", "turn off sandbox",
]

STATUS_KEYWORDS = [
    "zombie status", "clone status", "sandbox status",
    "is your zombie running", "is your clone running", "is zombie running",
    "is the sandbox up", "is sandbox running", "check sandbox",
    "is your clone up", "is clone up", "check your clone",
    "zombie running", "clone running",
]


def detect_sandbox_intent(prompt: str) -> tuple[str | None, Dict[str, Any]]:
    """Detect if a prompt is asking for a sandbox operation.
    
    Args:
        prompt: User's prompt text
    
    Returns:
        (tool_name, parameters) if sandbox intent detected, else (None, {})
    """
    prompt_lower = prompt.lower().strip()
    
    # Check for start intent
    for keyword in START_KEYWORDS:
        if keyword in prompt_lower:
            # Check if backend-only requested
            full_mode = "backend only" not in prompt_lower and "api only" not in prompt_lower
            return "start_sandbox_clone", {"full_mode": full_mode}
    
    # Check for stop intent
    for keyword in STOP_KEYWORDS:
        if keyword in prompt_lower:
            return "stop_sandbox_clone", {}
    
    # Check for status intent
    for keyword in STATUS_KEYWORDS:
        if keyword in prompt_lower:
            return "check_sandbox_status", {}
    
    return None, {}


def handle_sandbox_prompt(prompt: str) -> str | None:
    """Handle a prompt if it's a sandbox command.
    
    Args:
        prompt: User's prompt text
    
    Returns:
        Response message if handled, None if not a sandbox command
    """
    tool_name, parameters = detect_sandbox_intent(prompt)
    
    if tool_name is None:
        return None
    
    logger.info(f"Detected sandbox intent: {tool_name} with params {parameters}")
    return execute_sandbox_tool(tool_name, parameters)


__all__ = [
    "SANDBOX_TOOLS",
    "execute_sandbox_tool",
    "detect_sandbox_intent",
    "handle_sandbox_prompt",
]

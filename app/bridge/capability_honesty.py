# FILE: app/bridge/capability_honesty.py
"""
Bridge Capability Honesty — prevents the LLM from claiming it performed
actions that were never executed.

When a user requests a tool-backed action (web search, image generation,
file analysis, etc.), the bridge must either:
  1. Execute the tool and confirm with evidence
  2. Report that execution failed
  3. Report that the capability is not available on this route

The LLM must NEVER generate a confident "I did that" response when no
tool actually ran. This module enforces that contract.

v1.0 (2026-04-06): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Intents that require tool execution — the LLM must not bluff these
TOOL_BACKED_INTENTS = {
    "WEB_SEARCH",
    "DEEP_RESEARCH",
    "GENERATE_IMAGE",
    "BUILD_AND_DEPLOY",
    "RUN_PIPELINE",  # v5.4: unified pipeline intent
    "RUN_CRITICAL_PIPELINE_FOR_JOB",  # v5.4: deprecated alias (confirmed_intent path)
    "WEAVER_BUILD_SPEC",
    "SEND_TO_SPEC_GATE",
}

# Intents that are not yet wired through the bridge
UNSUPPORTED_BRIDGE_INTENTS = {
    "BUILD_AND_DEPLOY",
    "RUN_PIPELINE",  # v5.4: unified pipeline intent
    "RUN_CRITICAL_PIPELINE_FOR_JOB",  # v5.4: deprecated alias (confirmed_intent path)
    "WEAVER_BUILD_SPEC",
    "SEND_TO_SPEC_GATE",
}

# Human-friendly messages for unsupported capabilities
_UNSUPPORTED_MESSAGES = {
    "BUILD_AND_DEPLOY": (
        "Build and deploy operations aren't available from the Bridge app yet. "
        "Use the desktop builds tab for now."
    ),
    "RUN_PIPELINE": (
        "Pipeline operations aren't available from the Bridge app yet. "
        "Use the desktop builds tab for now."
    ),
    "RUN_CRITICAL_PIPELINE_FOR_JOB": (
        "Pipeline operations aren't available from the Bridge app yet. "
        "Use the desktop builds tab for now."
    ),
    "WEAVER_BUILD_SPEC": (
        "The weaver and spec gate aren't available from the Bridge app yet. "
        "Use the desktop for build pipeline operations."
    ),
    "SEND_TO_SPEC_GATE": (
        "The spec gate isn't available from the Bridge app yet. "
        "Use the desktop for build pipeline operations."
    ),
}

_SEARCH_FAILED_MESSAGE = (
    "I tried to search the web for you but the search didn't return results. "
    "I won't pretend I have current information — ask me again in a moment, "
    "or try from the desktop where the full search pipeline is available."
)

_SEARCH_NOT_DETECTED = (
    "I don't have live web search available on this request. "
    "I can share what I know from my training data, but I can't "
    "verify it's current. Want me to do that, or would you prefer "
    "to ask from the desktop where full search is available?"
)


def is_tool_backed(intent_value: Optional[str]) -> bool:
    """Check if this intent requires tool execution proof."""
    if not intent_value:
        return False
    return intent_value in TOOL_BACKED_INTENTS


def is_unsupported_on_bridge(intent_value: Optional[str]) -> bool:
    """Check if this intent is not yet wired through the bridge."""
    if not intent_value:
        return False
    return intent_value in UNSUPPORTED_BRIDGE_INTENTS


def get_unsupported_message(intent_value: str) -> str:
    """Get a human-friendly message for an unsupported capability."""
    return _UNSUPPORTED_MESSAGES.get(
        intent_value,
        "That capability isn't available from the Bridge app yet."
    )


def get_search_failed_message() -> str:
    """Get the message for when web search was attempted but failed."""
    return _SEARCH_FAILED_MESSAGE


def get_search_not_detected_message() -> str:
    """Get the message for when search intent wasn't detected but user expects it."""
    return _SEARCH_NOT_DETECTED


def build_honesty_system_addendum(
    intent_value: Optional[str],
    tool_executed: bool,
    tool_succeeded: bool,
) -> str:
    """Build a system prompt addendum that enforces honesty.

    This is injected into the LLM's system prompt so it knows
    whether a tool actually ran and what it should/shouldn't claim.
    """
    if not intent_value:
        return ""

    if intent_value in ("WEB_SEARCH", "DEEP_RESEARCH"):
        if tool_executed and tool_succeeded:
            return (
                "\n\nIMPORTANT: Web search was performed and results are "
                "included above. You may reference these results."
            )
        elif tool_executed and not tool_succeeded:
            return (
                "\n\nIMPORTANT: Web search was attempted but returned no "
                "results. Do NOT pretend you have current information. "
                "Be honest that the search did not succeed."
            )
        else:
            return (
                "\n\nIMPORTANT: No web search was performed for this request. "
                "Do NOT claim to have searched the internet or to have "
                "current information. If the user asked for live/current "
                "data, be honest that you cannot provide it right now."
            )

    return ""
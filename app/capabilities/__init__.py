# FILE: app/capabilities/__init__.py
"""
ASTRA Global Capability Layer Module

This module provides the capability layer that defines what ASTRA can and cannot do.
It is injected at the TOP of every system prompt across all pipeline stages.

The capability layer solves the problem of LLMs saying "I can't do that" when
ASTRA actually HAS the capability via its sandbox and pipeline infrastructure.

Usage:
    from app.capabilities import get_capability_context, inject_capabilities

    # Get the capability context string
    context = get_capability_context()

    # Inject into a system prompt
    enhanced_prompt = inject_capabilities(original_prompt)
"""

from .loader import (
    load_capabilities,
    get_capability_context,
    get_capability_summary,
    get_hard_safety_rules,
    check_capability,
)

from .injector import (
    inject_capabilities,
    enhance_system_prompt_with_capabilities,
)

__all__ = [
    'load_capabilities',
    'get_capability_context',
    'get_capability_summary', 
    'get_hard_safety_rules',
    'check_capability',
    'inject_capabilities',
    'enhance_system_prompt_with_capabilities',
]

__version__ = '1.0.0'

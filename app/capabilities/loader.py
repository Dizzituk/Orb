# FILE: app/capabilities/loader.py
"""
ASTRA Capability Layer Loader

Loads the capability definition from JSON and provides formatted context
for injection into LLM system prompts.

The capability layer is cached after first load for performance.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default paths - can be overridden via environment
DEFAULT_CAPABILITIES_PATH = os.getenv(
    "ASTRA_CAPABILITIES_PATH",
    r"D:\Orb\config\astra_capabilities.json"
)

# Fallback embedded capabilities (used if file not found)
# This ensures ASTRA always has SOME capability awareness
FALLBACK_CAPABILITIES = {
    "version": 1,
    "identity": {
        "name": "ASTRA",
        "core_truth": "ASTRA is a complete system with sandbox execution capabilities, not just a language model."
    },
    "environments": {
        "host_pc": {"allowed": False},
        "sandbox": {"allowed": True}
    },
    "hard_safety_rules": [
        {"id": "HSR-001", "rule": "NO_HOST_WRITE", "description": "Never write to host PC"},
        {"id": "HSR-003", "rule": "DELETE_REQUIRES_CONFIRMATION", "description": "Deletions need confirmation"}
    ]
}


@lru_cache(maxsize=1)
def load_capabilities(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load capabilities from JSON file.
    
    Uses LRU cache to avoid repeated file reads.
    Falls back to embedded defaults if file not found.
    
    Args:
        path: Optional path override. Uses ASTRA_CAPABILITIES_PATH env var or default.
        
    Returns:
        Dictionary containing full capability specification.
    """
    config_path = path or DEFAULT_CAPABILITIES_PATH
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            capabilities = json.load(f)
            logger.info(f"Loaded ASTRA capabilities v{capabilities.get('version', '?')} from {config_path}")
            return capabilities
    except FileNotFoundError:
        logger.warning(f"Capabilities file not found at {config_path}, using fallback")
        return FALLBACK_CAPABILITIES
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in capabilities file: {e}")
        return FALLBACK_CAPABILITIES
    except Exception as e:
        logger.error(f"Error loading capabilities: {e}")
        return FALLBACK_CAPABILITIES


def reload_capabilities() -> Dict[str, Any]:
    """Force reload capabilities (clears cache)."""
    load_capabilities.cache_clear()
    return load_capabilities()


def get_capability_context() -> str:
    """
    Get the full capability context formatted for system prompt injection.
    
    This is the PRIMARY function for getting capability context.
    Returns a formatted string ready to prepend to any system prompt.
    
    Returns:
        Formatted capability context string.
    """
    caps = load_capabilities()
    
    identity = caps.get('identity', {})
    environments = caps.get('environments', {})
    capabilities = caps.get('capabilities', {})
    safety_rules = caps.get('hard_safety_rules', [])
    behavioral = caps.get('behavioral_directives', {})
    
    lines = [
        "=" * 60,
        "ASTRA GLOBAL CAPABILITY LAYER",
        "=" * 60,
        "",
        "## IDENTITY",
        f"You are {identity.get('name', 'ASTRA')}.",
        identity.get('core_truth', ''),
        "",
        "## CORE TRUTH",
        "When users ask you to perform actions (write files, run code, modify repos),",
        "you DO have these capabilities via the ASTRA pipeline and sandbox infrastructure.",
        "Do NOT say 'I cannot do that' or 'As an AI, I don't have access'.",
        "Instead, check your capabilities and offer to execute via the appropriate mechanism.",
        "",
        "## ENVIRONMENTS",
    ]
    
    # Host PC
    host = environments.get('host_pc', {})
    lines.append(f"### Host PC: {'❌ FORBIDDEN' if not host.get('allowed', False) else '✅ ALLOWED'}")
    if not host.get('allowed', False):
        lines.append("   - You must NEVER read, write, or execute on the host PC filesystem.")
    lines.append("")
    
    # Sandbox
    sandbox = environments.get('sandbox', {})
    lines.append(f"### Sandbox: {'✅ ALLOWED' if sandbox.get('allowed', True) else '❌ FORBIDDEN'}")
    if sandbox.get('allowed', True):
        lines.append(f"   - Controller: {caps.get('infrastructure', {}).get('sandbox_controller', {}).get('url', 'http://192.168.250.2:8765')}")
        lines.append("   - You CAN: read files, write files, execute code, explore directories")
        lines.append("   - User space (Desktop, working dirs): ✅ ALLOWED")
        lines.append("   - System zone (OS files, registry): ❌ FORBIDDEN")
    lines.append("")
    
    # Hard Safety Rules
    lines.append("## HARD SAFETY RULES (NON-NEGOTIABLE)")
    for rule in safety_rules:
        rule_id = rule.get('id', 'RULE')
        rule_name = rule.get('rule', '')
        description = rule.get('description', '')
        lines.append(f"   [{rule_id}] {rule_name}: {description}")
    lines.append("")
    
    # Capabilities Summary
    lines.append("## WHAT YOU CAN DO")
    lines.append("   ✅ Create/read/modify files in sandbox user-space")
    lines.append("   ✅ Execute Python and shell commands in sandbox")
    lines.append("   ✅ Analyze repositories (zobie_mapper)")
    lines.append("   ✅ Generate specs (Weaver), validate (Spec Gate), architect (Critical), implement (Overwatcher)")
    lines.append("   ⚠️ Delete files (requires 'ASTRA confirm delete <token>')")
    lines.append("")
    
    # Behavioral Directive
    on_action = behavioral.get('on_action_request', {})
    if on_action:
        lines.append("## WHEN USER ASKS YOU TO DO SOMETHING")
        lines.append(f"   ✅ CORRECT: {on_action.get('correct_response', 'Check capabilities, offer to execute via pipeline')}")
        lines.append(f"   ❌ WRONG: {on_action.get('incorrect_response', 'Saying I cannot do that')}")
    lines.append("")
    
    lines.append("=" * 60)
    lines.append("")
    
    return "\n".join(lines)


def get_capability_summary() -> str:
    """
    Get a condensed capability summary for token-constrained contexts.
    
    Returns:
        Short capability summary string.
    """
    return """[ASTRA CAPABILITIES]
You are ASTRA - a system with real execution capabilities via sandbox.
✅ CAN: Read/write/execute in sandbox user-space, analyze repos, run pipelines
❌ CANNOT: Access host PC, modify sandbox OS, delete without confirmation
When asked to do something, CHECK capabilities and OFFER to execute - don't say "I can't".
[/ASTRA CAPABILITIES]
"""


def get_hard_safety_rules() -> List[Dict[str, Any]]:
    """
    Get just the hard safety rules for quick reference.
    
    Returns:
        List of safety rule dictionaries.
    """
    caps = load_capabilities()
    return caps.get('hard_safety_rules', [])


def check_capability(action: str, target: str = "sandbox") -> Dict[str, Any]:
    """
    Check if a specific action is allowed.
    
    Args:
        action: The action to check (e.g., 'write_files', 'delete_files')
        target: The target environment (e.g., 'sandbox', 'host_pc')
        
    Returns:
        Dict with 'allowed' bool and 'reason' string.
    """
    caps = load_capabilities()
    
    # Check environment first
    env = caps.get('environments', {}).get(target, {})
    if not env.get('allowed', False):
        return {
            'allowed': False,
            'reason': f"Environment '{target}' is not allowed. {env.get('reason', '')}"
        }
    
    # Check specific capability
    all_caps = caps.get('capabilities', {})
    for category, actions in all_caps.items():
        if action in actions:
            cap_info = actions[action]
            if cap_info.get('enabled', True):
                requires_confirm = cap_info.get('requires_confirmation', False)
                return {
                    'allowed': True,
                    'requires_confirmation': requires_confirm,
                    'confirmation_format': cap_info.get('confirmation_format'),
                    'via': cap_info.get('via', 'sandbox_controller'),
                    'reason': cap_info.get('description', '')
                }
    
    # Unknown capability - default to asking
    return {
        'allowed': None,  # Unknown
        'reason': f"Capability '{action}' not explicitly defined. Check with user."
    }


def get_capability_version() -> int:
    """Get the current capability layer version."""
    caps = load_capabilities()
    return caps.get('version', 0)


def format_for_sandbox_context() -> str:
    """
    Format capabilities specifically for sandbox execution context.
    
    This is a specialized format used when ASTRA is operating inside
    the sandbox environment.
    
    Returns:
        Sandbox-specific capability context.
    """
    caps = load_capabilities()
    
    lines = [
        "[ASTRA SANDBOX EXECUTION CONTEXT]",
        "",
        "You are operating in the ASTRA sandbox environment.",
        "Controller: http://192.168.250.2:8765",
        "",
        "ALLOWED OPERATIONS:",
        "  - Read/write files in user directories (Desktop, D:\\Orb, etc.)",
        "  - Execute Python scripts and shell commands",
        "  - List and explore directory structures",
        "",
        "FORBIDDEN OPERATIONS:",
        "  - Modifying Windows system files/folders",
        "  - Accessing registry or drivers",
        "  - Any operation on host PC (this sandbox is isolated)",
        "",
        "CONFIRMATION REQUIRED:",
        "  - File deletion: User must reply 'ASTRA confirm delete <token>'",
        "  - File moves: User confirmation recommended",
        "",
        "[/ASTRA SANDBOX EXECUTION CONTEXT]"
    ]
    
    return "\n".join(lines)

# FILE: app/llm/legacy_triggers.py
"""
Legacy trigger detection for stream router.

Used as fallback when translation layer is unavailable.
These are simple string-matching triggers that predate the
translation layer's rule-based intent classification.

v1.1 (2026-01): Cleaned up (removed host filesystem scan - sandbox only rule)
v1.0 (2026-01): Extracted from stream_router.py
"""

import os
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# TRIGGER SETS
# =============================================================================

_ZOBIE_TRIGGER_SET = {"zobie map", "zombie map", "zobie_map", "/zobie_map", "/zombie_map"}

# Import archmap triggers from local tools
try:
    from app.llm.local_tools.archmap_helpers import (
        _ARCHMAP_TRIGGER_SET,
        _UPDATE_ARCH_TRIGGER_SET,
        ARCHMAP_PROVIDER,
        ARCHMAP_MODEL,
    )
except ImportError:
    _ARCHMAP_TRIGGER_SET = {"create architecture map", "architecture map"}
    _UPDATE_ARCH_TRIGGER_SET = {"update architecture", "refresh architecture"}
    ARCHMAP_PROVIDER = "openai"
    ARCHMAP_MODEL = "gpt-4.1"

# =============================================================================
# ENVIRONMENT CONFIG
# =============================================================================

ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\tools\zobie_mapper\zobie_map.py")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", r"D:\tools\zobie_mapper\out")
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "300"))
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "").strip()
ZOBIE_MAPPER_ARGS = ZOBIE_MAPPER_ARGS_RAW.split() if ZOBIE_MAPPER_ARGS_RAW else []


# =============================================================================
# TRIGGER DETECTION FUNCTIONS
# =============================================================================

def is_zobie_map_trigger(msg: str) -> bool:
    """Check if message triggers zobie map."""
    return (msg or "").strip().lower() in _ZOBIE_TRIGGER_SET


def is_archmap_trigger(msg: str) -> bool:
    """Check if message triggers architecture map creation."""
    return (msg or "").strip().lower() in _ARCHMAP_TRIGGER_SET


def is_update_arch_trigger(msg: str) -> bool:
    """Check if message triggers architecture update."""
    return (msg or "").strip().lower() in _UPDATE_ARCH_TRIGGER_SET


def is_introspection_trigger(msg: str) -> bool:
    """Check if message triggers log introspection."""
    try:
        from app.introspection.chat_integration import detect_log_intent
        intent = detect_log_intent(msg)
        return intent.is_log_request
    except ImportError:
        return False


def is_sandbox_trigger(msg: str) -> bool:
    """Check if message triggers sandbox control."""
    try:
        from app.sandbox import detect_sandbox_intent
        tool, _ = detect_sandbox_intent(msg)
        return tool is not None
    except ImportError:
        return False


__all__ = [
    "is_zobie_map_trigger",
    "is_archmap_trigger", 
    "is_update_arch_trigger",
    "is_introspection_trigger",
    "is_sandbox_trigger",
    "ARCHMAP_PROVIDER",
    "ARCHMAP_MODEL",
    "ZOBIE_CONTROLLER_URL",
    "ZOBIE_MAPPER_SCRIPT",
    "ZOBIE_MAPPER_OUT_DIR",
    "ZOBIE_MAPPER_TIMEOUT_SEC",
    "ZOBIE_MAPPER_ARGS",
]
# Purpose: weaver stream utils 14
# Called-by: app.llm._weaver_stream_prepare, app.llm._weaver_stream_utils_13, app.llm._weaver_stream_utils_15, app.llm._weaver_stream_utils_17 (+1 more)
# Depends-on: app.llm._weaver_stream_utils_13, app.llm._weaver_stream_utils_15, app.llm.streaming
# Last-renovated: 2026-06-11
from __future__ import annotations
import hashlib
import logging
import re
from app.llm._weaver_stream_utils_13 import LEAKAGE_PATTERNS
from typing import Any, Dict
logger = logging.getLogger(__name__)
# Import streaming functions — these must be available for _get_streaming_function
try:
    from app.llm.streaming import stream_openai, stream_anthropic, stream_gemini
except ImportError:
    try:
        from app.llm.streaming import stream_openai
    except ImportError:
        stream_openai = None
    try:
        from app.llm.streaming import stream_anthropic
    except ImportError:
        stream_anthropic = None
    try:
        from app.llm.streaming import stream_gemini
    except ImportError:
        stream_gemini = None


def _get_streaming_function(provider: str):
    """Get the appropriate streaming function for the provider."""
    provider_lower = provider.lower()
    if provider_lower in ("openai", "openai-compatible"):
        return stream_openai
    elif provider_lower in ("anthropic", "claude"):
        return stream_anthropic
    elif provider_lower in ("google", "gemini"):
        return stream_gemini
    else:
        logger.warning("[WEAVER] Unknown provider '%s', defaulting to OpenAI", provider)
        return stream_openai

def _hash_message(msg: Dict[str, Any]) -> str:
    """
    Create a stable hash for a message.
    
    Uses role + normalized content to create a short hash.
    This allows us to track which messages have been woven,
    regardless of message ordering or count drift.
    """
    role = msg.get("role", "").strip().lower()
    content = msg.get("content", "").strip()
    # Normalize whitespace for stability
    content = " ".join(content.split())
    raw = f"{role}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

META_MODE_PATTERNS = [
    r"just\s+talk\s+about\s+it",
    r"no\s+code",
    r"don'?t\s+build\s+it\s+yet",
    r"just\s+planning",
    r"only\s+discuss",
    r"ask\s+me\s+questions\s+first",
    r"before\s+coding",
    r"don'?t\s+assume\s+too\s+much",
    r"discussion\s+only",
    r"no\s+implementation",
    r"planning\s+phase",
    r"just\s+the\s+idea",
    r"for\s+now",
    r"at\s+the\s+moment",
]

def _sanitize_weaver_output(output: str) -> str:
    """
    Sanitize weaver output to remove any prompt scaffold leakage.
    
    If the LLM accidentally echoes parts of the prompt template,
    this function strips them out.
    """
    lines = output.split("\n")
    cleaned_lines = []
    skip_until_content = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for leakage patterns
        is_leakage = False
        for pattern in LEAKAGE_PATTERNS:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_leakage = True
                # If we hit a scaffold header, skip until we see real content
                if "EXISTING" in line_stripped.upper() or "PREVIOUS" in line_stripped.upper():
                    skip_until_content = True
                break
        
        if is_leakage:
            continue
        
        # Skip separator lines when in skip mode
        if skip_until_content and line_stripped == "---":
            continue
        
        # Once we see real content, stop skipping
        if skip_until_content and line_stripped and line_stripped != "---":
            skip_until_content = False
        
        cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines).strip()
    
    # Log if we cleaned anything
    if result != output.strip():
        print("[WEAVER] Sanitized output - removed prompt leakage")
    
    return result

CORE_GOAL_VERBS = [
    "build", "create", "make", "add", "remove", "delete", "fix", "change",
    "update", "modify", "refactor", "implement", "write", "reply", "respond",
    "design", "develop", "integrate", "connect", "migrate", "convert", "generate",
    "set up", "setup", "configure", "install", "deploy", "test", "check",
    "analyze", "review", "edit", "improve", "optimize", "clean", "organize",
]

QUESTIONS_DISMISSED_PATTERNS = [
    r"questions?\s+(are\s+)?not\s+(really\s+)?needed",
    r"don'?t\s+need\s+(to\s+)?(ask|answer)\s+(those\s+)?questions?",
    r"reply\s+to\s+(your\s+)?questions?\s+(are\s+)?not\s+needed",
    r"no\s+need\s+(for|to\s+ask)\s+questions?",
    r"skip\s+(the\s+)?questions?",
    r"ignore\s+(the\s+)?questions?",
    r"questions?\s+aren'?t\s+(really\s+)?relevant",
]

def _normalize_typos(text: str) -> str:
    """
    Silently normalize common typos without flagging them (v3.6.0).
    
    Uses word boundaries to avoid substring collisions.
    """
    from ._weaver_stream_utils_15 import TYPO_NORMALIZATIONS
    result = text
    normalized_any = False
    
    for pattern, correction in TYPO_NORMALIZATIONS:
        if re.search(pattern, result, re.IGNORECASE):
            result = re.sub(pattern, correction, result, flags=re.IGNORECASE)
            normalized_any = True
    
    if normalized_any:
        print(f"[WEAVER] Normalized typos in input")
    
    return result

FEATURE_COMPONENT_INDICATORS = [
    "audio", "microphone", "recording", "capture",
    "transcription", "speech", "voice", "stt", "whisper",
    "button", "widget", "component", "panel",
    "endpoint", "api", "route", "handler",
    "provider", "service", "integration",
    "config", "settings", "environment",
    "stream", "websocket", "real-time", "realtime",
    "authentication", "auth", "permission",
    "notification", "alert", "feedback",
    "database", "storage", "persistence",
]

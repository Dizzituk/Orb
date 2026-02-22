from __future__ import annotations
import logging
import re
from app.llm._weaver_stream_utils_12 import _is_control_message
from app.llm._weaver_stream_utils_13 import BUILD_VERBS, MICRO_FILE_INDICATORS, NON_MICRO_INDICATORS, REFACTOR_INDICATORS
from app.llm._weaver_stream_utils_14 import FEATURE_COMPONENT_INDICATORS, META_MODE_PATTERNS, QUESTIONS_DISMISSED_PATTERNS
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_MEMORY_AVAILABLE = True
memory_service = None


def _gather_ramble_messages(db: Session, project_id: int, max_messages: int = 50) -> List[Dict[str, Any]]:
    """
    Gather recent conversation messages as the ramble input.
    
    This is the ONLY DB access Weaver does - reading its input.
    """
    if not _MEMORY_AVAILABLE or not memory_service:
        return []
    
    try:
        messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)
        messages_raw = list(reversed(messages_raw))  # Chronological order
        
        messages: List[Dict[str, Any]] = []
        for msg in messages_raw:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "") or ""
            
            if _is_control_message(role, content):
                continue
            
            messages.append({
                "role": role,
                "content": content,
            })
        
        return messages
    except Exception as e:
        logger.error("[WEAVER] Failed to gather messages: %s", e)
        return []

def _extract_vision_context(messages: List[Dict[str, Any]]) -> str:
    """
    Extract vision context from assistant messages for refactor tasks.
    
    v3.9.0: Returns a string describing UI elements that were identified
    from screenshot analysis. This context is passed to SpecGate.
    """
    from .weaver_stream import _is_vision_context
    vision_parts = []
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if _is_vision_context(content):
                # Extract relevant portions (first 1000 chars to avoid bloat)
                vision_parts.append(content[:1000])
    
    if vision_parts:
        return "\n\n".join(vision_parts)
    return ""

def _extract_meta_mode(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract meta/mode phrases from user messages (v3.5.0 - Bug 2 fix).
    
    Pipeline control language like "no code", "just talk about it" should NOT
    end up in the product spec. They are execution constraints.
    
    Returns:
        Tuple of (filtered_messages, extracted_modes)
    """
    filtered_messages = []
    extracted_modes = []
    
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        
        if role == "user" and content:
            original_content = content
            for pattern in META_MODE_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    matched_text = match.group(0)
                    if matched_text not in extracted_modes:
                        extracted_modes.append(matched_text)
                    # Remove the meta phrase from content
                    content = re.sub(pattern, "", content, flags=re.IGNORECASE)
            
            # Clean up any trailing punctuation artifacts
            content = re.sub(r'\s*[,.]\s*[,.]\s*', '. ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            content = re.sub(r'^[,.]\s*', '', content)
            content = re.sub(r'\s*[,.]$', '', content)
            
            if content != original_content:
                print(f"[WEAVER] Extracted meta-mode phrases: {extracted_modes}")
            
            msg = {**msg, "content": content}
        
        # Only keep non-empty messages
        if msg.get("content", "").strip():
            filtered_messages.append(msg)
    
    return filtered_messages, extracted_modes

DESIGN_PREF_WHITELIST_PATTERNS = [
    r"\bcolor\b", r"\bcolour\b", r"\bcolors\b", r"\bcolours\b",
    r"\bdark\s*mode\b", r"\blight\s*mode\b", r"\btheme\b", r"\bpalette\b",
    r"\bbrand\b",  # "brand colors"
    r"\blayout\b", r"\bsidebar\b", r"\btop\s*nav\b", r"\bcentered\b", r"\bgrid\b",
    r"\bstyle\b", r"\bminimal\b", r"\bmodern\b", r"\bplayful\b", r"\bclean\b",
    r"\bbig\s*buttons?\b", r"\bno\s*clutter\b", r"\bdead[\s-]*simple\b",
    r"\bui\s*(elements?|feel)\b", r"\bvisual\b", r"\baesthetic\b",
    r"\bsimple\b", r"\bfast\b", r"\bsleek\b", r"\belegant\b",
]

REFACTOR_ACTION_PATTERNS = [
    # "rename X to Y" / "rename all X to Y"
    r"\brename\b.{1,40}\bto\b",
    # "rebrand from X to Y" / "rebrand X as Y"
    r"\brebrand\b",
    # "refactor" + scope indicator (codebase, all files, across, everywhere)
    r"\brefactor\b.{0,30}\b(across|codebase|all\s+files|everywhere|project)\b",
    # "replace all X with Y" / "replace X with Y in all files"
    r"\breplace\s+all\b",
    r"\breplace\b.{1,40}\b(across|everywhere|all\s+files|codebase|all\s+occurrences)\b",
    # "change all X to Y" / "change X to Y across"
    r"\bchange\s+all\b.{1,40}\bto\b",
    r"\bchange\b.{1,40}\bto\b.{1,40}\b(across|everywhere|all\s+files|codebase)\b",
    # "search and replace" / "find and replace" + scope
    r"\b(search|find)\s+and\s+replace\b",
    # Explicit codebase-wide rename language
    r"\ball\s+occurrences\b.{0,30}\b(of|rename|replace|change)\b",
    r"\b(rename|replace|change)\b.{0,30}\ball\s+occurrences\b",
]

def _user_dismissed_questions(text: str) -> bool:
    """
    Detect if user explicitly dismissed or said questions aren't needed (v3.8.0).
    
    Patterns like:
    - "The reply to your questions are actually not needed"
    - "Questions aren't relevant"
    - "Don't need to ask those questions"
    
    Returns True if user dismissed questions.
    """
    text_lower = text.lower()
    
    for pattern in QUESTIONS_DISMISSED_PATTERNS:
        if re.search(pattern, text_lower):
            print(f"[WEAVER] v3.8 User dismissed questions (pattern matched)")
            return True
    
    return False

TYPO_NORMALIZATIONS = [
    (r"\bdeck\s*top\b", "desktop"),
    (r"\bdecktop\b", "desktop"),
    (r"\bdekstop\b", "desktop"),
    (r"\bdestop\b", "desktop"),
    (r"\bdesctop\b", "desktop"),
    (r"\bdocumets\b", "documents"),
    (r"\bdocments\b", "documents"),
    (r"\bfloder\b", "folder"),
    (r"\bfodler\b", "folder"),
    (r"\bfild\b", "file"),
    (r"\bflie\b", "file"),
    (r"\bmesage\b", "message"),
    (r"\bmessge\b", "message"),
    (r"\bmesssage\b", "message"),
    (r"\banser\b", "answer"),
    (r"\banwser\b", "answer"),
    (r"\brepley\b", "reply"),
    (r"\brelpy\b", "reply"),
    (r"\bwirte\b", "write"),
    (r"\bwrtie\b", "write"),
]

def _is_micro_file_task(text: str) -> bool:
    """
    Detect simple file operations that need no questions (v3.6.0).
    v3.6.1: Context-aware detection - "create a file" is micro, "create an app" is not.
    v3.7.0: Refactor/rename operations are NEVER micro-tasks.
    v3.11.0: NON_MICRO indicators now checked EVEN when file indicators are present.
             Multi-component feature detection prevents substantial features from
             being classified as micro tasks.
    
    Logic:
    - If REFACTOR_INDICATOR present → NOT micro (codebase-wide operation)
    - If 3+ FEATURE_COMPONENT_INDICATORS present → NOT micro (substantial feature)
    - If NON_MICRO indicator present → NOT micro (even if file indicators match)
    - If BUILD_VERB + NON_MICRO → NOT micro (it's a build job)
    - If "create" + "file" → IS micro (simple file creation)
    - If any MICRO_FILE_INDICATOR present (without non-micro) → IS micro
    - Otherwise → NOT micro
    
    CRITICAL: Context matters!
    - "create a file" → micro task
    - "create an app" → NOT micro task
    - "build a game" → NOT micro task
    - "find file on my system" → micro task
    - "rename Orb to Astra" → NOT micro task (refactor!)
    - "add voice-to-text to the desktop app" → NOT micro task (feature!)
    - "add push-to-talk with audio capture and transcription" → NOT micro task (multi-component!)
    """
    text_lower = text.lower()
    
    # v3.7.0: FIRST check for refactor/rename indicators - these are NEVER micro
    for indicator in REFACTOR_INDICATORS:
        if indicator in text_lower:
            print(f"[WEAVER] v3.7 NOT micro-task (refactor indicator: '{indicator}')")
            return False
    
    # v3.11.0: Check for multi-component feature requests
    # If 3+ distinct feature components are mentioned, this is a substantial feature
    component_matches = [ind for ind in FEATURE_COMPONENT_INDICATORS if ind in text_lower]
    if len(component_matches) >= 3:
        print(f"[WEAVER] v3.11 NOT micro-task (multi-component feature: {component_matches[:5]}...)")
        return False
    
    # v3.11.0: Check NON_MICRO indicators EARLY - these override file indicators
    # "desktop app", "desktop application", "desktop feature" are NOT file tasks
    has_non_micro = any(ind in text_lower for ind in NON_MICRO_INDICATORS)
    if has_non_micro:
        # Find which non-micro indicator matched for logging
        matched_non_micro = [ind for ind in NON_MICRO_INDICATORS if ind in text_lower]
        print(f"[WEAVER] v3.11 NOT micro-task (non-micro indicators present: {matched_non_micro})")
        return False
    
    # v3.6.1: Check for explicit file creation context
    # "create a file", "create new file", "make a file" are MICRO tasks
    file_creation_patterns = [
        r"create\s+(?:a\s+)?(?:new\s+)?file",
        r"make\s+(?:a\s+)?(?:new\s+)?file",
        r"write\s+(?:a\s+)?(?:new\s+)?file",
        r"create\s+(?:a\s+)?(?:text|txt|reply|response)\s+file",
    ]
    for pattern in file_creation_patterns:
        if re.search(pattern, text_lower):
            print("[WEAVER] Classified as MICRO_FILE_TASK (file creation pattern)")
            return True
    
    # Check for file operation indicators
    has_file_indicator = any(ind in text_lower for ind in MICRO_FILE_INDICATORS)
    
    # v3.6.1: Check for BUILD VERB (even without non-micro, build verbs suggest non-micro)
    has_build_verb = any(v in text_lower for v in BUILD_VERBS)
    if has_build_verb:
        print(f"[WEAVER] NOT micro-task (build verb present without file context)")
        return False
    
    # If file indicators present and no non-micro/build overrides, it's a micro task
    if has_file_indicator:
        print("[WEAVER] Classified as MICRO_FILE_TASK")
        return True
    
    return False

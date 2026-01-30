# FILE: app/pot_spec/grounded/tech_stack_detection.py
"""
Tech Stack Detection and Anchoring (v1.11)

Detects tech stack choices from conversation messages and Weaver output.
Populates implementation_stack field for stack anchoring in Critique v1.4.

Version Notes:
-------------
v1.11 (2026-01): Initial implementation - upstream capture
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# STACK DETECTION PATTERNS
# =============================================================================

# Maps keywords to canonical (language, framework) pairs
STACK_DETECTION_PATTERNS = {
    # Python ecosystem
    "python": ("Python", None),
    "pygame": ("Python", "Pygame"),
    "tkinter": ("Python", "Tkinter"),
    "pyqt": ("Python", "PyQt"),
    "flask": ("Python", "Flask"),
    "fastapi": ("Python", "FastAPI"),
    "django": ("Python", "Django"),
    "kivy": ("Python", "Kivy"),
    "pyside": ("Python", "PySide"),
    
    # JavaScript/TypeScript ecosystem
    "javascript": ("JavaScript", None),
    "typescript": ("TypeScript", None),
    "react": ("TypeScript", "React"),
    "electron": ("TypeScript", "Electron"),
    "node.js": ("JavaScript", "Node.js"),
    "nodejs": ("JavaScript", "Node.js"),
    "next.js": ("TypeScript", "Next.js"),
    "nextjs": ("TypeScript", "Next.js"),
    "vue": ("TypeScript", "Vue"),
    "angular": ("TypeScript", "Angular"),
    
    # Game engines
    "unity": ("C#", "Unity"),
    "godot": ("GDScript", "Godot"),
    "unreal": ("C++", "Unreal"),
    
    # Other stacks
    "rust": ("Rust", None),
    "go": ("Go", None),
    "golang": ("Go", None),
    "c++": ("C++", None),
    "cpp": ("C++", None),
    "c#": ("C#", None),
    "csharp": ("C#", None),
    "java": ("Java", None),
    "kotlin": ("Kotlin", None),
    "swift": ("Swift", None),
}

# Keywords that indicate user is CHOOSING a stack
STACK_CHOICE_INDICATORS = [
    "use ", "using ", "let's use ", "lets use ", "i want ", "i'd like ",
    "prefer ", "go with ", "choose ", "pick ", "selected ", "decided on ",
    "will use ", "gonna use ", "going to use ", "want to use ",
    "build with ", "make with ", "create with ", "develop with ",
    "in python", "in javascript", "in typescript", "in rust",
    "with python", "with pygame", "with react", "with electron",
]

# Keywords that indicate assistant proposed and user confirmed
CONFIRMATION_PATTERNS = [
    r"\b(yes|yeah|yep|sure|ok|okay|sounds good|that works|perfect|great)\b",
    r"\b(go ahead|do it|proceed|confirmed|correct|exactly)\b",
    r"\b(that's right|that is right|that's correct|that is correct)\b",
    r"\b(also|and|additionally|plus|what about|how about)\b",
]


# =============================================================================
# DETECTION FUNCTION
# =============================================================================

def detect_implementation_stack(
    messages: List[Dict[str, Any]],
    weaver_text: str,
    intent: Dict[str, Any],
    implementation_stack_class: Optional[type] = None,
) -> Optional[Any]:
    """
    v1.11: Detect tech stack from conversation messages and Weaver output.
    
    Detection Priority:
    1. Explicit user statement: "use Python", "I want Pygame", "let's use React"
    2. Assistant proposal + user confirmation (sets stack_locked=True)
    3. Strong implication from goal/requirements text
    4. Weaver captured stack hints
    
    Args:
        messages: Conversation messages (list of {role, content})
        weaver_text: Full Weaver job description text
        intent: Parsed intent dict
        implementation_stack_class: The ImplementationStack class to use (optional)
    
    Returns:
        ImplementationStack if detected, None otherwise
    """
    if implementation_stack_class is None:
        logger.warning("[tech_stack_detection] v1.11 ImplementationStack class not provided")
        return None
    
    detected_language = None
    detected_framework = None
    detected_runtime = None
    stack_locked = False
    source = None
    notes = None
    
    all_text = f"{weaver_text or ''} {intent.get('goal', '')} {intent.get('raw_text', '')}"
    all_text_lower = all_text.lower()
    
    # =========================================================================
    # Pass 1: Check for explicit user stack choice in messages
    # =========================================================================
    
    user_messages = [m.get("content", "") for m in (messages or []) if m.get("role") == "user"]
    assistant_messages = [m.get("content", "") for m in (messages or []) if m.get("role") == "assistant"]
    
    for user_msg in user_messages:
        if not user_msg:
            continue
        user_msg_lower = user_msg.lower()
        
        has_choice_indicator = any(ind in user_msg_lower for ind in STACK_CHOICE_INDICATORS)
        
        for keyword, (lang, framework) in STACK_DETECTION_PATTERNS.items():
            if keyword in user_msg_lower:
                if has_choice_indicator:
                    detected_language = lang
                    if framework:
                        detected_framework = framework
                    stack_locked = True
                    source = "user_explicit_choice"
                    notes = f"User explicitly chose: '{keyword}' in message"
                    logger.info(
                        "[tech_stack_detection] v1.11 EXPLICIT user stack choice: %s+%s (locked=True)",
                        detected_language, detected_framework
                    )
                    break
                elif not detected_language:
                    detected_language = lang
                    if framework:
                        detected_framework = framework
                    source = "user_message"
                    notes = f"Stack mentioned in user message: '{keyword}'"
        
        if stack_locked:
            break
    
    # =========================================================================
    # Pass 2: Check for assistant proposal + user confirmation
    # =========================================================================
    
    if not stack_locked and len(assistant_messages) > 0 and len(user_messages) > 1:
        for i, assistant_msg in enumerate(assistant_messages):
            if not assistant_msg:
                continue
            assistant_msg_lower = assistant_msg.lower()
            
            proposed_lang = None
            proposed_framework = None
            
            proposal_indicators = [
                "i suggest", "i recommend", "i'd recommend", "i would suggest",
                "we could use", "we can use", "let's use", "how about",
                "i'll use", "i will use", "using", "built with",
            ]
            
            has_proposal = any(ind in assistant_msg_lower for ind in proposal_indicators)
            
            if has_proposal:
                for keyword, (lang, framework) in STACK_DETECTION_PATTERNS.items():
                    if keyword in assistant_msg_lower:
                        proposed_lang = lang
                        if framework:
                            proposed_framework = framework
                        break
            
            if proposed_lang:
                if i < len(user_messages) - 1:
                    next_user_msg = user_messages[i + 1] if i + 1 < len(user_messages) else ""
                    next_user_lower = (next_user_msg or "").lower()
                    
                    is_confirmation = any(
                        re.search(pattern, next_user_lower)
                        for pattern in CONFIRMATION_PATTERNS
                    )
                    
                    continues_with_details = len(next_user_lower) > 10 and not any(
                        neg in next_user_lower for neg in ["no", "don't", "not", "instead", "rather"]
                    )
                    
                    if is_confirmation or continues_with_details:
                        detected_language = proposed_lang
                        detected_framework = proposed_framework
                        stack_locked = True
                        source = "assistant_proposal_confirmed"
                        notes = f"Assistant proposed {proposed_lang}+{proposed_framework or 'N/A'}, user confirmed"
                        logger.info(
                            "[tech_stack_detection] v1.11 Assistant proposal CONFIRMED: %s+%s (locked=True)",
                            detected_language, detected_framework
                        )
                        break
    
    # =========================================================================
    # Pass 3: Check Weaver text for stack hints (not locked)
    # =========================================================================
    
    if not detected_language:
        for keyword, (lang, framework) in STACK_DETECTION_PATTERNS.items():
            if keyword in all_text_lower:
                detected_language = lang
                if framework:
                    detected_framework = framework
                source = "weaver_text"
                notes = f"Stack detected in Weaver/intent: '{keyword}'"
                logger.info(
                    "[tech_stack_detection] v1.11 Stack detected from Weaver text: %s+%s (locked=%s)",
                    detected_language, detected_framework, stack_locked
                )
                break
    
    # =========================================================================
    # Pass 4: Build and return ImplementationStack if detected
    # =========================================================================
    
    if detected_language:
        try:
            impl_stack = implementation_stack_class(
                language=detected_language,
                framework=detected_framework,
                runtime=detected_runtime,
                stack_locked=stack_locked,
                source=source,
                notes=notes,
            )
            
            logger.info(
                "[tech_stack_detection] v1.11 Detected implementation_stack: %s+%s (locked=%s, source=%s)",
                detected_language, detected_framework or "N/A", stack_locked, source
            )
            
            return impl_stack
        except Exception as e:
            logger.warning("[tech_stack_detection] v1.11 Failed to create ImplementationStack: %s", e)
            return None
    
    logger.info("[tech_stack_detection] v1.11 No implementation_stack detected")
    return None

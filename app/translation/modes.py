# FILE: app/translation/modes.py
"""
Mode classification for ASTRA Translation Layer.
Every message must first be classified into Chat, Command-Capable, or Feedback.

v1.6 (2026-01): Added READ file patterns to IMPLICIT_COMMAND_PATTERNS
  - Routes "what's written in <path>" to COMMAND_CAPABLE mode
  - Routes "read file <path>" to COMMAND_CAPABLE mode
  - Routes "show contents of <path>" to COMMAND_CAPABLE mode
  - Routes "view/display/cat <path>" to COMMAND_CAPABLE mode
  - Without these, READ queries were falling through to CHAT mode
"""
from __future__ import annotations
import re
from typing import Tuple, Optional
from .schemas import TranslationMode


# =============================================================================
# WAKE PHRASE PATTERNS
# =============================================================================

# Command mode wake phrases
COMMAND_WAKE_PHRASES = [
    "Astra, command:",
    "astra, command:",
    "ASTRA, command:",
    "Astra command:",
    "astra command:",
]

# Feedback mode wake phrases  
FEEDBACK_WAKE_PHRASES = [
    "Astra, feedback:",
    "astra, feedback:",
    "ASTRA, feedback:",
    "Astra feedback:",
    "astra feedback:",
]

# Compiled patterns for efficiency
COMMAND_WAKE_PATTERN = re.compile(
    r"^[Aa][Ss][Tt][Rr][Aa],?\s*command:\s*",
    re.IGNORECASE
)

FEEDBACK_WAKE_PATTERN = re.compile(
    r"^[Aa][Ss][Tt][Rr][Aa],?\s*feedback:\s*",
    re.IGNORECASE
)


# =============================================================================
# IMPLICIT COMMAND PATTERNS (no wake phrase needed)
# =============================================================================

# These patterns trigger COMMAND mode without needing "Astra, command:"
# NOTE: Patterns here just enable COMMAND mode - actual intent resolution
#       happens in tier0_rules.py (tier0_classify function)
IMPLICIT_COMMAND_PATTERNS = [
    # Architecture commands (exact match, case matters)
    r"^CREATE ARCHITECTURE MAP$",  # ALL CAPS
    r"^[Cc]reate [Aa]rchitecture [Mm]ap$",
    r"^[Uu]pdate [Aa]rchitecture$",
    r"^[Ss]can [Ss]andbox$",
    r"^SCAN SANDBOX(?: STRUCTURE)?$",
    
    # RAG/Codebase search (v1.3)
    r"^[Ss]earch\s+(?:the\s+)?codebase:\s*.+",
    r"^[Aa]sk\s+about\s+(?:the\s+)?codebase:\s*.+",
    r"^[Cc]odebase\s+(?:search|query):\s*.+",
    r"^[Ii]n\s+(?:the|this)\s+codebase,?\s+.+",
    r"^[Ii]ndex\s+(?:the\s+)?(?:architecture|codebase|RAG)$",
    r"^[Rr]un\s+RAG\s+index$",
    
    # =========================================================================
    # Architecture questions - route to RAG (v1.5)
    # These patterns enable COMMAND mode so tier0_rules can match RAG intent
    # High-precision: anchored on "where/show/find" + codebase terms
    # =========================================================================
    
    # --- "Where is X" patterns (broad - catches most arch questions) ---
    # Match: "Where is the main chat streaming entrypoint and what calls it?"
    # Match: "Where is routing decided between local tools vs LLM providers?"
    r"^[Ww]here\s+is\s+(?:the\s+)?(?:main\s+)?(?:\w+\s+){0,6}(?:entrypoint|entry\s*point|router|stream|handler|function|class|module|file|config|constant|routing|implementation|trigger|pipeline|gate)[s]?",
    
    # --- "Show me where X" patterns ---
    # Match: "Show me where Spec Gate is implemented and how it's triggered."
    r"^[Ss]how\s+(?:me\s+)?where\s+.+(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed)",
    
    # --- "Find where X" patterns ---
    # Match: "Find where ARCHMAP_TIMEOUT_SECONDS is loaded/used."
    r"^[Ff]ind\s+(?:where|the\s+file\s+where)\s+.+(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed|routes)",
    
    # --- "Find the file where X" patterns ---
    # Match: "Find the file where stream_router routes intents to handlers."
    r"^[Ff]ind\s+(?:the\s+)?(?:file|module|class|function)\s+(?:where|that)\s+.+",
    
    # --- "Find/List/Show call sites/callers of X" patterns ---
    # Match: "Find call sites of streamChat( and list the calling files"
    # Match: "List callers of handleRouting"
    r"^(?:[Ff]ind|[Ll]ist|[Ss]how)\s+(?:the\s+)?(?:call\s*sites?|callers?)\s+(?:of|for)\s+.+",
    
    # --- "Who calls X" patterns ---
    # Match: "Who calls streamChat?"
    r"^[Ww]ho\s+calls\s+.+",
    
    # --- Codebase-specific questions with known terms ---
    # Match questions mentioning specific ASTRA components
    r"^(?:[Ww]here|[Hh]ow|[Ww]hat)\s+.+(?:[Ss]pec\s*[Gg]ate|[Oo]verwatcher|[Ww]eaver|[Cc]ritical\s*[Pp]ipeline|[Ss]tream\s*[Rr]outer|[Tt]ranslation\s*[Ll]ayer|[Rr][Aa][Gg]|[Ee]mbedding)\s*.+[?.!]?$",
    
    # --- Original specific patterns (kept for exact matches) ---
    r"^[Ww]hat\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?[?.!]?$",
    r"^[Ww]here\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?[?.!]?$",
    r"^[Ww]hat\s+(?:are|is)\s+(?:the\s+)?(?:potential\s+)?bottlenecks?[?.!]?$",
    r"^[Ww]here\s+(?:are|is)\s+(?:the\s+)?bottlenecks?[?.!]?$",
    r"^[Ww]hat\s+(?:function|class|method|module|file)s?\s+(?:handle|process|manage)s?\s+.+[?.!]?$",
    r"^[Ww]hich\s+(?:function|class|method|module|file)s?\s+(?:are\s+)?responsible\s+for\s+.+[?.!]?$",
    r"^[Ww]here\s+is\s+.+\s+(?:implemented|defined|located)[?.!]?$",
    r"^[Ww]here\s+should\s+(?:a\s+)?(?:new\s+)?(?:feature|functionality|code|module|component)\s+.+\s+(?:live|go)[?.!]?$",
    r"^[Ww]here\s+should\s+[Ii]\s+(?:put|add|place|implement)\s+.+[?.!]?$",
    r"^[Hh]ow\s+does\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:work|function|flow)[?.!]?$",
    r"^[Ww]hat\s+is\s+the\s+(?:purpose|role|responsibility)\s+of\s+.+[?.!]?$",
    r"^[Ww]hat\s+does\s+(?:the\s+)?(?:file|module|class|function)\s+.+\s+do[?.!]?$",
    r"^[Ss]how\s+(?:me\s+)?(?:the\s+)?(?:modules?|components?|services?|handlers?|routers?)[?.!]?$",
    r"^[Ll]ist\s+(?:all\s+)?(?:the\s+)?(?:modules?|components?|services?)[?.!]?$",
    r"^[Bb]ottlenecks?[?.!]?$",  # Single word
    
    # Spec Gate flow
    r"^[Hh]ow does that look all together\??$",
    r"^[Ss]end (?:that |this |it )?to [Ss]pec ?[Gg]ate$",
    r"^[Rr]un (?:the )?[Cc]ritical [Pp]ipeline$",
    r"^[Rr]un [Oo]verwatcher$",
    
    # Embedding commands (v1.3)
    r"^[Ee]mbedding[s]?\s+status$",
    r"^[Cc]heck\s+embedding[s]?$",
    r"^[Gg]enerate\s+embedding[s]?$",
    r"^[Rr]un\s+embedding[s]?$",
    r"^[Ss]tart\s+embedding[s]?$",
    
    # =========================================================================
    # Filesystem queries (v1.6) - route to local FILESYSTEM_QUERY handler
    # HIGH-PRECISION: Must include Windows drive path (C:\ or D:\) to avoid
    # false positives on generic chat like "find files with invoices"
    # Optional prefix: "After scan sandbox, " supported
    # =========================================================================
    
    # List/Show patterns with Windows path
    # Match: "List everything on C:\Users\dizzi\Desktop"
    # Match: "After scan sandbox, list everything on C:\Users\dizzi\Desktop"
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Ll]ist\s+(?:everything|all|contents?|files?(?:\s+and\s+folders?)?|folders?(?:\s+and\s+files?)?|top[- ]?level\s+folders?)\s+(?:on|in|at|under|inside)\s+[A-Za-z]:[/\\]",
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Ss]how\s+(?:me\s+)?(?:everything|all|contents?|files?|folders?)\s+(?:in|on|at|under|inside)\s+[A-Za-z]:[/\\]",
    
    # What's in patterns with Windows path
    # Match: "What's in C:\Users\dizzi\OneDrive"
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Ww]hat(?:'s|\s+is)\s+(?:in|on|at|inside)\s+[A-Za-z]:[/\\]",
    
    # Contents of patterns with Windows path
    # Match: "Contents of D:\Projects"
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Cc]ontents?\s+of\s+[A-Za-z]:[/\\]",
    
    # Find folder/file named X under/in path
    # Match: "Find folder named Jobs under C:\Users\dizzi"
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Ff]ind\s+(?:folder|file|directory)\s+(?:named?\s+)?[\w\s]+\s+(?:under|in|on|inside)\s+[A-Za-z]:[/\\]",
    
    # Find X under known folder (Desktop, OneDrive, etc.) - no explicit path needed
    # Match: "Find MBS Fitness under OneDrive"
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Ff]ind\s+[\w\s]+\s+(?:under|in|inside|on)\s+(?:my\s+)?(?:[Dd]esktop|[Oo]ne[Dd]rive|[Dd]ocuments|[Dd]ownloads)",
    
    # Generic find files with explicit Windows path
    # Match: "Find files with Amber in the name under C:\Users\dizzi"
    r"^(?:[Aa]fter\s+scan\s+sandbox,?\s*)?[Ff]ind\s+(?:all\s+)?files?\s+(?:with|containing|named)\s+.+\s+(?:under|in|on|inside)\s+[A-Za-z]:[/\\]",
    
    # =========================================================================
    # Filesystem READ queries (v1.6) - read file contents from DB
    # These enable COMMAND mode so tier0_rules can route to FILESYSTEM_QUERY
    # Patterns must include Windows drive path (C:\ or D:\)
    # =========================================================================
    
    # "what's written in <path>" / "whats written in <path>" / "what's inside <path>"
    # Match: "what's written in C:\Users\dizzi\file.txt"
    r"^[Ww]hat['']?s\s+(?:written|inside)\s+(?:in\s+)?[A-Za-z]:[/\\]",
    
    # "read [file] <path>" / "read the file <path>"
    # Match: "read file C:\Users\dizzi\file.txt"
    r"^[Rr]ead\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\]",
    
    # "show contents of <path>" (for reading file content, not listing)
    # Match: "show contents of C:\Users\dizzi\file.txt"
    r"^[Ss]how\s+(?:the\s+)?contents?\s+of\s+[A-Za-z]:[/\\]",
    
    # "view/display/cat/output/print <path>" (Unix-style commands)
    # Match: "cat C:\Users\dizzi\file.txt", "view C:\file.txt"
    r"^(?:[Vv]iew|[Dd]isplay|[Cc]at|[Oo]utput|[Pp]rint)\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\]",
    
    # "open <path>" with file extension
    # Match: "open C:\Users\dizzi\file.txt"
    r"^[Oo]pen\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\].+\.\w+",
    
    # "what does <path> say/contain"
    # Match: "what does C:\file.txt contain"
    r"^[Ww]hat\s+does\s+[A-Za-z]:[/\\].+\s+(?:say|contain)",
]

# Compile patterns for efficiency
_IMPLICIT_COMMAND_COMPILED = [re.compile(p) for p in IMPLICIT_COMMAND_PATTERNS]


def _is_implicit_command(text: str) -> bool:
    """Check if text matches an implicit command pattern."""
    for pattern in _IMPLICIT_COMMAND_COMPILED:
        if pattern.match(text.strip()):
            return True
    return False


# =============================================================================
# MODE CLASSIFICATION
# =============================================================================

def classify_mode(
    text: str,
    ui_command_context: bool = False
) -> Tuple[TranslationMode, Optional[str], str]:
    """
    Classify the mode of a message.
    
    Args:
        text: The user's message text
        ui_command_context: True if UI has placed user in command context
        
    Returns:
        Tuple of:
        - TranslationMode
        - Wake phrase detected (if any)
        - Remaining text after wake phrase
    """
    text = text.strip()
    
    # Check feedback mode first (takes priority)
    if _is_feedback_mode(text):
        wake_phrase = _extract_feedback_wake_phrase(text)
        remaining = _strip_wake_phrase(text, FEEDBACK_WAKE_PATTERN)
        return TranslationMode.FEEDBACK, wake_phrase, remaining
    
    # Check command mode via wake phrase
    if _is_command_wake_phrase(text):
        wake_phrase = _extract_command_wake_phrase(text)
        remaining = _strip_wake_phrase(text, COMMAND_WAKE_PATTERN)
        return TranslationMode.COMMAND_CAPABLE, wake_phrase, remaining
    
    # Check command mode via UI context
    if ui_command_context:
        return TranslationMode.COMMAND_CAPABLE, None, text
    
    # Check for implicit command patterns (no wake phrase needed)
    if _is_implicit_command(text):
        return TranslationMode.COMMAND_CAPABLE, None, text
    
    # Default to chat mode
    return TranslationMode.CHAT, None, text


def _is_feedback_mode(text: str) -> bool:
    """Check if text starts with feedback wake phrase."""
    return bool(FEEDBACK_WAKE_PATTERN.match(text))


def _is_command_wake_phrase(text: str) -> bool:
    """Check if text starts with command wake phrase."""
    return bool(COMMAND_WAKE_PATTERN.match(text))


def _extract_feedback_wake_phrase(text: str) -> Optional[str]:
    """Extract the feedback wake phrase from text."""
    match = FEEDBACK_WAKE_PATTERN.match(text)
    if match:
        return match.group(0).strip()
    return None


def _extract_command_wake_phrase(text: str) -> Optional[str]:
    """Extract the command wake phrase from text."""
    match = COMMAND_WAKE_PATTERN.match(text)
    if match:
        return match.group(0).strip()
    return None


def _strip_wake_phrase(text: str, pattern: re.Pattern) -> str:
    """Remove wake phrase from text, returning remaining content."""
    return pattern.sub("", text).strip()


# =============================================================================
# UI CONTEXT DETECTION
# =============================================================================

class UIContext:
    """
    Represents the UI context that may place user in command mode.
    This would be passed from the frontend.
    """
    
    def __init__(
        self,
        in_job_config: bool = False,
        in_sandbox_control: bool = False,
        in_pipeline_control: bool = False,
        active_job_id: Optional[str] = None,
        active_sandbox_id: Optional[str] = None,
    ):
        self.in_job_config = in_job_config
        self.in_sandbox_control = in_sandbox_control
        self.in_pipeline_control = in_pipeline_control
        self.active_job_id = active_job_id
        self.active_sandbox_id = active_sandbox_id
    
    @property
    def is_command_context(self) -> bool:
        """True if any UI context makes this command-capable."""
        return any([
            self.in_job_config,
            self.in_sandbox_control,
            self.in_pipeline_control,
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for context gate."""
        return {
            "job_id": self.active_job_id,
            "sandbox_id": self.active_sandbox_id,
        }


def classify_mode_with_ui(
    text: str,
    ui_context: Optional[UIContext] = None
) -> Tuple[TranslationMode, Optional[str], str]:
    """
    Classify mode considering UI context.
    
    Args:
        text: User message
        ui_context: Current UI context (if any)
        
    Returns:
        Same as classify_mode()
    """
    ui_command_context = ui_context.is_command_context if ui_context else False
    return classify_mode(text, ui_command_context=ui_command_context)

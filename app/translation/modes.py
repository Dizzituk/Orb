# FILE: app/translation/modes.py
"""
Mode classification for ASTRA Translation Layer.
Every message must first be classified into Chat, Command-Capable, or Feedback.
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
    
    # Architecture questions - route to RAG (v1.4)
    # These patterns enable COMMAND mode so tier0_rules can match RAG intent
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

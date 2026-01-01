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

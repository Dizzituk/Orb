# FILE: app/llm/fallbacks.py
"""
Fallback Handler for Orb Routing Pipeline.

Version: 1.0.0 - Critical Pipeline Spec Implementation

Implements Spec §11 (Fallback Behaviour):
- Structured reactions to different failure types
- Fallback chains per model role
- Graceful degradation with user notification

FALLBACK SCENARIOS (Spec §11):

1. Video Transcription Fails:
   - Re-route as if no video, keep text/code summaries
   - Log "VIDEO_PREPROCESSING_FAILED"
   - Do NOT abort task

2. Vision/OCR Fails:
   - Continue with available modalities
   - Include note: "Some images could not be processed"
   - Log "IMAGE_PREPROCESSING_FAILED"

3. Primary Model Unavailable:
   - Fallback chain: Sonnet → Opus → GPT
   - Log fallback event with original + fallback model
   - Continue with fallback model

4. Overwatcher Unavailable (Critical Pipeline):
   - Proceed with task
   - Mark output as "unchecked_by_overwatcher"
   - Log "OVERWATCHER_UNAVAILABLE"

FALLBACK CHAINS:
- Code tasks: Sonnet → Opus → GPT-4.1
- Vision tasks: Gemini 2.5 Pro → Gemini 2.0 Flash → GPT-4.1-Vision
- Video tasks: Gemini 3 Pro → Gemini 2.5 Pro
- Critical: Opus → Sonnet → GPT-4.1
- Text tasks: GPT → Sonnet

Usage:
    from app.llm.fallbacks import (
        FallbackHandler,
        get_fallback_chain,
        handle_preprocessing_failure,
    )
    
    handler = FallbackHandler()
    result = await handler.execute_with_fallback(
        primary_call=lambda: call_model(sonnet),
        fallback_chain=["opus", "gpt"],
        task_id="TASK_1",
    )
"""

import os
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Enable fallback behavior
FALLBACK_ENABLED = os.getenv("ORB_FALLBACK_ENABLED", "1") == "1"

# Maximum fallback attempts
MAX_FALLBACK_ATTEMPTS = int(os.getenv("ORB_MAX_FALLBACK_ATTEMPTS", "3"))

# Router debug mode
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


# =============================================================================
# FAILURE TYPES
# =============================================================================

class FailureType(str, Enum):
    """Types of failures that can trigger fallback."""
    # Preprocessing failures
    VIDEO_TRANSCRIPTION_FAILED = "VIDEO_TRANSCRIPTION_FAILED"
    IMAGE_PROCESSING_FAILED = "IMAGE_PROCESSING_FAILED"
    CODE_EXTRACTION_FAILED = "CODE_EXTRACTION_FAILED"
    TEXT_EXTRACTION_FAILED = "TEXT_EXTRACTION_FAILED"
    
    # Model failures
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_RATE_LIMITED = "MODEL_RATE_LIMITED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_ERROR = "MODEL_ERROR"
    
    # Pipeline failures
    ROUTING_ERROR = "ROUTING_ERROR"
    CONTEXT_TOO_LARGE = "CONTEXT_TOO_LARGE"
    
    # Critical pipeline specific
    OVERWATCHER_UNAVAILABLE = "OVERWATCHER_UNAVAILABLE"
    CRITIQUE_FAILED = "CRITIQUE_FAILED"
    REVISION_FAILED = "REVISION_FAILED"


class FallbackAction(str, Enum):
    """Actions to take on failure."""
    RETRY_SAME = "retry_same"           # Retry with same model
    USE_FALLBACK = "use_fallback"       # Use fallback chain
    SKIP_STEP = "skip_step"             # Skip this step, continue
    ABORT_TASK = "abort_task"           # Abort the task
    DEGRADE_GRACEFULLY = "degrade"      # Continue with reduced capability


# =============================================================================
# FALLBACK CHAINS
# =============================================================================

# Model fallback chains by role
FALLBACK_CHAINS = {
    # Code tasks: Claude Sonnet → Opus → GPT
    "code": [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("anthropic", "claude-opus-4-20250514"),
        ("openai", "gpt-4.1"),
    ],
    
    # Vision tasks: Gemini 2.5 Pro → Gemini 2.0 Flash → GPT-4.1-Vision
    "vision": [
        ("google", "gemini-2.5-pro-preview-05-06"),
        ("google", "gemini-2.0-flash"),
        ("openai", "gpt-4.1"),
    ],
    
    # Video tasks: Gemini 3 Pro → Gemini 2.5 Pro
    "video": [
        ("google", "gemini-3.0-pro-preview"),
        ("google", "gemini-2.5-pro-preview-05-06"),
    ],
    
    # Critical tasks: Opus → Sonnet → GPT
    "critical": [
        ("anthropic", "claude-opus-4-20250514"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4.1"),
    ],
    
    # Text/research tasks: GPT → Sonnet
    "text": [
        ("openai", "gpt-4.1"),
        ("anthropic", "claude-sonnet-4-20250514"),
    ],
    
    # Critique: Gemini 3 → Sonnet
    "critique": [
        ("google", "gemini-3.0-pro-preview"),
        ("anthropic", "claude-sonnet-4-20250514"),
    ],
}


def get_fallback_chain(role: str) -> List[Tuple[str, str]]:
    """
    Get fallback chain for a role.
    
    Args:
        role: Task role (code, vision, video, critical, text, critique)
    
    Returns:
        List of (provider, model) tuples in priority order
    """
    return FALLBACK_CHAINS.get(role, FALLBACK_CHAINS["text"])


# =============================================================================
# FAILURE HANDLING RULES
# =============================================================================

@dataclass
class FailureRule:
    """Rule for handling a specific failure type."""
    failure_type: FailureType
    action: FallbackAction
    user_message: Optional[str] = None
    log_level: str = "warning"
    retry_count: int = 0
    skip_if_other_modalities: bool = False


# Default rules for each failure type
FAILURE_RULES: Dict[FailureType, FailureRule] = {
    # Preprocessing failures - degrade gracefully
    FailureType.VIDEO_TRANSCRIPTION_FAILED: FailureRule(
        failure_type=FailureType.VIDEO_TRANSCRIPTION_FAILED,
        action=FallbackAction.DEGRADE_GRACEFULLY,
        user_message="Video could not be transcribed. Processing with available content.",
        log_level="warning",
        skip_if_other_modalities=True,
    ),
    
    FailureType.IMAGE_PROCESSING_FAILED: FailureRule(
        failure_type=FailureType.IMAGE_PROCESSING_FAILED,
        action=FallbackAction.DEGRADE_GRACEFULLY,
        user_message="Some images could not be processed.",
        log_level="warning",
        skip_if_other_modalities=True,
    ),
    
    FailureType.CODE_EXTRACTION_FAILED: FailureRule(
        failure_type=FailureType.CODE_EXTRACTION_FAILED,
        action=FallbackAction.SKIP_STEP,
        user_message=None,
        log_level="warning",
    ),
    
    FailureType.TEXT_EXTRACTION_FAILED: FailureRule(
        failure_type=FailureType.TEXT_EXTRACTION_FAILED,
        action=FallbackAction.SKIP_STEP,
        user_message=None,
        log_level="warning",
    ),
    
    # Model failures - use fallback chain
    FailureType.MODEL_UNAVAILABLE: FailureRule(
        failure_type=FailureType.MODEL_UNAVAILABLE,
        action=FallbackAction.USE_FALLBACK,
        user_message=None,
        log_level="warning",
    ),
    
    FailureType.MODEL_RATE_LIMITED: FailureRule(
        failure_type=FailureType.MODEL_RATE_LIMITED,
        action=FallbackAction.RETRY_SAME,
        retry_count=2,
        user_message=None,
        log_level="info",
    ),
    
    FailureType.MODEL_TIMEOUT: FailureRule(
        failure_type=FailureType.MODEL_TIMEOUT,
        action=FallbackAction.USE_FALLBACK,
        user_message="Request timed out. Trying alternative model.",
        log_level="warning",
    ),
    
    FailureType.MODEL_ERROR: FailureRule(
        failure_type=FailureType.MODEL_ERROR,
        action=FallbackAction.USE_FALLBACK,
        user_message=None,
        log_level="error",
    ),
    
    # Pipeline failures
    FailureType.ROUTING_ERROR: FailureRule(
        failure_type=FailureType.ROUTING_ERROR,
        action=FallbackAction.ABORT_TASK,
        user_message="Unable to process request. Please try again.",
        log_level="error",
    ),
    
    FailureType.CONTEXT_TOO_LARGE: FailureRule(
        failure_type=FailureType.CONTEXT_TOO_LARGE,
        action=FallbackAction.DEGRADE_GRACEFULLY,
        user_message="Content was truncated to fit context limits.",
        log_level="warning",
    ),
    
    # Critical pipeline failures
    FailureType.OVERWATCHER_UNAVAILABLE: FailureRule(
        failure_type=FailureType.OVERWATCHER_UNAVAILABLE,
        action=FallbackAction.SKIP_STEP,
        user_message=None,
        log_level="warning",
    ),
    
    FailureType.CRITIQUE_FAILED: FailureRule(
        failure_type=FailureType.CRITIQUE_FAILED,
        action=FallbackAction.USE_FALLBACK,
        user_message=None,
        log_level="warning",
    ),
    
    FailureType.REVISION_FAILED: FailureRule(
        failure_type=FailureType.REVISION_FAILED,
        action=FallbackAction.SKIP_STEP,
        user_message="Review step was skipped.",
        log_level="warning",
    ),
}


def get_failure_rule(failure_type: FailureType) -> FailureRule:
    """Get the handling rule for a failure type."""
    return FAILURE_RULES.get(failure_type, FailureRule(
        failure_type=failure_type,
        action=FallbackAction.ABORT_TASK,
        log_level="error",
    ))


# =============================================================================
# FALLBACK RESULT
# =============================================================================

@dataclass
class FallbackEvent:
    """Record of a fallback event."""
    timestamp: datetime
    failure_type: FailureType
    action_taken: FallbackAction
    
    # Original state
    original_provider: Optional[str] = None
    original_model: Optional[str] = None
    
    # Fallback state
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    
    # Details
    error_message: str = ""
    user_message: Optional[str] = None
    task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "failure_type": self.failure_type.value,
            "action_taken": self.action_taken.value,
            "original_provider": self.original_provider,
            "original_model": self.original_model,
            "fallback_provider": self.fallback_provider,
            "fallback_model": self.fallback_model,
            "error_message": self.error_message,
            "user_message": self.user_message,
            "task_id": self.task_id,
        }


@dataclass
class FallbackResult:
    """Result of fallback handling."""
    success: bool
    content: Any = None
    
    # What happened
    fallback_used: bool = False
    fallback_events: List[FallbackEvent] = field(default_factory=list)
    
    # Final state
    final_provider: Optional[str] = None
    final_model: Optional[str] = None
    
    # User messages to include
    user_messages: List[str] = field(default_factory=list)
    
    # Flags
    was_degraded: bool = False
    was_skipped: bool = False
    unchecked_by_overwatcher: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "fallback_used": self.fallback_used,
            "fallback_count": len(self.fallback_events),
            "final_provider": self.final_provider,
            "final_model": self.final_model,
            "was_degraded": self.was_degraded,
            "was_skipped": self.was_skipped,
            "unchecked_by_overwatcher": self.unchecked_by_overwatcher,
            "user_messages": self.user_messages,
            "events": [e.to_dict() for e in self.fallback_events],
        }


# =============================================================================
# PREPROCESSING FAILURE HANDLERS
# =============================================================================

def handle_preprocessing_failure(
    failure_type: FailureType,
    error_message: str,
    has_other_modalities: bool,
    task_id: Optional[str] = None,
) -> Tuple[FallbackAction, FallbackEvent]:
    """
    Handle a preprocessing failure.
    
    Args:
        failure_type: Type of preprocessing failure
        error_message: Error details
        has_other_modalities: Are there other modalities to fall back on?
        task_id: Task identifier
    
    Returns:
        (action, event) tuple
    """
    rule = get_failure_rule(failure_type)
    
    # If we have other modalities and rule allows skipping
    if has_other_modalities and rule.skip_if_other_modalities:
        action = FallbackAction.SKIP_STEP
    else:
        action = rule.action
    
    event = FallbackEvent(
        timestamp=datetime.utcnow(),
        failure_type=failure_type,
        action_taken=action,
        error_message=error_message,
        user_message=rule.user_message,
        task_id=task_id,
    )
    
    # Log appropriately
    log_fn = getattr(logger, rule.log_level, logger.warning)
    log_fn(f"[fallback] {failure_type.value}: {error_message} → {action.value}")
    
    return action, event


def handle_video_failure(
    error_message: str,
    has_code: bool = False,
    has_text: bool = False,
    has_images: bool = False,
    task_id: Optional[str] = None,
) -> Tuple[FallbackAction, FallbackEvent]:
    """
    Handle video transcription failure (Spec §11.1).
    
    Re-route as if no video, keep text/code summaries.
    """
    has_other = has_code or has_text or has_images
    return handle_preprocessing_failure(
        FailureType.VIDEO_TRANSCRIPTION_FAILED,
        error_message,
        has_other,
        task_id,
    )


def handle_vision_failure(
    error_message: str,
    has_code: bool = False,
    has_text: bool = False,
    has_video: bool = False,
    task_id: Optional[str] = None,
) -> Tuple[FallbackAction, FallbackEvent]:
    """
    Handle vision/OCR failure (Spec §11.2).
    
    Continue with available modalities, note images couldn't be processed.
    """
    has_other = has_code or has_text or has_video
    return handle_preprocessing_failure(
        FailureType.IMAGE_PROCESSING_FAILED,
        error_message,
        has_other,
        task_id,
    )


# =============================================================================
# MODEL FAILURE HANDLERS
# =============================================================================

def get_next_fallback(
    current_provider: str,
    current_model: str,
    role: str,
) -> Optional[Tuple[str, str]]:
    """
    Get next model in fallback chain.
    
    Args:
        current_provider: Current provider
        current_model: Current model
        role: Task role for chain selection
    
    Returns:
        (provider, model) tuple or None if no more fallbacks
    """
    chain = get_fallback_chain(role)
    
    # Find current position in chain
    current_pos = -1
    for i, (provider, model) in enumerate(chain):
        if provider == current_provider and model == current_model:
            current_pos = i
            break
    
    # Get next in chain
    if current_pos >= 0 and current_pos < len(chain) - 1:
        return chain[current_pos + 1]
    elif current_pos == -1 and chain:
        # Not in chain, start from beginning
        return chain[0]
    
    return None


# =============================================================================
# FALLBACK HANDLER CLASS
# =============================================================================

class FallbackHandler:
    """
    Handler for executing operations with fallback support.
    
    Usage:
        handler = FallbackHandler()
        
        result = await handler.execute_with_fallback(
            operation=my_async_operation,
            role="code",
            initial_provider="anthropic",
            initial_model="claude-sonnet-4-20250514",
        )
    """
    
    def __init__(self, max_attempts: int = MAX_FALLBACK_ATTEMPTS):
        self.max_attempts = max_attempts
        self.events: List[FallbackEvent] = []
    
    async def execute_with_fallback(
        self,
        operation: Callable[[str, str], Awaitable[Any]],
        role: str,
        initial_provider: str,
        initial_model: str,
        task_id: Optional[str] = None,
        check_availability: Optional[Callable[[str, str], bool]] = None,
    ) -> FallbackResult:
        """
        Execute operation with automatic fallback.
        
        Args:
            operation: Async callable(provider, model) -> result
            role: Task role for fallback chain
            initial_provider: Starting provider
            initial_model: Starting model
            task_id: Task identifier for logging
            check_availability: Optional function to check model availability
        
        Returns:
            FallbackResult with operation result or failure info
        """
        result = FallbackResult()
        
        current_provider = initial_provider
        current_model = initial_model
        attempts = 0
        
        while attempts < self.max_attempts:
            attempts += 1
            
            # Check availability if function provided
            if check_availability and not check_availability(current_provider, current_model):
                event = FallbackEvent(
                    timestamp=datetime.utcnow(),
                    failure_type=FailureType.MODEL_UNAVAILABLE,
                    action_taken=FallbackAction.USE_FALLBACK,
                    original_provider=current_provider,
                    original_model=current_model,
                    error_message=f"Model {current_provider}/{current_model} not available",
                    task_id=task_id,
                )
                result.fallback_events.append(event)
                
                # Get next fallback
                next_model = get_next_fallback(current_provider, current_model, role)
                if next_model:
                    current_provider, current_model = next_model
                    event.fallback_provider = current_provider
                    event.fallback_model = current_model
                    result.fallback_used = True
                    continue
                else:
                    result.success = False
                    return result
            
            try:
                # Execute operation
                content = await operation(current_provider, current_model)
                
                result.success = True
                result.content = content
                result.final_provider = current_provider
                result.final_model = current_model
                
                logger.debug(f"[fallback] Success with {current_provider}/{current_model}")
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Determine failure type
                if "rate limit" in error_str or "429" in error_str:
                    failure_type = FailureType.MODEL_RATE_LIMITED
                elif "timeout" in error_str:
                    failure_type = FailureType.MODEL_TIMEOUT
                elif "unavailable" in error_str or "503" in error_str:
                    failure_type = FailureType.MODEL_UNAVAILABLE
                else:
                    failure_type = FailureType.MODEL_ERROR
                
                rule = get_failure_rule(failure_type)
                
                event = FallbackEvent(
                    timestamp=datetime.utcnow(),
                    failure_type=failure_type,
                    action_taken=rule.action,
                    original_provider=current_provider,
                    original_model=current_model,
                    error_message=str(e),
                    task_id=task_id,
                )
                result.fallback_events.append(event)
                
                logger.warning(
                    f"[fallback] {failure_type.value} with {current_provider}/{current_model}: {e}"
                )
                
                # Handle based on rule
                if rule.action == FallbackAction.RETRY_SAME and attempts <= rule.retry_count:
                    continue
                    
                elif rule.action == FallbackAction.USE_FALLBACK:
                    next_model = get_next_fallback(current_provider, current_model, role)
                    if next_model:
                        current_provider, current_model = next_model
                        event.fallback_provider = current_provider
                        event.fallback_model = current_model
                        result.fallback_used = True
                        continue
                    else:
                        result.success = False
                        return result
                        
                elif rule.action == FallbackAction.ABORT_TASK:
                    result.success = False
                    return result
                    
                else:
                    result.success = False
                    return result
        
        # Max attempts reached
        result.success = False
        logger.error(f"[fallback] Max attempts ({self.max_attempts}) reached for {task_id}")
        return result
    
    def record_event(self, event: FallbackEvent) -> None:
        """Record a fallback event."""
        self.events.append(event)
    
    def get_events(self) -> List[FallbackEvent]:
        """Get all recorded events."""
        return self.events
    
    def clear_events(self) -> None:
        """Clear recorded events."""
        self.events.clear()


# =============================================================================
# CRITICAL PIPELINE HELPERS
# =============================================================================

def handle_overwatcher_failure(
    error_message: str,
    task_id: Optional[str] = None,
) -> FallbackEvent:
    """
    Handle overwatcher unavailability (Spec §11.4).
    
    Proceed with task but mark as unchecked.
    """
    event = FallbackEvent(
        timestamp=datetime.utcnow(),
        failure_type=FailureType.OVERWATCHER_UNAVAILABLE,
        action_taken=FallbackAction.SKIP_STEP,
        error_message=error_message,
        task_id=task_id,
    )
    
    logger.warning(f"[fallback] Overwatcher unavailable: {error_message}")
    return event


def handle_critique_failure(
    error_message: str,
    original_provider: str,
    original_model: str,
    task_id: Optional[str] = None,
) -> Tuple[FallbackAction, FallbackEvent, Optional[Tuple[str, str]]]:
    """
    Handle critique step failure in critical pipeline.
    
    Returns:
        (action, event, optional_fallback_model)
    """
    event = FallbackEvent(
        timestamp=datetime.utcnow(),
        failure_type=FailureType.CRITIQUE_FAILED,
        action_taken=FallbackAction.USE_FALLBACK,
        original_provider=original_provider,
        original_model=original_model,
        error_message=error_message,
        task_id=task_id,
    )
    
    next_model = get_next_fallback(original_provider, original_model, "critique")
    if next_model:
        event.fallback_provider = next_model[0]
        event.fallback_model = next_model[1]
    
    logger.warning(f"[fallback] Critique failed: {error_message}")
    return FallbackAction.USE_FALLBACK, event, next_model


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "FailureType",
    "FallbackAction",
    
    # Data classes
    "FailureRule",
    "FallbackEvent",
    "FallbackResult",
    
    # Chains
    "FALLBACK_CHAINS",
    "get_fallback_chain",
    "get_next_fallback",
    
    # Rules
    "FAILURE_RULES",
    "get_failure_rule",
    
    # Handlers
    "FallbackHandler",
    "handle_preprocessing_failure",
    "handle_video_failure",
    "handle_vision_failure",
    "handle_overwatcher_failure",
    "handle_critique_failure",
    
    # Configuration
    "FALLBACK_ENABLED",
    "MAX_FALLBACK_ATTEMPTS",
]
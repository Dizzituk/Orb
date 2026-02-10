# FILE: app/translation/integration.py
"""
Integration helpers for the Translation Layer.
Shows how to integrate with existing stream_router.py

This file provides drop-in replacements for the existing trigger detection
functions that use the new Translation Layer.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple, Dict, Any

from .translator import translate_message_sync, get_translator
from .schemas import TranslationResult, CanonicalIntent, TranslationMode
from .modes import UIContext

logger = logging.getLogger(__name__)


# =============================================================================
# DROP-IN REPLACEMENTS FOR STREAM_ROUTER TRIGGERS
# =============================================================================
# These functions replace the existing _is_*_trigger() functions in stream_router.py

def check_message_intent(
    text: str,
    user_id: str = "default",
    ui_command_context: bool = False,
    conversation_id: Optional[str] = None,
) -> TranslationResult:
    """
    Primary entry point for the translation layer.
    Replaces multiple _is_*_trigger() calls with single translation.
    
    Args:
        text: User message
        user_id: User identifier for phrase cache
        ui_command_context: True if UI has placed user in command context
        conversation_id: For confirmation state tracking
        
    Returns:
        TranslationResult with resolved intent and execution decision
    """
    ui_context = UIContext(
        in_job_config=ui_command_context,
        in_sandbox_control=ui_command_context,
        in_pipeline_control=ui_command_context,
    )
    
    return translate_message_sync(
        text=text,
        user_id=user_id,
        ui_context=ui_context if ui_command_context else None,
        conversation_id=conversation_id,
    )


def is_zombie_map_trigger(text: str) -> bool:
    """
    Drop-in replacement for _is_zobie_map_trigger().
    
    Returns True only if:
    1. Message is in command mode
    2. Resolves to START_SANDBOX_ZOMBIE_SELF or related
    3. All gates pass
    """
    result = check_message_intent(text)
    return (
        result.should_execute and 
        result.resolved_intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF
    )


def is_archmap_trigger(text: str) -> bool:
    """
    Drop-in replacement for _is_archmap_trigger().
    
    Returns True only for ARCHITECTURE_MAP_WITH_FILES (ALL CAPS).
    """
    result = check_message_intent(text)
    return (
        result.should_execute and
        result.resolved_intent == CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES
    )


def is_archmap_structure_trigger(text: str) -> bool:
    """
    Returns True for ARCHITECTURE_MAP_STRUCTURE_ONLY (normal case).
    """
    result = check_message_intent(text)
    return (
        result.should_execute and
        result.resolved_intent == CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY
    )


def is_update_arch_trigger(text: str) -> bool:
    """
    Drop-in replacement for _is_update_arch_trigger().
    """
    result = check_message_intent(text)
    return (
        result.should_execute and
        result.resolved_intent == CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY
    )


def is_critical_pipeline_trigger(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if message triggers critical pipeline.
    
    Returns:
        (should_trigger, job_id or None)
        
    Note: This is a HIGH-STAKES operation that requires confirmation.
    The first call will return (False, None) with awaiting_confirmation=True.
    The user must then confirm with "Yes" before execution.
    """
    result = check_message_intent(text)
    
    if result.resolved_intent not in (CanonicalIntent.RUN_PIPELINE, CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB):
        return False, None
    
    # Check if confirmation is pending
    if result.confirmation_gate and result.confirmation_gate.awaiting_confirmation:
        # Return the confirmation prompt instead
        logger.info(f"Critical pipeline requires confirmation: {result.confirmation_gate.confirmation_prompt}")
        return False, None
    
    if result.should_execute:
        job_id = result.extracted_context.get("job_id")
        return True, job_id
    
    return False, None


def is_sandbox_trigger(text: str) -> bool:
    """
    Drop-in replacement for _is_sandbox_trigger().
    """
    result = check_message_intent(text)
    return (
        result.should_execute and
        result.resolved_intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF
    )


# =============================================================================
# STREAM ROUTER INTEGRATION
# =============================================================================

def route_message(
    text: str,
    user_id: str = "default",
    conversation_id: Optional[str] = None,
    ui_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Route a message using the translation layer.
    
    Returns a dictionary with routing information:
    {
        "mode": "chat" | "command" | "feedback",
        "should_execute": bool,
        "intent": str,
        "action": str | None,  # What action to take
        "context": dict,       # Extracted context
        "awaiting_confirmation": bool,
        "confirmation_prompt": str | None,
    }
    """
    ui = None
    if ui_context:
        ui = UIContext(
            in_job_config=ui_context.get("in_job_config", False),
            in_sandbox_control=ui_context.get("in_sandbox_control", False),
            in_pipeline_control=ui_context.get("in_pipeline_control", False),
            active_job_id=ui_context.get("job_id"),
            active_sandbox_id=ui_context.get("sandbox_id"),
        )
    
    result = translate_message_sync(
        text=text,
        user_id=user_id,
        ui_context=ui,
        conversation_id=conversation_id,
    )
    
    # Map intent to action
    action = _intent_to_action(result.resolved_intent) if result.should_execute else None
    
    return {
        "mode": result.mode.value,
        "should_execute": result.should_execute,
        "intent": result.resolved_intent.value,
        "action": action,
        "context": result.extracted_context,
        "awaiting_confirmation": (
            result.confirmation_gate.awaiting_confirmation 
            if result.confirmation_gate else False
        ),
        "confirmation_prompt": (
            result.confirmation_gate.confirmation_prompt
            if result.confirmation_gate and result.confirmation_gate.awaiting_confirmation
            else None
        ),
        "latency_tier": result.latency_tier.value,
        "from_cache": result.from_phrase_cache,
    }


def _intent_to_action(intent: CanonicalIntent) -> Optional[str]:
    """Map canonical intent to action name."""
    mapping = {
        CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES: "generate_full_architecture_map",
        CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY: "generate_structure_map",
        CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY: "update_code_atlas",
        CanonicalIntent.START_SANDBOX_ZOMBIE_SELF: "start_sandbox_zombie",
        CanonicalIntent.RUN_PIPELINE: "run_pipeline",  # v5.4: unified
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB: "run_pipeline",  # v5.4: alias
        CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES: "execute_overwatcher_changes",
        CanonicalIntent.CHAT_ONLY: None,
        CanonicalIntent.USER_BEHAVIOR_FEEDBACK: "log_feedback",
    }
    return mapping.get(intent)


# =============================================================================
# EXAMPLE USAGE IN STREAM_ROUTER.PY
# =============================================================================
"""
# In stream_router.py, replace the old trigger detection with:

from app.translation.integration import route_message

async def stream_chat(request: StreamRequest, ...):
    # Route the message through translation layer
    routing = route_message(
        text=request.message,
        user_id=str(user.id),
        conversation_id=request.conversation_id,
    )
    
    if not routing["should_execute"]:
        # Chat mode or blocked by gates
        if routing["awaiting_confirmation"]:
            # Send confirmation prompt
            yield f"data: {json.dumps({'confirmation': routing['confirmation_prompt']})}\n\n"
            return
        
        # Normal chat - proceed to LLM
        async for chunk in generate_sse_stream(request, ...):
            yield chunk
        return
    
    # Command execution
    action = routing["action"]
    context = routing["context"]
    
    if action == "generate_full_architecture_map":
        async for chunk in generate_archmap_stream(request, with_files=True):
            yield chunk
    elif action == "generate_structure_map":
        async for chunk in generate_archmap_stream(request, with_files=False):
            yield chunk
    elif action == "start_sandbox_zombie":
        async for chunk in generate_sandbox_stream(request):
            yield chunk
    elif action == "run_critical_pipeline":
        job_id = context.get("job_id")
        async for chunk in run_high_stakes_pipeline(request, job_id):
            yield chunk
    # ... etc
"""

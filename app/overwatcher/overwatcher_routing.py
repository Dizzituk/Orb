# FILE: app/overwatcher/overwatcher_routing.py
"""Overwatcher Routing Integration: Connects to translation layer.

This module provides the routing hooks for the "run overwatcher" command
to be recognized by the translation layer and routed correctly.

Command patterns recognized:
    - "run overwatcher"
    - "astra, command: run overwatcher"
    - "execute overwatcher"
    - "start overwatcher"
    - "overwatcher run"

Integration with stream_router.py:
    1. Translation layer detects command pattern
    2. Routes to OVERWATCHER_EXECUTE_CHANGES command type
    3. stream_router._handle_command_execution calls generate_overwatcher_stream
    4. generate_overwatcher_stream invokes run_overwatcher_command

Routing rules:
    - If spec flow state is AWAITING_OVERWATCHER → route to overwatcher
    - If explicit "run overwatcher" command → route to overwatcher
    - Otherwise → normal routing continues

STAGE_TRACE logging:
    - OVERWATCHER_ROUTE_ENTER: Command detected, routing started
    - OVERWATCHER_SPEC_RESOLVE: Spec lookup result
    - OVERWATCHER_ENTER: Analysis starting
    - OVERWATCHER_EXIT: Analysis complete with decision
    - IMPLEMENTER_ENTER: Implementation starting
    - IMPLEMENTER_EXIT: Implementation complete
    - VERIFICATION_ENTER: Verification starting
    - VERIFICATION_EXIT: Verification complete
    - OVERWATCHER_ROUTE_EXIT: Command complete
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Command Detection
# =============================================================================

class OverwatcherCommandType(str, Enum):
    """Types of Overwatcher commands."""
    RUN = "run"
    STATUS = "status"
    RETRY = "retry"
    CANCEL = "cancel"


# Patterns that indicate "run overwatcher" command
OVERWATCHER_COMMAND_PATTERNS = [
    r"(?:astra[,:]?\s+)?(?:command[:\s]+)?run\s+overwatcher",
    r"(?:astra[,:]?\s+)?execute\s+overwatcher",
    r"(?:astra[,:]?\s+)?start\s+overwatcher",
    r"overwatcher\s+run",
    r"run\s+the\s+overwatcher",
    r"invoke\s+overwatcher",
    r"trigger\s+overwatcher",
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in OVERWATCHER_COMMAND_PATTERNS]


def detect_overwatcher_command(message: str) -> Optional[OverwatcherCommandType]:
    """Detect if message is an Overwatcher command.
    
    Args:
        message: User message to check
    
    Returns:
        OverwatcherCommandType if detected, None otherwise
    """
    message = message.strip()
    
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(message):
            return OverwatcherCommandType.RUN
    
    # Check for status/retry/cancel
    if re.search(r"overwatcher\s+status", message, re.IGNORECASE):
        return OverwatcherCommandType.STATUS
    
    if re.search(r"(?:retry|rerun)\s+overwatcher", message, re.IGNORECASE):
        return OverwatcherCommandType.RETRY
    
    if re.search(r"(?:cancel|stop)\s+overwatcher", message, re.IGNORECASE):
        return OverwatcherCommandType.CANCEL
    
    return None


def should_route_to_overwatcher(
    message: str,
    project_id: int,
    spec_flow_state: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    """Determine if message should route to Overwatcher.
    
    This is called by stream_router to decide routing.
    
    Args:
        message: User message
        project_id: Current project ID
        spec_flow_state: Current spec flow state (from spec_flow_state module)
    
    Returns:
        (should_route, reason) tuple
    """
    # Check for explicit command
    cmd = detect_overwatcher_command(message)
    if cmd == OverwatcherCommandType.RUN:
        return True, "explicit_command"
    
    # Check spec flow state
    if spec_flow_state:
        try:
            from app.llm.spec_flow_state import SpecFlowStage
            
            if spec_flow_state.stage == SpecFlowStage.AWAITING_OVERWATCHER:
                return True, "spec_flow_awaiting_overwatcher"
        except ImportError:
            logger.debug("[overwatcher_routing] spec_flow_state module not available")
    
    return False, None


# =============================================================================
# Stream Generation
# =============================================================================

async def generate_overwatcher_stream(
    project_id: int,
    message: str,
    db,  # Session type
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for Overwatcher execution.
    
    This is the main entry point called by stream_router when
    routing to Overwatcher.
    
    Flow:
    1. Receive SPot (validated spec) context from DB
    2. Run Overwatcher analysis
    3. If APPROVED → run Implementer
    4. Run verification
    5. Stream results as SSE events
    
    Args:
        project_id: Project ID
        message: User message (for context)
        db: Database session
        trace: Optional routing trace for debugging
        conversation_id: Conversation ID for history
        job_id: Job ID to continue (optional)
    
    Yields:
        SSE-formatted response chunks
    """
    import json
    from uuid import uuid4
    from app.overwatcher.overwatcher_command import run_overwatcher_command
    
    # Create stage trace logger if trace available
    def log_stage(stage: str, status: str, details: Optional[Dict] = None):
        if trace and hasattr(trace, 'log_stage'):
            trace.log_stage(stage, status, details or {})
        logger.info(f"[STAGE_TRACE] {stage}: {status} {details or ''}")
    
    log_stage("OVERWATCHER_ROUTE_ENTER", "started", {"project_id": project_id, "message": message[:100]})
    
    # Initialize
    job_id = job_id or str(uuid4())
    
    # Yield initial status
    yield f"data: {json.dumps({'type': 'status', 'stage': 'overwatcher_starting', 'job_id': job_id})}\n\n"
    
    try:
        # Build LLM call function from providers
        llm_call_fn = None
        try:
            from app.providers.registry import get_provider
            from app.llm.stage_models import get_overwatcher_config
            
            async def _llm_call(provider_id: str, model_id: str, messages: list, max_tokens: int = 2000, **kwargs):
                """Wrapper for LLM calls using provider registry."""
                provider = get_provider(provider_id)
                if not provider:
                    raise ValueError(f"Provider not found: {provider_id}")
                
                from app.llm.schemas import LLMTask
                task = LLMTask(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,  # Deterministic for Overwatcher
                )
                
                result = await provider.complete(model_id, task)
                return result
            
            llm_call_fn = _llm_call
            
        except ImportError as e:
            logger.warning(f"[overwatcher_routing] Provider imports failed: {e}, running without LLM")
        
        # Run the command
        result = await run_overwatcher_command(
            project_id=project_id,
            job_id=job_id,
            message=message,
            db_session=db,
            llm_call_fn=llm_call_fn,
            use_smoke_test=True,  # Enable smoke test for first run
        )
        
        # Stream stage trace events
        for trace_entry in result.stage_trace:
            yield f"data: {json.dumps({'type': 'stage_trace', **trace_entry})}\n\n"
        
        # Stream final result
        if result.success:
            log_stage("OVERWATCHER_ROUTE_EXIT", "success", {"job_id": job_id})
            
            yield f"data: {json.dumps({'type': 'result', 'success': True, 'job_id': job_id, 'message': 'Overwatcher job completed successfully. File verified.'})}\n\n"
            
            # Final text response
            response_text = f"""✓ **Overwatcher Job Complete**

**Job ID:** {job_id}
**Spec:** {result.spec.spec_id if result.spec else 'N/A'} ({result.spec.spec_hash[:16] if result.spec else 'N/A'}...)

**Overwatcher Decision:** {result.overwatcher_decision}
**Diagnosis:** {result.overwatcher_diagnosis}

**Implementation:**
- Output: {result.implementer_result.output_path if result.implementer_result else 'N/A'}
- Sandbox: {'Yes' if result.implementer_result and result.implementer_result.sandbox_used else 'No'}

**Verification:** ✓ PASSED
- File exists: {result.verification_result.file_exists if result.verification_result else False}
- Content matches: {result.verification_result.content_matches if result.verification_result else False}

The job has been logged and the file is available at the specified location.
"""
            yield f"data: {json.dumps({'type': 'text', 'content': response_text})}\n\n"
            
        else:
            log_stage("OVERWATCHER_ROUTE_EXIT", "failed", {"job_id": job_id, "error": result.error})
            
            yield f"data: {json.dumps({'type': 'result', 'success': False, 'job_id': job_id, 'error': result.error})}\n\n"
            
            # Final error response
            response_text = f"""✗ **Overwatcher Job Failed**

**Job ID:** {job_id}
**Error:** {result.error}

**Stage Trace:**
"""
            for entry in result.stage_trace[-5:]:  # Last 5 entries
                response_text += f"- {entry['stage']}: {entry['status']}\n"
            
            response_text += "\nPlease check the logs for more details."
            yield f"data: {json.dumps({'type': 'text', 'content': response_text})}\n\n"
        
    except Exception as e:
        logger.exception(f"[overwatcher_routing] Stream generation failed: {e}")
        log_stage("OVERWATCHER_ROUTE_EXIT", "error", {"error": str(e)})
        
        yield f"data: {json.dumps({'type': 'error', 'message': f'Overwatcher execution failed: {str(e)}'})}\n\n"
    
    # End of stream
    yield "data: [DONE]\n\n"


# =============================================================================
# Spec Flow State Integration
# =============================================================================

def advance_to_awaiting_overwatcher(
    project_id: int,
    job_id: str,
    spec_id: str,
    spec_hash: str,
    db_session=None,
) -> bool:
    """Advance spec flow state to AWAITING_OVERWATCHER.
    
    Called by Critical Pipeline when artifacts are ready for Overwatcher.
    
    Args:
        project_id: Project ID
        job_id: Job ID
        spec_id: Spec ID
        spec_hash: Spec hash
        db_session: Database session
    
    Returns:
        True if state was advanced successfully
    """
    try:
        from app.llm.spec_flow_state import SpecFlowStage, advance_stage
        
        return advance_stage(
            project_id=project_id,
            new_stage=SpecFlowStage.AWAITING_OVERWATCHER,
            job_id=job_id,
            spec_id=spec_id,
            spec_hash=spec_hash,
            db=db_session,
        )
    except ImportError:
        logger.warning("[overwatcher_routing] spec_flow_state not available")
        return False
    except Exception as e:
        logger.error(f"[overwatcher_routing] Failed to advance state: {e}")
        return False


# =============================================================================
# Translation Layer Registration
# =============================================================================

# Command type for translation layer
OVERWATCHER_COMMAND_TYPE = "OVERWATCHER_EXECUTE_CHANGES"


@dataclass
class TranslationResult:
    """Result from translation layer classification."""
    command_type: str
    confidence: float
    params: Dict[str, Any]
    raw_message: str


def register_overwatcher_command():
    """Register Overwatcher command with translation layer.
    
    This should be called during app initialization to ensure
    the translation layer knows about the Overwatcher command.
    """
    try:
        from app.llm.translation_routing import register_command_type
        
        register_command_type(
            command_type=OVERWATCHER_COMMAND_TYPE,
            patterns=OVERWATCHER_COMMAND_PATTERNS,
            handler="generate_overwatcher_stream",
            description="Execute Overwatcher verification and implementation pipeline",
        )
        
        logger.info("[overwatcher_routing] Registered OVERWATCHER_EXECUTE_CHANGES command type")
        return True
        
    except ImportError:
        logger.warning("[overwatcher_routing] translation_routing not available")
        return False
    except Exception as e:
        logger.warning(f"[overwatcher_routing] Failed to register command: {e}")
        return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Command detection
    "OverwatcherCommandType",
    "detect_overwatcher_command",
    "should_route_to_overwatcher",
    # Stream generation
    "generate_overwatcher_stream",
    # State management
    "advance_to_awaiting_overwatcher",
    # Translation layer
    "OVERWATCHER_COMMAND_TYPE",
    "TranslationResult",
    "register_overwatcher_command",
]

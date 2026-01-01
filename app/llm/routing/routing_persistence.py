# FILE: app/llm/routing/routing_persistence.py
"""
Routing Decision Persistence (Job 3)

Persists routing decisions to astra_memory for:
- Cross-job learning
- Audit trail
- Pattern detection

Integrates with:
- astra_memory.service for job/event storage
- Overwatcher for strike state
- Cost guard for usage tracking
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Import astra_memory service
try:
    from app.astra_memory.service import (
        create_job,
        update_job_status,
        link_spec_to_job,
        link_arch_to_job,
        project_event_to_db,
        record_overwatch_intervention,
        record_overwatch_pattern,
        get_job,
    )
    from app.astra_memory.models import Job
    ASTRA_MEMORY_AVAILABLE = True
except ImportError:
    ASTRA_MEMORY_AVAILABLE = False
    logger.warning("[routing_persistence] astra_memory not available")

# Import cost guard
try:
    from app.overwatcher.cost_guard import (
        record_usage,
        ModelRole,
        get_cost_guard,
    )
    COST_GUARD_AVAILABLE = True
except ImportError:
    COST_GUARD_AVAILABLE = False

# Import strike state
try:
    from app.overwatcher.strike_state import (
        get_strike_state,
        StrikeState,
    )
    STRIKE_STATE_AVAILABLE = True
except ImportError:
    STRIKE_STATE_AVAILABLE = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RoutingPersistence:
    """
    Persists routing decisions and LLM call metadata to astra_memory.
    
    Provides:
    - Job lifecycle tracking
    - Event projection for routing decisions
    - Usage/cost aggregation
    - Pattern recording for learning
    """
    
    def __init__(self, db_session=None):
        self._db = db_session
        self._pending_events: List[Dict[str, Any]] = []
    
    def set_db_session(self, db_session) -> None:
        """Set the database session for persistence."""
        self._db = db_session
    
    def record_routing_decision(
        self,
        job_id: str,
        provider: str,
        model: str,
        job_type: str,
        reason: str,
        *,
        user_override: bool = False,
        frontier_override: bool = False,
        break_glass: bool = False,
        attachments: Optional[List[str]] = None,
        spec_id: Optional[str] = None,
        spec_hash: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record a routing decision.
        
        Args:
            job_id: Job identifier
            provider: Selected provider
            model: Selected model
            job_type: Classified job type
            reason: Routing reason/explanation
            user_override: User explicitly requested provider/model
            frontier_override: OVERRIDE command used
            break_glass: Break-glass mode active
            attachments: List of attachment filenames
            spec_id: PoT spec ID if available
            spec_hash: PoT spec hash if available
        
        Returns:
            Event ID if persisted, None otherwise
        """
        event_id = str(uuid4())
        
        event_data = {
            "event_id": event_id,
            "event_type": "ROUTING_DECISION",
            "job_id": job_id,
            "provider": provider,
            "model": model,
            "job_type": job_type,
            "reason": reason,
            "user_override": user_override,
            "frontier_override": frontier_override,
            "break_glass": break_glass,
            "attachments": attachments or [],
            "spec_id": spec_id,
            "spec_hash": spec_hash,
            "timestamp": _utc_now().isoformat(),
        }
        
        # Project to database if available
        if ASTRA_MEMORY_AVAILABLE and self._db:
            try:
                project_event_to_db(
                    db=self._db,
                    job_id=job_id,
                    event_type="ROUTING_DECISION",
                    stage="routing",
                    severity="info",
                    status="ok",
                    spec_id=spec_id,
                )
            except Exception as e:
                logger.warning(f"[routing_persistence] Failed to project event: {e}")
        else:
            self._pending_events.append(event_data)
        
        logger.debug(
            f"[routing_persistence] Recorded: job={job_id}, "
            f"provider={provider}, model={model}, type={job_type}"
        )
        
        return event_id
    
    def record_llm_call_result(
        self,
        job_id: str,
        provider: str,
        model: str,
        role: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_estimate: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        stage: Optional[str] = None,
        break_glass_used: bool = False,
    ) -> None:
        """
        Record an LLM call result.
        
        Args:
            job_id: Job identifier
            provider: LLM provider used
            model: Model used
            role: Model role (overwatcher, implementer, etc.)
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            cost_estimate: Estimated cost in USD
            success: Whether call succeeded
            error_message: Error message if failed
            stage: Pipeline stage
            break_glass_used: Whether break-glass was used
        """
        # Record to cost guard
        if COST_GUARD_AVAILABLE:
            try:
                model_role = ModelRole(role.lower()) if role else ModelRole.IMPLEMENTER
            except ValueError:
                model_role = ModelRole.IMPLEMENTER
            
            record_usage(
                job_id=job_id,
                role=model_role,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_estimate=cost_estimate,
                break_glass_used=break_glass_used,
                stage=stage,
            )
        
        # Project event to database
        if ASTRA_MEMORY_AVAILABLE and self._db:
            try:
                project_event_to_db(
                    db=self._db,
                    job_id=job_id,
                    event_type="LLM_CALL_COMPLETED" if success else "LLM_CALL_FAILED",
                    stage=stage,
                    severity="info" if success else "error",
                    status="ok" if success else "failed",
                    error_message=error_message,
                )
            except Exception as e:
                logger.warning(f"[routing_persistence] Failed to project LLM result: {e}")
    
    def record_validation_failure(
        self,
        job_id: str,
        validation_type: str,
        violations: List[Dict[str, Any]],
        *,
        role: str = "overwatcher",
        reprompt_triggered: bool = True,
    ) -> None:
        """
        Record a validation failure (e.g., Overwatcher output contract violation).
        
        Args:
            job_id: Job identifier
            validation_type: Type of validation (output_contract, schema, etc.)
            violations: List of violations detected
            role: Role that produced the invalid output
            reprompt_triggered: Whether reprompt was triggered
        """
        if ASTRA_MEMORY_AVAILABLE and self._db:
            try:
                # This is an intervention, not a strike
                record_overwatch_intervention(
                    db=self._db,
                    job_id=job_id,
                    intervention_type="warning",
                    reason=f"Validation failure: {validation_type}",
                )
                
                project_event_to_db(
                    db=self._db,
                    job_id=job_id,
                    event_type="VALIDATION_FAILURE",
                    stage="validation",
                    severity="warn",
                    status="reprompted" if reprompt_triggered else "failed",
                    error_message=f"{validation_type}: {len(violations)} violations",
                )
            except Exception as e:
                logger.warning(f"[routing_persistence] Failed to record validation failure: {e}")
    
    def record_strike(
        self,
        job_id: str,
        error_signature: str,
        strike_count: int,
        *,
        stage: str = "verification",
        diagnosis: Optional[str] = None,
        hard_stopped: bool = False,
    ) -> None:
        """
        Record a strike event.
        
        Args:
            job_id: Job identifier
            error_signature: Error signature that triggered the strike
            strike_count: Current strike count for this signature
            stage: Pipeline stage
            diagnosis: Overwatcher diagnosis
            hard_stopped: Whether this triggered a hard stop
        """
        if ASTRA_MEMORY_AVAILABLE and self._db:
            try:
                intervention_type = "block" if strike_count >= 3 else "warning"
                
                record_overwatch_intervention(
                    db=self._db,
                    job_id=job_id,
                    intervention_type=intervention_type,
                    reason=f"Strike {strike_count} for signature: {error_signature[:50]}",
                    error_signature=error_signature,
                )
                
                project_event_to_db(
                    db=self._db,
                    job_id=job_id,
                    event_type="STRIKE_RECORDED" if not hard_stopped else "HARD_STOP",
                    stage=stage,
                    severity="error" if hard_stopped else "warn",
                    status="hard_stopped" if hard_stopped else "continuing",
                    error_message=diagnosis,
                )
            except Exception as e:
                logger.warning(f"[routing_persistence] Failed to record strike: {e}")
    
    def record_pattern(
        self,
        job_id: str,
        pattern_type: str,
        *,
        target_path: Optional[str] = None,
        target_model: Optional[str] = None,
        error_signature: Optional[str] = None,
    ) -> None:
        """
        Record a pattern for cross-job learning.
        
        Args:
            job_id: Job identifier
            pattern_type: Type of pattern (file_fragility, model_error, etc.)
            target_path: File path if file-related
            target_model: Model if model-related
            error_signature: Error signature if error-related
        """
        if ASTRA_MEMORY_AVAILABLE and self._db:
            try:
                record_overwatch_pattern(
                    db=self._db,
                    pattern_type=pattern_type,
                    job_id=job_id,
                    target_path=target_path,
                    target_model=target_model,
                    error_signature=error_signature,
                )
            except Exception as e:
                logger.warning(f"[routing_persistence] Failed to record pattern: {e}")
    
    def flush_pending(self, db_session) -> int:
        """
        Flush pending events to database.
        
        Returns number of events flushed.
        """
        if not ASTRA_MEMORY_AVAILABLE:
            return 0
        
        count = 0
        for event in self._pending_events:
            try:
                project_event_to_db(
                    db=db_session,
                    job_id=event.get("job_id", "unknown"),
                    event_type=event.get("event_type", "UNKNOWN"),
                    stage="routing",
                    severity="info",
                )
                count += 1
            except Exception as e:
                logger.warning(f"[routing_persistence] Failed to flush event: {e}")
        
        self._pending_events.clear()
        return count


# =============================================================================
# Global Instance
# =============================================================================

_persistence: Optional[RoutingPersistence] = None


def get_routing_persistence() -> RoutingPersistence:
    """Get the global RoutingPersistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = RoutingPersistence()
    return _persistence


def record_routing_decision(
    job_id: str,
    provider: str,
    model: str,
    job_type: str,
    reason: str,
    **kwargs,
) -> Optional[str]:
    """Convenience function to record routing decision."""
    return get_routing_persistence().record_routing_decision(
        job_id, provider, model, job_type, reason, **kwargs
    )


def record_llm_call_result(
    job_id: str,
    provider: str,
    model: str,
    role: str,
    **kwargs,
) -> None:
    """Convenience function to record LLM call result."""
    get_routing_persistence().record_llm_call_result(
        job_id, provider, model, role, **kwargs
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RoutingPersistence",
    "get_routing_persistence",
    "record_routing_decision",
    "record_llm_call_result",
]

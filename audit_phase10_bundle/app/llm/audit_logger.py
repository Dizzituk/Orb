# FILE: app/llm/audit_logger.py
"""
Audit Logger for Orb Routing Pipeline.

Version: 1.0.0 - Critical Pipeline Spec Implementation

Implements Spec §12 (Logging / Audit Trail):
- Structured logging for all routing decisions
- Per-task routing traces
- Preprocessing step tracking
- Token budget allocation logging
- Model usage logging
- Fallback tracking

Log Format (per task):
{
    "task_id": "TASK_1",
    "user_text_fragment": "...",
    "file_ids": ["[FILE_1]", "[FILE_3]"],
    "flags": {
        "HAS_TEXT": true,
        "HAS_CODE": true,
        "HAS_IMAGE": false,
        "HAS_VIDEO": true,
        "IS_CRITICAL": false,
        "SANDBOX_MODE": false
    },
    "relations": {
        "REL_IMAGE_TEXT": "unrelated",
        "REL_VIDEO_TEXT": "related",
        ...
    },
    "lane": "video+code → Sonnet",
    "preprocessing": {
        "video_transcribed": true,
        "video_summary_tokens": 600,
        ...
    },
    "token_budget": {
        "max_context_tokens": 200000,
        "used_tokens_estimate": 48000,
        "allocation": {...}
    },
    "models_used": {
        "routing_model": "gpt-4.1-mini",
        "video_model": "Gemini-3-Pro",
        ...
    },
    "fallbacks": [...]
}

Usage:
    from app.llm.audit_logger import AuditLogger, get_audit_logger
    
    logger = get_audit_logger()
    trace = logger.start_trace(session_id, project_id)
    trace.log_classification(result)
    trace.log_routing_decision(decision)
    trace.finalize()
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Enable detailed audit logging
AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

# Write audit logs to file (in addition to standard logging)
AUDIT_LOG_DIR = os.getenv("ORB_AUDIT_LOG_DIR", "")

# Maximum entries to keep in memory
AUDIT_MAX_MEMORY_ENTRIES = int(os.getenv("ORB_AUDIT_MAX_ENTRIES", "1000"))

# Router debug mode (more verbose)
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


# =============================================================================
# AUDIT ENTRY TYPES
# =============================================================================

class AuditEventType(str, Enum):
    """Types of audit events."""
    TRACE_START = "trace_start"
    TRACE_END = "trace_end"
    CLASSIFICATION = "classification"
    RELATIONSHIP_DETECTION = "relationship_detection"
    TASK_EXTRACTION = "task_extraction"
    ROUTING_DECISION = "routing_decision"
    PREPROCESSING = "preprocessing"
    TOKEN_BUDGET = "token_budget"
    MODEL_CALL = "model_call"
    FALLBACK = "fallback"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class AuditEntry:
    """Single audit log entry."""
    event_type: AuditEventType
    timestamp: datetime
    trace_id: str
    task_id: Optional[str]
    data: Dict[str, Any]
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# MODALITY FLAGS (Spec §1.2)
# =============================================================================

@dataclass
class ModalityFlags:
    """Modality flags for audit logging."""
    HAS_TEXT: bool = False
    HAS_CODE: bool = False
    HAS_IMAGE: bool = False
    HAS_VIDEO: bool = False
    HAS_MIXED: bool = False
    IS_CRITICAL: bool = False
    SANDBOX_MODE: bool = False
    
    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


# =============================================================================
# RELATIONSHIP FLAGS (Spec §3)
# =============================================================================

@dataclass
class RelationshipFlags:
    """Pairwise relationship flags for audit logging."""
    REL_IMAGE_TEXT: str = "unclear"
    REL_IMAGE_CODE: str = "unclear"
    REL_VIDEO_TEXT: str = "unclear"
    REL_VIDEO_CODE: str = "unclear"
    REL_CODE_TEXT: str = "unclear"
    REL_IMAGE_VIDEO: str = "unclear"
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# =============================================================================
# PREPROCESSING LOG (Spec §5)
# =============================================================================

@dataclass
class PreprocessingLog:
    """Preprocessing step tracking."""
    video_transcribed: bool = False
    video_summary_tokens: int = 0
    video_transcript_tokens: int = 0
    video_model: Optional[str] = None
    
    image_described: bool = False
    image_summary_tokens: int = 0
    image_model: Optional[str] = None
    
    code_summarized: bool = False
    code_summary_tokens: int = 0
    
    text_summarized: bool = False
    text_summary_tokens: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# TOKEN BUDGET LOG (Spec §7)
# =============================================================================

@dataclass
class TokenBudgetLog:
    """Token budget allocation tracking."""
    max_context_tokens: int = 0
    used_tokens_estimate: int = 0
    allocation: Dict[str, int] = field(default_factory=dict)
    truncations: List[str] = field(default_factory=list)
    over_budget: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# MODEL USAGE LOG
# =============================================================================

@dataclass
class ModelUsageLog:
    """Models used in the pipeline."""
    routing_model: Optional[str] = None
    relationship_model: Optional[str] = None
    task_extractor_model: Optional[str] = None
    video_model: Optional[str] = None
    vision_model: Optional[str] = None
    final_model: Optional[str] = None
    critique_model: Optional[str] = None
    revision_model: Optional[str] = None
    overwatcher_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# =============================================================================
# TASK AUDIT LOG (Spec §12 complete structure)
# =============================================================================

@dataclass
class TaskAuditLog:
    """
    Complete audit log for a single task (Spec §12).
    
    This is the authoritative structure for task-level logging.
    """
    task_id: str
    user_text_fragment: str = ""
    file_ids: List[str] = field(default_factory=list)
    
    # Flags
    flags: ModalityFlags = field(default_factory=ModalityFlags)
    
    # Relationships
    relations: RelationshipFlags = field(default_factory=RelationshipFlags)
    
    # Routing
    lane: str = ""
    routing_reason: str = ""
    
    # Preprocessing
    preprocessing: PreprocessingLog = field(default_factory=PreprocessingLog)
    
    # Token budget
    token_budget: TokenBudgetLog = field(default_factory=TokenBudgetLog)
    
    # Models
    models_used: ModelUsageLog = field(default_factory=ModelUsageLog)
    
    # Fallbacks
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    
    # Outcome
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to full audit log dictionary."""
        return {
            "task_id": self.task_id,
            "user_text_fragment": self.user_text_fragment,
            "file_ids": self.file_ids,
            "flags": self.flags.to_dict(),
            "relations": self.relations.to_dict(),
            "lane": self.lane,
            "routing_reason": self.routing_reason,
            "preprocessing": self.preprocessing.to_dict(),
            "token_budget": self.token_budget.to_dict(),
            "models_used": self.models_used.to_dict(),
            "fallbacks": self.fallbacks,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


# =============================================================================
# ROUTING TRACE (per request)
# =============================================================================

class RoutingTrace:
    """
    Trace for a single routing request (may contain multiple tasks).
    
    Usage:
        trace = logger.start_trace(session_id, project_id)
        
        # Log classification
        trace.log_classification(classification_result)
        
        # Log routing decision
        trace.log_routing_decision(job_type, provider, model, reason)
        
        # Start task tracking
        task_log = trace.start_task("TASK_1")
        task_log.log_preprocessing(...)
        task_log.log_model_call(...)
        trace.complete_task(task_log)
        
        # Finalize
        trace.finalize(success=True)
    """
    
    def __init__(
        self,
        trace_id: str,
        session_id: str,
        project_id: int,
        user_text: str = "",
        is_critical: bool = False,
        sandbox_mode: bool = False,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.project_id = project_id
        self.user_text = user_text
        self.is_critical = is_critical
        self.sandbox_mode = sandbox_mode
        
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        
        self.entries: List[AuditEntry] = []
        self.task_logs: Dict[str, TaskAuditLog] = {}
        
        self.classification_result: Optional[Dict] = None
        self.relationships: Optional[Dict] = None
        self.tasks_extracted: List[Dict] = []
        self.final_routing: Optional[Dict] = None
        
        self._add_entry(AuditEventType.TRACE_START, None, {
            "session_id": session_id,
            "project_id": project_id,
            "user_text_length": len(user_text),
            "is_critical": is_critical,
            "sandbox_mode": sandbox_mode,
        })
    
    def _add_entry(
        self,
        event_type: AuditEventType,
        task_id: Optional[str],
        data: Dict[str, Any],
        duration_ms: Optional[int] = None,
    ) -> AuditEntry:
        """Add an audit entry."""
        entry = AuditEntry(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            trace_id=self.trace_id,
            task_id=task_id,
            data=data,
            duration_ms=duration_ms,
        )
        self.entries.append(entry)
        
        # Also log to standard logger if debug enabled
        if ROUTER_DEBUG:
            logger.debug(f"[audit] {event_type.value}: {json.dumps(data, default=str)[:500]}")
        
        return entry
    
    def log_classification(
        self,
        classification_result: Dict[str, Any],
    ) -> None:
        """Log file classification result."""
        self.classification_result = classification_result
        self._add_entry(AuditEventType.CLASSIFICATION, None, classification_result)
    
    def log_relationships(
        self,
        relationships: Dict[str, str],
    ) -> None:
        """Log relationship detection result."""
        self.relationships = relationships
        self._add_entry(AuditEventType.RELATIONSHIP_DETECTION, None, relationships)
    
    def log_task_extraction(
        self,
        tasks: List[Dict[str, Any]],
    ) -> None:
        """Log extracted tasks."""
        self.tasks_extracted = tasks
        self._add_entry(AuditEventType.TASK_EXTRACTION, None, {"tasks": tasks})
    
    def log_routing_decision(
        self,
        job_type: str,
        provider: str,
        model: str,
        reason: str,
        lane: str = "",
        task_id: Optional[str] = None,
    ) -> None:
        """Log routing decision."""
        data = {
            "job_type": job_type,
            "provider": provider,
            "model": model,
            "reason": reason,
            "lane": lane,
        }
        self.final_routing = data
        self._add_entry(AuditEventType.ROUTING_DECISION, task_id, data)
    
    def start_task(self, task_id: str, user_text_fragment: str = "") -> TaskAuditLog:
        """Start tracking a task."""
        task_log = TaskAuditLog(
            task_id=task_id,
            user_text_fragment=user_text_fragment[:200],
            started_at=datetime.utcnow(),
        )
        task_log.flags.IS_CRITICAL = self.is_critical
        task_log.flags.SANDBOX_MODE = self.sandbox_mode
        
        self.task_logs[task_id] = task_log
        return task_log
    
    def complete_task(
        self,
        task_log: TaskAuditLog,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Complete a task and log its audit data."""
        task_log.completed_at = datetime.utcnow()
        task_log.success = success
        task_log.error = error
        
        if task_log.started_at:
            delta = task_log.completed_at - task_log.started_at
            task_log.duration_ms = int(delta.total_seconds() * 1000)
        
        self._add_entry(
            AuditEventType.TRACE_END,
            task_log.task_id,
            task_log.to_dict(),
            task_log.duration_ms,
        )
    
    def log_preprocessing(
        self,
        task_id: Optional[str],
        step: str,
        data: Dict[str, Any],
        duration_ms: Optional[int] = None,
    ) -> None:
        """Log a preprocessing step."""
        self._add_entry(
            AuditEventType.PREPROCESSING,
            task_id,
            {"step": step, **data},
            duration_ms,
        )
    
    def log_model_call(
        self,
        task_id: Optional[str],
        provider: str,
        model: str,
        role: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log a model call."""
        self._add_entry(
            AuditEventType.MODEL_CALL,
            task_id,
            {
                "provider": provider,
                "model": model,
                "role": role,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "success": success,
                "error": error,
            },
            duration_ms,
        )
    
    def log_fallback(
        self,
        task_id: Optional[str],
        from_provider: str,
        from_model: str,
        to_provider: str,
        to_model: str,
        reason: str,
    ) -> None:
        """Log a fallback event."""
        data = {
            "from_provider": from_provider,
            "from_model": from_model,
            "to_provider": to_provider,
            "to_model": to_model,
            "reason": reason,
        }
        self._add_entry(AuditEventType.FALLBACK, task_id, data)
        
        # Also add to task log if exists
        if task_id and task_id in self.task_logs:
            self.task_logs[task_id].fallbacks.append(data)
    
    def log_error(
        self,
        task_id: Optional[str],
        error_type: str,
        error_message: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Log an error."""
        self._add_entry(
            AuditEventType.ERROR,
            task_id,
            {
                "error_type": error_type,
                "error_message": error_message,
                "details": details or {},
            },
        )
    
    def log_warning(
        self,
        task_id: Optional[str],
        warning_type: str,
        message: str,
    ) -> None:
        """Log a warning."""
        self._add_entry(
            AuditEventType.WARNING,
            task_id,
            {"warning_type": warning_type, "message": message},
        )
    
    def finalize(
        self,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Finalize the trace and return summary.
        
        Returns complete trace data for storage/analysis.
        """
        self.completed_at = datetime.utcnow()
        
        duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        
        summary = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
            "is_critical": self.is_critical,
            "sandbox_mode": self.sandbox_mode,
            "classification": self.classification_result,
            "relationships": self.relationships,
            "tasks_count": len(self.task_logs),
            "tasks": {k: v.to_dict() for k, v in self.task_logs.items()},
            "final_routing": self.final_routing,
            "entry_count": len(self.entries),
        }
        
        self._add_entry(
            AuditEventType.TRACE_END,
            None,
            {"success": success, "error": error, "duration_ms": duration_ms},
            duration_ms,
        )
        
        # Log summary
        if AUDIT_ENABLED:
            logger.info(f"[audit] Trace {self.trace_id} complete: {duration_ms}ms, success={success}")
            if ROUTER_DEBUG:
                logger.debug(f"[audit] Summary: {json.dumps(summary, default=str)[:1000]}")
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "user_text": self.user_text[:500],
            "is_critical": self.is_critical,
            "sandbox_mode": self.sandbox_mode,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "classification": self.classification_result,
            "relationships": self.relationships,
            "tasks_extracted": self.tasks_extracted,
            "task_logs": {k: v.to_dict() for k, v in self.task_logs.items()},
            "final_routing": self.final_routing,
            "entries": [e.to_dict() for e in self.entries],
        }


# =============================================================================
# AUDIT LOGGER (Global)
# =============================================================================

class AuditLogger:
    """
    Global audit logger for routing pipeline.
    
    Thread-safe, supports:
    - In-memory trace storage (bounded)
    - File-based logging (optional)
    - Trace retrieval for debugging
    """
    
    def __init__(
        self,
        max_entries: int = AUDIT_MAX_MEMORY_ENTRIES,
        log_dir: Optional[str] = None,
    ):
        self.max_entries = max_entries
        self.log_dir = log_dir or AUDIT_LOG_DIR
        
        self._traces: Dict[str, RoutingTrace] = {}
        self._completed_traces: List[Dict] = []
        self._lock = threading.Lock()
        self._trace_counter = 0
        
        # Create log directory if specified
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def start_trace(
        self,
        session_id: str,
        project_id: int,
        user_text: str = "",
        is_critical: bool = False,
        sandbox_mode: bool = False,
    ) -> RoutingTrace:
        """Start a new routing trace."""
        with self._lock:
            self._trace_counter += 1
            trace_id = f"trace_{int(time.time() * 1000)}_{self._trace_counter}"
        
        trace = RoutingTrace(
            trace_id=trace_id,
            session_id=session_id,
            project_id=project_id,
            user_text=user_text,
            is_critical=is_critical,
            sandbox_mode=sandbox_mode,
        )
        
        with self._lock:
            self._traces[trace_id] = trace
        
        if AUDIT_ENABLED:
            logger.info(f"[audit] Started trace {trace_id} for session {session_id}")
        
        return trace
    
    def complete_trace(
        self,
        trace: RoutingTrace,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete a trace and archive it."""
        summary = trace.finalize(success, error)
        
        with self._lock:
            # Remove from active traces
            self._traces.pop(trace.trace_id, None)
            
            # Add to completed traces (bounded)
            self._completed_traces.append(summary)
            if len(self._completed_traces) > self.max_entries:
                self._completed_traces = self._completed_traces[-self.max_entries:]
        
        # Write to file if configured
        if self.log_dir:
            self._write_trace_to_file(trace)
        
        return summary
    
    def _write_trace_to_file(self, trace: RoutingTrace) -> None:
        """Write trace to log file."""
        try:
            date_str = trace.started_at.strftime("%Y-%m-%d")
            log_file = Path(self.log_dir) / f"audit_{date_str}.jsonl"
            
            with open(log_file, "a") as f:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")
                
        except Exception as e:
            logger.warning(f"[audit] Failed to write trace to file: {e}")
    
    def get_trace(self, trace_id: str) -> Optional[RoutingTrace]:
        """Get an active trace by ID."""
        with self._lock:
            return self._traces.get(trace_id)
    
    def get_recent_traces(self, limit: int = 20) -> List[Dict]:
        """Get recent completed traces."""
        with self._lock:
            return self._completed_traces[-limit:]
    
    def get_traces_for_session(self, session_id: str) -> List[Dict]:
        """Get all traces for a session."""
        with self._lock:
            return [t for t in self._completed_traces if t.get("session_id") == session_id]
    
    def get_active_trace_count(self) -> int:
        """Get count of active traces."""
        with self._lock:
            return len(self._traces)
    
    def get_completed_trace_count(self) -> int:
        """Get count of completed traces in memory."""
        with self._lock:
            return len(self._completed_traces)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_audit_logger: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    
    with _audit_lock:
        if _audit_logger is None:
            _audit_logger = AuditLogger()
    
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset the global audit logger (for testing)."""
    global _audit_logger
    
    with _audit_lock:
        _audit_logger = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def start_trace(
    session_id: str,
    project_id: int,
    user_text: str = "",
    is_critical: bool = False,
    sandbox_mode: bool = False,
) -> RoutingTrace:
    """Start a new routing trace (convenience function)."""
    return get_audit_logger().start_trace(
        session_id=session_id,
        project_id=project_id,
        user_text=user_text,
        is_critical=is_critical,
        sandbox_mode=sandbox_mode,
    )


def complete_trace(
    trace: RoutingTrace,
    success: bool = True,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete a trace (convenience function)."""
    return get_audit_logger().complete_trace(trace, success, error)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Event types
    "AuditEventType",
    
    # Data structures
    "AuditEntry",
    "ModalityFlags",
    "RelationshipFlags",
    "PreprocessingLog",
    "TokenBudgetLog",
    "ModelUsageLog",
    "TaskAuditLog",
    
    # Trace
    "RoutingTrace",
    
    # Logger
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
    
    # Convenience
    "start_trace",
    "complete_trace",
    
    # Configuration
    "AUDIT_ENABLED",
    "ROUTER_DEBUG",
]
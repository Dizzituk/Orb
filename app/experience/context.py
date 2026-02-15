# FILE: app/experience/context.py
"""
Job context for journal emission.

Pipeline stages that don't have direct access to job_id can use the
thread-local context to emit journal entries. The context is set at
the start of a pipeline run and cleared at the end.

Usage:
    # At pipeline entry point (overwatcher_command, etc.):
    set_job_context(job_id="sg-abc123", job_dir="D:/Orb/jobs/jobs/sg-abc123")

    # In any pipeline stage:
    ctx = get_job_context()
    if ctx:
        emit_journal_entry(job_id=ctx.job_id, job_dir=ctx.job_dir, ...)

    # Or use the shortcut that handles missing context gracefully:
    journal_emit(stage="phase_checkout", event_type="boot_failure", ...)

    # At pipeline exit:
    clear_job_context()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Thread-local storage for job context
_thread_local = threading.local()


@dataclass
class JobContext:
    """Active job context for journal emission."""
    job_id: str
    job_dir: str
    job_type: str = ""
    language: str = ""


def set_job_context(
    job_id: str,
    job_dir: str,
    job_type: str = "",
    language: str = "",
) -> None:
    """Set the current job context (thread-local)."""
    _thread_local.job_context = JobContext(
        job_id=job_id,
        job_dir=job_dir,
        job_type=job_type,
        language=language,
    )
    logger.debug(f"[journal_ctx] Set job context: {job_id}")


def get_job_context() -> Optional[JobContext]:
    """Get the current job context, or None if not set."""
    return getattr(_thread_local, "job_context", None)


def clear_job_context() -> None:
    """Clear the current job context."""
    _thread_local.job_context = None


def journal_emit(
    *,
    stage: str,
    event_type: str,
    severity: str = "info",
    description: str = "",
    root_cause: str = "",
    resolution: str = "",
    error_signature: str = "",
    file_scope: str = "",
    segment_id: str = "",
    strategy_used: str = "",
    strike_number: int = 0,
    duration_ms: int = 0,
    details: Optional[Dict[str, Any]] = None,
    # Override context if available:
    job_id: Optional[str] = None,
    job_dir: Optional[str] = None,
    job_type: Optional[str] = None,
    language: Optional[str] = None,
) -> bool:
    """
    Emit a journal entry using the current job context.

    This is the recommended entry point for pipeline stages that don't
    have direct access to job_id/job_dir. Falls back gracefully if no
    context is set — returns False but never raises.

    Args:
        stage, event_type, etc.: Journal entry fields
        job_id, job_dir: Override context values (optional)
        job_type, language: Override context values (optional)
    """
    try:
        ctx = get_job_context()

        # Resolve job_id and job_dir from context or overrides
        _job_id = job_id or (ctx.job_id if ctx else None)
        _job_dir = job_dir or (ctx.job_dir if ctx else None)
        _job_type = job_type or (ctx.job_type if ctx else "")
        _language = language or (ctx.language if ctx else "")

        if not _job_id:
            return False  # No context, no journal — silently skip

        from .journal_writer import emit_journal_entry
        return emit_journal_entry(
            job_id=_job_id,
            job_dir=_job_dir,
            stage=stage,
            event_type=event_type,
            severity=severity,
            description=description,
            root_cause=root_cause,
            resolution=resolution,
            error_signature=error_signature,
            file_scope=file_scope,
            segment_id=segment_id,
            job_type=_job_type,
            language=_language,
            strategy_used=strategy_used,
            strike_number=strike_number,
            duration_ms=duration_ms,
            details=details,
        )
    except Exception:
        return False

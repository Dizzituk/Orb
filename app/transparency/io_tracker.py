# FILE: app/transparency/io_tracker.py
# Purpose: IOTracker — context-variable-based IO operation tracker.
# Called-by: app.builds.stage_hooks, app.sandbox.client, app.sandbox_fs
# Depends-on: app.transparency.io_events
# Last-renovated: 2026-06-11
"""
IOTracker — context-variable-based IO operation tracker.

Uses Python's contextvars so that pipeline stages set the tracker once
at stage entry, and all downstream sandbox_fs / SandboxClient calls
automatically record their IO operations without any parameter changes.

Usage in a pipeline stage:

    from app.transparency.io_tracker import IOTracker

    collector = ReasoningCollector(...)
    with IOTracker(collector, "specgate") as tracker:
        # All sandbox_read_text() calls inside this block
        # are automatically logged via the context var.
        content = sandbox_read_text("/some/path")

    # After the block, tracker.get_io_summary() has the totals.

Usage from sandbox_fs.py (automatic — no code changes needed at call sites):

    from app.transparency.io_tracker import get_active_tracker

    tracker = get_active_tracker()
    if tracker:
        tracker.record_read(path, "sandbox", ...)

v1.0 (2026-03): Initial implementation — Pipeline Logging Overhaul.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from app.transparency.io_events import (
    IOEvent,
    make_dir_scan_event,
    make_exists_event,
    make_read_event,
    make_write_event,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT VARIABLE
# =============================================================================

# The active IOTracker for the current async/thread context.
# Set by IOTracker.__enter__, cleared by IOTracker.__exit__.
# sandbox_fs.py reads this to decide whether to log IO operations.
_active_tracker: ContextVar[Optional["IOTracker"]] = ContextVar(
    "active_io_tracker", default=None,
)


def get_active_tracker() -> Optional["IOTracker"]:
    """Get the IOTracker for the current context, or None."""
    return _active_tracker.get(None)


# =============================================================================
# IO TRACKER
# =============================================================================

class IOTracker:
    """Tracks IO operations for a single pipeline stage execution.

    Acts as a context manager that sets/clears the context variable.
    Records every IO event and forwards them to the ReasoningCollector
    for SSE emission and DB persistence.
    """

    def __init__(
        self,
        collector: Any = None,
        stage_name: str = "",
    ):
        """
        Args:
            collector: A ReasoningCollector instance (or None for dry-run).
            stage_name: The pipeline stage name (e.g. "specgate").
        """
        self._collector = collector
        self._stage_name = stage_name
        self._events: List[IOEvent] = []
        self._token = None  # contextvars.Token for reset

        # Running totals
        self._sandbox_reads = 0
        self._sandbox_writes = 0
        self._host_reads = 0
        self._host_writes = 0
        self._violations = 0

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    def __enter__(self) -> "IOTracker":
        self._token = _active_tracker.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _active_tracker.reset(self._token)
            self._token = None
        return None  # don't suppress exceptions

    # =========================================================================
    # RECORD OPERATIONS
    # =========================================================================

    def record_read(
        self,
        path: str,
        source: str = "sandbox",
        purpose: str = "",
        content_summary: str = "",
        bytes_count: int = 0,
    ) -> IOEvent:
        """Record a file read operation."""
        event = make_read_event(
            path=path,
            source=source,
            purpose=purpose,
            content_summary=content_summary,
            bytes_count=bytes_count,
            stage_name=self._stage_name,
        )
        self._store_and_emit(event)
        return event

    def record_write(
        self,
        path: str,
        target: str = "sandbox",
        intent: str = "",
        content_summary: str = "",
        bytes_count: int = 0,
    ) -> IOEvent:
        """Record a file write operation."""
        event = make_write_event(
            path=path,
            target=target,
            intent=intent,
            content_summary=content_summary,
            bytes_count=bytes_count,
            stage_name=self._stage_name,
        )
        self._store_and_emit(event)
        return event

    def record_exists_check(
        self,
        path: str,
        source: str = "sandbox",
        result: bool = False,
        purpose: str = "",
    ) -> IOEvent:
        """Record a file/directory existence check."""
        event = make_exists_event(
            path=path,
            source=source,
            result=result,
            purpose=purpose,
            stage_name=self._stage_name,
        )
        self._store_and_emit(event)
        return event

    def record_dir_scan(
        self,
        path: str,
        source: str = "sandbox",
        file_count: int = 0,
        purpose: str = "",
    ) -> IOEvent:
        """Record a directory listing operation."""
        event = make_dir_scan_event(
            path=path,
            source=source,
            file_count=file_count,
            purpose=purpose,
            stage_name=self._stage_name,
        )
        self._store_and_emit(event)
        return event

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _store_and_emit(self, event: IOEvent) -> None:
        """Store the event, update totals, emit via collector."""
        self._events.append(event)
        self._update_totals(event)

        if event.is_violation:
            logger.warning(
                "[io_tracker] SANDBOX VIOLATION: %s %s from %s (%s)",
                event.operation, event.path, event.source, event.purpose,
            )

        # Forward to ReasoningCollector for SSE + DB persistence
        if self._collector is not None:
            try:
                self._collector.add_io_event(event)
            except Exception as e:
                logger.debug("[io_tracker] Failed to emit IO event: %s", e)

    def _update_totals(self, event: IOEvent) -> None:
        """Update running totals from an event."""
        if event.is_violation:
            self._violations += 1

        source = event.source.split(":")[0]  # "host:VIOLATION" → "host"
        is_write = event.operation == "write"

        if source == "sandbox":
            if is_write:
                self._sandbox_writes += 1
            else:
                self._sandbox_reads += 1
        elif source == "host":
            if is_write:
                self._host_writes += 1
            else:
                self._host_reads += 1

    # =========================================================================
    # SUMMARY / RETRIEVAL
    # =========================================================================

    @property
    def events(self) -> List[IOEvent]:
        """All IO events recorded by this tracker."""
        return list(self._events)

    @property
    def violation_count(self) -> int:
        return self._violations

    def get_io_summary(self) -> Dict[str, Any]:
        """Get summary totals for report compilation.

        Returns:
            Dict with read/write/violation counts, broken down by
            sandbox vs host.
        """
        return {
            "total_reads": self._sandbox_reads + self._host_reads,
            "total_writes": self._sandbox_writes + self._host_writes,
            "sandbox_reads": self._sandbox_reads,
            "sandbox_writes": self._sandbox_writes,
            "host_reads_operational": self._host_reads,
            "host_writes_operational": self._host_writes,
            "host_reads_violation": sum(
                1 for e in self._events
                if e.is_violation and e.operation != "write"
            ),
            "host_writes_violation": sum(
                1 for e in self._events
                if e.is_violation and e.operation == "write"
            ),
            "violations": self._violations,
            "total_events": len(self._events),
        }

    def get_events_for_stage(self, stage_name: str) -> List[IOEvent]:
        """Filter events by stage name."""
        return [e for e in self._events if e.stage_name == stage_name]


__all__ = [
    "IOTracker",
    "get_active_tracker",
]

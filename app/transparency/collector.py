# FILE: app/transparency/collector.py
"""
ReasoningCollector — central hub for emitting reasoning events.

Pipeline stages call collector.emit() to:
1. Persist the event to the reasoning_events table
2. Emit via SSE to the frontend in real-time

Also provides helpers for:
- Starting/finishing stage traces with reasoning
- Streaming reasoning text in real-time (Layer 1)
- Recording evidence and decisions (Layer 2)

v1.0 (2026-02): Initial implementation
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.transparency.schemas import (
    DecisionPoint,
    EvidenceSource,
    ReasoningEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SSE CALLBACK TYPE
# =============================================================================

# SSE emit callback: async function that sends an SSE event to the client.
# Set by the stream handler before pipeline stages run.
SSECallback = Optional[Callable[[dict], Any]]


# =============================================================================
# REASONING COLLECTOR
# =============================================================================

class ReasoningCollector:
    """
    Collects and emits reasoning events from pipeline stages.

    Usage:
        collector = ReasoningCollector(
            job_id="job_123",
            run_id="run_001",
            build_project_id="bp_abc",
        )

        # Start a stage
        collector.start_stage("specgate", stage_index=2)

        # Stream reasoning text (Layer 1 — types out like LLM thinking)
        collector.add_reasoning("Checking router.py for auth patterns...")

        # Record evidence (Layer 2 — collapsible dropdown)
        collector.add_evidence(EvidenceSource(
            source_type="file_read",
            reference="app/endpoints/chat.py",
            summary="Read lines 45-120, found route handlers",
            bytes_read=3200,
        ))

        # Record a decision
        collector.add_decision(DecisionPoint(
            question="Where is auth handled?",
            options_considered=["decorators", "middleware"],
            chosen="middleware",
            reasoning="No @auth decorators found",
            confidence=0.78,
        ))

        # Finish the stage
        await collector.finish_stage(
            status="passed",
            confidence_score=0.78,
            model_used="gemini-2.5-flash",
        )
    """

    def __init__(
        self,
        job_id: str = "",
        run_id: str = "",
        build_project_id: str = "",
        sse_callback: SSECallback = None,
    ):
        self.job_id = job_id
        self.run_id = run_id
        self.build_project_id = build_project_id
        self._sse_callback = sse_callback

        # Current stage being traced
        self._current_event: Optional[ReasoningEvent] = None
        self._stage_start_time: float = 0.0

        # All events for this run
        self.events: List[ReasoningEvent] = []

    def set_sse_callback(self, callback: SSECallback) -> None:
        """Set the SSE emit callback (called by stream handlers)."""
        self._sse_callback = callback

    # =========================================================================
    # STAGE LIFECYCLE
    # =========================================================================

    def start_stage(
        self,
        stage_name: str,
        stage_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningEvent:
        """Start tracing a new pipeline stage."""
        self._stage_start_time = time.time()

        event = ReasoningEvent(
            job_id=self.job_id,
            run_id=self.run_id,
            build_project_id=self.build_project_id,
            stage_name=stage_name,
            stage_index=stage_index,
            status="running",
            metadata=metadata or {},
        )

        self._current_event = event
        self._emit_sse(event.to_sse_dict())

        logger.info(
            "[transparency] Stage STARTED: %s (job=%s, run=%s)",
            stage_name, self.job_id, self.run_id,
        )

        return event

    async def finish_stage(
        self,
        status: str = "passed",
        confidence_score: float = 0.0,
        model_used: str = "",
        token_cost_usd: float = 0.0,
        reasoning_summary: str = "",
    ) -> Optional[ReasoningEvent]:
        """Finish the current stage trace and persist it."""
        if not self._current_event:
            return None

        event = self._current_event
        event.status = status
        event.confidence_score = confidence_score
        event.model_used = model_used
        event.token_cost_usd = token_cost_usd
        event.duration_ms = int((time.time() - self._stage_start_time) * 1000)

        if reasoning_summary:
            event.reasoning_summary = reasoning_summary

        # Auto-generate summary if not provided
        if not event.reasoning_summary and event.reasoning_detail:
            lines = event.reasoning_detail.strip().split("\n")
            event.reasoning_summary = lines[0][:200] if lines else ""

        # Persist to DB
        await self._persist(event)

        # Emit final state via SSE
        self._emit_sse(event.to_sse_dict())

        # Store in run history
        self.events.append(event)
        self._current_event = None

        logger.info(
            "[transparency] Stage FINISHED: %s — %s (%.0f%% confidence, %dms)",
            event.stage_name, status,
            confidence_score * 100, event.duration_ms,
        )

        return event

    # =========================================================================
    # REASONING (Layer 1 — streams out like LLM thinking)
    # =========================================================================

    def add_reasoning(self, text: str) -> None:
        """Append reasoning text to the current stage trace."""
        if not self._current_event:
            return

        self._current_event.reasoning_detail += text

        # Stream the reasoning text chunk via SSE
        self._emit_sse({
            "type": "reasoning_stream",
            "stage_name": self._current_event.stage_name,
            "content": text,
        })

    # =========================================================================
    # EVIDENCE (Layer 2 — collapsible dropdowns)
    # =========================================================================

    def add_evidence(self, evidence: EvidenceSource) -> None:
        """Record an evidence source gathered by the current stage."""
        if not self._current_event:
            return

        self._current_event.evidence_sources.append(evidence)

        # Emit as collapsible data operation via SSE
        self._emit_sse({
            "type": "evidence_gathered",
            "stage_name": self._current_event.stage_name,
            "evidence": evidence.to_dict(),
        })

    def add_decision(self, decision: DecisionPoint) -> None:
        """Record a decision point made by the current stage."""
        if not self._current_event:
            return

        self._current_event.decisions.append(decision)

        # Emit as collapsible data operation via SSE
        self._emit_sse({
            "type": "decision_made",
            "stage_name": self._current_event.stage_name,
            "decision": decision.to_dict(),
        })

    # =========================================================================
    # SSE EMISSION
    # =========================================================================

    def _emit_sse(self, data: dict) -> None:
        """Send an SSE event to the frontend."""
        if self._sse_callback:
            try:
                self._sse_callback(data)
            except Exception as e:
                logger.debug("[transparency] SSE emit failed: %s", e)

    def make_sse_bytes(self, data: dict) -> bytes:
        """Create SSE event as bytes for direct stream injection."""
        return f"data: {json.dumps(data)}\n\n".encode("utf-8")

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def _persist(self, event: ReasoningEvent) -> None:
        """Persist a reasoning event to the database."""
        try:
            from app.db import get_db_session
            from app.transparency.models import ReasoningEventModel

            db = get_db_session()
            try:
                row = ReasoningEventModel(
                    event_id=event.event_id,
                    job_id=event.job_id,
                    run_id=event.run_id,
                    build_project_id=event.build_project_id,
                    stage_name=event.stage_name,
                    stage_index=event.stage_index,
                    status=event.status,
                    confidence_score=event.confidence_score,
                    reasoning_summary=event.reasoning_summary,
                    reasoning_detail=event.reasoning_detail,
                    evidence_sources=[e.to_dict() for e in event.evidence_sources],
                    decisions=[d.to_dict() for d in event.decisions],
                    model_used=event.model_used,
                    token_cost_usd=event.token_cost_usd,
                    duration_ms=event.duration_ms,
                    metadata_json=event.metadata,
                )
                db.add(row)
                db.commit()
            finally:
                db.close()

        except Exception as e:
            logger.warning("[transparency] Failed to persist reasoning event: %s", e)

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    @staticmethod
    async def get_run_trace(job_id: str, run_id: str) -> List[dict]:
        """Retrieve full reasoning trace for a pipeline run."""
        try:
            from app.db import get_db_session
            from app.transparency.models import ReasoningEventModel

            db = get_db_session()
            try:
                rows = (
                    db.query(ReasoningEventModel)
                    .filter_by(job_id=job_id, run_id=run_id)
                    .order_by(ReasoningEventModel.stage_index)
                    .all()
                )
                return [_row_to_dict(r) for r in rows]
            finally:
                db.close()
        except Exception as e:
            logger.warning("[transparency] Failed to get run trace: %s", e)
            return []

    @staticmethod
    async def get_project_trace(build_project_id: str) -> List[dict]:
        """Retrieve all reasoning events for a build project."""
        try:
            from app.db import get_db_session
            from app.transparency.models import ReasoningEventModel

            db = get_db_session()
            try:
                rows = (
                    db.query(ReasoningEventModel)
                    .filter_by(build_project_id=build_project_id)
                    .order_by(ReasoningEventModel.created_at)
                    .all()
                )
                return [_row_to_dict(r) for r in rows]
            finally:
                db.close()
        except Exception as e:
            logger.warning("[transparency] Failed to get project trace: %s", e)
            return []


def _row_to_dict(row) -> dict:
    """Convert a ReasoningEventModel row to a dict."""
    return {
        "event_id": row.event_id or "",
        "job_id": row.job_id or "",
        "run_id": row.run_id or "",
        "build_project_id": row.build_project_id or "",
        "stage_name": row.stage_name or "",
        "stage_index": row.stage_index or 0,
        "status": row.status or "running",
        "confidence_score": row.confidence_score or 0.0,
        "reasoning_summary": row.reasoning_summary or "",
        "reasoning_detail": row.reasoning_detail or "",
        "evidence_sources": row.evidence_sources or [],
        "decisions": row.decisions or [],
        "model_used": row.model_used or "",
        "token_cost_usd": row.token_cost_usd or 0.0,
        "duration_ms": row.duration_ms or 0,
        "metadata": row.metadata_json or {},
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "corrections": [],  # populated by router from CorrectionStore
    }


__all__ = [
    "ReasoningCollector",
]

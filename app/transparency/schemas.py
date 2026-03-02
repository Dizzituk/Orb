# FILE: app/transparency/schemas.py
"""
Data structures for the Pipeline Transparency system.

Defines:
- EvidenceSource: What data was gathered and from where
- DecisionPoint: A reasoning decision with options considered
- ReasoningEvent: Full stage reasoning trace emitted via SSE
- UserCorrection: Feedback pinned to a specific reasoning event
- CorrectionMatch: A relevant past correction found by the matcher

v1.0 (2026-02): Initial implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _uuid() -> str:
    return str(uuid4())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# EVIDENCE SOURCE
# =============================================================================

@dataclass
class EvidenceSource:
    """A piece of evidence gathered during pipeline reasoning."""

    source_type: str  # "file_read" | "rag_chunk" | "db_query" | "import_graph" | "web_search"
    reference: str    # file path, chunk ID, query string, URL
    summary: str      # what was found / why it was relevant
    bytes_read: int = 0
    content_preview: str = ""  # first N chars of content (for UI display)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "reference": self.reference,
            "summary": self.summary,
            "bytes_read": self.bytes_read,
            "content_preview": self.content_preview,
        }


# =============================================================================
# DECISION POINT
# =============================================================================

@dataclass
class DecisionPoint:
    """A specific decision made during pipeline reasoning."""

    question: str              # "Where is auth handled?"
    options_considered: List[str] = field(default_factory=list)  # ["decorators", "middleware"]
    chosen: str = ""           # "middleware"
    reasoning: str = ""        # "No @auth decorators found..."
    confidence: float = 0.0    # 0.0 - 1.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "options_considered": self.options_considered,
            "chosen": self.chosen,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


# =============================================================================
# REASONING EVENT
# =============================================================================

@dataclass
class ReasoningEvent:
    """
    Full reasoning trace for a single pipeline stage execution.

    Emitted via SSE to the frontend for real-time display.
    Persisted to DB for audit trail and correction matching.
    """

    # Identity
    event_id: str = field(default_factory=_uuid)
    job_id: str = ""
    run_id: str = ""
    build_project_id: str = ""

    # Stage info
    stage_name: str = ""       # "weaver" | "specgate" | "critical" | etc.
    stage_index: int = 0       # sequential order in pipeline

    # Timing
    timestamp: str = field(default_factory=_utc_now)

    # Status
    status: str = "running"    # "running" | "passed" | "failed" | "warning"
    confidence_score: float = 0.0

    # Reasoning content — Layer 1 (streams out like LLM thinking)
    reasoning_summary: str = ""   # 1-2 line collapsed view
    reasoning_detail: str = ""    # full reasoning chain (streams in real-time)

    # Data operations — Layer 2 (collapsible dropdowns)
    evidence_sources: List[EvidenceSource] = field(default_factory=list)
    decisions: List[DecisionPoint] = field(default_factory=list)

    # Cost/performance
    model_used: str = ""          # "gemini-2.5-flash" | "deterministic" | etc.
    token_cost_usd: float = 0.0
    duration_ms: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "build_project_id": self.build_project_id,
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "timestamp": self.timestamp,
            "status": self.status,
            "confidence_score": self.confidence_score,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_detail": self.reasoning_detail,
            "evidence_sources": [e.to_dict() for e in self.evidence_sources],
            "decisions": [d.to_dict() for d in self.decisions],
            "model_used": self.model_used,
            "token_cost_usd": self.token_cost_usd,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    def to_sse_dict(self) -> dict:
        """Format for SSE emission — includes type field."""
        d = self.to_dict()
        d["type"] = "reasoning_event"
        return d


# =============================================================================
# USER CORRECTION
# =============================================================================

@dataclass
class UserCorrection:
    """
    User feedback pinned to a specific reasoning event.

    Stored in DB and queried by CorrectionMatcher during future runs.
    """

    correction_id: str = field(default_factory=_uuid)
    reasoning_event_id: str = ""
    job_id: str = ""
    run_id: str = ""
    build_project_id: str = ""

    # What stage / decision was wrong
    stage_name: str = ""
    stage_index: int = 0
    decision_index: Optional[int] = None  # if correcting a specific decision

    # The correction
    user_comment: str = ""
    severity: str = "note"          # "note" | "wrong_output" | "broke_things"
    correction_type: str = "wrong_decision"  # "wrong_evidence" | "wrong_decision" | "missing_context" | "wrong_output"

    # For matching
    context_keywords: List[str] = field(default_factory=list)

    # Timing
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "correction_id": self.correction_id,
            "reasoning_event_id": self.reasoning_event_id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "build_project_id": self.build_project_id,
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "decision_index": self.decision_index,
            "user_comment": self.user_comment,
            "severity": self.severity,
            "correction_type": self.correction_type,
            "context_keywords": self.context_keywords,
            "created_at": self.created_at,
        }


# =============================================================================
# CORRECTION MATCH (returned by matcher)
# =============================================================================

@dataclass
class CorrectionMatch:
    """A relevant past correction found during pipeline execution."""

    correction: UserCorrection
    relevance_score: float = 0.0   # 0.0 - 1.0
    original_context: str = ""     # what the pipeline was doing when corrected

    def to_prompt_injection(self) -> str:
        """Format for injection into LLM prompts."""
        severity_label = self.correction.severity.upper().replace("_", " ")
        return (
            f"PREVIOUS CORRECTION ({severity_label}): "
            f"Stage '{self.correction.stage_name}' — "
            f"{self.correction.user_comment}"
        )


__all__ = [
    "EvidenceSource",
    "DecisionPoint",
    "ReasoningEvent",
    "UserCorrection",
    "CorrectionMatch",
]

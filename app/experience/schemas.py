# FILE: app/experience/schemas.py
# Purpose: Data structures for the Experience Database.
# Called-by: app.experience, app.experience.distillation, app.experience.journal_writer
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Data structures for the Experience Database.

Defines:
- BuildJournalEntry: The atomic unit of pipeline observation
- EventSeverity: How critical an event is
- JournalEventType: Catalogue of event types emitted per stage

Design: Every field exists because a retrieval query will filter on it.
Nothing is stored "just in case."
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import json


# =============================================================================
# EVENT SEVERITY
# =============================================================================

class EventSeverity(str, Enum):
    """Severity of a journal event. Used for confidence scoring during distillation."""
    CRITICAL = "critical"   # Fatal errors, pipeline-stopping failures
    ERROR = "error"         # Errors that triggered retries/strikes
    WARNING = "warning"     # Non-fatal issues, suboptimal outcomes
    INFO = "info"           # Normal pipeline decisions and progress
    DEBUG = "debug"         # Detailed diagnostic data (not distilled)


# =============================================================================
# EVENT TYPES BY STAGE (Section 3.1.2 of spec)
# =============================================================================

class JournalEventType(str, Enum):
    """
    Structured event types emitted by pipeline stages.

    The event_type is the primary classifier for retrieval — when the
    Critical Pipeline queries memory, it filters on event types produced
    by past Critical Pipeline and Critique runs.
    """

    # --- Critical Pipeline ---
    ARCHITECTURE_DECISION = "architecture_decision"
    ARCHITECTURE_CONSTRAINT = "architecture_constraint"
    ARCHITECTURE_PARSE_ERROR = "architecture_parse_error"

    # --- Critique ---
    CRITIQUE_FINDING = "critique_finding"
    CRITIQUE_FALSE_POSITIVE = "critique_false_positive"
    CRITIQUE_REVISION_TRIGGER = "critique_revision_trigger"

    # --- Cohesion Check ---
    COHESION_MISMATCH = "cohesion_mismatch"
    COHESION_NAMING_DRIFT = "cohesion_naming_drift"
    COHESION_INTERFACE_BREAK = "cohesion_interface_break"

    # --- Overwatcher / Implementer ---
    IMPLEMENTATION_STRATEGY = "implementation_strategy"
    IMPLEMENTATION_STRIKE = "implementation_strike"
    IMPLEMENTATION_SUCCESS = "implementation_success"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    EDIT_MODE_FALLBACK = "edit_mode_fallback"
    VERBATIM_EXTRACTION = "verbatim_extraction"

    # --- Phase Checkout (Boot Test) ---
    BOOT_PASS = "boot_pass"
    BOOT_FAILURE = "boot_failure"
    BOOT_FIX_ATTEMPT = "boot_fix_attempt"
    BOOT_FIX_SUCCESS = "boot_fix_success"
    BOOT_FIX_FAILED = "boot_fix_failed"

    # --- Final Checkout ---
    SPEC_COVERAGE_MISS = "spec_coverage_miss"
    AI_REVIEW_FINDING = "ai_review_finding"
    FINAL_BOOT_RESULT = "final_boot_result"

    # --- General ---
    STAGE_ENTER = "stage_enter"
    STAGE_EXIT = "stage_exit"
    ERROR_GENERIC = "error_generic"
    DECISION = "decision"


# =============================================================================
# BUILD JOURNAL ENTRY (Section 3.1.1 of spec)
# =============================================================================

@dataclass
class BuildJournalEntry:
    """
    The atomic unit of pipeline observation.

    Each entry captures a single event: a decision, an error, a fix attempt,
    a success. The journal is append-only — entries are never modified.

    Fields are designed for retrieval:
    - stage + event_type: Primary filter keys (indexed)
    - job_type + language: Secondary filter keys
    - error_signature: Exact-match lookup for boot fix patterns
    - file_scope: File-level filtering for Implementer patterns
    """

    # --- Identity ---
    job_id: str
    stage: str                          # Pipeline stage: "critical_pipeline", "implementer", etc.
    event_type: str                     # From JournalEventType
    timestamp: str = ""                 # ISO 8601, filled on creation

    # --- Classification ---
    severity: str = "info"              # From EventSeverity
    job_type: str = ""                  # "refactor", "app_build", "feature_add", etc.
    language: str = ""                  # "python", "typescript", etc.
    segment_id: str = ""                # e.g. "seg-01-architecture-executor"

    # --- Scoping ---
    file_scope: str = ""                # File path this event relates to
    error_signature: str = ""           # Deterministic hash of the error for exact-match lookup

    # --- Content ---
    description: str = ""               # Human-readable summary of what happened
    root_cause: str = ""                # Why it happened (causal chain)
    resolution: str = ""                # What fixed it (if applicable)
    details: Dict[str, Any] = field(default_factory=dict)  # Arbitrary structured data

    # --- Context ---
    strategy_used: str = ""             # What approach was taken
    strike_number: int = 0              # Which attempt (0 = first try)
    duration_ms: int = 0                # How long this event took

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dict, omitting empty fields."""
        d = asdict(self)
        # Remove empty strings and empty dicts to keep NDJSON compact
        return {k: v for k, v in d.items() if v or v == 0}

    def to_json(self) -> str:
        """Serialise to compact JSON string for NDJSON output."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildJournalEntry":
        """Reconstruct from a dict (e.g. parsed from NDJSON line)."""
        # Only pass fields that the dataclass knows about
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, line: str) -> "BuildJournalEntry":
        """Parse a single NDJSON line."""
        return cls.from_dict(json.loads(line))


# =============================================================================
# TRACE → JOURNAL MAPPING HELPERS
# =============================================================================

# Maps existing add_trace() stage names to journal event types.
# Used by the journal hook to automatically classify trace events.
TRACE_STAGE_TO_EVENT_TYPE: Dict[str, str] = {
    # Architecture execution
    "ARCHITECTURE_EXECUTION_START": JournalEventType.STAGE_ENTER,
    "ARCHITECTURE_PARSE": JournalEventType.ARCHITECTURE_DECISION,
    "ARCHITECTURE_EXECUTION": JournalEventType.ERROR_GENERIC,
    "SANDBOX_CONNECTED": JournalEventType.STAGE_ENTER,
    "SANDBOX_BASE_RESOLVED": JournalEventType.STAGE_ENTER,
    "AUTO_INIT_PY": JournalEventType.IMPLEMENTATION_STRATEGY,
    "MODULE_SHADOW_BLOCKED": JournalEventType.IMPORT_ERROR,

    # File task execution
    "FILE_TASK_START": JournalEventType.STAGE_ENTER,
    "FILE_TASK_STRIKE": JournalEventType.IMPLEMENTATION_STRIKE,
    "FILE_TASK_SUCCESS": JournalEventType.IMPLEMENTATION_SUCCESS,
    "FILE_TASK_FAILED": JournalEventType.ERROR_GENERIC,
    "VERBATIM_EXTRACTION": JournalEventType.VERBATIM_EXTRACTION,
    "MODIFY_EDIT_MODE": JournalEventType.EDIT_MODE_FALLBACK,
    "EDIT_PARSE_FALLBACK": JournalEventType.EDIT_MODE_FALLBACK,
    "EDIT_PARTIAL": JournalEventType.IMPLEMENTATION_STRIKE,
    "QUARANTINE_SKIP": JournalEventType.DECISION,
    "TWO_PASS_CONTEXT_REFRESH": JournalEventType.IMPLEMENTATION_STRATEGY,

    # Boot check
    "BOOT_CHECK_START": JournalEventType.STAGE_ENTER,
    "BOOT_CHECK_COMPLETE": JournalEventType.BOOT_PASS,
    "BOOT_CHECK_FAIL": JournalEventType.BOOT_FAILURE,
    "BOOT_FIX_ATTEMPT": JournalEventType.BOOT_FIX_ATTEMPT,

    # Job-level checks
    "JOB_CHECK_PASS": JournalEventType.STAGE_EXIT,
    "JOB_CHECK_FAIL": JournalEventType.ERROR_GENERIC,

    # Overwatcher stages
    "OVERWATCHER_ENTER": JournalEventType.STAGE_ENTER,
    "IMPLEMENTER_ENTER": JournalEventType.STAGE_ENTER,
    "VERIFICATION_ENTER": JournalEventType.STAGE_ENTER,
    "SPEC_RESOLVE": JournalEventType.ERROR_GENERIC,

    # Build validation
    "BUILD_VALIDATION": JournalEventType.DECISION,
    "BUILD_FIX_ATTEMPT": JournalEventType.BOOT_FIX_ATTEMPT,
    "BUILD_FIX_DIAGNOSIS": JournalEventType.DECISION,
    "BUILD_FIX_EXECUTED": JournalEventType.BOOT_FIX_ATTEMPT,
    "BUILD_FIX_SUCCESS": JournalEventType.BOOT_FIX_SUCCESS,
    "BUILD_FIX_EXHAUSTED": JournalEventType.BOOT_FIX_FAILED,
}

# Maps trace status values to severity
TRACE_STATUS_TO_SEVERITY: Dict[str, str] = {
    "started": EventSeverity.INFO,
    "running": EventSeverity.INFO,
    "success": EventSeverity.INFO,
    "complete": EventSeverity.INFO,
    "pass": EventSeverity.INFO,
    "processing": EventSeverity.INFO,
    "enabled": EventSeverity.INFO,
    "skipped": EventSeverity.INFO,
    "verified": EventSeverity.INFO,
    "idempotent": EventSeverity.INFO,
    "disabled": EventSeverity.INFO,

    "failed": EventSeverity.ERROR,
    "fatal": EventSeverity.CRITICAL,
    "parse_failed": EventSeverity.ERROR,
    "some_failed": EventSeverity.WARNING,
    "still_failing": EventSeverity.ERROR,
    "exhausted_strikes": EventSeverity.CRITICAL,
    "warning_no_projects": EventSeverity.WARNING,
    "no_fixes": EventSeverity.WARNING,

    "strike_1": EventSeverity.WARNING,
    "strike_2": EventSeverity.ERROR,
    "strike_3": EventSeverity.CRITICAL,
}

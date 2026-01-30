# FILE: app/pot_spec/spec_gate_types.py
"""
Spec Gate v2 - Types and Constants

Contains:
- Result dataclass
- Blocking validation config
- Placeholder patterns and detection

v1.14 (2026-01-24): Added REWRITE_IN_PLACE output mode for multi-question file edits
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# OUTPUT MODE ENUM (v1.14 - REWRITE_IN_PLACE for multi-question edits)
# ---------------------------------------------------------------------------

class OutputMode(str, Enum):
    """
    v1.14: Output mode for MICRO_FILE_TASK jobs (sandbox file discovery jobs).
    v1.24: Added OVERWRITE_FULL for destructive replacement operations.
    
    Determines where and how the reply should be written:
    
    - OVERWRITE_FULL: Replace entire file with new content (destructive)
      Use for: "overwrite the file", "delete everything and leave", "replace contents"
      HIGHEST WRITE PRIORITY - bypasses Q&A analysis entirely
      
    - APPEND_IN_PLACE: Append reply to END of file (Add-Content)
      Use for: Single-question files, "write reply at the end"
      
    - REWRITE_IN_PLACE: Read file, insert answers under each question, write back entire file
      Use for: Multi-question files, "answer every question", "under each question"
      Requires: sandbox_output_path == sandbox_input_path
      
    - SEPARATE_REPLY_FILE: Write to a separate reply.txt file
      Use for: "save to reply.txt", "create a reply file"
      
    - CHAT_ONLY: No file output, reply in chat only
      Use for: "just answer here", "don't change the file"
    
    Detection keywords (v1.24 priority order):
    
    1. CHAT_ONLY (explicit no-file-change request) - SAFETY OVERRIDE:
       "just answer here", "don't change the file", "chat only"
       
    2. OVERWRITE_FULL (destructive replacement - v1.24) - HIGHEST WRITE PRIORITY:
       "overwrite the file", "overwrite file", "replace contents",
       "delete everything", "remove everything", "just leave",
       "replace with", "contain only", "leave only"
       
    3. REWRITE_IN_PLACE (multi-question file edits - v1.14):
       "answer every question", "answer each question", "answer all questions",
       "under each question", "beneath each question", "below each question",
       "directly under each question", "put answer under each"
       
    4. SEPARATE_REPLY_FILE:
       "save to reply.txt", "write to a new file", "create a reply file"
       
    5. APPEND_IN_PLACE (single insertion at EOF):
       "write under", "append", "add below", "beneath the question"
       (only when NOT multi-question context)
    
    Default: CHAT_ONLY (safest - no file modification)
    """
    OVERWRITE_FULL = "overwrite_full"  # v1.24: Destructive full replacement
    APPEND_IN_PLACE = "append_in_place"
    REWRITE_IN_PLACE = "rewrite_in_place"  # v1.14: Multi-question insert
    SEPARATE_REPLY_FILE = "separate_reply_file"
    CHAT_ONLY = "chat_only"


# ---------------------------------------------------------------------------
# FILESYSTEM EVIDENCE TYPES (v1.25 - Evidence-First Implementation)
# ---------------------------------------------------------------------------

class FilesystemEvidenceSource(str, Enum):
    """
    v1.25: Source of filesystem evidence for grounding specs.
    
    Distinguishes HOW we know about a file's existence/state:
    - RAG_HINT: Found via RAG query (may be stale)
    - LIVE_FS: Confirmed via live filesystem check (ground truth)
    - RAG_CONFIRMED: RAG hint validated by live filesystem
    - SEARCH_FOUND: Discovered via live filesystem search
    - NOT_FOUND: Path was checked but does not exist
    - USER_PROVIDED: Explicitly provided by user in message
    """
    RAG_HINT = "rag_hint"
    LIVE_FS = "live_filesystem"
    RAG_CONFIRMED = "rag_hint_confirmed_by_live_fs"
    SEARCH_FOUND = "live_fs_search"
    NOT_FOUND = "not_found"
    USER_PROVIDED = "user_provided"


@dataclass
class FileEvidence:
    """
    v1.25: Evidence about a single file's existence and state.
    
    This is the core unit of filesystem evidence. Each FileEvidence
    represents what we KNOW (not guess) about a file.
    
    Fields:
        original_reference: What user said ("Desktop/Test/test.txt")
        resolved_path: Actual filesystem path (C:/Users/.../test.txt) or None if unresolved
        source: How we know about this file (FilesystemEvidenceSource)
        exists: Whether the file exists on the filesystem
        readable: Whether we can read the file
        writable: Whether we can write to the file
        size_bytes: File size in bytes
        mtime: Last modified time (ISO format string)
        content_preview: First ~500 chars of content (for structure detection)
        content_hash: SHA256 hash of content (for change detection)
        detected_structure: Detected format ("qa_format", "python", "json", etc.)
        metadata: Additional metadata dict for extensibility
    """
    original_reference: str
    resolved_path: Optional[str] = None
    source: FilesystemEvidenceSource = FilesystemEvidenceSource.NOT_FOUND
    exists: bool = False
    readable: bool = False
    writable: bool = False
    size_bytes: Optional[int] = None
    mtime: Optional[str] = None
    content_preview: Optional[str] = None
    content_hash: Optional[str] = None
    detected_structure: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_evidence_line(self) -> str:
        """Format as a single-line evidence summary."""
        status = "✅ EXISTS" if self.exists else "❌ NOT FOUND"
        parts = [f"`{self.resolved_path or self.original_reference}` {status}"]
        
        if self.exists:
            if self.size_bytes is not None:
                parts.append(f"({self.size_bytes} bytes)")
            if self.detected_structure:
                parts.append(f"[{self.detected_structure}]")
        
        parts.append(f"source: {self.source.value}")
        return " ".join(parts)


@dataclass
class EvidencePackage:
    """
    v1.25: Complete evidence package gathered for a spec.
    
    This is the collection of ALL evidence gathered during the
    reconnaissance phase, BEFORE the LLM is called. The LLM receives
    this as FACT, not as something to discover.
    
    Fields:
        task_type: Detected task type ("file_operation", "code_analysis", etc.)
        target_files: List of FileEvidence for all referenced files
        rag_hints_used: RAG hints that were consulted
        search_queries_run: Filesystem searches that were executed
        validation_errors: Hard errors that block spec generation
        warnings: Non-blocking warnings
        ground_truth_timestamp: When this evidence was gathered (ISO format)
    """
    task_type: str = "unknown"
    target_files: List[FileEvidence] = field(default_factory=list)
    rag_hints_used: List[dict] = field(default_factory=list)
    search_queries_run: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ground_truth_timestamp: str = ""
    
    def has_valid_targets(self) -> bool:
        """Check if we have at least one confirmed existing file."""
        return any(f.exists for f in self.target_files)
    
    def get_primary_target(self) -> Optional[FileEvidence]:
        """Get the first existing file target, or None."""
        for f in self.target_files:
            if f.exists:
                return f
        return None
    
    def to_summary(self) -> str:
        """Generate a brief summary of the evidence package."""
        existing = sum(1 for f in self.target_files if f.exists)
        total = len(self.target_files)
        return (
            f"EvidencePackage: {existing}/{total} files exist, "
            f"{len(self.validation_errors)} errors, "
            f"{len(self.warnings)} warnings"
        )


# ---------------------------------------------------------------------------
# BLOCKING VALIDATION CONFIG (v2.1)
# ---------------------------------------------------------------------------

BLOCKING_FIELDS = {
    "steps": {"min_count": 1, "description": "execution steps"},
    "outputs": {"min_count": 1, "description": "output artifacts"},
    "acceptance_criteria": {"min_count": 1, "description": "verification criteria"},
}

PLACEHOLDER_PATTERNS = [
    r"^\s*\(?\s*not\s+specified\s*\)?\s*$",
    r"^\s*\(?\s*tbd\s*\)?\s*$",
    r"^\s*\(?\s*to\s+be\s+determined\s*\)?\s*$",
    r"^\s*\(?\s*n/?a\s*\)?\s*$",
    r"^\s*\(?\s*unknown\s*\)?\s*$",
    r"^\s*\(?\s*unspecified\s*\)?\s*$",
    r"^\s*\.\.\.\s*$",
    r"^\s*-\s*$",
    r"^\s*$",
]

PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SpecGateResult:
    """
    Result from Spec Gate v2 processing.
    
    v2.2 (2026-01-21): Added grounding_data field for Critical Pipeline job classification.
    This enables micro vs architecture routing by persisting sandbox discovery results.
    """
    ready_for_pipeline: bool = False
    open_questions: List[str] = field(default_factory=list)
    spot_markdown: Optional[str] = None
    db_persisted: bool = False
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: Optional[int] = None
    hard_stopped: bool = False
    hard_stop_reason: Optional[str] = None
    notes: Optional[str] = None
    blocking_issues: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, needs_clarification, validated, blocked
    
    # v2.2: Grounding data for Critical Pipeline job classification
    # Contains sandbox resolution fields needed for micro vs architecture routing
    grounding_data: Optional[dict] = None


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

def is_placeholder(value: Any) -> bool:
    """Check if a value is a placeholder that shouldn't count as real content."""
    if value is None:
        return True
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.match(value.strip()))
    if isinstance(value, dict):
        name = value.get("name", "")
        if is_placeholder(name):
            return True
    return False


def count_real_items(items: List[Any]) -> int:
    """Count items that are not placeholders."""
    if not items or not isinstance(items, list):
        return 0
    return sum(1 for item in items if not is_placeholder(item))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_blocking_fields(
    outputs: List[dict],
    steps: List[str],
    acceptance: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that blocking fields have real content.
    
    Returns:
        (is_valid, blocking_issues, clarification_questions)
    """
    blocking_issues: List[str] = []
    questions: List[str] = []
    
    real_outputs = count_real_items(outputs)
    if real_outputs < BLOCKING_FIELDS["outputs"]["min_count"]:
        blocking_issues.append(f"BLOCKING: No valid outputs specified (found {real_outputs} real items)")
        questions.append("What exact output artifacts should exist when done? (file names + locations)")
    
    real_steps = count_real_items(steps)
    if real_steps < BLOCKING_FIELDS["steps"]["min_count"]:
        blocking_issues.append(f"BLOCKING: No valid steps specified (found {real_steps} real items)")
        questions.append("What exact steps should the system take, in order?")
    
    real_acceptance = count_real_items(acceptance)
    if real_acceptance < BLOCKING_FIELDS["acceptance_criteria"]["min_count"]:
        blocking_issues.append(f"BLOCKING: No valid acceptance criteria (found {real_acceptance} real items)")
        questions.append("How should we verify success? (exact file content, path, etc.)")
    
    return len(blocking_issues) == 0, blocking_issues, questions


def derive_required_questions(
    outputs: List[dict], 
    steps: List[str], 
    acceptance: List[str]
) -> List[str]:
    """Legacy function - derive questions for missing fields."""
    q: List[str] = []
    if not outputs:
        q.append("What exact output artifacts should exist when the job is done (file/folder names + locations)?")
    if not steps:
        q.append("What exact steps should the system take, in order? (keep it short)")
    if not acceptance:
        q.append("How should we verify success? (e.g., exact file content, exact path, overwrite behaviour)")
    return q


__all__ = [
    "BLOCKING_FIELDS",
    "PLACEHOLDER_RE",
    "SpecGateResult",
    "OutputMode",  # v1.14: Now includes REWRITE_IN_PLACE
    # v1.25: Evidence-First types
    "FilesystemEvidenceSource",
    "FileEvidence",
    "EvidencePackage",
    # Functions
    "is_placeholder",
    "count_real_items",
    "validate_blocking_fields",
    "derive_required_questions",
]

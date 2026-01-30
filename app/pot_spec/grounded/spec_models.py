# FILE: app/pot_spec/grounded/spec_models.py
"""
SpecGate Data Models

Contains all dataclasses and enums used by the SpecGate grounding system.

Classes:
--------
- QuestionCategory: Enum for categorizing questions
- GroundedFact: A verified fact from evidence
- GroundedAssumption: A safe default applied instead of asking (v1.4)
- GroundedQuestion: A high-impact question requiring human input
- FileTarget: Target file with individual anchor (v1.21 - Level 2.5 multi-target read)
- MultiFileOperation: Multi-file operation spec (v1.20 - Level 3)
- GroundedPOTSpec: The main Point-of-Truth spec dataclass

Version Notes:
-------------
v1.21 (2026-01-29): Added FileTarget for Level 2.5 multi-target file read
    - Supports reading N specific named files from different locations
    - Each target has its own anchor (desktop, D:, etc.)
    - Bridges gap between single-file (Level 2) and pattern-search (Level 3)
v1.20 (2026-01-28): Added MultiFileOperation for Level 3 multi-file operations
v1.4 (2026-01): Added GroundedAssumption for tracking safe defaults
v1.5 (2026-01): Added spec.decisions dict for resolved blocking forks
v1.11 (2026-01): Added implementation_stack field for tech stack anchoring
v1.13 (2026-01): Added sandbox_output_mode and sandbox_insertion_format
v1.19 (2026-01): Added scan_* fields for SCAN_ONLY jobs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_QUESTIONS = 0
MAX_QUESTIONS = 7


# =============================================================================
# ENUMS
# =============================================================================

class QuestionCategory(str, Enum):
    """Categories for questions (allowed types)."""
    PREFERENCE = "preference"           # UI style, tone, naming preference
    MISSING_PRODUCT_DECISION = "product_decision"  # New workflow, manual vs auto
    AMBIGUOUS_EVIDENCE = "ambiguous"    # Map says X, code says Y
    SAFETY_CONSTRAINT = "safety"        # Sandbox vs main, backwards compat
    DECISION_FORK = "decision_fork"     # v1.2: Bounded A/B/C product decision


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class FileTarget:
    """
    v1.21: Target file with individual anchor for multi-target read operations.
    
    Used for Level 2.5 operations like "read test on desktop and test2 on D drive".
    Each target has its own anchor, allowing files from different locations.
    
    Attributes:
        name: The filename or reference (e.g., "test", "test2.txt")
        anchor: Location anchor ("desktop", "documents", "D:", "C:", etc.)
        subfolder: Optional subfolder within anchor (e.g., "Test" in Desktop/Test/)
        explicit_path: Full path if user provided one (e.g., "D:\\test2.txt")
        resolved_path: Actual filesystem path after resolution
        found: Whether the file was found during resolution
        content: File content if read successfully
        error: Error message if resolution/read failed
    """
    name: str
    anchor: Optional[str] = None        # "desktop", "documents", "D:", "C:", etc.
    subfolder: Optional[str] = None     # Subfolder within anchor
    explicit_path: Optional[str] = None # Full path if provided by user
    resolved_path: Optional[str] = None # Actual path after resolution
    found: bool = False                 # Was file found?
    content: Optional[str] = None       # File content if read
    error: Optional[str] = None         # Error message if failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "anchor": self.anchor,
            "subfolder": self.subfolder,
            "explicit_path": self.explicit_path,
            "resolved_path": self.resolved_path,
            "found": self.found,
            "content": self.content[:500] if self.content else None,  # Truncate for storage
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileTarget":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", ""),
            anchor=data.get("anchor"),
            subfolder=data.get("subfolder"),
            explicit_path=data.get("explicit_path"),
            resolved_path=data.get("resolved_path"),
            found=data.get("found", False),
            content=data.get("content"),
            error=data.get("error"),
        )


@dataclass
class GroundedFact:
    """A verified fact grounded in evidence."""
    description: str
    source: str  # e.g., "codebase_report", "architecture_map", "file_read"
    path: Optional[str] = None  # File path if applicable
    confidence: str = "confirmed"  # "confirmed", "inferred", "uncertain"


@dataclass
class GroundedAssumption:
    """
    v1.4: A safe default applied instead of asking a non-blocking question.
    
    These are recorded in the spec so the user can override if needed,
    but they don't block spec completion.
    """
    topic: str              # e.g., "sync_behaviour", "pay_variation"
    assumed_value: str      # The default we applied
    reason: str             # Why this is safe for v1
    can_override: bool = True  # User can change this later


@dataclass
class GroundedQuestion:
    """A high-impact question that requires human input."""
    question: str
    category: QuestionCategory
    why_it_matters: str
    evidence_found: str  # What SpecGate found so far
    options: Optional[List[str]] = None  # A/B options if applicable

    def format(self) -> str:
        """Format question for POT spec output."""
        lines = [f"**Q:** {self.question}"]
        lines.append(f"  - *Why it matters:* {self.why_it_matters}")
        lines.append(f"  - *Evidence found:* {self.evidence_found}")
        if self.options:
            lines.append(f"  - *Options:* " + " / ".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(self.options)))
        return "\n".join(lines)


@dataclass
class MultiFileOperation:
    """Multi-file operation specification (v1.20 - Level 3).
    
    Used when SpecGate detects a multi-file intent and runs file discovery
    to build a target file list for human review before execution.
    
    Attributes:
        is_multi_file: Whether this is a multi-file operation
        operation_type: "search" (read-only) or "refactor" (write)
        search_pattern: Pattern to find in files
        replacement_pattern: Replacement text (empty for search operations)
        target_files: List of file paths to process
        total_files: Total number of matching files found
        total_occurrences: Total occurrences across all files
        file_filter: Optional extension filter (e.g., "*.py")
        file_preview: Human-readable preview of matches
        discovery_truncated: True if results were truncated
        discovery_duration_ms: How long discovery took
        roots_searched: Directories that were searched
        requires_confirmation: True for refactor operations (safety)
        confirmed: Whether user has approved the operation
        error_message: Set if discovery encountered issues
    """
    # Operation type
    is_multi_file: bool = False
    operation_type: str = ""  # "search" or "refactor"
    
    # Patterns
    search_pattern: str = ""
    replacement_pattern: str = ""  # Empty for search operations
    
    # Discovery results
    target_files: List[str] = field(default_factory=list)
    total_files: int = 0
    total_occurrences: int = 0
    
    # File filter (e.g., "*.py")
    file_filter: Optional[str] = None
    
    # Preview for human review (first N files with samples)
    file_preview: str = ""
    
    # Discovery metadata
    discovery_truncated: bool = False
    discovery_duration_ms: int = 0
    roots_searched: List[str] = field(default_factory=list)
    
    # Confirmation flags (refactor operations require explicit approval)
    requires_confirmation: bool = False
    confirmed: bool = False
    
    # Error handling
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/DB storage."""
        return {
            "is_multi_file": self.is_multi_file,
            "operation_type": self.operation_type,
            "search_pattern": self.search_pattern,
            "replacement_pattern": self.replacement_pattern,
            "target_files": self.target_files,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "file_filter": self.file_filter,
            "file_preview": self.file_preview,
            "discovery_truncated": self.discovery_truncated,
            "discovery_duration_ms": self.discovery_duration_ms,
            "roots_searched": self.roots_searched,
            "requires_confirmation": self.requires_confirmation,
            "confirmed": self.confirmed,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiFileOperation":
        """Deserialize from dictionary."""
        return cls(
            is_multi_file=data.get("is_multi_file", False),
            operation_type=data.get("operation_type", ""),
            search_pattern=data.get("search_pattern", ""),
            replacement_pattern=data.get("replacement_pattern", ""),
            target_files=data.get("target_files", []),
            total_files=data.get("total_files", 0),
            total_occurrences=data.get("total_occurrences", 0),
            file_filter=data.get("file_filter"),
            file_preview=data.get("file_preview", ""),
            discovery_truncated=data.get("discovery_truncated", False),
            discovery_duration_ms=data.get("discovery_duration_ms", 0),
            roots_searched=data.get("roots_searched", []),
            requires_confirmation=data.get("requires_confirmation", False),
            confirmed=data.get("confirmed", False),
            error_message=data.get("error_message"),
        )


@dataclass
class GroundedPOTSpec:
    """
    Point-of-Truth Spec grounded in repo evidence.
    
    This is the main output of SpecGate - a complete specification
    that anchors intent in filesystem reality.
    """
    # Core
    goal: str
    
    # Grounded reality
    confirmed_components: List[GroundedFact] = field(default_factory=list)
    what_exists: List[str] = field(default_factory=list)
    what_missing: List[str] = field(default_factory=list)
    
    # v1.11: Tech stack anchoring (prevents architecture drift)
    # Type hint uses string to avoid circular import - actual type is ImplementationStack
    implementation_stack: Optional[Any] = None
    
    # Scope
    in_scope: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    
    # Constraints
    constraints_from_intent: List[str] = field(default_factory=list)
    constraints_from_repo: List[str] = field(default_factory=list)
    
    # Evidence
    # Type hint uses Any to avoid import of EvidenceBundle
    evidence_bundle: Optional[Any] = None
    
    # Plan
    proposed_steps: List[str] = field(default_factory=list)
    acceptance_tests: List[str] = field(default_factory=list)
    
    # Risks
    risks: List[Dict[str, str]] = field(default_factory=list)
    refactor_flags: List[str] = field(default_factory=list)
    
    # Questions (human decisions only)
    open_questions: List[GroundedQuestion] = field(default_factory=list)
    
    # v1.4: Assumptions (safe defaults applied instead of asking)
    assumptions: List[GroundedAssumption] = field(default_factory=list)
    
    # v1.5: Resolved decisions (explicit answers to blocking forks)
    decisions: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: int = 1
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation
    is_complete: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    
    # Evidence completeness tracking (v1.1)
    evidence_complete: bool = True
    evidence_gaps: List[str] = field(default_factory=list)
    
    # v1.19: SCAN_ONLY job parameters
    scan_roots: List[str] = field(default_factory=list)       # Paths to scan, e.g. ["D:\\Orb"]
    scan_terms: List[str] = field(default_factory=list)       # Search terms, e.g. ["Orb", "ORB", "orb"]
    scan_targets: List[str] = field(default_factory=list)     # What to search: ["names"], ["contents"], or both
    scan_case_mode: Optional[str] = None                       # "case_sensitive" or "case_insensitive"
    scan_exclusions: List[str] = field(default_factory=list)  # Patterns to skip, e.g. [".git", "node_modules"]
    
    # Sandbox resolution (v1.3 - for sandbox file jobs)
    sandbox_input_path: Optional[str] = None       # Full path to input file in sandbox
    sandbox_output_path: Optional[str] = None      # Full path for output (same folder as input)
    sandbox_folder_path: Optional[str] = None      # Folder containing input
    sandbox_anchor: Optional[str] = None           # "desktop", "documents", etc.
    sandbox_subfolder: Optional[str] = None        # Subfolder name if specified
    sandbox_selected_type: Optional[str] = None    # MESSAGE, CODE, etc.
    sandbox_selection_confidence: float = 0.0      # Classification confidence
    sandbox_input_excerpt: Optional[str] = None    # First ~500 chars of input
    sandbox_input_full_content: Optional[str] = None  # v1.7: Full content of input file
    sandbox_generated_reply: Optional[str] = None  # v1.7: Generated reply (read-only, included in SPoT)
    sandbox_discovery_used: bool = False           # True if sandbox discovery was run
    sandbox_ambiguity: Optional[str] = None        # Ambiguity reason if any
    sandbox_discovery_status: Optional[str] = None # v1.6: not_attempted, attempted, success, no_match, error
    sandbox_skip_reason: Optional[str] = None      # v1.6: Why discovery was skipped
    sandbox_output_mode: Optional[str] = None      # v1.13: append_in_place, separate_reply_file, chat_only
    sandbox_insertion_format: Optional[str] = None # v1.13: Planned insertion format for APPEND_IN_PLACE
    
    # v1.20: Multi-file operations (Level 3)
    multi_file: Optional[MultiFileOperation] = None
    
    # v1.21: Multi-target file read (Level 2.5)
    # Supports reading N specific named files from different locations
    multi_target_files: List[FileTarget] = field(default_factory=list)
    is_multi_target_read: bool = False
    
    def get_multi_file_summary(self) -> str:
        """Generate human-readable summary for multi-file operations.
        
        Returns empty string if this is not a multi-file operation.
        Used for POT spec display and human review.
        """
        if not self.multi_file or not self.multi_file.is_multi_file:
            return ""
        
        mf = self.multi_file
        lines = [
            f"## Multi-File Operation: {mf.operation_type.upper()}",
            "",
            f"**Pattern:** `{mf.search_pattern}`",
        ]
        
        if mf.operation_type == "refactor" and mf.replacement_pattern:
            lines.append(f"**Replace with:** `{mf.replacement_pattern}`")
        elif mf.operation_type == "refactor":
            lines.append("**Action:** Remove all occurrences")
        
        lines.extend([
            "",
            f"**Scope:** {mf.total_files} files, {mf.total_occurrences} occurrences",
        ])
        
        if mf.roots_searched:
            lines.append(f"**Roots:** {', '.join(mf.roots_searched)}")
        
        if mf.file_filter:
            lines.append(f"**Filter:** {mf.file_filter}")
        
        if mf.discovery_truncated:
            lines.append(f"")
            lines.append(f"⚠️ Results truncated (showing first {len(mf.target_files)} of {mf.total_files} files)")
        
        if mf.file_preview:
            lines.extend([
                "",
                "### File Preview",
                "```",
                mf.file_preview,
                "```",
            ])
        
        if mf.requires_confirmation and not mf.confirmed:
            lines.extend([
                "",
                "⚠️ **This operation modifies files and requires confirmation before execution.**",
            ])
        elif mf.confirmed:
            lines.extend([
                "",
                "✅ **Operation confirmed by user.**",
            ])
        
        if mf.error_message:
            lines.extend([
                "",
                f"❌ **Discovery error:** {mf.error_message}",
            ])
        
        return "\n".join(lines)
    
    def get_multi_target_read_summary(self) -> str:
        """
        v1.21: Generate human-readable summary for multi-target read operations.
        
        Returns empty string if this is not a multi-target read.
        Used for POT spec display and human review.
        """
        if not self.is_multi_target_read or not self.multi_target_files:
            return ""
        
        found_count = sum(1 for f in self.multi_target_files if f.found)
        total_count = len(self.multi_target_files)
        
        lines = [
            f"## Multi-Target File Read (Level 2.5)",
            "",
            f"**Targets:** {found_count}/{total_count} files found",
            "",
        ]
        
        # List each target with its status
        for ft in self.multi_target_files:
            status = "✓" if ft.found else "✗"
            anchor_str = f" ({ft.anchor})" if ft.anchor else ""
            path_str = ft.resolved_path or ft.explicit_path or ft.name
            
            if ft.found and ft.content:
                preview = ft.content[:100].replace('\n', ' ')
                if len(ft.content) > 100:
                    preview += "..."
                lines.append(f"  {status} **{ft.name}**{anchor_str}: `{path_str}`")
                lines.append(f"    > {preview}")
            elif ft.found:
                lines.append(f"  {status} **{ft.name}**{anchor_str}: `{path_str}`")
            else:
                error_str = f" - {ft.error}" if ft.error else ""
                lines.append(f"  {status} **{ft.name}**{anchor_str}: NOT FOUND{error_str}")
        
        return "\n".join(lines)
    
    def get_multi_target_generated_reply(self) -> Optional[str]:
        """
        v1.21: Generate combined reply content from all found multi-target files.
        
        Returns None if no files were found.
        """
        if not self.is_multi_target_read or not self.multi_target_files:
            return None
        
        found_files = [f for f in self.multi_target_files if f.found and f.content]
        if not found_files:
            return None
        
        lines = []
        for ft in found_files:
            path_display = ft.resolved_path or ft.name
            lines.append(f"\n=== {path_display} ===")
            lines.append(ft.content)
        
        return "\n".join(lines)

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


MIN_QUESTIONS = 0

MAX_QUESTIONS = 7

class QuestionCategory(str, Enum):
    """Categories for questions (allowed types)."""
    PREFERENCE = "preference"           # UI style, tone, naming preference
    MISSING_PRODUCT_DECISION = "product_decision"  # New workflow, manual vs auto
    AMBIGUOUS_EVIDENCE = "ambiguous"    # Map says X, code says Y
    SAFETY_CONSTRAINT = "safety"        # Sandbox vs main, backwards compat
    DECISION_FORK = "decision_fork"     # v1.2: Bounded A/B/C product decision

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
    """Multi-file operation specification (v1.20 - Level 3, v2.0 - Intelligent Classification).
    
    Used when SpecGate detects a multi-file intent and runs file discovery
    to build a target file list for human review before execution.
    
    v2.0 (2026-02-01): Added intelligent LLM-powered classification
        - refactor_plan: Full classification with CHANGE/SKIP/FLAG decisions
        - classification_markdown: Human-readable analysis
        - classification_json: Machine-readable structured output
        - confirmation_message: Concise user confirmation prompt
    
    Attributes:
        is_multi_file: Whether this is a multi-file operation
        operation_type: "search" (read-only) or "refactor" (write)
        search_pattern: Pattern to find in files
        replacement_pattern: Replacement text (empty for search operations)
        target_files: List of file paths to process (CHANGE matches only in v2.0)
        total_files: Total number of matching files found
        total_occurrences: Total occurrences across all files
        file_filter: Optional extension filter (e.g., "*.py")
        file_preview: Human-readable preview (classification markdown in v2.0)
        discovery_truncated: True if results were truncated
        discovery_duration_ms: How long discovery took
        roots_searched: Directories that were searched
        requires_confirmation: True for refactor operations (safety)
        confirmed: Whether user has approved the operation
        error_message: Set if discovery encountered issues
        refactor_plan: v2.0 - Full RefactorPlan with classified matches
        classification_markdown: v2.0 - Human-readable classification analysis
        classification_json: v2.0 - Machine-readable structured output
        confirmation_message: v2.0 - Concise confirmation prompt for user
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
    
    # Preview for human review (classification markdown in v2.0)
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
    
    # v2.0: Intelligent classification (LLM-powered analysis)
    # Type hint uses Any to avoid circular import - actual type is RefactorPlan
    refactor_plan: Optional[Any] = None
    classification_markdown: str = ""   # Human-readable analysis
    classification_json: Dict[str, Any] = field(default_factory=dict)  # Machine-readable
    confirmation_message: str = ""      # Concise user prompt
    
    # v3.0: Raw matches for planner-first flow
    raw_matches: List[Any] = field(default_factory=list)  # RawMatch objects from discovery
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/DB storage."""
        result = {
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
            # v2.0: Classification fields
            "classification_markdown": self.classification_markdown,
            "classification_json": self.classification_json,
            "confirmation_message": self.confirmation_message,
        }
        # Serialize refactor_plan if present
        if self.refactor_plan is not None and hasattr(self.refactor_plan, 'to_dict'):
            result["refactor_plan"] = self.refactor_plan.to_dict()
        else:
            result["refactor_plan"] = None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiFileOperation":
        """Deserialize from dictionary."""
        # Note: refactor_plan deserialization requires RefactorPlan.from_dict()
        # which would create a circular import. Leave as dict for now.
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
            # v2.0: Classification fields (refactor_plan left as None/dict)
            refactor_plan=data.get("refactor_plan"),
            classification_markdown=data.get("classification_markdown", ""),
            classification_json=data.get("classification_json", {}),
            confirmation_message=data.get("confirmation_message", ""),
        )
    
    def has_classification(self) -> bool:
        """v2.0: Check if intelligent classification was performed."""
        return self.refactor_plan is not None or bool(self.classification_json)
    
    def get_change_count(self) -> int:
        """v2.0: Get count of matches that will be changed."""
        if self.refactor_plan and hasattr(self.refactor_plan, 'get_change_matches'):
            return len(self.refactor_plan.get_change_matches())
        # Fallback to classification_json
        if self.classification_json:
            summary = self.classification_json.get("refactor_summary", {})
            return summary.get("change_count", self.total_occurrences)
        return self.total_occurrences
    
    def get_skip_count(self) -> int:
        """v2.0: Get count of matches that will be skipped."""
        if self.refactor_plan and hasattr(self.refactor_plan, 'get_skip_matches'):
            return len(self.refactor_plan.get_skip_matches())
        if self.classification_json:
            summary = self.classification_json.get("refactor_summary", {})
            return summary.get("skip_count", 0)
        return 0
    
    def get_flag_count(self) -> int:
        """v2.0: Get count of matches flagged for user attention."""
        if self.refactor_plan and hasattr(self.refactor_plan, 'get_flag_matches'):
            return len(self.refactor_plan.get_flag_matches())
        if self.classification_json:
            summary = self.classification_json.get("refactor_summary", {})
            return summary.get("flag_count", 0)
        return 0

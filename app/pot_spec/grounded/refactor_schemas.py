# FILE: app/pot_spec/grounded/refactor_schemas.py
"""
SpecGate v2.0 - Refactor Classification Schemas

Dataclasses and enums for intelligent refactor analysis.
These schemas support LLM-powered classification with strict schema enforcement.

v2.0 (2026-02-01): Initial implementation
    - MatchBucket enum for categorizing match types
    - ClassifiedMatch for individual match classification
    - RefactorPlan for aggregated refactor decisions
    - RefactorSummary for human-readable output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# ENUMS
# =============================================================================

class MatchBucket(str, Enum):
    """
    Categories for classifying code references.
    
    The LLM assigns each match to one of these buckets based on context.
    """
    # Code structure
    CODE_IDENTIFIER = "code_identifier"      # Variable names, function names, class names
    IMPORT_PATH = "import_path"              # from X import Y, import X
    MODULE_PACKAGE = "module_package"        # Folder/package names in code structure
    
    # Configuration
    ENV_VAR_KEY = "env_var_key"              # ORB_DATABASE_URL, ORB_JOB_ARTIFACT_ROOT
    CONFIG_KEY = "config_key"                # JSON/YAML config keys
    API_ROUTE = "api_route"                  # URL paths, endpoint names
    
    # Files and data
    FILE_FOLDER_NAME = "file_folder_name"    # Actual file/folder naming
    DATABASE_ARTIFACT = "database_artifact"  # .db files, schema names, stored data
    HISTORICAL_DATA = "historical_data"      # Job outputs, logs, cached reports, backups
    
    # Content
    DOCUMENTATION = "documentation"          # Comments, docstrings, README content
    UI_LABEL = "ui_label"                    # User-facing strings, branding
    TEST_ASSERTION = "test_assertion"        # Test strings, expected values
    
    # Fallback
    UNKNOWN = "unknown"                      # Could not classify


class ChangeDecision(str, Enum):
    """
    Decision for each classified match.
    
    The LLM decides whether to CHANGE, SKIP, or FLAG each match.
    """
    CHANGE = "change"   # Safe to modify
    SKIP = "skip"       # Leave alone (with explanation)
    FLAG = "flag"       # Inform user of impact but proceed if approved


class RiskLevel(str, Enum):
    """
    Risk level for each match or bucket.
    """
    LOW = "low"           # No code impact, easy to change
    MEDIUM = "medium"     # May have downstream references
    HIGH = "high"         # External dependencies, careful handling needed
    CRITICAL = "critical" # Data migration required, high chance of breakage


# =============================================================================
# DEFAULT RISK MAPPINGS
# =============================================================================

# Default risk level per bucket (can be overridden by LLM based on context)
DEFAULT_BUCKET_RISKS: Dict[MatchBucket, RiskLevel] = {
    MatchBucket.UI_LABEL: RiskLevel.LOW,
    MatchBucket.DOCUMENTATION: RiskLevel.LOW,
    MatchBucket.TEST_ASSERTION: RiskLevel.LOW,
    MatchBucket.CODE_IDENTIFIER: RiskLevel.MEDIUM,
    MatchBucket.CONFIG_KEY: RiskLevel.MEDIUM,
    MatchBucket.IMPORT_PATH: RiskLevel.HIGH,
    MatchBucket.MODULE_PACKAGE: RiskLevel.HIGH,
    MatchBucket.ENV_VAR_KEY: RiskLevel.HIGH,
    MatchBucket.API_ROUTE: RiskLevel.HIGH,
    MatchBucket.FILE_FOLDER_NAME: RiskLevel.HIGH,
    MatchBucket.DATABASE_ARTIFACT: RiskLevel.CRITICAL,
    MatchBucket.HISTORICAL_DATA: RiskLevel.CRITICAL,
    MatchBucket.UNKNOWN: RiskLevel.MEDIUM,
}

# Default change decision per bucket (LLM can override based on context)
DEFAULT_BUCKET_DECISIONS: Dict[MatchBucket, ChangeDecision] = {
    MatchBucket.UI_LABEL: ChangeDecision.CHANGE,
    MatchBucket.DOCUMENTATION: ChangeDecision.CHANGE,
    MatchBucket.TEST_ASSERTION: ChangeDecision.CHANGE,
    MatchBucket.CODE_IDENTIFIER: ChangeDecision.CHANGE,
    MatchBucket.CONFIG_KEY: ChangeDecision.CHANGE,
    MatchBucket.IMPORT_PATH: ChangeDecision.FLAG,  # Needs import update coordination
    MatchBucket.MODULE_PACKAGE: ChangeDecision.FLAG,  # Cascading import failures
    MatchBucket.ENV_VAR_KEY: ChangeDecision.SKIP,  # Recommend alias approach
    MatchBucket.API_ROUTE: ChangeDecision.FLAG,  # External consumers may depend
    MatchBucket.FILE_FOLDER_NAME: ChangeDecision.FLAG,  # Path references throughout
    MatchBucket.DATABASE_ARTIFACT: ChangeDecision.SKIP,  # Requires data migration
    MatchBucket.HISTORICAL_DATA: ChangeDecision.SKIP,  # No value in changing
    MatchBucket.UNKNOWN: ChangeDecision.FLAG,  # Need human review
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ClassifiedMatch:
    """
    A single match that has been classified by the LLM.
    
    Attributes:
        file_path: Full path to the file
        line_number: Line number in the file (1-indexed)
        line_content: The actual line content
        match_text: The specific text that matched
        bucket: Category this match belongs to
        confidence: Classification confidence (0.0-1.0)
        change_decision: CHANGE, SKIP, or FLAG
        reasoning: Why this decision was made
        risk_level: LOW, MEDIUM, HIGH, or CRITICAL
        impact_note: Optional note about what happens if changed
        migration_hint: Optional hint for how to migrate safely
    """
    file_path: str
    line_number: int
    line_content: str
    match_text: str
    bucket: MatchBucket
    confidence: float = 0.8
    change_decision: ChangeDecision = ChangeDecision.CHANGE
    reasoning: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    impact_note: Optional[str] = None
    migration_hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_text": self.match_text,
            "bucket": self.bucket.value,
            "confidence": self.confidence,
            "change_decision": self.change_decision.value,
            "reasoning": self.reasoning,
            "risk_level": self.risk_level.value,
            "impact_note": self.impact_note,
            "migration_hint": self.migration_hint,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassifiedMatch":
        """Deserialize from dictionary."""
        return cls(
            file_path=data.get("file_path", ""),
            line_number=data.get("line_number", 0),
            line_content=data.get("line_content", ""),
            match_text=data.get("match_text", ""),
            bucket=MatchBucket(data.get("bucket", "unknown")),
            confidence=data.get("confidence", 0.8),
            change_decision=ChangeDecision(data.get("change_decision", "flag")),
            reasoning=data.get("reasoning", ""),
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            impact_note=data.get("impact_note"),
            migration_hint=data.get("migration_hint"),
        )


@dataclass
class BucketSummary:
    """
    Summary of a single bucket's matches.
    
    Aggregates all matches of a particular type for overview reporting.
    """
    bucket: MatchBucket
    total_count: int
    change_count: int
    skip_count: int
    flag_count: int
    risk_level: RiskLevel
    decision: ChangeDecision  # Overall decision for this bucket
    reasoning: str  # Why this overall decision
    sample_files: List[str] = field(default_factory=list)  # Top N files
    sample_lines: List[str] = field(default_factory=list)  # Sample line contents
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bucket": self.bucket.value,
            "total_count": self.total_count,
            "change_count": self.change_count,
            "skip_count": self.skip_count,
            "flag_count": self.flag_count,
            "risk_level": self.risk_level.value,
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "sample_files": self.sample_files,
            "sample_lines": self.sample_lines,
        }


@dataclass
class RefactorFlag:
    """
    A flag to inform the user of an impact.
    
    Flags don't block execution but inform the user what will happen.
    """
    flag_type: str  # e.g., "DATA_IMPACT", "ENV_VAR_ALIAS", "IMPORT_CASCADE"
    message: str
    severity: str  # "INFO", "WARNING", "CAUTION"
    affected_files: List[str] = field(default_factory=list)
    affected_count: int = 0
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "flag_type": self.flag_type,
            "message": self.message,
            "severity": self.severity,
            "affected_files": self.affected_files,
            "affected_count": self.affected_count,
            "recommendation": self.recommendation,
        }


@dataclass
class RefactorPlan:
    """
    Complete refactor plan with all classified matches and decisions.
    
    This is the main output of the classification process, containing
    both human-readable summaries and machine-readable structured data.
    """
    # Core metadata
    search_term: str
    replace_term: str
    total_files: int
    total_occurrences: int
    
    # Classification results
    classified_matches: List[ClassifiedMatch] = field(default_factory=list)
    bucket_summaries: Dict[str, BucketSummary] = field(default_factory=dict)
    
    # Aggregated decisions
    change_count: int = 0
    skip_count: int = 0
    flag_count: int = 0
    
    # Files by decision
    files_to_change: List[str] = field(default_factory=list)
    files_to_skip: List[str] = field(default_factory=list)
    files_to_flag: List[str] = field(default_factory=list)
    
    # Flags for user information
    flags: List[RefactorFlag] = field(default_factory=list)
    
    # Execution phases (recommended order)
    execution_phases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Exclusions (what we're NOT changing and why)
    exclusions: List[Dict[str, str]] = field(default_factory=list)
    
    # Classification metadata
    classification_model: str = ""
    classification_duration_ms: int = 0
    classification_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/DB storage."""
        return {
            "search_term": self.search_term,
            "replace_term": self.replace_term,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "classified_matches": [m.to_dict() for m in self.classified_matches],
            "bucket_summaries": {k: v.to_dict() for k, v in self.bucket_summaries.items()},
            "change_count": self.change_count,
            "skip_count": self.skip_count,
            "flag_count": self.flag_count,
            "files_to_change": self.files_to_change,
            "files_to_skip": self.files_to_skip,
            "files_to_flag": self.files_to_flag,
            "flags": [f.to_dict() for f in self.flags],
            "execution_phases": self.execution_phases,
            "exclusions": self.exclusions,
            "classification_model": self.classification_model,
            "classification_duration_ms": self.classification_duration_ms,
            "classification_timestamp": self.classification_timestamp.isoformat(),
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefactorPlan":
        """Deserialize from dictionary."""
        plan = cls(
            search_term=data.get("search_term", ""),
            replace_term=data.get("replace_term", ""),
            total_files=data.get("total_files", 0),
            total_occurrences=data.get("total_occurrences", 0),
            change_count=data.get("change_count", 0),
            skip_count=data.get("skip_count", 0),
            flag_count=data.get("flag_count", 0),
            files_to_change=data.get("files_to_change", []),
            files_to_skip=data.get("files_to_skip", []),
            files_to_flag=data.get("files_to_flag", []),
            execution_phases=data.get("execution_phases", []),
            exclusions=data.get("exclusions", []),
            classification_model=data.get("classification_model", ""),
            classification_duration_ms=data.get("classification_duration_ms", 0),
            is_valid=data.get("is_valid", True),
            validation_errors=data.get("validation_errors", []),
        )
        
        # Deserialize classified matches
        for m_data in data.get("classified_matches", []):
            plan.classified_matches.append(ClassifiedMatch.from_dict(m_data))
        
        # Deserialize flags
        for f_data in data.get("flags", []):
            plan.flags.append(RefactorFlag(
                flag_type=f_data.get("flag_type", ""),
                message=f_data.get("message", ""),
                severity=f_data.get("severity", "INFO"),
                affected_files=f_data.get("affected_files", []),
                affected_count=f_data.get("affected_count", 0),
                recommendation=f_data.get("recommendation"),
            ))
        
        # Parse timestamp
        ts_str = data.get("classification_timestamp")
        if ts_str:
            try:
                plan.classification_timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        return plan
    
    def get_change_matches(self) -> List[ClassifiedMatch]:
        """Get all matches marked for CHANGE."""
        return [m for m in self.classified_matches if m.change_decision == ChangeDecision.CHANGE]
    
    def get_skip_matches(self) -> List[ClassifiedMatch]:
        """Get all matches marked for SKIP."""
        return [m for m in self.classified_matches if m.change_decision == ChangeDecision.SKIP]
    
    def get_flag_matches(self) -> List[ClassifiedMatch]:
        """Get all matches marked for FLAG."""
        return [m for m in self.classified_matches if m.change_decision == ChangeDecision.FLAG]
    
    def get_matches_by_bucket(self, bucket: MatchBucket) -> List[ClassifiedMatch]:
        """Get all matches for a specific bucket."""
        return [m for m in self.classified_matches if m.bucket == bucket]


# =============================================================================
# RAW MATCH (pre-classification)
# =============================================================================

@dataclass
class RawMatch:
    """
    A raw match from file discovery (before classification).
    
    This is the input to the classification process.
    """
    file_path: str
    line_number: int
    line_content: str
    match_text: str
    file_extension: Optional[str] = None
    relative_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_text": self.match_text,
            "file_extension": self.file_extension,
            "relative_path": self.relative_path,
        }


# =============================================================================
# V3.0 PLANNER-FIRST TYPES (2026-02-01)
# =============================================================================

class RiskClass(str, Enum):
    """
    v3.0: Risk classification for gating decisions.
    
    Unlike RiskLevel (per-match), RiskClass is computed from the ENTIRE plan
    and determines whether SpecGate blocks or proceeds.
    """
    LOW = "low"           # Proceed automatically
    MEDIUM = "medium"     # Proceed automatically
    HIGH = "high"         # Block unless Weaver explicitly allows "auto proceed"
    CRITICAL = "critical" # Always block - requires human review


class ExpansionFailureReason(str, Enum):
    """
    v3.0: Reasons why evidence expansion failed for a match.
    
    When expansion fails, this becomes a blocking issue (not a question).
    """
    READ_FAILED = "read_failed"           # File couldn't be read
    BINARY_FILE = "binary_file"           # Non-text file
    HUGE_FILE = "huge_file"               # File too large to read
    INSUFFICIENT_CONTEXT = "insufficient_context"  # Context doesn't clarify
    PARSE_ERROR = "parse_error"           # Couldn't parse file structure
    ACCESS_DENIED = "access_denied"       # Permission issue
    FILE_NOT_FOUND = "file_not_found"     # File doesn't exist


class ReasonCode(str, Enum):
    """
    v3.0: Structured reason codes for classification decisions.
    
    Provides testable, deterministic reasoning (not free text).
    """
    # CHANGE reasons
    UI_TEXT_VISIBLE_TO_USER = "ui_text_visible_to_user"
    DOCUMENTATION_STRING = "documentation_string"
    COMMENT_TEXT = "comment_text"
    TEST_ASSERTION_VALUE = "test_assertion_value"
    INTERNAL_IDENTIFIER = "internal_identifier"
    
    # PRESERVE (SKIP) reasons
    STORAGE_KEY_IN_USE = "storage_key_in_use"
    ENV_VAR_EXTERNAL_DEPENDENCY = "env_var_external_dependency"
    TOKEN_PREFIX_VALIDATION = "token_prefix_validation"
    DATABASE_ARTIFACT = "database_artifact"
    HISTORICAL_DATA_NO_VALUE = "historical_data_no_value"
    API_KEY_OR_SECRET = "api_key_or_secret"
    
    # FLAG reasons
    PATH_LITERAL_WORKS_NOW = "path_literal_works_now"
    IMPORT_PATH_CASCADE = "import_path_cascade"
    API_ROUTE_EXTERNAL_CONSUMERS = "api_route_external_consumers"
    PACKAGE_NAME_BREAKING = "package_name_breaking"
    
    # Fallback
    UNKNOWN_CONTEXT = "unknown_context"
    EXPANSION_FAILED = "expansion_failed"


@dataclass
class ExpansionResult:
    """
    v3.0: Result of evidence expansion for a match.
    
    Expansion fetches ±N lines of context to enable confident classification.
    If expansion fails, it's a blocking issue, NOT a question.
    """
    attempted: bool
    succeeded: bool
    context: Optional[str] = None
    failure_reason: Optional[ExpansionFailureReason] = None
    lines_fetched: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "context": self.context,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "lines_fetched": self.lines_fetched,
        }


@dataclass
class BlockingIssue:
    """
    v3.0: A blocking issue that prevents proceeding.
    
    Unlike questions, blocking issues don't ask the user - they report
    why the operation cannot proceed.
    """
    file_path: str
    line_number: Optional[int] = None
    reason: str = ""
    what_was_needed: str = ""
    failure_type: Optional[ExpansionFailureReason] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "reason": self.reason,
            "what_was_needed": self.what_was_needed,
            "failure_type": self.failure_type.value if self.failure_type else None,
        }
    
    def __str__(self) -> str:
        loc = f"{self.file_path}:{self.line_number}" if self.line_number else self.file_path
        return f"[{loc}] {self.reason}"


@dataclass
class ClassifiedMatchV3:
    """
    v3.0: Plan-ready classified match with expansion support.
    
    All fields are deterministic and testable. Extends ClassifiedMatch
    with context expansion and structured reason codes.
    """
    # Location
    file_path: str
    line_number: int
    line_content: str
    match_text: str
    
    # Context (if expanded)
    context_snippet: Optional[str] = None
    expansion_attempted: bool = False
    expansion_succeeded: bool = False
    expansion_failure: Optional[ExpansionFailureReason] = None
    
    # Classification
    bucket: MatchBucket = MatchBucket.UNKNOWN
    pre_bucket: Optional[MatchBucket] = None  # Heuristic bucket before LLM
    confidence: float = 0.8
    
    # Decision
    decision: ChangeDecision = ChangeDecision.FLAG
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Reasoning (structured + human-readable)
    reason_code: ReasonCode = ReasonCode.UNKNOWN_CONTEXT
    reason_text: str = ""
    
    # Impact (for FLAG/PRESERVE)
    impact_note: Optional[str] = None
    
    # Unresolved flag
    is_unresolved: bool = False  # True if expansion failed and classification uncertain
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_text": self.match_text,
            "context_snippet": self.context_snippet,
            "expansion_attempted": self.expansion_attempted,
            "expansion_succeeded": self.expansion_succeeded,
            "expansion_failure": self.expansion_failure.value if self.expansion_failure else None,
            "bucket": self.bucket.value,
            "pre_bucket": self.pre_bucket.value if self.pre_bucket else None,
            "confidence": self.confidence,
            "decision": self.decision.value,
            "risk_level": self.risk_level.value,
            "reason_code": self.reason_code.value,
            "reason_text": self.reason_text,
            "impact_note": self.impact_note,
            "is_unresolved": self.is_unresolved,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassifiedMatchV3":
        return cls(
            file_path=data.get("file_path", ""),
            line_number=data.get("line_number", 0),
            line_content=data.get("line_content", ""),
            match_text=data.get("match_text", ""),
            context_snippet=data.get("context_snippet"),
            expansion_attempted=data.get("expansion_attempted", False),
            expansion_succeeded=data.get("expansion_succeeded", False),
            expansion_failure=ExpansionFailureReason(data["expansion_failure"]) if data.get("expansion_failure") else None,
            bucket=MatchBucket(data.get("bucket", "unknown")),
            pre_bucket=MatchBucket(data["pre_bucket"]) if data.get("pre_bucket") else None,
            confidence=data.get("confidence", 0.8),
            decision=ChangeDecision(data.get("decision", "flag")),
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            reason_code=ReasonCode(data.get("reason_code", "unknown_context")),
            reason_text=data.get("reason_text", ""),
            impact_note=data.get("impact_note"),
            is_unresolved=data.get("is_unresolved", False),
        )
    
    def to_v2(self) -> ClassifiedMatch:
        """Convert to v2 ClassifiedMatch for backward compatibility."""
        return ClassifiedMatch(
            file_path=self.file_path,
            line_number=self.line_number,
            line_content=self.line_content,
            match_text=self.match_text,
            bucket=self.bucket,
            confidence=self.confidence,
            change_decision=self.decision,
            reasoning=self.reason_text,
            risk_level=self.risk_level,
            impact_note=self.impact_note,
        )


@dataclass
class RefactorPlanV3:
    """
    v3.0: Complete refactor plan with evidence expansion and risk classification.
    
    This is the planner-first output: a complete, deterministic execution plan
    that exists BEFORE gating decision is made.
    """
    # Core metadata
    search_term: str
    replace_term: str
    total_files: int
    total_occurrences: int
    
    # Classification results (v3 matches)
    classified_matches: List[ClassifiedMatchV3] = field(default_factory=list)
    
    # Expansion tracking
    expansion_failures: List[BlockingIssue] = field(default_factory=list)
    matches_needing_expansion: int = 0
    matches_expanded_successfully: int = 0
    
    # Aggregated decisions
    change_count: int = 0
    skip_count: int = 0
    flag_count: int = 0
    unresolved_count: int = 0  # Matches that couldn't be confidently classified
    
    # Files by decision
    files_to_change: List[str] = field(default_factory=list)
    files_to_skip: List[str] = field(default_factory=list)
    files_to_flag: List[str] = field(default_factory=list)
    
    # Flags for user information
    flags: List[RefactorFlag] = field(default_factory=list)
    
    # Risk assessment
    computed_risk_class: RiskClass = RiskClass.LOW
    risk_factors: List[str] = field(default_factory=list)
    
    # Constraints snapshot (from Weaver)
    constraints_snapshot: Dict[str, Any] = field(default_factory=dict)
    allows_migration: bool = False
    text_only: bool = False
    no_renames: bool = False
    questions_none: bool = False
    
    # Classification metadata
    classification_model: str = ""
    classification_duration_ms: int = 0
    classification_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def has_unresolved_unknowns(self) -> bool:
        """Check if any matches couldn't be confidently classified."""
        return self.unresolved_count > 0 or len(self.expansion_failures) > 0
    
    def has_critical_changes(self) -> bool:
        """Check if any CHANGE decisions are on CRITICAL risk items."""
        for m in self.classified_matches:
            if m.decision == ChangeDecision.CHANGE and m.risk_level == RiskLevel.CRITICAL:
                return True
        return False
    
    def get_blocking_issues(self) -> List[BlockingIssue]:
        """Get all issues that should block the operation."""
        issues = list(self.expansion_failures)
        
        # Add unresolved matches as blocking issues
        for m in self.classified_matches:
            if m.is_unresolved:
                issues.append(BlockingIssue(
                    file_path=m.file_path,
                    line_number=m.line_number,
                    reason=f"Classification uncertain: {m.reason_text}",
                    what_was_needed="Additional context or human review",
                ))
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_term": self.search_term,
            "replace_term": self.replace_term,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "classified_matches": [m.to_dict() for m in self.classified_matches],
            "expansion_failures": [f.to_dict() for f in self.expansion_failures],
            "matches_needing_expansion": self.matches_needing_expansion,
            "matches_expanded_successfully": self.matches_expanded_successfully,
            "change_count": self.change_count,
            "skip_count": self.skip_count,
            "flag_count": self.flag_count,
            "unresolved_count": self.unresolved_count,
            "files_to_change": self.files_to_change,
            "files_to_skip": self.files_to_skip,
            "files_to_flag": self.files_to_flag,
            "flags": [f.to_dict() for f in self.flags],
            "computed_risk_class": self.computed_risk_class.value,
            "risk_factors": self.risk_factors,
            "constraints_snapshot": self.constraints_snapshot,
            "allows_migration": self.allows_migration,
            "text_only": self.text_only,
            "no_renames": self.no_renames,
            "questions_none": self.questions_none,
            "classification_model": self.classification_model,
            "classification_duration_ms": self.classification_duration_ms,
            "classification_timestamp": self.classification_timestamp.isoformat(),
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }
    
    def to_v2(self) -> RefactorPlan:
        """Convert to v2 RefactorPlan for backward compatibility."""
        return RefactorPlan(
            search_term=self.search_term,
            replace_term=self.replace_term,
            total_files=self.total_files,
            total_occurrences=self.total_occurrences,
            classified_matches=[m.to_v2() for m in self.classified_matches],
            change_count=self.change_count,
            skip_count=self.skip_count,
            flag_count=self.flag_count,
            files_to_change=self.files_to_change,
            files_to_skip=self.files_to_skip,
            files_to_flag=self.files_to_flag,
            flags=self.flags,
            classification_model=self.classification_model,
            classification_duration_ms=self.classification_duration_ms,
            classification_timestamp=self.classification_timestamp,
            is_valid=self.is_valid,
            validation_errors=self.validation_errors,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MatchBucket",
    "ChangeDecision",
    "RiskLevel",
    # v3.0 Enums
    "RiskClass",
    "ExpansionFailureReason",
    "ReasonCode",
    # Defaults
    "DEFAULT_BUCKET_RISKS",
    "DEFAULT_BUCKET_DECISIONS",
    # Dataclasses
    "ClassifiedMatch",
    "BucketSummary",
    "RefactorFlag",
    "RefactorPlan",
    "RawMatch",
    # v3.0 Dataclasses
    "ExpansionResult",
    "BlockingIssue",
    "ClassifiedMatchV3",
    "RefactorPlanV3",
]

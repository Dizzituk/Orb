from __future__ import annotations
from app.pot_spec.grounded.refactor_schemas import ChangeDecision, ClassifiedMatchV3, ExpansionFailureReason, MatchBucket, RefactorPlan, RiskLevel
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


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

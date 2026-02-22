from __future__ import annotations
from .refactor_schemas import BucketSummary, ChangeDecision, ClassifiedMatch, MatchBucket, RiskLevel
from .refactor_schemas import ClassifiedMatchV3, ExpansionFailureReason, ExpansionResult, ReasonCode
from collections import defaultdict
from typing import Any, Dict, List, Optional


REFACTOR_CLASSIFIER_BUILD_ID = "2026-02-01-v2.1-vision-context"

def _build_bucket_summaries(matches: List[ClassifiedMatch]) -> Dict[str, BucketSummary]:
    """Build summary statistics for each bucket."""
    summaries = {}
    
    # Group by bucket
    by_bucket: Dict[MatchBucket, List[ClassifiedMatch]] = defaultdict(list)
    for m in matches:
        by_bucket[m.bucket].append(m)
    
    for bucket, bucket_matches in by_bucket.items():
        change_count = sum(1 for m in bucket_matches if m.change_decision == ChangeDecision.CHANGE)
        skip_count = sum(1 for m in bucket_matches if m.change_decision == ChangeDecision.SKIP)
        flag_count = sum(1 for m in bucket_matches if m.change_decision == ChangeDecision.FLAG)
        
        # Determine overall decision based on majority
        if skip_count >= change_count and skip_count >= flag_count:
            decision = ChangeDecision.SKIP
        elif flag_count >= change_count:
            decision = ChangeDecision.FLAG
        else:
            decision = ChangeDecision.CHANGE
        
        # Get risk level (use highest in bucket)
        risk_levels = [m.risk_level for m in bucket_matches]
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        risk = max(risk_levels, key=lambda r: risk_order.index(r))
        
        # Build reasoning
        if decision == ChangeDecision.SKIP:
            reasoning = f"Skipping {len(bucket_matches)} {bucket.value} matches - too risky to change"
        elif decision == ChangeDecision.FLAG:
            reasoning = f"Flagging {len(bucket_matches)} {bucket.value} matches - review recommended"
        else:
            reasoning = f"Safe to change {len(bucket_matches)} {bucket.value} matches"
        
        # Sample files and lines
        sample_files = list(set(m.file_path for m in bucket_matches))[:5]
        sample_lines = [m.line_content[:100] for m in bucket_matches[:3]]
        
        summaries[bucket.value] = BucketSummary(
            bucket=bucket,
            total_count=len(bucket_matches),
            change_count=change_count,
            skip_count=skip_count,
            flag_count=flag_count,
            risk_level=risk,
            decision=decision,
            reasoning=reasoning,
            sample_files=sample_files,
            sample_lines=sample_lines,
        )
    
    return summaries

def _build_execution_phases(
    matches: List[ClassifiedMatch],
    search_term: str,
    replace_term: str,
) -> List[Dict[str, Any]]:
    """Build recommended execution phases for the refactor."""
    phases = []
    
    # Phase 1: Documentation and UI labels (safest)
    safe_buckets = {MatchBucket.DOCUMENTATION, MatchBucket.UI_LABEL, MatchBucket.TEST_ASSERTION}
    safe_matches = [m for m in matches if m.bucket in safe_buckets and m.change_decision == ChangeDecision.CHANGE]
    if safe_matches:
        phases.append({
            "phase": 1,
            "name": "Safe content updates",
            "description": "Update documentation, UI labels, and test assertions",
            "file_count": len(set(m.file_path for m in safe_matches)),
            "occurrence_count": len(safe_matches),
            "risk": "low",
        })
    
    # Phase 2: Code identifiers (medium risk)
    code_buckets = {MatchBucket.CODE_IDENTIFIER, MatchBucket.CONFIG_KEY}
    code_matches = [m for m in matches if m.bucket in code_buckets and m.change_decision == ChangeDecision.CHANGE]
    if code_matches:
        phases.append({
            "phase": 2,
            "name": "Code identifier updates",
            "description": "Update internal variable names and config keys",
            "file_count": len(set(m.file_path for m in code_matches)),
            "occurrence_count": len(code_matches),
            "risk": "medium",
        })
    
    # Phase 3: Import paths (high risk, needs coordination)
    import_matches = [m for m in matches if m.bucket == MatchBucket.IMPORT_PATH and m.change_decision in (ChangeDecision.CHANGE, ChangeDecision.FLAG)]
    if import_matches:
        phases.append({
            "phase": 3,
            "name": "Import path updates",
            "description": "Update import statements (may require restart)",
            "file_count": len(set(m.file_path for m in import_matches)),
            "occurrence_count": len(import_matches),
            "risk": "high",
        })
    
    # Phase 4: Env var aliases (if any flagged)
    env_matches = [m for m in matches if m.bucket == MatchBucket.ENV_VAR_KEY and m.change_decision == ChangeDecision.FLAG]
    if env_matches:
        phases.append({
            "phase": 4,
            "name": "Environment variable aliases",
            "description": f"Add {replace_term}_* aliases with fallback to {search_term}_*",
            "file_count": len(set(m.file_path for m in env_matches)),
            "occurrence_count": len(env_matches),
            "risk": "medium",
        })
    
    return phases

def _build_exclusions(matches: List[ClassifiedMatch]) -> List[Dict[str, str]]:
    """Build list of excluded items with reasons."""
    exclusions = []
    
    skip_matches = [m for m in matches if m.change_decision == ChangeDecision.SKIP]
    
    # Group by bucket
    by_bucket: Dict[MatchBucket, List[ClassifiedMatch]] = defaultdict(list)
    for m in skip_matches:
        by_bucket[m.bucket].append(m)
    
    for bucket, bucket_matches in by_bucket.items():
        # Get a representative reason
        reasons = [m.reasoning for m in bucket_matches if m.reasoning]
        reason = reasons[0] if reasons else "Too risky to change automatically"
        
        exclusions.append({
            "category": bucket.value,
            "count": len(bucket_matches),
            "reason": reason,
            "sample_files": ", ".join(list(set(m.file_path for m in bucket_matches))[:3]),
        })
    
    return exclusions

REFACTOR_CLASSIFIER_V3_BUILD_ID = "2026-02-01-v3.2-text-only-leniency"

EXPANSION_HINT_BUCKETS = {
    MatchBucket.ENV_VAR_KEY,
    MatchBucket.CONFIG_KEY,
    MatchBucket.DATABASE_ARTIFACT,
    MatchBucket.HISTORICAL_DATA,
}

def _read_file_snippet(
    file_path: str,
    line_number: int,
    context_lines: int,
    sandbox_client: Any,
) -> ExpansionResult:
    """
    v3.0: Read ±N lines around a specific line for evidence expansion.
    
    Pure slicing on backend - no new sandbox endpoint needed.
    """
    from .refactor_classifier import _read_file_lines_from_sandbox
    lines = _read_file_lines_from_sandbox(file_path, sandbox_client)
    
    if lines is None:
        return ExpansionResult(
            attempted=True,
            succeeded=False,
            failure_reason=ExpansionFailureReason.READ_FAILED,
        )
    
    if not lines:
        return ExpansionResult(
            attempted=True,
            succeeded=False,
            failure_reason=ExpansionFailureReason.READ_FAILED,
        )
    
    # Calculate range (0-indexed internally)
    total_lines = len(lines)
    target_idx = line_number - 1  # Convert to 0-indexed
    
    if target_idx < 0 or target_idx >= total_lines:
        return ExpansionResult(
            attempted=True,
            succeeded=False,
            failure_reason=ExpansionFailureReason.READ_FAILED,
        )
    
    start_idx = max(0, target_idx - context_lines)
    end_idx = min(total_lines, target_idx + context_lines + 1)
    
    snippet_lines = lines[start_idx:end_idx]
    snippet = "\n".join(snippet_lines)
    
    return ExpansionResult(
        attempted=True,
        succeeded=True,
        context=snippet,
        lines_fetched=len(snippet_lines),
    )

def _convert_to_v3_match(
    v2_match: ClassifiedMatch,
    expansion_attempted: bool = False,
    expansion_succeeded: bool = False,
    context_snippet: Optional[str] = None,
) -> ClassifiedMatchV3:
    """
    v3.0: Convert v2 ClassifiedMatch to v3 ClassifiedMatchV3.
    """
    # Map v2 reasoning to v3 reason code
    reason_code = ReasonCode.UNKNOWN_CONTEXT
    reasoning_lower = (v2_match.reasoning or "").lower()
    
    if "ui" in reasoning_lower or "user-facing" in reasoning_lower or "visible" in reasoning_lower:
        reason_code = ReasonCode.UI_TEXT_VISIBLE_TO_USER
    elif "doc" in reasoning_lower or "comment" in reasoning_lower:
        reason_code = ReasonCode.DOCUMENTATION_STRING
    elif "test" in reasoning_lower:
        reason_code = ReasonCode.TEST_ASSERTION_VALUE
    elif "storage" in reasoning_lower or "localstorage" in reasoning_lower:
        reason_code = ReasonCode.STORAGE_KEY_IN_USE
    elif "env" in reasoning_lower:
        reason_code = ReasonCode.ENV_VAR_EXTERNAL_DEPENDENCY
    elif "token" in reasoning_lower or "prefix" in reasoning_lower:
        reason_code = ReasonCode.TOKEN_PREFIX_VALIDATION
    elif "database" in reasoning_lower or "db" in reasoning_lower:
        reason_code = ReasonCode.DATABASE_ARTIFACT
    elif "historical" in reasoning_lower or "artifact" in reasoning_lower:
        reason_code = ReasonCode.HISTORICAL_DATA_NO_VALUE
    elif "path" in reasoning_lower:
        reason_code = ReasonCode.PATH_LITERAL_WORKS_NOW
    elif "import" in reasoning_lower:
        reason_code = ReasonCode.IMPORT_PATH_CASCADE
    elif "api" in reasoning_lower or "route" in reasoning_lower:
        reason_code = ReasonCode.API_ROUTE_EXTERNAL_CONSUMERS
    elif "identifier" in reasoning_lower:
        reason_code = ReasonCode.INTERNAL_IDENTIFIER
    
    return ClassifiedMatchV3(
        file_path=v2_match.file_path,
        line_number=v2_match.line_number,
        line_content=v2_match.line_content,
        match_text=v2_match.match_text,
        context_snippet=context_snippet,
        expansion_attempted=expansion_attempted,
        expansion_succeeded=expansion_succeeded,
        bucket=v2_match.bucket,
        pre_bucket=None,
        confidence=v2_match.confidence,
        decision=v2_match.change_decision,
        risk_level=v2_match.risk_level,
        reason_code=reason_code,
        reason_text=v2_match.reasoning or "",
        impact_note=v2_match.impact_note,
        is_unresolved=False,
    )

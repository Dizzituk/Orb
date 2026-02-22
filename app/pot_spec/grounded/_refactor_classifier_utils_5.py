from __future__ import annotations
import logging
import re
from .refactor_schemas import BlockingIssue, ExpansionFailureReason, RefactorPlanV3, RiskClass
from .refactor_schemas import ChangeDecision, ClassifiedMatch, DEFAULT_BUCKET_DECISIONS, DEFAULT_BUCKET_RISKS, MatchBucket, RawMatch, RiskLevel
from app.pot_spec.grounded._refactor_classifier_utils import FILE_PATTERN_HINTS, LINE_PATTERN_HINTS, _read_file_lines_from_sandbox
from app.pot_spec.grounded.refactor_classifier import _needs_expansion, logger
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _get_heuristic_hint(match: RawMatch) -> Tuple[Optional[MatchBucket], float]:
    """
    Get a heuristic hint for bucket classification based on file/line patterns.
    
    Returns (bucket_hint, confidence) or (None, 0.0) if no hint.
    This helps the LLM but doesn't override its decision.
    """
    # Check file patterns
    file_path_lower = match.file_path.lower()
    for pattern, bucket in FILE_PATTERN_HINTS.items():
        if re.search(pattern, file_path_lower, re.IGNORECASE):
            return bucket, 0.6  # Medium confidence hint
    
    # Check line patterns
    line = match.line_content.strip()
    for pattern, bucket in LINE_PATTERN_HINTS.items():
        if re.search(pattern, line, re.IGNORECASE):
            return bucket, 0.5  # Lower confidence hint
    
    return None, 0.0

def _apply_heuristic_exclusions(matches: List[RawMatch]) -> Tuple[List[RawMatch], List[ClassifiedMatch]]:
    """
    Apply heuristic exclusions for obvious cases.
    
    Some matches can be pre-classified without LLM:
    - Database files → SKIP (historical data)
    - Backup files → SKIP (historical data)
    - Job output folders → SKIP (historical data)
    
    Returns:
        (matches_for_llm, pre_classified_matches)
    """
    matches_for_llm = []
    pre_classified = []
    
    for match in matches:
        file_lower = match.file_path.lower()
        
        # Skip patterns that are obviously historical/data
        skip_patterns = [
            (r"\.db$", "Database file - requires migration"),
            (r"\.sqlite$", "SQLite database - requires migration"),
            (r"_backup", "Backup file - historical data"),
            (r"\\jobs\\[^\\]+\\", "Job output - historical artifact"),
            (r"\\data\\.*\.json$", "Data file - may contain user data"),
        ]
        
        skipped = False
        for pattern, reason in skip_patterns:
            if re.search(pattern, file_lower, re.IGNORECASE):
                pre_classified.append(ClassifiedMatch(
                    file_path=match.file_path,
                    line_number=match.line_number,
                    line_content=match.line_content,
                    match_text=match.match_text,
                    bucket=MatchBucket.HISTORICAL_DATA,
                    confidence=0.9,
                    change_decision=ChangeDecision.SKIP,
                    reasoning=f"Heuristic: {reason}",
                    risk_level=RiskLevel.CRITICAL,
                    impact_note="Changing historical data has no value and risks corruption",
                ))
                skipped = True
                break
        
        if not skipped:
            matches_for_llm.append(match)
    
    logger.info(
        "[refactor_classifier] Heuristic pre-classification: %d skipped, %d for LLM",
        len(pre_classified), len(matches_for_llm)
    )
    
    return matches_for_llm, pre_classified

def _fallback_heuristic_classification(
    matches: List[RawMatch],
    search_term: str,
    replace_term: str,
) -> List[ClassifiedMatch]:
    """
    Fallback classification using heuristics when LLM is unavailable.
    """
    classified = []
    
    for match in matches:
        hint_bucket, hint_conf = _get_heuristic_hint(match)
        bucket = hint_bucket or MatchBucket.CODE_IDENTIFIER
        
        # Get default decision and risk for this bucket
        decision = DEFAULT_BUCKET_DECISIONS.get(bucket, ChangeDecision.FLAG)
        risk = DEFAULT_BUCKET_RISKS.get(bucket, RiskLevel.MEDIUM)
        
        classified.append(ClassifiedMatch(
            file_path=match.file_path,
            line_number=match.line_number,
            line_content=match.line_content,
            match_text=match.match_text,
            bucket=bucket,
            confidence=hint_conf or 0.5,  # Lower confidence for heuristic
            change_decision=decision,
            reasoning="Heuristic classification (LLM unavailable)",
            risk_level=risk,
        ))
    
    return classified

async def _expand_evidence_batch(
    matches: List[RawMatch],
    sandbox_client: Any,
    context_lines: int = 10,
) -> Tuple[List[RawMatch], List[BlockingIssue]]:
    """
    v3.0: Batch evidence expansion per file to reduce IO.
    
    Groups matches by file, fetches ONE window covering all matches,
    then distributes context to individual matches.
    
    Returns:
        (expanded_matches, expansion_failures)
    """
    from collections import defaultdict
    
    # Group matches by file that need expansion
    matches_by_file: Dict[str, List[RawMatch]] = defaultdict(list)
    no_expansion_needed = []
    
    for match in matches:
        if _needs_expansion(match):
            matches_by_file[match.file_path].append(match)
        else:
            no_expansion_needed.append(match)
    
    expanded = []
    failures = []
    
    for file_path, file_matches in matches_by_file.items():
        # Calculate window that covers all matches in this file
        min_line = min(m.line_number for m in file_matches)
        max_line = max(m.line_number for m in file_matches)
        
        # Fetch once with padding
        context_start = max(1, min_line - context_lines)
        context_end = max_line + context_lines
        
        # Read lines for this file
        lines = _read_file_lines_from_sandbox(file_path, sandbox_client)
        
        if lines is None:
            # All matches in this file failed expansion
            for match in file_matches:
                failures.append(BlockingIssue(
                    file_path=file_path,
                    line_number=match.line_number,
                    reason=f"Evidence expansion failed: could not read file",
                    what_was_needed="File read capability",
                    failure_type=ExpansionFailureReason.READ_FAILED,
                ))
                # Still add match but mark expansion failed
                match.expanded_context = None
                match.expansion_attempted = True
                match.expansion_failed = True
                expanded.append(match)
        else:
            # Extract context for each match
            for match in file_matches:
                target_idx = match.line_number - 1
                start_idx = max(0, target_idx - context_lines)
                end_idx = min(len(lines), target_idx + context_lines + 1)
                
                if 0 <= target_idx < len(lines):
                    context = "\n".join(lines[start_idx:end_idx])
                    match.expanded_context = context
                    match.expansion_attempted = True
                    match.expansion_failed = False
                else:
                    match.expanded_context = None
                    match.expansion_attempted = True
                    match.expansion_failed = True
                    failures.append(BlockingIssue(
                        file_path=file_path,
                        line_number=match.line_number,
                        reason=f"Line {match.line_number} out of range (file has {len(lines)} lines)",
                        what_was_needed="Valid line number",
                        failure_type=ExpansionFailureReason.READ_FAILED,
                    ))
                
                expanded.append(match)
    
    # Add matches that didn't need expansion
    for match in no_expansion_needed:
        match.expanded_context = None
        match.expansion_attempted = False
        match.expansion_failed = False
        expanded.append(match)
    
    logger.info(
        "[refactor_classifier] v3.0 Evidence expansion: %d needed, %d succeeded, %d failed",
        len(matches_by_file),
        len(expanded) - len(no_expansion_needed) - len(failures),
        len(failures),
    )
    
    return expanded, failures

def _compute_risk_class(
    plan: RefactorPlanV3,
    constraints: Dict[str, Any],
) -> RiskClass:
    """
    v3.0: Compute risk class from the final refactor plan.
    v3.2: Be more lenient for text_only jobs with expansion failures.
    
    Block when:
    - Critical change is planned, OR
    - Unknowns remain after expansion (but NOT for text_only jobs), OR
    - Operation violates declared constraints
    
    For text_only jobs:
    - Expansion failures downgraded from CRITICAL to MEDIUM
    - Classified matches already provide sufficient confidence
    """
    risk_factors = []
    
    # Check constraints for leniency
    text_only = constraints.get("text_only", False) or plan.text_only
    questions_none = constraints.get("questions_none", False) or plan.questions_none
    
    # Check for critical-risk decisions (CHANGE on CRITICAL items)
    for match in plan.classified_matches:
        if match.decision == ChangeDecision.CHANGE:
            if match.risk_level == RiskLevel.CRITICAL:
                risk_factors.append(f"Critical change: {match.file_path}:{match.line_number}")
                plan.risk_factors = risk_factors
                return RiskClass.CRITICAL
    
    # Check for unresolved unknowns
    # v3.2: For text_only jobs, unresolved unknowns don't block
    if plan.has_unresolved_unknowns():
        if text_only:
            logger.info(
                "[refactor_classifier] v3.2 Unresolved unknowns (%d) downgraded for text_only job",
                plan.unresolved_count,
            )
            risk_factors.append(f"Unresolved unknowns (text_only leniency): {plan.unresolved_count} matches")
            # Continue checking - don't return CRITICAL
        else:
            risk_factors.append(f"Unresolved unknowns: {plan.unresolved_count} matches")
            plan.risk_factors = risk_factors
            return RiskClass.CRITICAL
    
    # Check for expansion failures
    # v3.2: For text_only jobs, expansion failures don't block
    if plan.expansion_failures:
        if text_only:
            logger.info(
                "[refactor_classifier] v3.2 Expansion failures (%d) downgraded for text_only job",
                len(plan.expansion_failures),
            )
            print(f"[refactor_classifier] v3.2 EXPANSION_FAILURES downgraded (text_only=True): {len(plan.expansion_failures)} failures")
            risk_factors.append(f"Expansion failures (text_only leniency): {len(plan.expansion_failures)} files")
            # Continue checking - don't return CRITICAL
        else:
            risk_factors.append(f"Expansion failures: {len(plan.expansion_failures)} files")
            plan.risk_factors = risk_factors
            return RiskClass.CRITICAL
    
    # Check constraint violations
    no_renames = constraints.get("no_renames", False) or plan.no_renames
    
    # Count high-risk items being changed
    high_count = sum(
        1 for m in plan.classified_matches 
        if m.risk_level == RiskLevel.HIGH and m.decision == ChangeDecision.CHANGE
    )
    
    if high_count > 0:
        risk_factors.append(f"High-risk changes: {high_count}")
        plan.risk_factors = risk_factors
        return RiskClass.HIGH
    
    # Medium risk: many changes or some flags
    if plan.change_count > 100 or plan.flag_count > 10:
        risk_factors.append(f"Large scope: {plan.change_count} changes, {plan.flag_count} flags")
        plan.risk_factors = risk_factors
        return RiskClass.MEDIUM
    
    # v3.2: If text_only had downgraded failures, return MEDIUM instead of LOW
    if text_only and (plan.expansion_failures or plan.has_unresolved_unknowns()):
        risk_factors.append("Text-only job with minor issues")
        plan.risk_factors = risk_factors
        return RiskClass.MEDIUM
    
    plan.risk_factors = risk_factors
    return RiskClass.LOW

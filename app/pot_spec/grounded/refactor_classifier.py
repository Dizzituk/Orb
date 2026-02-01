# FILE: app/pot_spec/grounded/refactor_classifier.py
"""
SpecGate v2.0 - Refactor Match Classifier

LLM-powered classification of refactor matches with strict schema enforcement.
Uses the intelligence of larger models to classify, assess risk, and make decisions.

v2.0 (2026-02-01): Initial implementation
    - classify_matches_batch() for LLM classification
    - build_refactor_plan() for aggregating decisions
    - generate_flags() for impact notifications
    - Structured JSON output with schema validation

v2.1 (2026-02-01): VISION CONTEXT FLOW FIX
    - Added vision_context parameter to _build_classification_prompt()
    - Added vision_context parameter to classify_matches_batch()
    - Added vision_context parameter to build_refactor_plan()
    - Vision context enables intelligent UI vs internal path classification
    - Classifier now knows which matches are USER-VISIBLE UI elements

Design Principles:
- LLM makes the technical decisions (user trusts AI judgment)
- CHANGE: Safe to modify (UI labels, docs, internal identifiers)
- SKIP: Leave alone with explanation (env vars, historical data)
- FLAG: Inform user of impact (imports, packages, routes)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# BUILD VERIFICATION
# =============================================================================
REFACTOR_CLASSIFIER_BUILD_ID = "2026-02-01-v2.1-vision-context"
print(f"[REFACTOR_CLASSIFIER_LOADED] BUILD_ID={REFACTOR_CLASSIFIER_BUILD_ID}")

# =============================================================================
# IMPORTS
# =============================================================================

from .refactor_schemas import (
    MatchBucket,
    ChangeDecision,
    RiskLevel,
    DEFAULT_BUCKET_RISKS,
    DEFAULT_BUCKET_DECISIONS,
    ClassifiedMatch,
    BucketSummary,
    RefactorFlag,
    RefactorPlan,
    RawMatch,
)

# LLM call function import
try:
    from app.providers.registry import llm_call
    _LLM_CALL_AVAILABLE = True
except ImportError as e:
    logger.warning("[refactor_classifier] llm_call not available: %s", e)
    _LLM_CALL_AVAILABLE = False
    llm_call = None


# =============================================================================
# CONSTANTS
# =============================================================================

# Batch size for classification (balance token limits vs API calls)
CLASSIFICATION_BATCH_SIZE = 40

# File patterns for heuristic pre-classification
FILE_PATTERN_HINTS = {
    # Data/artifacts (likely SKIP)
    r"\.db$": MatchBucket.DATABASE_ARTIFACT,
    r"\.sqlite$": MatchBucket.DATABASE_ARTIFACT,
    r"_backup": MatchBucket.HISTORICAL_DATA,
    r"\\jobs\\": MatchBucket.HISTORICAL_DATA,
    r"\\data\\": MatchBucket.HISTORICAL_DATA,
    r"\\logs\\": MatchBucket.HISTORICAL_DATA,
    r"\\cache\\": MatchBucket.HISTORICAL_DATA,
    # Config (likely FLAG)
    r"\.env": MatchBucket.ENV_VAR_KEY,
    r"config\.": MatchBucket.CONFIG_KEY,
    r"settings\.": MatchBucket.CONFIG_KEY,
    # Docs (likely CHANGE)
    r"README": MatchBucket.DOCUMENTATION,
    r"\.md$": MatchBucket.DOCUMENTATION,
    r"CHANGELOG": MatchBucket.DOCUMENTATION,
    # Tests (likely CHANGE)
    r"test_": MatchBucket.TEST_ASSERTION,
    r"_test\.py": MatchBucket.TEST_ASSERTION,
    r"\\tests\\": MatchBucket.TEST_ASSERTION,
}

# Line content patterns for heuristic hints
LINE_PATTERN_HINTS = {
    # Imports
    r"^\s*(?:from|import)\s+": MatchBucket.IMPORT_PATH,
    # Env vars (all caps with underscores)
    r"[A-Z][A-Z0-9_]{3,}\s*=": MatchBucket.ENV_VAR_KEY,
    r"os\.(?:getenv|environ)": MatchBucket.ENV_VAR_KEY,
    # Comments/docs
    r"^\s*#": MatchBucket.DOCUMENTATION,
    r'"""': MatchBucket.DOCUMENTATION,
    r"'''": MatchBucket.DOCUMENTATION,
    # UI strings
    r"(?:title|label|message|text)\s*[=:]": MatchBucket.UI_LABEL,
    r"(?:print|logger\.)": MatchBucket.UI_LABEL,
    # Test assertions
    r"(?:assert|expect|assertEqual)": MatchBucket.TEST_ASSERTION,
    # API routes
    r"@(?:app|router)\.(?:get|post|put|delete)": MatchBucket.API_ROUTE,
    r"(?:path|route)\s*=": MatchBucket.API_ROUTE,
}


# =============================================================================
# HEURISTIC PRE-CLASSIFICATION
# =============================================================================

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


# =============================================================================
# LLM CLASSIFICATION
# =============================================================================

def _build_classification_prompt(
    matches: List[RawMatch],
    search_term: str,
    replace_term: str,
    context: str,
    vision_context: str = "",
) -> str:
    """
    Build the classification prompt for the LLM.
    
    The prompt asks the LLM to classify each match and make a decision.
    
    Args:
        matches: List of raw matches to classify
        search_term: The term being searched for
        replace_term: The replacement term
        context: Additional context about the refactor job
        vision_context: v2.1 - Screenshot/UI analysis from Gemini Vision
    
    v2.1: Added vision_context parameter for intelligent UI classification.
    If vision context is present, the LLM can distinguish between:
    - USER-VISIBLE UI elements (title bars, headings, buttons) → likely CHANGE
    - Internal code paths/identifiers → assess based on risk
    """
    # Build match list with heuristic hints
    match_entries = []
    for i, m in enumerate(matches):
        hint_bucket, hint_conf = _get_heuristic_hint(m)
        hint_str = f" [hint: {hint_bucket.value}]" if hint_bucket else ""
        
        entry = f"""[{i}] {m.file_path}:{m.line_number}{hint_str}
    Line: {m.line_content[:150]}{'...' if len(m.line_content) > 150 else ''}
    Match: "{m.match_text}"
"""
        match_entries.append(entry)
    
    matches_text = "\n".join(match_entries)
    
    # v2.1: Build vision context section if available
    vision_section = ""
    if vision_context:
        vision_section = f"""
## VISION CONTEXT (Screenshot Analysis)
The user uploaded a screenshot. Here's what was visible in the UI:
{vision_context}

Use this to determine which matches are USER-VISIBLE UI elements:
- Text visible in title bars, headings, buttons, labels → bucket: ui_label, decision: CHANGE
- Text in window titles, status bars, menu text → bucket: ui_label, decision: CHANGE  
- Internal code identifiers not shown to user → assess risk as normal
- File paths, storage keys, env vars → appropriate bucket, likely SKIP

IMPORTANT: If vision context mentions specific UI text like "title bar shows '{search_term}'" or "heading says '{search_term}'", 
matches in code that render that UI text should be classified as ui_label with decision CHANGE.
"""
    
    prompt = f"""You are an expert code refactoring analyst. Classify each match and decide whether to CHANGE, SKIP, or FLAG it.
{vision_section}
## Context
We are renaming "{search_term}" to "{replace_term}" across a codebase.
{context}

## Your Task
For each match below, provide:
1. bucket: Category (code_identifier, import_path, module_package, env_var_key, config_key, api_route, file_folder_name, database_artifact, historical_data, documentation, ui_label, test_assertion, unknown)
2. decision: CHANGE (safe to modify), SKIP (leave alone), or FLAG (inform user but proceed if approved)
3. risk: low, medium, high, or critical
4. reasoning: Brief explanation (1 sentence)
5. impact: What happens if changed (optional, for FLAG/SKIP)

## Decision Guidelines
- CHANGE: UI labels, documentation, comments, internal variable names, test assertions
- SKIP: Env var keys (recommend alias), database files (migration needed), historical logs/artifacts (no value)
- FLAG: Import paths (cascade updates needed), package names (breaking change), API routes (external consumers)

## Matches to Classify
{matches_text}

## Response Format
Return ONLY valid JSON array. No markdown, no explanation. Each object must have:
{{"index": 0, "bucket": "...", "decision": "change|skip|flag", "risk": "low|medium|high|critical", "reasoning": "...", "impact": "..."}}

Example:
[
  {{"index": 0, "bucket": "ui_label", "decision": "change", "risk": "low", "reasoning": "User-facing string with no code dependencies"}},
  {{"index": 1, "bucket": "env_var_key", "decision": "skip", "risk": "high", "reasoning": "External systems may read this key", "impact": "Breaking change for .env files"}}
]

Classify all {len(matches)} matches:"""
    
    return prompt


async def classify_matches_batch(
    matches: List[RawMatch],
    search_term: str,
    replace_term: str,
    context: str = "",
    provider_id: str = "openai",
    model_id: str = "gpt-5.2-pro",
    llm_call_func: Optional[Callable] = None,
    vision_context: str = "",
) -> List[ClassifiedMatch]:
    """
    Classify a batch of matches using LLM with structured schema enforcement.
    
    Args:
        matches: List of raw matches to classify
        search_term: The term being searched for
        replace_term: The replacement term
        context: Additional context about the refactor job
        provider_id: LLM provider
        model_id: LLM model
        llm_call_func: Optional LLM call function override
        vision_context: v2.1 - Screenshot/UI analysis from Gemini Vision
    
    Returns:
        List of ClassifiedMatch objects
    
    v2.1: Added vision_context parameter for intelligent UI classification.
    """
    if not matches:
        return []
    
    # Use provided or default LLM call function
    call_func = llm_call_func or llm_call
    if not call_func:
        logger.warning("[refactor_classifier] LLM not available, using heuristic-only classification")
        return _fallback_heuristic_classification(matches, search_term, replace_term)
    
    # Build prompt (v2.1: now includes vision_context)
    prompt = _build_classification_prompt(matches, search_term, replace_term, context, vision_context)
    
    if vision_context:
        logger.info(
            "[refactor_classifier] v2.1 Vision context included in prompt (%d chars)",
            len(vision_context)
        )
    
    try:
        # Call LLM
        logger.info(
            "[refactor_classifier] Calling LLM for %d matches (model=%s)",
            len(matches), model_id
        )
        
        response = await call_func(
            provider_id=provider_id,
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=4000,
        )
        
        # Extract response text
        response_text = ""
        if isinstance(response, dict):
            response_text = response.get("content", "") or response.get("text", "")
        elif hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse JSON response
        classifications = _parse_classification_response(response_text, matches)
        
        logger.info(
            "[refactor_classifier] LLM classified %d/%d matches",
            len(classifications), len(matches)
        )
        
        return classifications
        
    except Exception as e:
        logger.error("[refactor_classifier] LLM classification failed: %s", e)
        return _fallback_heuristic_classification(matches, search_term, replace_term)


def _parse_classification_response(
    response_text: str,
    original_matches: List[RawMatch],
) -> List[ClassifiedMatch]:
    """
    Parse the LLM's JSON response into ClassifiedMatch objects.
    """
    classified = []
    
    # Try to extract JSON array from response
    try:
        # Clean response - remove markdown code blocks if present
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            # Remove markdown code blocks
            clean_text = re.sub(r"```(?:json)?\n?", "", clean_text)
            clean_text = clean_text.strip()
        
        # Parse JSON
        classifications = json.loads(clean_text)
        
        if not isinstance(classifications, list):
            raise ValueError("Response is not a JSON array")
        
        # Map classifications back to matches
        for item in classifications:
            idx = item.get("index", -1)
            if idx < 0 or idx >= len(original_matches):
                continue
            
            match = original_matches[idx]
            
            # Parse bucket
            bucket_str = item.get("bucket", "unknown").lower()
            try:
                bucket = MatchBucket(bucket_str)
            except ValueError:
                bucket = MatchBucket.UNKNOWN
            
            # Parse decision
            decision_str = item.get("decision", "flag").lower()
            try:
                decision = ChangeDecision(decision_str)
            except ValueError:
                decision = ChangeDecision.FLAG
            
            # Parse risk
            risk_str = item.get("risk", "medium").lower()
            try:
                risk = RiskLevel(risk_str)
            except ValueError:
                risk = RiskLevel.MEDIUM
            
            classified.append(ClassifiedMatch(
                file_path=match.file_path,
                line_number=match.line_number,
                line_content=match.line_content,
                match_text=match.match_text,
                bucket=bucket,
                confidence=0.85,  # LLM classification confidence
                change_decision=decision,
                reasoning=item.get("reasoning", ""),
                risk_level=risk,
                impact_note=item.get("impact"),
            ))
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("[refactor_classifier] Failed to parse LLM response: %s", e)
        logger.debug("[refactor_classifier] Raw response: %s", response_text[:500])
        # Fallback to heuristics
        return _fallback_heuristic_classification(original_matches, "", "")
    
    return classified


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


# =============================================================================
# REFACTOR PLAN BUILDING
# =============================================================================

async def build_refactor_plan(
    raw_matches: List[RawMatch],
    search_term: str,
    replace_term: str,
    context: str = "",
    provider_id: str = "openai",
    model_id: str = "gpt-5.2-pro",
    llm_call_func: Optional[Callable] = None,
    vision_context: str = "",
) -> RefactorPlan:
    """
    Build a complete refactor plan from raw matches.
    
    This is the main entry point for refactor classification.
    
    Args:
        raw_matches: List of RawMatch from file discovery
        search_term: The term being searched for
        replace_term: The replacement term
        context: Additional context about the job
        provider_id: LLM provider
        model_id: LLM model
        llm_call_func: Optional LLM call function override
        vision_context: v2.1 - Screenshot/UI analysis from Gemini Vision
    
    Returns:
        RefactorPlan with all classifications and decisions
    
    v2.1: Added vision_context parameter for intelligent UI classification.
    When vision context is present, the classifier can distinguish between
    user-visible UI text (title bars, headings) and internal code paths.
    """
    start_time = time.time()
    
    logger.info(
        "[refactor_classifier] Building refactor plan: %d matches, search='%s', replace='%s', vision_context=%d chars",
        len(raw_matches), search_term, replace_term, len(vision_context)
    )
    
    if vision_context:
        print(f"[refactor_classifier] v2.1 VISION CONTEXT available for classification ({len(vision_context)} chars)")
    
    # Step 1: Apply heuristic exclusions for obvious cases
    matches_for_llm, pre_classified = _apply_heuristic_exclusions(raw_matches)
    
    # Step 2: Classify remaining matches in batches
    all_classified = list(pre_classified)
    
    for i in range(0, len(matches_for_llm), CLASSIFICATION_BATCH_SIZE):
        batch = matches_for_llm[i:i + CLASSIFICATION_BATCH_SIZE]
        batch_classified = await classify_matches_batch(
            matches=batch,
            search_term=search_term,
            replace_term=replace_term,
            context=context,
            provider_id=provider_id,
            model_id=model_id,
            llm_call_func=llm_call_func,
            vision_context=vision_context,  # v2.1: Pass vision context
        )
        all_classified.extend(batch_classified)
    
    # Step 3: Build bucket summaries
    bucket_summaries = _build_bucket_summaries(all_classified)
    
    # Step 4: Generate flags
    flags = _generate_flags(all_classified, search_term, replace_term)
    
    # Step 5: Build execution phases
    execution_phases = _build_execution_phases(all_classified, search_term, replace_term)
    
    # Step 6: Build exclusions list
    exclusions = _build_exclusions(all_classified)
    
    # Step 7: Aggregate counts
    change_count = sum(1 for m in all_classified if m.change_decision == ChangeDecision.CHANGE)
    skip_count = sum(1 for m in all_classified if m.change_decision == ChangeDecision.SKIP)
    flag_count = sum(1 for m in all_classified if m.change_decision == ChangeDecision.FLAG)
    
    # Get unique file lists
    files_to_change = list(set(m.file_path for m in all_classified if m.change_decision == ChangeDecision.CHANGE))
    files_to_skip = list(set(m.file_path for m in all_classified if m.change_decision == ChangeDecision.SKIP))
    files_to_flag = list(set(m.file_path for m in all_classified if m.change_decision == ChangeDecision.FLAG))
    
    # Build plan
    duration_ms = int((time.time() - start_time) * 1000)
    
    plan = RefactorPlan(
        search_term=search_term,
        replace_term=replace_term,
        total_files=len(set(m.file_path for m in raw_matches)),
        total_occurrences=len(raw_matches),
        classified_matches=all_classified,
        bucket_summaries=bucket_summaries,
        change_count=change_count,
        skip_count=skip_count,
        flag_count=flag_count,
        files_to_change=files_to_change,
        files_to_skip=files_to_skip,
        files_to_flag=files_to_flag,
        flags=flags,
        execution_phases=execution_phases,
        exclusions=exclusions,
        classification_model=f"{provider_id}/{model_id}",
        classification_duration_ms=duration_ms,
    )
    
    logger.info(
        "[refactor_classifier] Plan built in %dms: change=%d, skip=%d, flag=%d",
        duration_ms, change_count, skip_count, flag_count
    )
    
    return plan


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


def _generate_flags(
    matches: List[ClassifiedMatch],
    search_term: str,
    replace_term: str,
) -> List[RefactorFlag]:
    """Generate informational flags for the user."""
    flags = []
    
    # Check for env var changes
    env_matches = [m for m in matches if m.bucket == MatchBucket.ENV_VAR_KEY]
    if env_matches:
        skip_count = sum(1 for m in env_matches if m.change_decision == ChangeDecision.SKIP)
        if skip_count > 0:
            flags.append(RefactorFlag(
                flag_type="ENV_VAR_ALIAS",
                message=f"Found {len(env_matches)} environment variable keys. Recommend alias approach: introduce {replace_term}_* while keeping {search_term}_* as fallback.",
                severity="WARNING",
                affected_files=list(set(m.file_path for m in env_matches))[:5],
                affected_count=len(env_matches),
                recommendation=f"Add fallback logic: read {replace_term}_* first, fall back to {search_term}_*",
            ))
    
    # Check for import path changes
    import_matches = [m for m in matches if m.bucket == MatchBucket.IMPORT_PATH]
    if len(import_matches) > 10:
        flags.append(RefactorFlag(
            flag_type="IMPORT_CASCADE",
            message=f"Found {len(import_matches)} import statements. These will be updated as part of the refactor.",
            severity="INFO",
            affected_files=list(set(m.file_path for m in import_matches))[:10],
            affected_count=len(import_matches),
        ))
    
    # Check for database/historical data
    data_matches = [m for m in matches if m.bucket in (MatchBucket.DATABASE_ARTIFACT, MatchBucket.HISTORICAL_DATA)]
    if data_matches:
        skip_count = sum(1 for m in data_matches if m.change_decision == ChangeDecision.SKIP)
        if skip_count > 0:
            flags.append(RefactorFlag(
                flag_type="DATA_EXCLUDED",
                message=f"Excluding {skip_count} matches in database files, backups, and job artifacts. These contain historical data and won't be modified.",
                severity="INFO",
                affected_files=list(set(m.file_path for m in data_matches if m.change_decision == ChangeDecision.SKIP))[:5],
                affected_count=skip_count,
            ))
    
    # Check for API route changes
    route_matches = [m for m in matches if m.bucket == MatchBucket.API_ROUTE]
    if route_matches:
        flags.append(RefactorFlag(
            flag_type="API_ROUTES",
            message=f"Found {len(route_matches)} API route definitions. External consumers may need to be updated.",
            severity="CAUTION",
            affected_files=list(set(m.file_path for m in route_matches))[:5],
            affected_count=len(route_matches),
            recommendation="Consider adding route aliases for backward compatibility",
        ))
    
    return flags


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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "classify_matches_batch",
    "build_refactor_plan",
    "CLASSIFICATION_BATCH_SIZE",
]

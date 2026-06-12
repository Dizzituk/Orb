# FILE: app/pot_spec/grounded/refactor_classifier.py
# Purpose: SpecGate v2.0 - Refactor Match Classifier
# Called-by: app.pot_spec.grounded._multi_file_detection_utils_2, app.pot_spec.grounded._refactor_classifier_utils_6, app.pot_spec.grounded._refactor_classifier_utils_7, app.pot_spec.grounded._refactor_classifier_utils_8 (+1 more)
# Depends-on: app.pot_spec.grounded._refactor_classifier_utils_6, app.pot_spec.grounded._refactor_classifier_utils_7, app.pot_spec.grounded._refactor_classifier_utils_8, app.pot_spec.grounded.refactor_schemas (+1 more)
# Last-renovated: 2026-06-11
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

import logging
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.pot_spec.grounded._refactor_classifier_utils_6 import EXPANSION_HINT_BUCKETS, REFACTOR_CLASSIFIER_BUILD_ID, REFACTOR_CLASSIFIER_V3_BUILD_ID, _build_bucket_summaries, _build_exclusions, _build_execution_phases, _convert_to_v3_match, _read_file_snippet
from app.pot_spec.grounded._refactor_classifier_utils_7 import EXPANSION_NEEDED_PATTERNS, FILE_PATTERN_HINTS, LINE_PATTERN_HINTS, _build_classification_prompt, _generate_flags, _parse_classification_response, _read_file_lines_from_sandbox
from app.pot_spec.grounded._refactor_classifier_utils_8 import _apply_heuristic_exclusions, _compute_risk_class, _expand_evidence_batch, _fallback_heuristic_classification, _get_heuristic_hint

logger = logging.getLogger(__name__)

# =============================================================================
# BUILD VERIFICATION
# =============================================================================
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

# Line content patterns for heuristic hints


# =============================================================================
# HEURISTIC PRE-CLASSIFICATION
# =============================================================================


# =============================================================================
# LLM CLASSIFICATION
# =============================================================================


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


# =============================================================================
# V3.0 PLANNER-FIRST CLASSIFICATION (2026-02-01)
# =============================================================================

# Build ID for v3 verification

# Import v3 types
from .refactor_schemas import (
    RiskClass,
    ExpansionFailureReason,
    ReasonCode,
    ExpansionResult,
    BlockingIssue,
    ClassifiedMatchV3,
    RefactorPlanV3,
)

# Patterns that indicate a match NEEDS evidence expansion
# (Can't confidently classify from single line alone)

# Buckets that typically need expansion for confident classification


def _needs_expansion(match: RawMatch) -> bool:
    """
    v3.0: Determine if a match needs evidence expansion for confident classification.
    
    Returns True if:
    - Line contains patterns suggesting operational use (keys, tokens, env vars)
    - Heuristic bucket suggests potentially risky category
    - Single line doesn't provide enough context
    """
    line = match.line_content.strip()
    
    # Check expansion-needed patterns
    for pattern in EXPANSION_NEEDED_PATTERNS.keys():
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    # Check heuristic bucket
    hint_bucket, _ = _get_heuristic_hint(match)
    if hint_bucket in EXPANSION_HINT_BUCKETS:
        return True
    
    return False


async def build_refactor_plan_v3(
    raw_matches: List[RawMatch],
    search_term: str,
    replace_term: str,
    context: str = "",
    constraints: Optional[Dict[str, Any]] = None,
    sandbox_client: Any = None,
    provider_id: str = "openai",
    model_id: str = "gpt-5.2-pro",
    llm_call_func: Optional[Callable] = None,
    vision_context: str = "",
) -> RefactorPlanV3:
    """
    v3.0: Build a complete refactor plan with evidence expansion.
    
    This is the planner-first entry point:
    1. Scan → matches
    2. Expansion → context for ambiguous matches
    3. Classify → decisions (v2 classifier with expansion context)
    4. Build v3 plan with risk assessment
    
    PLAN GENERATION HAPPENS BEFORE GATING.
    """
    start_time = time.time()
    constraints = constraints or {}
    
    logger.info(
        "[refactor_classifier] v3.0 Building planner-first refactor plan: %d matches",
        len(raw_matches),
    )
    print(f"[refactor_classifier] v3.0 PLANNER-FIRST: {len(raw_matches)} matches")
    
    # STEP 1: Evidence expansion for matches that need it
    expanded_matches, expansion_failures = await _expand_evidence_batch(
        matches=raw_matches,
        sandbox_client=sandbox_client,
        context_lines=10,
    )
    
    matches_needing_expansion = sum(1 for m in raw_matches if _needs_expansion(m))
    matches_expanded_ok = matches_needing_expansion - len(expansion_failures)
    
    # STEP 2: Apply heuristic exclusions
    matches_for_llm, pre_classified = _apply_heuristic_exclusions(expanded_matches)
    
    # STEP 3: LLM classification (with expanded context in prompt)
    all_v2_classified = list(pre_classified)
    
    for i in range(0, len(matches_for_llm), CLASSIFICATION_BATCH_SIZE):
        batch = matches_for_llm[i:i + CLASSIFICATION_BATCH_SIZE]
        
        # Build context including expansion info
        expanded_context = context
        for m in batch:
            if hasattr(m, 'expanded_context') and m.expanded_context:
                expanded_context += f"\n\nExpanded context for {m.file_path}:{m.line_number}:\n{m.expanded_context[:500]}"
        
        batch_classified = await classify_matches_batch(
            matches=batch,
            search_term=search_term,
            replace_term=replace_term,
            context=expanded_context[:4000],  # Limit total context size
            provider_id=provider_id,
            model_id=model_id,
            llm_call_func=llm_call_func,
            vision_context=vision_context,
        )
        all_v2_classified.extend(batch_classified)
    
    # STEP 4: Convert to v3 matches
    v3_matches = []
    for v2_match in all_v2_classified:
        # Find corresponding raw match for expansion info
        raw = next(
            (m for m in expanded_matches 
             if m.file_path == v2_match.file_path and m.line_number == v2_match.line_number),
            None
        )
        
        expansion_attempted = hasattr(raw, 'expansion_attempted') and raw.expansion_attempted if raw else False
        expansion_succeeded = hasattr(raw, 'expansion_failed') and not raw.expansion_failed if raw else False
        context_snippet = raw.expanded_context if raw and hasattr(raw, 'expanded_context') else None
        
        v3_match = _convert_to_v3_match(
            v2_match,
            expansion_attempted=expansion_attempted,
            expansion_succeeded=expansion_succeeded,
            context_snippet=context_snippet,
        )
        
        # Mark as unresolved if expansion was needed but failed
        if expansion_attempted and not expansion_succeeded:
            if _needs_expansion(raw) if raw else False:
                v3_match.is_unresolved = True
                v3_match.reason_code = ReasonCode.EXPANSION_FAILED
        
        v3_matches.append(v3_match)
    
    # STEP 5: Build v3 plan
    change_count = sum(1 for m in v3_matches if m.decision == ChangeDecision.CHANGE)
    skip_count = sum(1 for m in v3_matches if m.decision == ChangeDecision.SKIP)
    flag_count = sum(1 for m in v3_matches if m.decision == ChangeDecision.FLAG)
    unresolved_count = sum(1 for m in v3_matches if m.is_unresolved)
    
    files_to_change = list(set(m.file_path for m in v3_matches if m.decision == ChangeDecision.CHANGE))
    files_to_skip = list(set(m.file_path for m in v3_matches if m.decision == ChangeDecision.SKIP))
    files_to_flag = list(set(m.file_path for m in v3_matches if m.decision == ChangeDecision.FLAG))
    
    # Build flags
    flags = _generate_flags(
        [m.to_v2() for m in v3_matches],
        search_term,
        replace_term,
    )
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    plan = RefactorPlanV3(
        search_term=search_term,
        replace_term=replace_term,
        total_files=len(set(m.file_path for m in raw_matches)),
        total_occurrences=len(raw_matches),
        classified_matches=v3_matches,
        expansion_failures=expansion_failures,
        matches_needing_expansion=matches_needing_expansion,
        matches_expanded_successfully=matches_expanded_ok,
        change_count=change_count,
        skip_count=skip_count,
        flag_count=flag_count,
        unresolved_count=unresolved_count,
        files_to_change=files_to_change,
        files_to_skip=files_to_skip,
        files_to_flag=files_to_flag,
        flags=flags,
        constraints_snapshot=constraints,
        allows_migration=constraints.get("allows_migration", False),
        text_only=constraints.get("text_only", False),
        no_renames=constraints.get("no_renames", False),
        questions_none=constraints.get("questions_none", False),
        classification_model=f"{provider_id}/{model_id}",
        classification_duration_ms=duration_ms,
    )
    
    # STEP 6: Compute risk class
    plan.computed_risk_class = _compute_risk_class(plan, constraints)
    
    logger.info(
        "[refactor_classifier] v3.0 Plan built in %dms: change=%d, skip=%d, flag=%d, unresolved=%d, risk=%s",
        duration_ms, change_count, skip_count, flag_count, unresolved_count, plan.computed_risk_class.value,
    )
    print(f"[refactor_classifier] v3.0 PLAN COMPLETE: risk={plan.computed_risk_class.value}")
    
    return plan


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "classify_matches_batch",
    "build_refactor_plan",
    "CLASSIFICATION_BATCH_SIZE",
    # v3.0 exports
    "build_refactor_plan_v3",
    "REFACTOR_CLASSIFIER_V3_BUILD_ID",
    "_needs_expansion",
    "_compute_risk_class",
]

# FILE: app/experience/experience_store.py
# Purpose: CRUD operations for Experience Patterns.
# Called-by: app.experience.distillation, app.experience.retrieval
# Depends-on: app.experience.models
# Last-renovated: 2026-06-11
"""
CRUD operations for Experience Patterns.

Handles:
- create_pattern: Store a new pattern from distillation
- validate_pattern: Increase confidence when a pattern is confirmed
- contradict_pattern: Decrease confidence when a pattern fails
- record_injection: Track when a pattern is injected into a prompt
- record_usefulness: Track whether injection led to success
- query_patterns: Stage 1 indexed filter for retrieval
- get_pattern_by_error_sig: Exact-match lookup for boot fix patterns
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .models import ExperiencePattern

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIDENCE CONSTANTS (Section 3.3)
# =============================================================================

# Initial confidence for new patterns
INITIAL_CONFIDENCE_ERROR_FIX = 0.6      # Error→fix pairs start medium
INITIAL_CONFIDENCE_STRATEGY = 0.5       # Strategy patterns start lower
INITIAL_CONFIDENCE_ARCHITECTURE = 0.5   # Architecture decisions start medium
INITIAL_CONFIDENCE_FALSE_POSITIVE = 0.7 # Known false positives start high (observed)

# Confidence adjustments
VALIDATION_BOOST = 0.1                  # +0.1 per validation (capped at 1.0)
CONTRADICTION_PENALTY = 0.15            # -0.15 per contradiction
INJECTION_SUCCESS_BOOST = 0.05          # +0.05 when injection led to success
MONTHLY_DECAY = 0.02                    # -0.02 per month of non-use

# Thresholds
CONFIDENCE_SKIP_THRESHOLD = 0.3         # Below this: never inject
CONFIDENCE_DEMOTE_THRESHOLD = 0.15      # Below this: pattern is effectively dead


# =============================================================================
# CREATE
# =============================================================================

def create_pattern(
    db: Session,
    *,
    category: str,
    stage: str,
    description: str,
    root_cause: str = "",
    resolution: str = "",
    job_type: Optional[str] = None,
    language: Optional[str] = None,
    error_signature: Optional[str] = None,
    file_scope: Optional[str] = None,
    source_job_id: Optional[str] = None,
    source_journal_index: Optional[int] = None,
    initial_confidence: Optional[float] = None,
) -> ExperiencePattern:
    """
    Create a new experience pattern.

    If initial_confidence is not provided, it's inferred from category.
    """
    if initial_confidence is None:
        initial_confidence = {
            "error_fix": INITIAL_CONFIDENCE_ERROR_FIX,
            "boot_resolution": INITIAL_CONFIDENCE_ERROR_FIX,
            "implementation_strategy": INITIAL_CONFIDENCE_STRATEGY,
            "architecture_decision": INITIAL_CONFIDENCE_ARCHITECTURE,
            "false_positive": INITIAL_CONFIDENCE_FALSE_POSITIVE,
            "performance_pattern": INITIAL_CONFIDENCE_STRATEGY,
        }.get(category, 0.5)

    pattern = ExperiencePattern(
        category=category,
        stage=stage,
        description=description,
        root_cause=root_cause,
        resolution=resolution,
        job_type=job_type,
        language=language,
        error_signature=error_signature,
        file_scope=file_scope,
        source_job_id=source_job_id,
        source_journal_index=source_journal_index,
        confidence=initial_confidence,
    )

    db.add(pattern)
    db.flush()

    logger.info(
        f"[experience] Created pattern #{pattern.id}: "
        f"category={category}, stage={stage}, confidence={initial_confidence:.2f}"
    )
    return pattern


# =============================================================================
# VALIDATE / CONTRADICT
# =============================================================================

def validate_pattern(db: Session, pattern_id: int) -> Optional[ExperiencePattern]:
    """
    Increase confidence when a pattern is confirmed by a subsequent job.
    Caps at 1.0.
    """
    pattern = db.query(ExperiencePattern).get(pattern_id)
    if not pattern:
        return None

    pattern.times_validated += 1
    pattern.confidence = min(1.0, pattern.confidence + VALIDATION_BOOST)
    pattern.last_validated_at = datetime.now(timezone.utc)
    pattern.updated_at = datetime.now(timezone.utc)

    db.flush()
    logger.info(
        f"[experience] Validated pattern #{pattern_id}: "
        f"confidence now {pattern.confidence:.2f} "
        f"(validated {pattern.times_validated}x)"
    )
    return pattern


def contradict_pattern(db: Session, pattern_id: int) -> Optional[ExperiencePattern]:
    """
    Decrease confidence when a pattern is contradicted.
    Does not delete — patterns below threshold are just never injected.
    """
    pattern = db.query(ExperiencePattern).get(pattern_id)
    if not pattern:
        return None

    pattern.times_contradicted += 1
    pattern.confidence = max(0.0, pattern.confidence - CONTRADICTION_PENALTY)
    pattern.updated_at = datetime.now(timezone.utc)

    db.flush()
    logger.info(
        f"[experience] Contradicted pattern #{pattern_id}: "
        f"confidence now {pattern.confidence:.2f} "
        f"(contradicted {pattern.times_contradicted}x)"
    )
    return pattern


# =============================================================================
# INJECTION TRACKING
# =============================================================================

def record_injection(db: Session, pattern_id: int) -> None:
    """Record that a pattern was injected into a pipeline prompt."""
    pattern = db.query(ExperiencePattern).get(pattern_id)
    if pattern:
        pattern.times_injected += 1
        pattern.last_injected_at = datetime.now(timezone.utc)
        pattern.updated_at = datetime.now(timezone.utc)
        db.flush()


def record_usefulness(db: Session, pattern_id: int, was_useful: bool) -> None:
    """
    Record whether an injected pattern led to a successful outcome.
    Adjusts confidence accordingly.
    """
    pattern = db.query(ExperiencePattern).get(pattern_id)
    if not pattern:
        return

    if was_useful:
        pattern.times_useful += 1
        pattern.confidence = min(1.0, pattern.confidence + INJECTION_SUCCESS_BOOST)
    else:
        pattern.confidence = max(0.0, pattern.confidence - INJECTION_SUCCESS_BOOST)

    pattern.updated_at = datetime.now(timezone.utc)
    db.flush()


# =============================================================================
# QUERY (Stage 1 — Indexed Filter)
# =============================================================================

def query_patterns(
    db: Session,
    *,
    stage: Optional[str] = None,
    stages: Optional[List[str]] = None,
    category: Optional[str] = None,
    categories: Optional[List[str]] = None,
    job_type: Optional[str] = None,
    language: Optional[str] = None,
    error_signature: Optional[str] = None,
    min_confidence: float = CONFIDENCE_SKIP_THRESHOLD,
    limit: int = 50,
) -> List[ExperiencePattern]:
    """
    Stage 1 retrieval: indexed filter query.

    Returns patterns matching the filter criteria, ordered by confidence desc.
    This is the fast pre-filter before semantic ranking (Stage 2).

    Args:
        stage: Single stage filter (e.g. "critical_pipeline")
        stages: Multiple stage filter (OR logic)
        category: Single category filter
        categories: Multiple category filter (OR logic)
        job_type: Filter by job type. Also includes universal patterns (job_type=NULL).
        language: Filter by language. Also includes language-agnostic (language=NULL).
        error_signature: Exact-match for boot error patterns
        min_confidence: Minimum confidence threshold (default: 0.3 = SKIP threshold)
        limit: Max results (default: 50 — Stage 2 will narrow further)
    """
    query = db.query(ExperiencePattern).filter(
        ExperiencePattern.confidence >= min_confidence
    )

    # Stage filter
    if stage:
        query = query.filter(ExperiencePattern.stage == stage)
    elif stages:
        query = query.filter(ExperiencePattern.stage.in_(stages))

    # Category filter
    if category:
        query = query.filter(ExperiencePattern.category == category)
    elif categories:
        query = query.filter(ExperiencePattern.category.in_(categories))

    # Job type: include matching + universal (NULL)
    if job_type:
        query = query.filter(
            (ExperiencePattern.job_type == job_type) |
            (ExperiencePattern.job_type.is_(None))
        )

    # Language: include matching + agnostic (NULL)
    if language:
        query = query.filter(
            (ExperiencePattern.language == language) |
            (ExperiencePattern.language.is_(None))
        )

    # Error signature: exact match
    if error_signature:
        query = query.filter(
            ExperiencePattern.error_signature == error_signature
        )

    # Order by confidence desc, limit
    return query.order_by(
        ExperiencePattern.confidence.desc()
    ).limit(limit).all()


def get_pattern_by_error_sig(
    db: Session,
    error_signature: str,
    min_confidence: float = CONFIDENCE_SKIP_THRESHOLD,
) -> Optional[ExperiencePattern]:
    """
    Exact-match lookup by error signature.
    Used by Phase Checkout for boot fix patterns.
    Returns the highest-confidence match.
    """
    return db.query(ExperiencePattern).filter(
        ExperiencePattern.error_signature == error_signature,
        ExperiencePattern.confidence >= min_confidence,
    ).order_by(
        ExperiencePattern.confidence.desc()
    ).first()


def find_similar_pattern(
    db: Session,
    *,
    category: str,
    stage: str,
    description: str,
    error_signature: Optional[str] = None,
) -> Optional[ExperiencePattern]:
    """
    Find an existing pattern that matches a candidate (for deduplication).

    During distillation, before creating a new pattern, check if a similar
    one already exists. If so, validate it instead of creating a duplicate.

    Match criteria (in priority order):
    1. Exact error_signature match (if provided)
    2. Same category + stage + first 100 chars of description
    """
    # Try exact error signature match first
    if error_signature:
        match = db.query(ExperiencePattern).filter(
            ExperiencePattern.error_signature == error_signature,
            ExperiencePattern.category == category,
        ).first()
        if match:
            return match

    # Fall back to category + stage + description prefix
    desc_prefix = (description or "")[:100]
    if desc_prefix:
        matches = db.query(ExperiencePattern).filter(
            ExperiencePattern.category == category,
            ExperiencePattern.stage == stage,
            ExperiencePattern.description.like(f"{desc_prefix}%"),
        ).all()
        if matches:
            return matches[0]

    return None


# =============================================================================
# MAINTENANCE
# =============================================================================

def apply_monthly_decay(db: Session) -> Dict[str, int]:
    """
    Apply monthly decay to all patterns.
    Patterns below CONFIDENCE_DEMOTE_THRESHOLD are effectively dead
    but kept for deduplication.

    Returns stats: {"decayed": N, "demoted": N}
    """
    patterns = db.query(ExperiencePattern).filter(
        ExperiencePattern.confidence > 0.0
    ).all()

    decayed = 0
    demoted = 0

    for p in patterns:
        old_conf = p.confidence
        p.confidence = max(0.0, p.confidence - MONTHLY_DECAY)
        p.updated_at = datetime.now(timezone.utc)
        decayed += 1

        if old_conf >= CONFIDENCE_DEMOTE_THRESHOLD and p.confidence < CONFIDENCE_DEMOTE_THRESHOLD:
            demoted += 1

    db.commit()

    logger.info(
        f"[experience] Monthly decay applied: "
        f"{decayed} patterns decayed, {demoted} newly demoted"
    )
    return {"decayed": decayed, "demoted": demoted}


def get_pattern_stats(db: Session) -> Dict[str, Any]:
    """Get summary statistics for the experience database."""
    from sqlalchemy import func

    total = db.query(func.count(ExperiencePattern.id)).scalar() or 0
    injectable = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.confidence >= CONFIDENCE_SKIP_THRESHOLD
    ).scalar() or 0
    embedded = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.embedded == True
    ).scalar() or 0
    avg_confidence = db.query(func.avg(ExperiencePattern.confidence)).scalar() or 0.0

    by_category = dict(
        db.query(
            ExperiencePattern.category,
            func.count(ExperiencePattern.id)
        ).group_by(ExperiencePattern.category).all()
    )

    by_stage = dict(
        db.query(
            ExperiencePattern.stage,
            func.count(ExperiencePattern.id)
        ).group_by(ExperiencePattern.stage).all()
    )

    return {
        "total_patterns": total,
        "injectable_patterns": injectable,
        "embedded_patterns": embedded,
        "avg_confidence": round(avg_confidence, 3),
        "by_category": by_category,
        "by_stage": by_stage,
    }

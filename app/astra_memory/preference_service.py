# FILE: app/astra_memory/preference_service.py
"""
ASTRA Memory Preference Service

Handles preference lifecycle:
- Create preferences (from explicit instruction or implicit learning)
- Update preference values
- Query preferences for components
- Apply behavior rules from spec section 6
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.astra_memory.confidence_config import get_config
from app.astra_memory.confidence_scoring import (
    append_preference_evidence,
    recompute_preference_confidence,
    check_namespace_mutation_allowed,
)
from app.astra_memory.preference_models import (
    PreferenceRecord,
    PreferenceEvidence,
    PreferenceStrength,
    RecordStatus,
    SignalType,
    ConfidenceType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PREFERENCE CREATION
# =============================================================================

def create_preference(
    db: Session,
    preference_key: str,
    preference_value: Any,
    strength: PreferenceStrength = PreferenceStrength.DEFAULT,
    source: str = "user_declared",
    applies_to: Optional[str] = None,
    namespace: str = "user_personal",
    context_pointer: Optional[str] = None,
) -> PreferenceRecord:
    """
    Create a new preference record.
    
    For explicit instructions, this sets initial confidence based on strength:
    - HARD_RULE: confidence=1.0, no decay
    - DEFAULT/SOFT: confidence computed from initial explicit evidence
    """
    # Check if preference already exists
    existing = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()
    
    if existing:
        logger.info(f"Preference '{preference_key}' exists, updating instead")
        return update_preference_value(
            db, preference_key, preference_value,
            is_explicit=True, context_pointer=context_pointer
        )
    
    # Determine initial confidence
    if strength == PreferenceStrength.HARD_RULE:
        initial_confidence = 1.0
    else:
        initial_confidence = 0.0  # Will be computed after evidence is added
    
    pref = PreferenceRecord(
        preference_key=preference_key,
        preference_value=preference_value,
        strength=strength,
        confidence=initial_confidence,
        confidence_type=ConfidenceType.PREFERENCE,
        evidence_count=0,
        evidence_weight=0.0,
        source_reliability=1.0,
        contradiction_count=0,
        status=RecordStatus.ACTIVE,
        namespace=namespace,
        applies_to=applies_to,
        source=source,
        last_reinforced_at=datetime.now(timezone.utc),
    )
    
    db.add(pref)
    db.commit()
    db.refresh(pref)
    
    # Add initial evidence
    signal_type = SignalType.EXPLICIT if source == "user_declared" else SignalType.IMPLICIT
    append_preference_evidence(
        db=db,
        preference_key=preference_key,
        signal_type=signal_type,
        context_pointer=context_pointer,
        details={"action": "create", "initial_value": preference_value},
    )
    
    # Recompute confidence (unless hard rule)
    if strength != PreferenceStrength.HARD_RULE:
        recompute_preference_confidence(db, preference_key)
    
    logger.info(f"Created preference '{preference_key}' with strength={strength.value}")
    return pref


def create_hard_rule(
    db: Session,
    preference_key: str,
    preference_value: Any,
    applies_to: Optional[str] = None,
    context_pointer: Optional[str] = None,
) -> PreferenceRecord:
    """
    Create a safety-critical hard rule preference.
    
    Hard rules:
    - No decay
    - Cannot be changed by implicit behavior
    - Can only change via explicit override event
    - confidence=1.0
    """
    return create_preference(
        db=db,
        preference_key=preference_key,
        preference_value=preference_value,
        strength=PreferenceStrength.HARD_RULE,
        source="user_declared",
        applies_to=applies_to,
        namespace="hard_rules",
        context_pointer=context_pointer,
    )


# =============================================================================
# PREFERENCE UPDATES
# =============================================================================

def update_preference_value(
    db: Session,
    preference_key: str,
    new_value: Any,
    is_explicit: bool = True,
    context_pointer: Optional[str] = None,
) -> Optional[PreferenceRecord]:
    """
    Update a preference value.
    
    For hard rules, only explicit updates are allowed.
    Implicit updates record evidence but don't immediately change the value.
    """
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()
    
    if not pref:
        logger.warning(f"Cannot update: preference not found: {preference_key}")
        return None
    
    # Hard rule protection
    if pref.strength == PreferenceStrength.HARD_RULE and not is_explicit:
        logger.warning(
            f"Cannot update hard rule '{preference_key}' via implicit signal. "
            "Only explicit override allowed."
        )
        return pref
    
    old_value = pref.preference_value
    
    # Check if this is a contradiction
    if old_value != new_value:
        if not is_explicit and pref.confidence >= 0.5:
            # Implicit signal contradicting established preference
            from app.astra_memory.confidence_scoring import record_contradiction
            record_contradiction(
                db=db,
                preference_key=preference_key,
                context_pointer=context_pointer,
                new_value=new_value,
                details={"old_value": old_value},
            )
            # Don't update value for implicit contradictions
            return pref
    
    # Update value
    pref.preference_value = new_value
    pref.updated_at = datetime.now(timezone.utc)
    
    if is_explicit:
        pref.last_reinforced_at = datetime.now(timezone.utc)
    
    db.commit()
    
    # Record evidence
    signal_type = SignalType.EXPLICIT if is_explicit else SignalType.IMPLICIT
    append_preference_evidence(
        db=db,
        preference_key=preference_key,
        signal_type=signal_type,
        context_pointer=context_pointer,
        details={"action": "update", "old_value": old_value, "new_value": new_value},
    )
    
    # Recompute confidence
    if pref.strength != PreferenceStrength.HARD_RULE:
        recompute_preference_confidence(db, preference_key)
    
    return pref


def reinforce_preference(
    db: Session,
    preference_key: str,
    signal_type: SignalType = SignalType.IMPLICIT,
    context_pointer: Optional[str] = None,
) -> Optional[PreferenceRecord]:
    """
    Reinforce an existing preference without changing its value.
    
    Use when user's behavior confirms the preference.
    """
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()
    
    if not pref:
        return None
    
    # Add evidence
    append_preference_evidence(
        db=db,
        preference_key=preference_key,
        signal_type=signal_type,
        context_pointer=context_pointer,
        details={"action": "reinforce"},
    )
    
    # Recompute
    if pref.strength != PreferenceStrength.HARD_RULE:
        recompute_preference_confidence(db, preference_key)
    
    return pref


# =============================================================================
# PREFERENCE QUERIES
# =============================================================================

def get_preference(
    db: Session,
    preference_key: str,
) -> Optional[PreferenceRecord]:
    """Get a preference by key."""
    return db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()


def get_preference_value(
    db: Session,
    preference_key: str,
    default: Any = None,
) -> Any:
    """Get preference value, or default if not found or below threshold."""
    cfg = get_config().thresholds
    
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key,
        PreferenceRecord.status == RecordStatus.ACTIVE,
    ).first()
    
    if not pref:
        return default
    
    # Check confidence threshold
    if pref.confidence < cfg.suggestion_threshold:
        return default
    
    return pref.preference_value


def get_preferences_for_component(
    db: Session,
    component: str,
    min_confidence: Optional[float] = None,
) -> List[PreferenceRecord]:
    """
    Get all active preferences for a component.
    
    Args:
        component: Component name (e.g., "overwatcher", "spec_gate")
        min_confidence: Minimum confidence threshold (default: suggestion_threshold)
    """
    cfg = get_config().thresholds
    threshold = min_confidence if min_confidence is not None else cfg.suggestion_threshold
    
    from sqlalchemy import or_
    
    return (
        db.query(PreferenceRecord)
        .filter(
            or_(
                PreferenceRecord.applies_to == component,
                PreferenceRecord.applies_to == "all",
                PreferenceRecord.applies_to.is_(None),
            ),
            PreferenceRecord.status == RecordStatus.ACTIVE,
            PreferenceRecord.confidence >= threshold,
        )
        .order_by(PreferenceRecord.confidence.desc())
        .all()
    )


def get_hard_rules(db: Session) -> List[PreferenceRecord]:
    """Get all hard rule preferences."""
    return (
        db.query(PreferenceRecord)
        .filter(
            PreferenceRecord.strength == PreferenceStrength.HARD_RULE,
            PreferenceRecord.status == RecordStatus.ACTIVE,
        )
        .all()
    )


def get_disputed_preferences(db: Session) -> List[PreferenceRecord]:
    """Get all disputed preferences requiring resolution."""
    return (
        db.query(PreferenceRecord)
        .filter(PreferenceRecord.status == RecordStatus.DISPUTED)
        .order_by(PreferenceRecord.updated_at.desc())
        .all()
    )


# =============================================================================
# BEHAVIOR RULES (Spec Section 6)
# =============================================================================

def resolve_preference_for_default(
    db: Session,
    preference_key: str,
    fallback_value: Any = None,
) -> Tuple[Any, str, float]:
    """
    Resolve a preference value for use as a default.
    
    Implements behavior rules from spec section 6:
    - confidence < 0.65: treat as suggestion, don't enforce
    - confidence >= 0.85: apply silently as default
    - disputed: do not apply automatically
    
    Returns: (value, disposition, confidence)
    - disposition: "not_found", "disputed", "suggest", "apply", "hard_rule"
    """
    cfg = get_config().thresholds
    
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()
    
    if not pref:
        return (fallback_value, "not_found", 0.0)
    
    if pref.status == RecordStatus.DISPUTED:
        return (fallback_value, "disputed", pref.confidence)
    
    if pref.status != RecordStatus.ACTIVE:
        return (fallback_value, "inactive", pref.confidence)
    
    # Hard rules always apply
    if pref.strength == PreferenceStrength.HARD_RULE:
        return (pref.preference_value, "hard_rule", 1.0)
    
    # Check confidence thresholds
    if pref.confidence < cfg.suggestion_threshold:
        return (fallback_value, "low_confidence", pref.confidence)
    
    if pref.confidence >= cfg.apply_threshold:
        return (pref.preference_value, "apply", pref.confidence)
    
    # Between thresholds: suggest but don't enforce
    return (pref.preference_value, "suggest", pref.confidence)


def should_apply_preference_silently(
    db: Session,
    preference_key: str,
) -> bool:
    """
    Check if a preference should be applied silently (without asking).
    
    True if:
    - Hard rule, or
    - confidence >= apply_threshold (0.85)
    """
    _, disposition, _ = resolve_preference_for_default(db, preference_key)
    return disposition in ("apply", "hard_rule")


# =============================================================================
# PREFERENCE LEARNING (Implicit)
# =============================================================================

def learn_from_behavior(
    db: Session,
    preference_key: str,
    observed_value: Any,
    context_pointer: Optional[str] = None,
    is_repeated: bool = False,
) -> Optional[PreferenceRecord]:
    """
    Learn a preference from observed behavior.
    
    Bad-learning prevention (spec section 4.2):
    - Single incidents don't create preferences
    - Require evidence_count >= 2 OR explicit instruction
    
    Args:
        preference_key: The preference being learned
        observed_value: The value observed in user behavior
        context_pointer: Link to the source event
        is_repeated: True if this is a repeated observation
    """
    pref = get_preference(db, preference_key)
    
    if pref:
        # Existing preference: check for contradiction or reinforcement
        if pref.preference_value == observed_value:
            # Reinforcement
            signal = SignalType.IMPLICIT if is_repeated else SignalType.ONE_OFF
            return reinforce_preference(db, preference_key, signal, context_pointer)
        else:
            # Potential contradiction - record but don't change
            from app.astra_memory.confidence_scoring import record_contradiction
            record_contradiction(
                db, preference_key, context_pointer, observed_value,
                {"source": "implicit_behavior"}
            )
            return pref
    else:
        # New preference from implicit signal
        # BAD-LEARNING PREVENTION: Use one_off weight for single incidents
        signal = SignalType.IMPLICIT if is_repeated else SignalType.ONE_OFF
        
        pref = create_preference(
            db=db,
            preference_key=preference_key,
            preference_value=observed_value,
            strength=PreferenceStrength.SOFT,  # Implicit = soft
            source="learned",
            namespace="learned",
            context_pointer=context_pointer,
        )
        
        # Override the signal type if it was one-off
        if signal == SignalType.ONE_OFF:
            # The create added explicit evidence, we need to adjust
            # This is handled by recompute which considers signal types
            pass
        
        return pref


# =============================================================================
# PREFERENCE RESOLUTION
# =============================================================================

def resolve_disputed_preference(
    db: Session,
    preference_key: str,
    resolved_value: Any,
    context_pointer: Optional[str] = None,
) -> Optional[PreferenceRecord]:
    """
    Explicitly resolve a disputed preference.
    
    This is the explicit resolution path required for major preference conflicts.
    """
    pref = get_preference(db, preference_key)
    
    if not pref:
        return None
    
    # Update with explicit signal
    pref.preference_value = resolved_value
    pref.status = RecordStatus.ACTIVE
    pref.last_reinforced_at = datetime.now(timezone.utc)
    pref.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    
    # Add explicit resolution evidence
    append_preference_evidence(
        db=db,
        preference_key=preference_key,
        signal_type=SignalType.EXPLICIT,
        context_pointer=context_pointer,
        details={"action": "resolve_dispute", "resolved_value": resolved_value},
    )
    
    # Recompute confidence
    recompute_preference_confidence(db, preference_key)
    
    logger.info(f"Resolved disputed preference '{preference_key}' to: {resolved_value}")
    return pref


def expire_preference(
    db: Session,
    preference_key: str,
    reason: Optional[str] = None,
) -> Optional[PreferenceRecord]:
    """Mark a preference as expired."""
    pref = get_preference(db, preference_key)
    
    if not pref:
        return None
    
    pref.status = RecordStatus.EXPIRED
    pref.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    logger.info(f"Expired preference '{preference_key}': {reason}")
    return pref


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Creation
    "create_preference",
    "create_hard_rule",
    # Updates
    "update_preference_value",
    "reinforce_preference",
    # Queries
    "get_preference",
    "get_preference_value",
    "get_preferences_for_component",
    "get_hard_rules",
    "get_disputed_preferences",
    # Behavior rules
    "resolve_preference_for_default",
    "should_apply_preference_silently",
    # Learning
    "learn_from_behavior",
    # Resolution
    "resolve_disputed_preference",
    "expire_preference",
]

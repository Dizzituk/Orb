# FILE: app/astra_memory/confidence_scoring.py
"""
ASTRA Memory Confidence Scoring

Implements the deterministic scoring function from spec section 4.

Core formula:
    decay_i = exp(-age_days / half_life_days)  # for weak/soft signals
    S = Σ(weight_i * decay_i)
    confidence = 1 - exp(-k * max(S, 0))

Rules:
- Hard rules: no decay, confidence=1.0, only explicit override can modify
- Bad-learning prevention: requires evidence_count >= 2 OR one explicit instruction
- Contradictions: mark disputed, reduce confidence, don't overwrite immediately
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.astra_memory.confidence_config import (
    get_config,
    get_evidence_weight,
)
from app.astra_memory.preference_models import (
    PreferenceRecord,
    PreferenceEvidence,
    PreferenceStrength,
    RecordStatus,
    SignalType,
    ConfidenceType,
    MemoryRecordConfidence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CORE SCORING FUNCTIONS
# =============================================================================

def compute_decay(age_days: float, half_life_days: float) -> float:
    """
    Compute exponential decay factor using half-life semantics.
    
    decay = 2^(-age_days / half_life_days) = exp(-ln(2) * age_days / half_life_days)
    
    At age=half_life, decay=0.5
    At age=0, decay=1.0
    """
    if half_life_days <= 0:
        return 1.0
    # Use ln(2) ≈ 0.693 for proper half-life decay
    return math.exp(-math.log(2) * age_days / half_life_days)


def compute_weighted_sum(
    evidence_list: List[Tuple[float, float, bool]],
    half_life_days: float,
) -> float:
    """
    Compute weighted sum of evidence with decay.
    
    Args:
        evidence_list: List of (weight, age_days, is_hard_rule)
        half_life_days: Decay half-life
        
    Returns:
        S = Σ(weight_i * decay_i)
    """
    total = 0.0
    for weight, age_days, is_hard_rule in evidence_list:
        if is_hard_rule:
            # Hard rules: no decay
            decay = 1.0
        else:
            decay = compute_decay(age_days, half_life_days)
        total += weight * decay
    return total


def weighted_sum_to_confidence(s: float, k: float) -> float:
    """
    Map weighted sum to confidence score.
    
    confidence = 1 - exp(-k * max(S, 0))
    
    Properties:
    - S <= 0 → confidence = 0
    - S → ∞ → confidence → 1
    - k controls saturation speed
    """
    if s <= 0:
        return 0.0
    return 1.0 - math.exp(-k * s)


def compute_confidence_score(
    evidence_list: List[Tuple[float, float, bool]],
    half_life_days: Optional[float] = None,
    k: Optional[float] = None,
) -> float:
    """
    Compute final confidence score from evidence.
    
    Args:
        evidence_list: List of (weight, age_days, is_hard_rule)
        half_life_days: Override config value
        k: Override config value
        
    Returns:
        Confidence score in [0.0, 1.0]
    """
    cfg = get_config()
    hl = half_life_days if half_life_days is not None else cfg.decay.half_life_days
    k_val = k if k is not None else cfg.decay.k
    
    s = compute_weighted_sum(evidence_list, hl)
    return weighted_sum_to_confidence(s, k_val)


# =============================================================================
# PREFERENCE CONFIDENCE COMPUTATION
# =============================================================================

def recompute_preference_confidence(
    db: Session,
    preference_key: str,
    now: Optional[datetime] = None,
) -> Optional[PreferenceRecord]:
    """
    Recompute confidence for a preference from its evidence ledger.
    
    This is the main entry point for confidence recalculation.
    Call after adding new evidence or periodically for decay updates.
    """
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()
    
    if not pref:
        logger.warning(f"Preference not found: {preference_key}")
        return None
    
    # Hard rules always have confidence=1.0
    if pref.strength == PreferenceStrength.HARD_RULE:
        if pref.confidence != 1.0:
            pref.confidence = 1.0
            db.commit()
        return pref
    
    # Get all evidence for this preference
    evidence_records = db.query(PreferenceEvidence).filter(
        PreferenceEvidence.preference_key == preference_key
    ).all()
    
    if not evidence_records:
        # No evidence = no confidence
        pref.confidence = 0.0
        pref.evidence_count = 0
        pref.evidence_weight = 0.0
        db.commit()
        return pref
    
    # Build evidence list for scoring
    now = now or datetime.now(timezone.utc)
    evidence_list = []
    
    total_weight = 0.0
    contradiction_count = 0
    has_explicit = False
    
    for ev in evidence_records:
        # Calculate age in days
        ev_time = ev.timestamp
        if ev_time.tzinfo is None:
            ev_time = ev_time.replace(tzinfo=timezone.utc)
        age_days = (now - ev_time).total_seconds() / 86400.0
        
        # Track explicit signals
        if ev.signal_type in (SignalType.EXPLICIT, SignalType.APPROVAL):
            has_explicit = True
        
        # Track contradictions
        if ev.signal_type == SignalType.CONTRADICTION:
            contradiction_count += 1
        
        # Hard rule signals don't decay
        is_hard_rule = (ev.signal_type == SignalType.EXPLICIT and 
                       pref.strength == PreferenceStrength.HARD_RULE)
        
        evidence_list.append((ev.weight, age_days, is_hard_rule))
        total_weight += ev.weight
    
    # Bad-learning prevention: require evidence_count >= 2 OR explicit instruction
    evidence_count = len(evidence_records)
    cfg = get_config()
    
    if evidence_count < cfg.thresholds.min_evidence_count and not has_explicit:
        # Not enough evidence for implicit-only preference
        confidence = 0.0
    else:
        confidence = compute_confidence_score(evidence_list)
    
    # Handle disputed status
    if contradiction_count > 0:
        # Check if mostly contradictions
        if contradiction_count >= evidence_count / 2:
            pref.status = RecordStatus.DISPUTED
    
    # Update preference record
    pref.confidence = round(confidence, 4)
    pref.evidence_count = evidence_count
    pref.evidence_weight = round(total_weight, 4)
    pref.contradiction_count = contradiction_count
    pref.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    return pref


def batch_recompute_confidence(
    db: Session,
    preference_keys: Optional[List[str]] = None,
    namespace: Optional[str] = None,
) -> int:
    """
    Recompute confidence for multiple preferences.
    
    Use for periodic batch updates or after bulk evidence imports.
    
    Returns number of preferences updated.
    """
    query = db.query(PreferenceRecord)
    
    if preference_keys:
        query = query.filter(PreferenceRecord.preference_key.in_(preference_keys))
    
    if namespace:
        query = query.filter(PreferenceRecord.namespace == namespace)
    
    # Exclude hard rules (always 1.0)
    query = query.filter(PreferenceRecord.strength != PreferenceStrength.HARD_RULE)
    
    prefs = query.all()
    count = 0
    
    for pref in prefs:
        result = recompute_preference_confidence(db, pref.preference_key)
        if result:
            count += 1
    
    return count


# =============================================================================
# EVIDENCE RECORDING
# =============================================================================

def append_preference_evidence(
    db: Session,
    preference_key: str,
    signal_type: SignalType,
    context_pointer: Optional[str] = None,
    details: Optional[dict] = None,
    weight_override: Optional[float] = None,
) -> Optional[PreferenceEvidence]:
    """
    Append evidence to the preference ledger.
    
    This is the ONLY way to add evidence. The ledger is append-only.
    
    Args:
        preference_key: The preference this evidence applies to
        signal_type: Type of signal (explicit, implicit, etc.)
        context_pointer: Link to source (message:123, job:abc, etc.)
        details: Optional additional context
        weight_override: Override the default weight for this signal type
        
    Returns:
        The created evidence record, or None if preference doesn't exist
    """
    # Verify preference exists
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key
    ).first()
    
    if not pref:
        logger.warning(f"Cannot add evidence: preference not found: {preference_key}")
        return None
    
    # Check hard rule constraints
    if pref.strength == PreferenceStrength.HARD_RULE:
        if signal_type not in (SignalType.EXPLICIT, SignalType.APPROVAL):
            logger.warning(
                f"Hard rule '{preference_key}' cannot be modified by {signal_type}. "
                "Only explicit override events allowed."
            )
            return None
    
    # Determine weight
    if weight_override is not None:
        weight = weight_override
    else:
        weight = get_evidence_weight(signal_type.value)
    
    # Create evidence record
    evidence = PreferenceEvidence(
        preference_key=preference_key,
        signal_type=signal_type,
        weight=weight,
        timestamp=datetime.now(timezone.utc),
        context_pointer=context_pointer,
        details=details,
    )
    
    db.add(evidence)
    db.commit()
    db.refresh(evidence)
    
    # Update preference's last_reinforced_at
    if weight > 0:  # Positive evidence
        pref.last_reinforced_at = datetime.now(timezone.utc)
        db.commit()
    
    logger.debug(
        f"Added evidence for '{preference_key}': {signal_type.value} "
        f"weight={weight} context={context_pointer}"
    )
    
    return evidence


def record_contradiction(
    db: Session,
    preference_key: str,
    context_pointer: Optional[str] = None,
    new_value: Optional[any] = None,
    details: Optional[dict] = None,
) -> Optional[PreferenceEvidence]:
    """
    Record a contradiction event for a preference.
    
    Contradictions:
    - Add negative weight evidence
    - May trigger disputed status
    - Do NOT immediately overwrite the preference value
    """
    full_details = details or {}
    if new_value is not None:
        full_details["contradicting_value"] = new_value
    
    evidence = append_preference_evidence(
        db=db,
        preference_key=preference_key,
        signal_type=SignalType.CONTRADICTION,
        context_pointer=context_pointer,
        details=full_details,
    )
    
    if evidence:
        # Recompute to potentially mark as disputed
        recompute_preference_confidence(db, preference_key)
    
    return evidence


# =============================================================================
# RECORD CONFIDENCE (Non-preference memory records)
# =============================================================================

def get_or_create_record_confidence(
    db: Session,
    source_type: str,
    source_id: int,
    namespace: str = "general",
) -> MemoryRecordConfidence:
    """
    Get or create confidence metadata for a memory record.
    """
    conf = db.query(MemoryRecordConfidence).filter(
        MemoryRecordConfidence.source_type == source_type,
        MemoryRecordConfidence.source_id == source_id,
    ).first()
    
    if not conf:
        conf = MemoryRecordConfidence(
            source_type=source_type,
            source_id=source_id,
            namespace=namespace,
            confidence=0.5,  # Default neutral confidence
            confidence_type=ConfidenceType.RECORD,
            evidence_count=1,
            evidence_weight=1.0,
            source_reliability=1.0,
        )
        db.add(conf)
        db.commit()
        db.refresh(conf)
    
    return conf


def update_record_confidence(
    db: Session,
    source_type: str,
    source_id: int,
    confidence_delta: float = 0.0,
    verified: bool = False,
    contradicted: bool = False,
    source_reliability: Optional[float] = None,
) -> Optional[MemoryRecordConfidence]:
    """
    Update confidence for a memory record.
    
    Args:
        confidence_delta: Direct adjustment to confidence
        verified: If True, boost confidence
        contradicted: If True, mark disputed and reduce confidence
        source_reliability: Update source reliability factor
    """
    conf = get_or_create_record_confidence(db, source_type, source_id)
    
    if verified:
        conf.evidence_count += 1
        conf.evidence_weight += 2.0
        conf.confidence = min(1.0, conf.confidence + 0.1)
    
    if contradicted:
        conf.contradiction_count += 1
        conf.status = RecordStatus.DISPUTED
        conf.confidence = max(0.0, conf.confidence - 0.2)
    
    if source_reliability is not None:
        conf.source_reliability = max(0.0, min(1.0, source_reliability))
        # Reliability affects confidence
        conf.confidence *= conf.source_reliability
    
    if confidence_delta != 0:
        conf.confidence = max(0.0, min(1.0, conf.confidence + confidence_delta))
    
    conf.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    return conf


# =============================================================================
# NAMESPACE ENFORCEMENT
# =============================================================================

def check_namespace_mutation_allowed(
    target_namespace: str,
    source_namespace: str,
    is_explicit: bool = False,
) -> bool:
    """
    Check if a mutation from source namespace can affect target namespace.
    
    Namespace separation rules:
    - Protected namespaces cannot be mutated by repo scans
    - Repo-mutable namespaces can only be updated by repo_derived sources
    - Explicit user actions can always promote/mutate
    """
    cfg = get_config().namespaces
    
    # Explicit user actions bypass namespace restrictions
    if is_explicit:
        return True
    
    # Protected namespaces cannot be mutated by non-explicit sources
    if target_namespace in cfg.protected_namespaces:
        return False
    
    # Repo-mutable namespaces can only be updated by repo sources
    if target_namespace in cfg.repo_mutable_namespaces:
        return source_namespace in ("repo_derived", "atlas", "code_analysis")
    
    # General namespaces allow mutation from same namespace
    return target_namespace == source_namespace


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core scoring
    "compute_decay",
    "compute_weighted_sum",
    "weighted_sum_to_confidence",
    "compute_confidence_score",
    # Preference confidence
    "recompute_preference_confidence",
    "batch_recompute_confidence",
    # Evidence recording
    "append_preference_evidence",
    "record_contradiction",
    # Record confidence
    "get_or_create_record_confidence",
    "update_record_confidence",
    # Namespace
    "check_namespace_mutation_allowed",
]

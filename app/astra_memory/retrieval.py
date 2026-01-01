# FILE: app/astra_memory/retrieval.py
"""
ASTRA Memory Retrieval Layer

Implements fast recall architecture from spec section 5:
1. Intent depth classification (D0-D4)
2. Hot/cold storage separation
3. 2-stage retrieval (cheap candidate selection → depth-gated expansion)
4. Summary pyramid selection

Key invariant: D0/D1 MUST NEVER fetch cold artifacts.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.astra_memory.confidence_config import get_config
from app.astra_memory.preference_models import (
    IntentDepth,
    RetrievalCost,
    HotIndex,
    SummaryPyramid,
    PreferenceRecord,
    RecordStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT DEPTH CLASSIFICATION
# =============================================================================

# Keywords that indicate different depth levels
DEPTH_KEYWORDS = {
    IntentDepth.D0: {
        "triggers": ["hi", "hello", "hey", "thanks", "bye", "ok", "sure"],
        "patterns": [r"^(hi|hello|hey|thanks|bye|ok|sure|yes|no)[\s!?.]*$"],
    },
    IntentDepth.D1: {
        "triggers": ["briefly", "quick", "simple", "short", "summary", "tldr", "recap"],
        "patterns": [r"tell me (?:a )?(?:bit|little)", r"what(?:'s| is) .{1,30}\?$"],
    },
    IntentDepth.D2: {
        "triggers": ["explain", "describe", "how", "why", "what", "current", "status"],
        "patterns": [r"how (?:do|does|can|should)", r"what (?:is|are) the"],
    },
    IntentDepth.D3: {
        "triggers": ["deep", "detailed", "full", "complete", "comprehensive", "in-depth",
                    "spec", "specification", "architecture", "all"],
        "patterns": [r"give me (?:the )?full", r"(?:deep|detailed) (?:dive|analysis)"],
    },
    IntentDepth.D4: {
        "triggers": ["forensic", "audit", "evidence", "history", "timeline", "ledger",
                    "debug", "investigate", "all changes", "diff history"],
        "patterns": [r"show me (?:all|every)", r"what (?:changed|happened)"],
    },
}

# Explicit command tokens for depth override
EXPLICIT_DEPTH_TOKENS = {
    "/brief": IntentDepth.D1,
    "/normal": IntentDepth.D2,
    "/deep": IntentDepth.D3,
    "/forensic": IntentDepth.D4,
    "/nomem": IntentDepth.D0,
    "[brief]": IntentDepth.D1,
    "[deep]": IntentDepth.D3,
    "[forensic]": IntentDepth.D4,
}


def classify_intent_depth(message: str) -> IntentDepth:
    """
    Classify user message into intent depth level.
    
    Default: D0/D1 (minimal memory) unless user explicitly asks for more.
    
    Args:
        message: User's message text
        
    Returns:
        IntentDepth enum value
    """
    message_lower = message.lower().strip()
    
    # Check for explicit command tokens first
    for token, depth in EXPLICIT_DEPTH_TOKENS.items():
        if token in message_lower:
            logger.debug(f"Explicit depth token '{token}' → {depth}")
            return depth
    
    # Check patterns and triggers from highest to lowest depth
    for depth in [IntentDepth.D4, IntentDepth.D3, IntentDepth.D2, IntentDepth.D1, IntentDepth.D0]:
        config = DEPTH_KEYWORDS.get(depth, {})
        
        # Check triggers
        triggers = config.get("triggers", [])
        for trigger in triggers:
            if trigger in message_lower:
                logger.debug(f"Depth trigger '{trigger}' → {depth}")
                return depth
        
        # Check patterns
        patterns = config.get("patterns", [])
        for pattern in patterns:
            if re.search(pattern, message_lower):
                logger.debug(f"Depth pattern '{pattern}' → {depth}")
                return depth
    
    # Default: D1 (brief) - minimal memory load
    return IntentDepth.D1


# =============================================================================
# RETRIEVAL RESULT TYPES
# =============================================================================

@dataclass
class RetrievalCandidate:
    """Candidate from stage 1 (hot index)."""
    record_type: str
    record_id: str
    title: str
    one_liner: Optional[str] = None
    relevance_score: float = 0.0
    retrieval_cost: RetrievalCost = RetrievalCost.TINY
    hot_index_id: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class ExpandedRecord:
    """Expanded record from stage 2."""
    record_type: str
    record_id: str
    title: str
    content: str  # Actual content (summary level depends on depth)
    summary_level: int = 0  # L0, L1, L2, or L3
    relevance_score: float = 0.0
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Final retrieval result."""
    depth: IntentDepth
    candidates_searched: int
    records_expanded: int
    token_estimate: int
    records: List[ExpandedRecord] = field(default_factory=list)
    preferences_applied: List[str] = field(default_factory=list)


# =============================================================================
# STAGE 1: CHEAP CANDIDATE SELECTION
# =============================================================================

def stage1_candidate_selection(
    db: Session,
    query_tags: Optional[List[str]] = None,
    query_entities: Optional[List[str]] = None,
    record_types: Optional[List[str]] = None,
    namespace_filter: Optional[str] = None,
    max_candidates: int = 100,
    recency_weight: float = 0.3,
) -> List[RetrievalCandidate]:
    """
    Stage 1: Fast candidate selection using hot index only.
    
    Uses only hot index fields:
    - tags/entities matching
    - record type filtering
    - recency
    - namespace filters
    
    Returns top K candidates sorted by relevance.
    """
    query = db.query(HotIndex)
    
    # Filter by record type
    if record_types:
        query = query.filter(HotIndex.record_type.in_(record_types))
    
    candidates = []
    
    for hot in query.limit(max_candidates * 2).all():  # Over-fetch for scoring
        # Calculate relevance score
        score = hot.retrieval_priority
        
        # Tag matching
        if query_tags and hot.tags:
            matching_tags = set(query_tags) & set(hot.tags)
            score += len(matching_tags) * 0.2
        
        # Entity matching
        if query_entities and hot.entities:
            matching_entities = set(query_entities) & set(hot.entities)
            score += len(matching_entities) * 0.3
        
        # Recency boost
        if hot.updated_at:
            age_days = (datetime.now(timezone.utc) - hot.updated_at.replace(tzinfo=timezone.utc)).days
            recency_score = max(0, 1 - (age_days / 30)) * recency_weight
            score += recency_score
        
        candidates.append(RetrievalCandidate(
            record_type=hot.record_type,
            record_id=hot.record_id,
            title=hot.title,
            one_liner=hot.one_liner,
            relevance_score=score,
            retrieval_cost=hot.retrieval_cost or RetrievalCost.TINY,
            hot_index_id=hot.id,
            tags=hot.tags or [],
        ))
    
    # Sort by relevance
    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
    
    return candidates[:max_candidates]


# =============================================================================
# STAGE 2: DEPTH-GATED EXPANSION
# =============================================================================

def get_summary_for_depth(
    db: Session,
    artifact_type: str,
    artifact_id: str,
    depth: IntentDepth,
) -> Tuple[str, int]:
    """
    Get appropriate summary level for depth.
    
    Returns (content, summary_level)
    """
    pyramid = db.query(SummaryPyramid).filter(
        SummaryPyramid.artifact_type == artifact_type,
        SummaryPyramid.artifact_id == artifact_id,
    ).first()
    
    if not pyramid:
        return ("", -1)
    
    cfg = get_config().retrieval
    
    if depth == IntentDepth.D1:
        # L0/L1: sentence or bullets
        if pyramid.l0_sentence:
            return (pyramid.l0_sentence, 0)
        elif pyramid.l1_bullets:
            return ("\n".join(f"• {b}" for b in pyramid.l1_bullets), 1)
    
    elif depth == IntentDepth.D2:
        # L1/L2: bullets or paragraphs
        if pyramid.l1_bullets:
            content = "\n".join(f"• {b}" for b in pyramid.l1_bullets)
            if pyramid.l2_paragraphs:
                content += f"\n\n{pyramid.l2_paragraphs}"
            return (content, 2)
    
    elif depth in (IntentDepth.D3, IntentDepth.D4):
        # L2/L3: paragraphs + full text
        parts = []
        if pyramid.l2_paragraphs:
            parts.append(pyramid.l2_paragraphs)
        # Note: L3 full text requires cold storage fetch
        # For D4, we'd also load the full artifact
        return ("\n\n".join(parts), 3 if depth == IntentDepth.D4 else 2)
    
    # Fallback: L0
    return (pyramid.l0_sentence or "", 0)


def stage2_expand_candidates(
    db: Session,
    candidates: List[RetrievalCandidate],
    depth: IntentDepth,
) -> List[ExpandedRecord]:
    """
    Stage 2: Expand candidates based on depth.
    
    CRITICAL: D0/D1 must NEVER fetch cold artifacts.
    """
    cfg = get_config().retrieval
    
    # Determine max items to expand
    max_items = {
        IntentDepth.D0: cfg.d0_max_items,
        IntentDepth.D1: cfg.d1_max_items,
        IntentDepth.D2: cfg.d2_max_items,
        IntentDepth.D3: cfg.d3_max_items,
        IntentDepth.D4: cfg.d4_max_items,
    }.get(depth, cfg.d1_max_items)
    
    if max_items == 0:
        return []  # D0: no memory
    
    expanded = []
    
    for candidate in candidates[:max_items]:
        # D0/D1: Use hot layer only
        if depth in (IntentDepth.D0, IntentDepth.D1):
            # INVARIANT: Never fetch cold for D0/D1
            expanded.append(ExpandedRecord(
                record_type=candidate.record_type,
                record_id=candidate.record_id,
                title=candidate.title,
                content=candidate.one_liner or candidate.title,
                summary_level=0,
                relevance_score=candidate.relevance_score,
            ))
        else:
            # D2+: Can expand from summary pyramid
            content, level = get_summary_for_depth(
                db, candidate.record_type, candidate.record_id, depth
            )
            
            if not content:
                # Fallback to hot layer
                content = candidate.one_liner or candidate.title
                level = 0
            
            expanded.append(ExpandedRecord(
                record_type=candidate.record_type,
                record_id=candidate.record_id,
                title=candidate.title,
                content=content,
                summary_level=level,
                relevance_score=candidate.relevance_score,
            ))
    
    return expanded


def apply_cost_ranking(
    candidates: List[RetrievalCandidate],
    depth: IntentDepth,
) -> List[RetrievalCandidate]:
    """
    Adjust ranking based on retrieval cost.
    
    D1: Favor high relevance + low cost
    D3+: Favor relevance regardless of cost
    """
    if depth == IntentDepth.D1:
        # Penalize high-cost candidates for brief queries
        for c in candidates:
            if c.retrieval_cost == RetrievalCost.LARGE:
                c.relevance_score *= 0.5
            elif c.retrieval_cost == RetrievalCost.MEDIUM:
                c.relevance_score *= 0.8
    
    # Re-sort
    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
    return candidates


# =============================================================================
# MAIN RETRIEVAL FUNCTION
# =============================================================================

def retrieve_for_query(
    db: Session,
    user_message: str,
    query_tags: Optional[List[str]] = None,
    query_entities: Optional[List[str]] = None,
    record_types: Optional[List[str]] = None,
    depth_override: Optional[IntentDepth] = None,
) -> RetrievalResult:
    """
    Main retrieval entry point.
    
    1. Classify intent depth
    2. Stage 1: Candidate selection (hot index)
    3. Apply cost ranking
    4. Stage 2: Depth-gated expansion
    5. Return results with token estimate
    """
    # Step 1: Determine depth
    depth = depth_override or classify_intent_depth(user_message)
    logger.info(f"Retrieval depth: {depth.value} for: {user_message[:50]}...")
    
    # D0: No memory
    if depth == IntentDepth.D0:
        return RetrievalResult(
            depth=depth,
            candidates_searched=0,
            records_expanded=0,
            token_estimate=0,
            records=[],
        )
    
    # Step 2: Stage 1 - Candidate selection
    candidates = stage1_candidate_selection(
        db=db,
        query_tags=query_tags,
        query_entities=query_entities,
        record_types=record_types,
    )
    
    # Step 3: Apply cost ranking
    candidates = apply_cost_ranking(candidates, depth)
    
    # Step 4: Stage 2 - Expand
    expanded = stage2_expand_candidates(db, candidates, depth)
    
    # Step 5: Estimate tokens
    token_estimate = sum(len(r.content) // 4 for r in expanded)  # ~4 chars/token
    
    return RetrievalResult(
        depth=depth,
        candidates_searched=len(candidates),
        records_expanded=len(expanded),
        token_estimate=token_estimate,
        records=expanded,
    )


# =============================================================================
# PREFERENCE RETRIEVAL
# =============================================================================

def get_applicable_preferences(
    db: Session,
    component: str,
    include_disputed: bool = False,
) -> List[PreferenceRecord]:
    """
    Get preferences applicable to a component.
    
    Args:
        component: Component name (e.g., "overwatcher", "spec_gate", "llm_router")
        include_disputed: Whether to include disputed preferences
        
    Returns:
        List of applicable preferences sorted by confidence
    """
    cfg = get_config().thresholds
    
    query = db.query(PreferenceRecord).filter(
        or_(
            PreferenceRecord.applies_to == component,
            PreferenceRecord.applies_to == "all",
            PreferenceRecord.applies_to.is_(None),
        ),
    )
    
    if not include_disputed:
        query = query.filter(PreferenceRecord.status != RecordStatus.DISPUTED)
    
    # Only get preferences with sufficient confidence
    query = query.filter(PreferenceRecord.confidence >= cfg.suggestion_threshold)
    
    return query.order_by(PreferenceRecord.confidence.desc()).all()


def get_highest_confidence_preference(
    db: Session,
    preference_key: str,
    min_confidence: Optional[float] = None,
) -> Optional[PreferenceRecord]:
    """
    Get preference by key if confidence meets threshold.
    
    Returns None if preference doesn't exist, is disputed, or below threshold.
    """
    cfg = get_config().thresholds
    threshold = min_confidence if min_confidence is not None else cfg.suggestion_threshold
    
    pref = db.query(PreferenceRecord).filter(
        PreferenceRecord.preference_key == preference_key,
        PreferenceRecord.status == RecordStatus.ACTIVE,
        PreferenceRecord.confidence >= threshold,
    ).first()
    
    return pref


def should_apply_preference(pref: PreferenceRecord) -> Tuple[bool, str]:
    """
    Determine if a preference should be applied.
    
    Returns (should_apply, reason)
    """
    cfg = get_config().thresholds
    
    if pref.status == RecordStatus.DISPUTED:
        return (False, "disputed")
    
    if pref.status != RecordStatus.ACTIVE:
        return (False, f"status={pref.status.value}")
    
    if pref.confidence < cfg.suggestion_threshold:
        return (False, f"confidence={pref.confidence:.2f} < {cfg.suggestion_threshold}")
    
    if pref.confidence >= cfg.apply_threshold:
        return (True, "apply_silently")
    
    return (True, "suggest_only")


# =============================================================================
# HOT INDEX MANAGEMENT
# =============================================================================

def upsert_hot_index(
    db: Session,
    record_type: str,
    record_id: str,
    title: str,
    one_liner: Optional[str] = None,
    bullets_5: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    entities: Optional[List[str]] = None,
    cold_storage_path: Optional[str] = None,
    retrieval_priority: float = 0.5,
    retrieval_cost: RetrievalCost = RetrievalCost.TINY,
) -> HotIndex:
    """
    Create or update a hot index entry.
    """
    hot = db.query(HotIndex).filter(
        HotIndex.record_type == record_type,
        HotIndex.record_id == record_id,
    ).first()
    
    if hot:
        hot.title = title
        hot.one_liner = one_liner
        hot.bullets_5 = bullets_5
        hot.tags = tags
        hot.entities = entities
        hot.cold_storage_path = cold_storage_path
        hot.retrieval_priority = retrieval_priority
        hot.retrieval_cost = retrieval_cost
        hot.updated_at = datetime.now(timezone.utc)
    else:
        hot = HotIndex(
            record_type=record_type,
            record_id=record_id,
            title=title,
            one_liner=one_liner,
            bullets_5=bullets_5,
            tags=tags,
            entities=entities,
            cold_storage_path=cold_storage_path,
            retrieval_priority=retrieval_priority,
            retrieval_cost=retrieval_cost,
        )
        db.add(hot)
    
    db.commit()
    db.refresh(hot)
    return hot


def upsert_summary_pyramid(
    db: Session,
    artifact_type: str,
    artifact_id: str,
    l0_sentence: Optional[str] = None,
    l1_bullets: Optional[List[str]] = None,
    l2_paragraphs: Optional[str] = None,
    l3_cold_path: Optional[str] = None,
    l3_token_estimate: Optional[int] = None,
    sections: Optional[List[dict]] = None,
    source_hash: Optional[str] = None,
) -> SummaryPyramid:
    """
    Create or update a summary pyramid.
    """
    pyramid = db.query(SummaryPyramid).filter(
        SummaryPyramid.artifact_type == artifact_type,
        SummaryPyramid.artifact_id == artifact_id,
    ).first()
    
    if pyramid:
        pyramid.l0_sentence = l0_sentence
        pyramid.l1_bullets = l1_bullets
        pyramid.l2_paragraphs = l2_paragraphs
        pyramid.l3_cold_path = l3_cold_path
        pyramid.l3_token_estimate = l3_token_estimate
        pyramid.sections = sections
        pyramid.source_hash = source_hash
        pyramid.generated_at = datetime.now(timezone.utc)
    else:
        pyramid = SummaryPyramid(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            l0_sentence=l0_sentence,
            l1_bullets=l1_bullets,
            l2_paragraphs=l2_paragraphs,
            l3_cold_path=l3_cold_path,
            l3_token_estimate=l3_token_estimate,
            sections=sections,
            source_hash=source_hash,
        )
        db.add(pyramid)
    
    db.commit()
    db.refresh(pyramid)
    return pyramid


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Intent classification
    "classify_intent_depth",
    "EXPLICIT_DEPTH_TOKENS",
    # Retrieval types
    "RetrievalCandidate",
    "ExpandedRecord",
    "RetrievalResult",
    # Core retrieval
    "stage1_candidate_selection",
    "stage2_expand_candidates",
    "retrieve_for_query",
    # Preference retrieval
    "get_applicable_preferences",
    "get_highest_confidence_preference",
    "should_apply_preference",
    # Hot index management
    "upsert_hot_index",
    "upsert_summary_pyramid",
]

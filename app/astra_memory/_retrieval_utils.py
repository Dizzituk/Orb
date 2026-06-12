# Purpose: retrieval utils
# Called-by: app.astra_memory.retrieval
# Depends-on: app.astra_memory.confidence_config, app.astra_memory.preference_models, app.memory.models
# Last-renovated: 2026-06-11
from __future__ import annotations
import logging
from app.astra_memory.confidence_config import get_config
from app.astra_memory.preference_models import IntentDepth, PreferenceRecord, RecordStatus
from sqlalchemy import or_
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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

def _fetch_cold_message(db: Session, record_id: str) -> Optional[str]:
    """Fetch full message content from Messages table."""
    try:
        from app.memory.models import Message
        msg = db.query(Message).filter(Message.id == int(record_id)).first()
        if msg and msg.content:
            # Check for encryption marker
            if msg.content.startswith('ENC:') or '[ENCRYPTED' in msg.content:
                return None  # Encryption not available
            return msg.content
    except Exception as e:
        logger.warning(f"Cold fetch message {record_id} failed: {e}")
    return None

def _fetch_cold_note(db: Session, record_id: str) -> Optional[str]:
    """Fetch full note content from Notes table."""
    try:
        from app.memory.models import Note
        note = db.query(Note).filter(Note.id == int(record_id)).first()
        if note and note.content:
            if note.content.startswith('ENC:') or '[ENCRYPTED' in note.content:
                return None
            return f"# {note.title}\n\n{note.content}"
    except Exception as e:
        logger.warning(f"Cold fetch note {record_id} failed: {e}")
    return None

def _fetch_cold_document(db: Session, record_id: str) -> Optional[str]:
    """Fetch full document content from DocumentContent table."""
    try:
        from app.memory.models import DocumentContent
        doc = db.query(DocumentContent).filter(DocumentContent.id == int(record_id)).first()
        if doc:
            parts = []
            if doc.filename:
                parts.append(f"# Document: {doc.filename}")
            if doc.summary and not doc.summary.startswith('ENC:'):
                parts.append(f"\n## Summary\n{doc.summary}")
            if doc.raw_text and not doc.raw_text.startswith('ENC:'):
                # Truncate very long documents
                text = doc.raw_text
                if len(text) > 10000:
                    text = text[:8000] + "\n\n[...truncated...]\n\n" + text[-2000:]
                parts.append(f"\n## Content\n{text}")
            if parts:
                return "\n".join(parts)
    except Exception as e:
        logger.warning(f"Cold fetch document {record_id} failed: {e}")
    return None

def _fetch_cold_project(db: Session, record_id: str) -> Optional[str]:
    """Fetch project details."""
    try:
        from app.memory.models import Project
        proj = db.query(Project).filter(Project.id == int(record_id)).first()
        if proj:
            parts = [f"# Project: {proj.name}"]
            if proj.description:
                parts.append(f"\n{proj.description}")
            # Get message count
            msg_count = len(proj.messages) if proj.messages else 0
            note_count = len(proj.notes) if proj.notes else 0
            file_count = len(proj.files) if proj.files else 0
            parts.append(f"\nStats: {msg_count} messages, {note_count} notes, {file_count} files")
            return "\n".join(parts)
    except Exception as e:
        logger.warning(f"Cold fetch project {record_id} failed: {e}")
    return None

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
# RETRIEVAL REINFORCEMENT — Job 2 unified strength model (2026-06-10)
# =============================================================================

def reinforce_accessed(db, records, bump: float = 0.02, cap: float = 1.0) -> int:
    """
    Touch hot-index entries that were actually returned to the model.

    The counter-force to _demote_stale_hot_index in decay_job.py: memories
    that get USED have updated_at refreshed (resetting the staleness clock)
    and retrieval_priority bumped slightly (capped). Memories that are never
    retrieved drift down via demotion; memories you keep returning to stay
    strong — human-like recall strengthening.

    Called only for topical retrievals (tag/entity matched) or deep-context
    queries (D2+) — never for the default D1 hot-layer skim, which would
    otherwise reinforce the same top-N records on every casual message
    (rich-get-richer feedback loop).

    Best-effort: failures roll back and are swallowed by the caller.
    Returns count of records touched.
    """
    from datetime import datetime, timezone
    from app.astra_memory.preference_models import HotIndex

    if not records:
        return 0

    now = datetime.now(timezone.utc)
    touched = 0
    try:
        for record in records:
            hot = db.query(HotIndex).filter(
                HotIndex.record_type == record.record_type,
                HotIndex.record_id == record.record_id,
            ).first()
            if not hot:
                continue
            hot.updated_at = now
            hot.retrieval_priority = min((hot.retrieval_priority or 0.0) + bump, cap)
            touched += 1
        if touched:
            db.commit()
    except Exception:
        db.rollback()
        raise
    return touched

# Purpose: retrieval utils
# Called-by: app.astra_memory.retrieval
# Depends-on: app.astra_memory.confidence_config, app.astra_memory.preference_models, app.memory.models
# Last-renovated: 2026-06-25 (get_applicable_preferences: ACTIVE-only, drop SUPERSEDED/EXPIRED)
from __future__ import annotations
import logging
from app.astra_memory.confidence_config import get_config
from app.astra_memory.preference_models import IntentDepth, PreferenceRecord, RecordStatus
from sqlalchemy import or_
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


# 2026-06-25 gate retune. The previous lists triggered on ubiquitous
# conversational words — D2 on bare "how/why/what", D3 on "all/full/
# architecture", D4 on "history/evidence/debug" — so ordinary spoken turns
# (incl. "watch this video") landed at D3/D4 and pulled 17–44 cold records,
# producing a 20–24KB block that swamped the live message and caused topic
# drift. The rule now: everyday talk falls through to D1 (still enriched —
# hot layer, ~5 records); D2/D3/D4 require an explicit explanatory / deep /
# forensic FRAME, not a single common word. Explicit /deep, /forensic etc.
# tokens (EXPLICIT_DEPTH_TOKENS) remain the manual override.
DEPTH_KEYWORDS = {
    # D0 — pure pleasantries / acknowledgements (no memory at all).
    IntentDepth.D0: {
        "triggers": ["thank you", "thanks", "cheers", "goodbye", "good night",
                     "sounds good", "no worries"],
        "patterns": [
            r"^(hi|hii|hey|hello|yo|sup|morning|thanks|thank you|cheers|bye|"
            r"goodbye|ok|okay|kk|sure|yep|yes|no|nope|nah|cool|nice|great|"
            r"perfect|lol|haha|right|gotcha|good night)[\s!?.]*$",
        ],
    },
    # D1 — DEFAULT tier. Everyday conversation gets light, on-topic enrichment
    # (hot layer, ~5 records). No triggers needed — this is the fall-through.
    # A couple of "keep it tight" cues live here so they never escalate.
    IntentDepth.D1: {
        "triggers": ["briefly", "quick question", "in short", "one-liner", "tldr"],
        "patterns": [],
    },
    # D2 — explicit request to EXPLAIN / WALK THROUGH something. Needs an
    # explanatory frame; bare what/why/how (which appear in normal speech) do
    # NOT escalate — only an explanatory verb structure does.
    IntentDepth.D2: {
        "triggers": ["explain", "walk me through", "talk me through", "break it down",
                     "break down", "how does", "how do i", "how do you", "step by step"],
        "patterns": [
            r"\bwhy (?:do|does|did|is|are|would|won't|can't) \w",
            r"\bwhat(?:'s| is| are) the (?:difference|point|deal|trade-?off|"
            r"reason|status|best way|catch)\b",
        ],
    },
    # D3 — explicit request for DEPTH / the full picture on a topic.
    IntentDepth.D3: {
        "triggers": ["deep dive", "in detail", "in depth", "in-depth",
                     "detailed breakdown", "full breakdown", "full picture",
                     "comprehensive", "thorough", "everything about",
                     "full spec", "full specification", "full rundown"],
        "patterns": [
            r"\b(?:deep|detailed|thorough) (?:dive|analysis|breakdown|look|review)\b",
            r"\bgive me (?:the )?(?:full|whole|complete|entire) \w",
            r"\bwalk me through (?:the )?(?:whole|entire|full|complete)\b",
            r"\b(?:tell|explain|give) me everything\b",
        ],
    },
    # D4 — explicit FORENSIC / AUDIT intent: evidence, full history, every change.
    IntentDepth.D4: {
        "triggers": ["forensic", "full audit", "audit trail", "every change",
                     "all changes", "diff history", "change history",
                     "complete timeline", "full history", "full timeline",
                     "line by line", "exhaustive"],
        "patterns": [
            r"\bshow me (?:all|every|each) \w",
            r"\bwhat exactly (?:changed|happened|broke)\b",
            r"\b(?:full|complete|entire) (?:audit|history|timeline|ledger|record)\b",
        ],
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
    """Fetch full document content from DocumentContent table.

    record_id is the FILE id (the key the embeddings table and the hot
    index use for documents, 2026-06-12); falls back to DocumentContent.id
    for any legacy references. Includes the file path so ASTRA can say
    "we have a document about this — here it is".
    """
    try:
        from app.memory.models import DocumentContent, File
        doc = db.query(DocumentContent).filter(
            DocumentContent.file_id == int(record_id)
        ).first()
        if not doc:
            doc = db.query(DocumentContent).filter(
                DocumentContent.id == int(record_id)
            ).first()
        if doc:
            parts = []
            if doc.filename:
                parts.append(f"# Document: {doc.filename}")
            try:
                file_rec = db.query(File).filter(File.id == doc.file_id).first()
                if file_rec and file_rec.path:
                    parts.append(f"Path: {file_rec.path}")
            except Exception:
                pass
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

def _fetch_cold_manifest(db: Session, record_id: str) -> Optional[str]:
    """Fetch the capability manifest from its cold file (Job 4, 2026-06-12)."""
    try:
        from app.self_model.capability_manifest import read_manifest
        return read_manifest()
    except Exception as e:
        logger.warning(f"Cold fetch manifest failed: {e}")
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
    
    # Only currently-applicable records. ACTIVE by default; SUPERSEDED (a newer
    # value already replaced this one — see document_knowledge_promoter) and
    # EXPIRED (decayed below belief) must never be injected, yet the old
    # `!= DISPUTED` filter let them through (777 stale SUPERSEDED rows live = the
    # bulk of the per-turn memory-block bloat). include_disputed also surfaces
    # DISPUTED records for callers that weigh contradictions.
    allowed_statuses = [RecordStatus.ACTIVE]
    if include_disputed:
        allowed_statuses.append(RecordStatus.DISPUTED)
    query = query.filter(PreferenceRecord.status.in_(allowed_statuses))
    
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

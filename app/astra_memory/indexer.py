# FILE: app/astra_memory/indexer.py
# Purpose: ASTRA Memory Hot Index Population
# Called-by: app.astra_memory.router, main
# Depends-on: app.astra_memory.models, app.astra_memory.preference_models, app.astra_memory.retrieval, app.astra_memory.topic_tagger (+2 more)
# Last-renovated: 2026-06-12
"""
ASTRA Memory Hot Index Population

Indexes existing data into the hot index for fast retrieval:
- Messages (assistant responses)
- Projects
- Notes
- Jobs (when available)
- Specs/Architecture (when available)

Run with: python -m app.astra_memory.indexer
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db import SessionLocal
from app.astra_memory.retrieval import (
    upsert_hot_index,
    upsert_summary_pyramid,
    RetrievalCost,
)
from app.astra_memory.preference_models import HotIndex, SummaryPyramid

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE INDEXING
# =============================================================================

def _summarize_message(content: str, max_len: int = 100) -> str:
    """Create a one-liner summary from message content."""
    # Strip common prefixes
    content = content.strip()
    
    # Take first sentence or max_len chars
    for end in ['. ', '.\n', '! ', '? ']:
        idx = content.find(end)
        if 0 < idx < max_len:
            return content[:idx + 1]
    
    if len(content) <= max_len:
        return content
    
    return content[:max_len].rsplit(' ', 1)[0] + '...'


# Phase 9 retrieval-quality work (2026-05-13):
# Replaced the hardcoded six-tag local extractor with the topic_tagger
# module. Old extractor only knew code/python/testing/architecture/
# documentation/debugging — so every TAO / immigration / legal / lifestyle
# discussion was tagged with a code label or untagged, making it
# invisible to topic-driven retrieval. The new tagger covers ~11 topic
# domains plus the legacy code tags, and also produces entity arrays
# (specific names like TAO, Yodel, Leigh Day, ASTRA) for high-precision
# match.
from app.astra_memory.topic_tagger import extract_tags, extract_entities


def _extract_tags_from_content(content: str) -> List[str]:
    """
    Wrapper kept for backward compatibility with any external callers.
    New code should call topic_tagger.extract_tags directly.
    """
    return extract_tags(content)


def index_messages(db: Session, project_id: Optional[int] = None, limit: int = 1000) -> int:
    """
    Index assistant messages to hot index.
    
    Returns count of messages indexed.
    """
    from app.memory.models import Message
    
    query = db.query(Message).filter(Message.role == 'assistant')
    
    if project_id:
        query = query.filter(Message.project_id == project_id)
    
    messages = query.order_by(Message.created_at.desc()).limit(limit).all()
    
    count = 0
    for msg in messages:
        content = msg.content or ""
        if len(content) < 50:  # Skip very short messages
            continue
        
        one_liner = _summarize_message(content)
        # Phase 9 (2026-05-13): also extract entities so retrieval can
        # match on specific names (TAO, Yodel etc.) rather than only
        # by coarse domain. Empty list — NOT None — when nothing matches
        # so the SQLite JSON column stores [] instead of 'null'.
        tags = extract_tags(content)
        entities = extract_entities(content)
        
        # Create title from first line or summary
        first_line = content.split('\n')[0][:80]
        title = first_line if len(first_line) > 20 else one_liner[:80]
        
        upsert_hot_index(
            db=db,
            record_type="message",
            record_id=str(msg.id),
            title=title,
            one_liner=one_liner,
            tags=tags,
            entities=entities,
            retrieval_priority=0.4,  # Messages are lower priority than structured data
            retrieval_cost=RetrievalCost.TINY,
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} messages")
    return count


# =============================================================================
# PROJECT INDEXING
# =============================================================================

def index_projects(db: Session) -> int:
    """Index projects to hot index."""
    from app.memory.models import Project
    
    projects = db.query(Project).all()
    
    count = 0
    for proj in projects:
        description = proj.description or f"Project: {proj.name}"

        # Combine name + description for tag/entity extraction
        combined = f"{proj.name}\n{description}"
        tags = extract_tags(combined)
        if "project" not in tags:
            tags = ["project"] + tags
        entities = extract_entities(combined)

        upsert_hot_index(
            db=db,
            record_type="project",
            record_id=str(proj.id),
            title=proj.name,
            one_liner=description[:200],
            tags=tags,
            entities=entities,
            retrieval_priority=0.6,  # Projects are higher priority
            retrieval_cost=RetrievalCost.TINY,
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} projects")
    return count


# =============================================================================
# NOTE INDEXING
# =============================================================================

def index_notes(db: Session, project_id: Optional[int] = None) -> int:
    """Index notes to hot index."""
    from app.memory.models import Note
    
    query = db.query(Note)
    if project_id:
        query = query.filter(Note.project_id == project_id)
    
    notes = query.all()
    
    count = 0
    for note in notes:
        content = note.content or ""
        one_liner = _summarize_message(content)

        # Parse tags from note.tags field, plus topic-tagger output, plus
        # entities from content.
        explicit_tags: List[str] = []
        if note.tags:
            explicit_tags = [t.strip() for t in note.tags.split(',') if t.strip()]
        tags = sorted(set(explicit_tags) | set(extract_tags(content)))[:8]
        entities = extract_entities(content)

        upsert_hot_index(
            db=db,
            record_type="note",
            record_id=str(note.id),
            title=note.title,
            one_liner=one_liner,
            tags=tags,
            entities=entities,
            retrieval_priority=0.7,  # Notes are high priority (user-created)
            retrieval_cost=RetrievalCost.TINY,
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} notes")
    return count


# =============================================================================
# DOCUMENT INDEXING (Job 3, 2026-06-12)
# =============================================================================

def index_documents(db: Session, project_id: Optional[int] = None) -> int:
    """Index DocumentContent records to the hot index.

    Documents (including enriched images/audio) were previously invisible
    to conversational retrieval: 526 'file' embeddings existed but zero
    'document' hot rows, so both the keyword channel and the semantic
    channel dropped them. Delegates to enrichment.hot_index_document so
    the record key (file_id) stays consistent everywhere.
    """
    from app.memory.models import DocumentContent, File
    from app.memory.enrichment.media_enricher import hot_index_document

    query = db.query(DocumentContent)
    if project_id:
        query = query.filter(DocumentContent.project_id == project_id)

    count = 0
    for doc in query.all():
        try:
            path = None
            file_rec = db.query(File).filter(File.id == doc.file_id).first()
            if file_rec:
                path = file_rec.path
            if hot_index_document(db, doc, doc.summary, path):
                count += 1
        except Exception as e:
            logger.warning(f"[indexer] Document {doc.id} failed: {e}")

    logger.info(f"[indexer] Indexed {count} documents")
    return count


# =============================================================================
# JOB INDEXING
# =============================================================================

def index_jobs(db: Session, limit: int = 500) -> int:
    """Index ASTRA jobs to hot index."""
    from app.astra_memory.models import AstraJob
    
    jobs = db.query(AstraJob).order_by(AstraJob.created_at.desc()).limit(limit).all()
    
    count = 0
    for job in jobs:
        intent = job.user_intent or f"Job {job.job_id}"
        one_liner = _summarize_message(intent)

        tags = ["job", job.status]
        if job.primary_provider:
            tags.append(job.primary_provider)
        # Phase 9 (2026-05-13): include topic tags so jobs are findable by
        # what they were about, not only by status.
        tags = sorted(set(tags) | set(extract_tags(intent)))[:8]
        entities = extract_entities(intent)

        upsert_hot_index(
            db=db,
            record_type="job",
            record_id=job.job_id,
            title=f"Job: {one_liner[:50]}",
            one_liner=one_liner,
            tags=tags,
            entities=entities,
            retrieval_priority=0.5,
            retrieval_cost=RetrievalCost.MEDIUM,  # Jobs may have more data
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} jobs")
    return count


# =============================================================================
# GLOBAL PREF INDEXING (Legacy)
# =============================================================================

def index_global_prefs(db: Session) -> int:
    """Index legacy global prefs to hot index."""
    from app.astra_memory.models import GlobalPref
    
    prefs = db.query(GlobalPref).filter(GlobalPref.active == True).all()
    
    count = 0
    for pref in prefs:
        value_str = str(pref.value)
        topic_tags = extract_tags(f"{pref.key} {value_str}")
        base_tags = ["preference", pref.category] if pref.category else ["preference"]
        tags = sorted(set(base_tags) | set(topic_tags))[:8]
        entities = extract_entities(f"{pref.key} {value_str}")

        upsert_hot_index(
            db=db,
            record_type="global_pref",
            record_id=pref.key,
            title=pref.key,
            one_liner=value_str[:200],
            tags=tags,
            entities=entities,
            retrieval_priority=0.8,  # Prefs are high priority
            retrieval_cost=RetrievalCost.TINY,
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} global prefs")
    return count


# =============================================================================
# MAIN INDEXER
# =============================================================================

def run_full_index(
    db: Optional[Session] = None,
    cleanup_first: bool = False,
) -> Dict[str, int]:
    """
    Run full indexing of all available data.

    cleanup_first: accepted for the /index endpoint, which has passed it
    since v1 (the missing parameter made that endpoint 500 on every call —
    fixed 2026-06-12). Upserts already refresh every record in place;
    stale-row removal is the decay job's responsibility, so True adds
    nothing beyond the refresh today.

    Returns dict of counts by type.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        results = {
            "projects": index_projects(db),
            "notes": index_notes(db),
            "messages": index_messages(db),
            "documents": index_documents(db),
            "jobs": index_jobs(db),
            "global_prefs": index_global_prefs(db),
        }
        
        total = sum(results.values())
        print(f"[indexer] Total indexed: {total}")
        
        return results
        
    finally:
        if close_db:
            db.close()


def get_index_stats(db: Optional[Session] = None) -> Dict[str, Any]:
    """Get stats about the current hot index."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        total = db.query(HotIndex).count()
        
        # Count by type
        by_type = {}
        for row in db.execute(text(
            "SELECT record_type, COUNT(*) FROM astra_hot_index GROUP BY record_type"
        )):
            by_type[row[0]] = row[1]
        
        return {
            "total": total,
            "by_type": by_type,
        }
        
    finally:
        if close_db:
            db.close()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== ASTRA Memory Hot Index Population ===")
    print()
    
    # Show current stats
    stats_before = get_index_stats()
    print(f"Before: {stats_before['total']} records in hot index")
    print(f"  By type: {stats_before['by_type']}")
    print()
    
    # Run indexing
    results = run_full_index()
    print()
    print("Indexed:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    # Show new stats
    stats_after = get_index_stats()
    print()
    print(f"After: {stats_after['total']} records in hot index")
    print(f"  By type: {stats_after['by_type']}")

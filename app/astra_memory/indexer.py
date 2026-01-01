# FILE: app/astra_memory/indexer.py
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


def _extract_tags_from_content(content: str) -> List[str]:
    """Extract simple tags from content."""
    tags = []
    
    # Code-related
    if any(kw in content.lower() for kw in ['def ', 'class ', 'import ', 'function', 'async ']):
        tags.append('code')
    if 'python' in content.lower():
        tags.append('python')
    if any(kw in content.lower() for kw in ['test', 'pytest', 'unittest']):
        tags.append('testing')
    
    # Architecture
    if any(kw in content.lower() for kw in ['architect', 'design', 'structure', 'pattern']):
        tags.append('architecture')
    
    # Documentation
    if any(kw in content.lower() for kw in ['document', 'readme', 'guide', 'tutorial']):
        tags.append('documentation')
    
    # Error/debug
    if any(kw in content.lower() for kw in ['error', 'bug', 'fix', 'debug', 'traceback']):
        tags.append('debugging')
    
    return tags[:5]  # Max 5 tags


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
        tags = _extract_tags_from_content(content)
        
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
        
        upsert_hot_index(
            db=db,
            record_type="project",
            record_id=str(proj.id),
            title=proj.name,
            one_liner=description[:200],
            tags=["project"],
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
        
        # Parse tags from note.tags field
        tags = []
        if note.tags:
            tags = [t.strip() for t in note.tags.split(',')]
        tags.extend(_extract_tags_from_content(content))
        tags = list(set(tags))[:5]
        
        upsert_hot_index(
            db=db,
            record_type="note",
            record_id=str(note.id),
            title=note.title,
            one_liner=one_liner,
            tags=tags,
            retrieval_priority=0.7,  # Notes are high priority (user-created)
            retrieval_cost=RetrievalCost.TINY,
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} notes")
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
        
        upsert_hot_index(
            db=db,
            record_type="job",
            record_id=job.job_id,
            title=f"Job: {one_liner[:50]}",
            one_liner=one_liner,
            tags=tags,
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
        upsert_hot_index(
            db=db,
            record_type="global_pref",
            record_id=pref.key,
            title=pref.key,
            one_liner=str(pref.value)[:200],
            tags=["preference", pref.category] if pref.category else ["preference"],
            retrieval_priority=0.8,  # Prefs are high priority
            retrieval_cost=RetrievalCost.TINY,
        )
        count += 1
    
    logger.info(f"[indexer] Indexed {count} global prefs")
    return count


# =============================================================================
# MAIN INDEXER
# =============================================================================

def run_full_index(db: Optional[Session] = None) -> Dict[str, int]:
    """
    Run full indexing of all available data.
    
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
    import sys
    
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

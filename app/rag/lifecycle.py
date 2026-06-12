# FILE: app/rag/lifecycle.py
# Purpose: RAG Lifecycle Manager.
# Called-by: app.memory.commands.purge, app.memory.domains.architecture, app.memory.lifecycle_hook
# Depends-on: app.memory.architecture_models, app.rag.models
# Last-renovated: 2026-06-11
"""
RAG Lifecycle Manager.

Handles status transitions for architecture entries during refactoring:
  - quarantine_file: Mark a monolith's entries as quarantined after refactor
  - activate_file: Index new package files as active
  - purge_quarantined: Permanently remove quarantined entries (manual trigger)
  - rescan_file: Update RAG when a file changes on disk

Status flow: ACTIVE -> QUARANTINED -> PURGED

All queries should filter by status='active' to exclude stale data.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.rag.models import ArchCodeChunk
from app.memory.architecture_models import ArchitectureFileIndex

logger = logging.getLogger(__name__)

# Status constants
ACTIVE = "active"
QUARANTINED = "quarantined"
PURGED = "purged"


def quarantine_file(
    db: Session,
    file_path: str,
    refactor_job_id: str,
) -> int:
    """
    Quarantine all entries for a file that has been refactored.
    
    Marks the file and all its code chunks as quarantined.
    They stop appearing in queries but remain for rollback.
    
    Args:
        db: Database session
        file_path: Absolute path of the monolith being replaced
        refactor_job_id: ID of the refactor job doing this
        
    Returns:
        Number of chunks quarantined
    """
    now = datetime.utcnow()
    
    # Quarantine file index entry
    file_entries = db.query(ArchitectureFileIndex).filter(
        and_(
            ArchitectureFileIndex.path == file_path,
            ArchitectureFileIndex.status == ACTIVE,
        )
    ).all()
    
    for entry in file_entries:
        entry.status = QUARANTINED
        entry.quarantined_at = now
        entry.refactor_job_id = refactor_job_id
    
    # Quarantine all code chunks for this file
    chunks = db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.file_path == file_path,
            ArchCodeChunk.status == ACTIVE,
        )
    ).all()
    
    for chunk in chunks:
        chunk.status = QUARANTINED
        chunk.refactor_job_id = refactor_job_id
    
    db.commit()
    
    count = len(chunks)
    logger.info(
        f"[rag_lifecycle] Quarantined {count} chunks + {len(file_entries)} file entries "
        f"for {file_path} (job={refactor_job_id})"
    )
    return count


def activate_new_file(
    db: Session,
    file_path: str,
    source_monolith: Optional[str] = None,
    refactor_job_id: Optional[str] = None,
    package_role: Optional[str] = None,
) -> None:
    """
    Mark a newly created file's entries as active with refactor metadata.
    
    Called after the architecture scan indexes new package files.
    Sets the source_monolith, refactor_job_id, and package_role on all
    chunks belonging to this file.
    
    Args:
        db: Database session
        file_path: Absolute path of the new file
        source_monolith: Path of the original monolith this was extracted from
        refactor_job_id: ID of the refactor job that created this
        package_role: Role in package (core, models, utils, etc.)
    """
    chunks = db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.file_path == file_path,
            ArchCodeChunk.status == ACTIVE,
        )
    ).all()
    
    for chunk in chunks:
        chunk.source_monolith = source_monolith
        chunk.refactor_job_id = refactor_job_id
        chunk.package_role = package_role
    
    # Also tag the file index entry
    file_entries = db.query(ArchitectureFileIndex).filter(
        and_(
            ArchitectureFileIndex.path == file_path,
            ArchitectureFileIndex.status == ACTIVE,
        )
    ).all()
    
    for entry in file_entries:
        entry.source_monolith = source_monolith
        entry.refactor_job_id = refactor_job_id
    
    db.commit()
    
    logger.info(
        f"[rag_lifecycle] Activated {len(chunks)} chunks for {file_path} "
        f"(from={source_monolith}, role={package_role})"
    )


def rollback_quarantine(
    db: Session,
    refactor_job_id: str,
) -> int:
    """
    Rollback a quarantine — restore entries to active status.
    
    Used when a refactor fails and we need to undo the quarantine.
    Also removes any new entries tagged with this job ID.
    
    Args:
        db: Database session
        refactor_job_id: The job to rollback
        
    Returns:
        Number of entries restored
    """
    # Restore quarantined chunks
    quarantined_chunks = db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.refactor_job_id == refactor_job_id,
            ArchCodeChunk.status == QUARANTINED,
        )
    ).all()
    
    for chunk in quarantined_chunks:
        chunk.status = ACTIVE
        chunk.refactor_job_id = None
    
    # Restore quarantined file entries
    quarantined_files = db.query(ArchitectureFileIndex).filter(
        and_(
            ArchitectureFileIndex.refactor_job_id == refactor_job_id,
            ArchitectureFileIndex.status == QUARANTINED,
        )
    ).all()
    
    for entry in quarantined_files:
        entry.status = ACTIVE
        entry.quarantined_at = None
        entry.refactor_job_id = None
    
    # Remove new entries that were created by this job
    # (the new package files that shouldn't exist if we're rolling back)
    new_chunks = db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.refactor_job_id == refactor_job_id,
            ArchCodeChunk.status == ACTIVE,
            ArchCodeChunk.source_monolith.isnot(None),
        )
    ).all()
    
    for chunk in new_chunks:
        db.delete(chunk)
    
    db.commit()
    
    restored = len(quarantined_chunks)
    removed = len(new_chunks)
    logger.info(
        f"[rag_lifecycle] Rolled back job {refactor_job_id}: "
        f"restored {restored} chunks, removed {removed} new chunks"
    )
    return restored


def purge_quarantined(
    db: Session,
    refactor_job_id: Optional[str] = None,
) -> int:
    """
    Permanently remove quarantined entries.
    
    This is a manual, irreversible operation. Only call after confirming
    the refactored packages are stable.
    
    Args:
        db: Database session
        refactor_job_id: Optional — purge only this job's quarantined entries.
                         If None, purge ALL quarantined entries.
                         
    Returns:
        Number of entries purged
    """
    chunk_filter = ArchCodeChunk.status == QUARANTINED
    file_filter = ArchitectureFileIndex.status == QUARANTINED
    
    if refactor_job_id:
        chunk_filter = and_(chunk_filter, ArchCodeChunk.refactor_job_id == refactor_job_id)
        file_filter = and_(file_filter, ArchitectureFileIndex.refactor_job_id == refactor_job_id)
    
    chunk_count = db.query(ArchCodeChunk).filter(chunk_filter).count()
    file_count = db.query(ArchitectureFileIndex).filter(file_filter).count()
    
    if chunk_count > 0:
        db.query(ArchCodeChunk).filter(chunk_filter).delete(synchronize_session="fetch")
    if file_count > 0:
        db.query(ArchitectureFileIndex).filter(file_filter).delete(synchronize_session="fetch")
    
    db.flush()
    db.commit()
    
    logger.info(
        f"[rag_lifecycle] Purged {chunk_count} chunks + {file_count} file entries"
        + (f" for job {refactor_job_id}" if refactor_job_id else " (all quarantined)")
    )
    return chunk_count


def get_lifecycle_stats(db: Session) -> dict:
    """
    Get current lifecycle statistics.
    
    Returns counts of active, quarantined, and total entries.
    """
    from sqlalchemy import func
    
    chunk_stats = db.query(
        ArchCodeChunk.status,
        func.count(ArchCodeChunk.id),
    ).group_by(ArchCodeChunk.status).all()
    
    file_stats = db.query(
        ArchitectureFileIndex.status,
        func.count(ArchitectureFileIndex.id),
    ).group_by(ArchitectureFileIndex.status).all()
    
    return {
        "chunks": {s: c for s, c in chunk_stats},
        "files": {s: c for s, c in file_stats},
    }

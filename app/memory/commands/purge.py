# FILE: app/memory/commands/purge.py
# Purpose: Purge quarantine command handler.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.db, app.memory.architecture_models, app.memory.rag_entries_model, app.rag.lifecycle (+1 more)
# Last-renovated: 2026-06-11
"""
Purge quarantine command handler.

Triggered by: "Astra, command: purge quarantine"

Permanently deletes all QUARANTINED entries from:
  - arch_code_chunks (architecture domain)
  - architecture_file_index (file metadata)
  - rag_entries (unified memory table)

Also cleans up any .quarantined/ directories left on disk.

This is irreversible. Only call after confirming refactored packages
are stable and boot-tested.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db import get_db_session
from app.rag.lifecycle import purge_quarantined as purge_arch_quarantined
from app.rag.models import ArchCodeChunk
from app.memory.rag_entries_model import RAGEntry
from app.memory.architecture_models import ArchitectureFileIndex

logger = logging.getLogger(__name__)

# Project root for .quarantined/ cleanup
PROJECT_ROOT = r"D:\Orb"


@dataclass
class PurgeResult:
    """Result of a purge operation."""
    arch_chunks_purged: int = 0
    arch_files_purged: int = 0
    rag_entries_purged: int = 0
    quarantine_dirs_cleaned: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def total_purged(self) -> int:
        return (
            self.arch_chunks_purged
            + self.arch_files_purged
            + self.rag_entries_purged
        )

    def summary(self) -> str:
        """Human-readable summary for chat response."""
        parts = []
        if self.arch_chunks_purged:
            parts.append(f"{self.arch_chunks_purged} architecture chunks")
        if self.arch_files_purged:
            parts.append(f"{self.arch_files_purged} file index entries")
        if self.rag_entries_purged:
            parts.append(f"{self.rag_entries_purged} RAG entries")
        if self.quarantine_dirs_cleaned:
            parts.append(
                f"{self.quarantine_dirs_cleaned} .quarantined directories"
            )

        if not parts:
            return "No quarantined entries found. Nothing to purge."

        return f"Purged: {', '.join(parts)}."


def execute_purge(
    refactor_job_id: Optional[str] = None,
) -> PurgeResult:
    """
    Execute full purge of all quarantined entries.

    Args:
        refactor_job_id: Optional — purge only this job's entries.
                         If None, purge ALL quarantined entries.

    Returns:
        PurgeResult with counts and any errors.
    """
    result = PurgeResult()
    db = get_db_session()

    try:
        # ── Purge arch_code_chunks + architecture_file_index ────────
        try:
            count = purge_arch_quarantined(db, refactor_job_id)
            result.arch_chunks_purged = count
        except Exception as e:
            msg = f"Architecture purge failed: {e}"
            logger.error("[purge] %s", msg)
            result.errors.append(msg)

        # ── Purge rag_entries ───────────────────────────────────────
        try:
            count = _purge_rag_entries(db, refactor_job_id)
            result.rag_entries_purged = count
        except Exception as e:
            msg = f"RAG entries purge failed: {e}"
            logger.error("[purge] %s", msg)
            result.errors.append(msg)

        # ── Clean .quarantined/ dirs from disk ──────────────────────
        try:
            count = _clean_quarantine_dirs()
            result.quarantine_dirs_cleaned = count
        except Exception as e:
            msg = f"Quarantine dir cleanup failed: {e}"
            logger.warning("[purge] %s", msg)
            result.errors.append(msg)

        db.commit()

    except Exception as e:
        db.rollback()
        result.errors.append(f"Purge transaction failed: {e}")
        logger.error("[purge] Transaction failed: %s", e)
    finally:
        db.close()

    logger.info("[purge] %s", result.summary())
    return result


def get_quarantine_stats() -> dict:
    """
    Get counts of quarantined entries across all tables.

    Returns dict with per-table counts, useful for "show quarantine status".
    """
    db = get_db_session()
    try:
        arch_chunks = db.query(func.count(ArchCodeChunk.id)).filter(
            ArchCodeChunk.status == "quarantined"
        ).scalar() or 0

        arch_files = db.query(func.count(ArchitectureFileIndex.id)).filter(
            ArchitectureFileIndex.status == "quarantined"
        ).scalar() or 0

        rag_entries = db.query(func.count(RAGEntry.id)).filter(
            RAGEntry.status == "QUARANTINED"
        ).scalar() or 0

        return {
            "arch_chunks": arch_chunks,
            "arch_files": arch_files,
            "rag_entries": rag_entries,
            "total": arch_chunks + arch_files + rag_entries,
        }
    finally:
        db.close()


def _purge_rag_entries(
    db: Session,
    refactor_job_id: Optional[str] = None,
) -> int:
    """Delete QUARANTINED rag_entries (and redirect entries for purged paths)."""
    filters = [RAGEntry.status == "QUARANTINED"]

    if refactor_job_id:
        filters.append(RAGEntry.refactor_job_id == refactor_job_id)

    count = db.query(RAGEntry).filter(*filters).count()

    if count > 0:
        db.query(RAGEntry).filter(*filters).delete(
            synchronize_session="fetch"
        )

    # Also purge redirect entries for the same job
    redirect_filters = [
        RAGEntry.domain == "redirect",
    ]
    if refactor_job_id:
        redirect_filters.append(RAGEntry.refactor_job_id == refactor_job_id)

    redirect_count = db.query(RAGEntry).filter(*redirect_filters).count()
    if redirect_count > 0:
        db.query(RAGEntry).filter(*redirect_filters).delete(
            synchronize_session="fetch"
        )
        count += redirect_count

    return count


def _clean_quarantine_dirs() -> int:
    """
    Remove .quarantined/ directories from the project tree.

    These are left behind by package_quarantine.py during refactoring.
    Only removes empty or all-quarantined directories.
    """
    cleaned = 0

    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        # Skip .git and node_modules
        if ".git" in dirpath or "node_modules" in dirpath:
            continue

        if ".quarantined" in dirnames:
            q_path = os.path.join(dirpath, ".quarantined")
            try:
                shutil.rmtree(q_path)
                cleaned += 1
                logger.info("[purge] Removed %s", q_path)
            except Exception as e:
                logger.warning("[purge] Failed to remove %s: %s", q_path, e)

    return cleaned

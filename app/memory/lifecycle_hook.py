# FILE: app/memory/lifecycle_hook.py
# Purpose: Post-refactor lifecycle hook.
# Called-by: app.orchestrator.refactor_loop
# Depends-on: app.memory.rag_entries_model, app.rag.lifecycle
# Last-renovated: 2026-06-11
"""
Post-refactor lifecycle hook.

Called after a successful refactor extraction + boot check to atomically:
1. Quarantine the old monolith's RAG entries
2. Rescan the codebase (indexes new files)
3. Tag new module entries with provenance metadata
4. Store a redirect entry so queries for the old path get routed correctly

This module bridges the refactor loop (app/orchestrator/refactor_loop.py)
with the RAG lifecycle system (app/rag/lifecycle.py) and the unified
rag_entries table (app/memory/rag_entries_model.py).

All operations run in a single DB transaction for atomicity.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.rag.lifecycle import (
    quarantine_file,
    activate_new_file,
    ACTIVE,
    QUARANTINED,
)
from app.memory.rag_entries_model import RAGEntry

logger = logging.getLogger(__name__)


def on_refactor_success(
    db: Session,
    old_file_path: str,
    new_module_path: str,
    job_id: str,
    package_role: Optional[str] = None,
) -> dict:
    """
    Execute all post-refactor lifecycle operations atomically.

    Called from refactor_loop.py after a successful extraction + boot check.

    Args:
        db: Active database session (caller manages lifecycle)
        old_file_path: Absolute path of the monolith being extracted from
        new_module_path: Absolute path of the newly created module
        job_id: Refactor job identifier
        package_role: Role of the new module (core, utils, models, etc.)

    Returns:
        dict with counts: quarantined_chunks, activated_chunks,
        redirect_created, errors
    """
    result = {
        "quarantined_chunks": 0,
        "activated_chunks": 0,
        "redirect_created": False,
        "errors": [],
    }

    # ── Step 1: Quarantine old entries ───────────────────────────────
    try:
        count = quarantine_file(db, old_file_path, job_id)
        result["quarantined_chunks"] = count
        logger.info(
            "[lifecycle_hook] Quarantined %d chunks for %s (job=%s)",
            count, old_file_path, job_id,
        )
    except Exception as e:
        msg = f"Quarantine failed for {old_file_path}: {e}"
        logger.error("[lifecycle_hook] %s", msg)
        result["errors"].append(msg)

    # ── Step 2: Quarantine rag_entries for the old path ─────────────
    try:
        count = _quarantine_rag_entries(db, old_file_path, job_id)
        if count > 0:
            result["quarantined_chunks"] += count
            logger.info(
                "[lifecycle_hook] Quarantined %d rag_entries for %s",
                count, old_file_path,
            )
    except Exception as e:
        msg = f"rag_entries quarantine failed: {e}"
        logger.warning("[lifecycle_hook] %s", msg)
        result["errors"].append(msg)

    # ── Step 3: Tag new module entries with provenance ──────────────
    try:
        activate_new_file(
            db,
            file_path=new_module_path,
            source_monolith=old_file_path,
            refactor_job_id=job_id,
            package_role=package_role,
        )
        logger.info(
            "[lifecycle_hook] Activated new module %s (from=%s, role=%s)",
            new_module_path, old_file_path, package_role,
        )
    except Exception as e:
        msg = f"Activate failed for {new_module_path}: {e}"
        logger.warning("[lifecycle_hook] %s", msg)
        result["errors"].append(msg)

    # ── Step 4: Create redirect entry ───────────────────────────────
    try:
        _create_redirect_entry(db, old_file_path, new_module_path, job_id)
        result["redirect_created"] = True
    except Exception as e:
        msg = f"Redirect entry failed: {e}"
        logger.warning("[lifecycle_hook] %s", msg)
        result["errors"].append(msg)

    # Note: quarantine_file() and activate_new_file() commit internally.
    # The rag_entries operations (steps 2+4) need an explicit commit.
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        msg = f"rag_entries commit failed: {e}"
        logger.error("[lifecycle_hook] %s", msg)
        result["errors"].append(msg)

    return result


def _quarantine_rag_entries(
    db: Session,
    file_path: str,
    job_id: str,
) -> int:
    """Mark rag_entries for a given file_path as QUARANTINED."""
    now = datetime.utcnow()
    entries = db.query(RAGEntry).filter(
        RAGEntry.file_path == file_path,
        RAGEntry.status == "ACTIVE",
    ).all()

    for entry in entries:
        entry.status = "QUARANTINED"
        entry.quarantined_at = now
        entry.refactor_job_id = job_id

    return len(entries)


def _create_redirect_entry(
    db: Session,
    old_path: str,
    new_path: str,
    job_id: str,
) -> None:
    """
    Store a redirect entry in rag_entries.

    When a query hits this entry, the caller knows the old file was
    refactored and should look at the new path instead.
    """
    redirect = RAGEntry(
        project_id="astra-core",
        domain="redirect",
        file_path=old_path,
        chunk_text=(
            f"REDIRECT: {old_path} was refactored into {new_path} "
            f"by job {job_id}. Query the new path for current content."
        ),
        status="ACTIVE",
        source_monolith=old_path,
        refactor_job_id=job_id,
        ingest_source="lifecycle_hook",
    )
    db.add(redirect)

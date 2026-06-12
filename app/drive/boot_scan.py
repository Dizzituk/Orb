# FILE: app/drive/boot_scan.py
# Purpose: Boot-time filesystem scan orchestrator.
# Called-by: main
# Depends-on: app.db, app.drive.content_indexer, app.drive.manifest_models, app.drive.manifest_scanner
# Last-renovated: 2026-06-11
"""
Boot-time filesystem scan orchestrator.

Called once during ASTRA startup. Runs a tiered scan:

  Tier 1 (SYNC, every boot): Manifest scan — walk all category paths,
      record file metadata, detect new/modified/deleted files.
      Target: <5 seconds for ~17,000 files.

  Tier 2 (BACKGROUND, new/modified docs only): Content indexing —
      extract text from document files, generate embeddings, promote
      knowledge to ASTRA memory. Capped at 50 files per boot.

  Tier 3 (FUTURE): Media cataloguing — EXIF, ID3 tags, video metadata.

  Tier 4 (FUTURE): Code project indexing — extend rescan to cover
      Android project alongside ASTRA backend.

CRITICAL RULE: This scan operates on the HOST filesystem only.
When ASTRA works on its own code, it MUST use the SANDBOX clone
(192.168.250.2:8765), never the host. This scan is READ-ONLY
awareness — it never modifies host files.

Called from: main.py → on_startup()
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Any

from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


def run_boot_scan(db: DbSession) -> Dict[str, Any]:
    """
    Run the synchronous Tier 1 manifest scan at startup.

    Returns scan report dict for startup logging.
    Spawns Tier 2 content indexing as a background thread.
    """
    result: Dict[str, Any] = {
        "manifest": None,
        "content_indexing": "deferred",
    }

    # ── Tier 1: Manifest scan (synchronous, fast) ────────────────
    try:
        # Ensure table exists
        from app.drive.manifest_models import DriveFileManifest  # noqa: F401
        from app.db import engine, Base
        Base.metadata.create_all(bind=engine, tables=[
            DriveFileManifest.__table__
        ])

        from app.drive.manifest_scanner import scan_manifest
        report = scan_manifest(db)

        result["manifest"] = {
            "total": report.total_files,
            "new": report.new_files,
            "modified": report.modified_files,
            "deleted": report.deleted_files,
            "unchanged": report.unchanged_files,
            "duration_ms": report.duration_ms,
            "categories": report.categories,
        }

        logger.info(
            "[boot_scan] Manifest: %d files (%d new, %d modified, "
            "%d deleted) in %dms",
            report.total_files, report.new_files, report.modified_files,
            report.deleted_files, report.duration_ms,
        )

    except Exception as e:
        logger.error("[boot_scan] Manifest scan failed: %s", e)
        result["manifest"] = {"error": str(e)}

    # ── Tier 2: Content indexing (background thread) ─────────────
    # Only runs if there are unindexed documents.
    # Uses a separate DB session to avoid blocking the main thread.
    try:
        _spawn_content_indexing()
        result["content_indexing"] = "started"
    except Exception as e:
        logger.warning("[boot_scan] Content indexing spawn failed: %s", e)
        result["content_indexing"] = {"error": str(e)}

    return result


def _spawn_content_indexing():
    """
    Spawn content indexing as a daemon background thread.

    Uses a fresh DB session. Non-blocking — startup continues
    immediately. If it fails, it logs and exits silently.
    """
    def _run():
        try:
            from app.db import SessionLocal
            bg_db = SessionLocal()
            try:
                from app.drive.content_indexer import run_content_indexing
                report = run_content_indexing(bg_db)

                if report.files_indexed > 0:
                    logger.info(
                        "[boot_scan] Background indexing: %d document(s) "
                        "indexed, %d promoted (%dms)",
                        report.files_indexed, report.files_promoted,
                        report.duration_ms,
                    )
            finally:
                bg_db.close()
        except Exception as e:
            logger.error("[boot_scan] Background indexing crashed: %s", e)

    thread = threading.Thread(target=_run, name="boot-content-indexer", daemon=True)
    thread.start()
    logger.info("[boot_scan] Content indexing thread started")

# FILE: app/finance/services/drive_scheduler.py
"""
Scheduled polling for Google Drive statement folders.

Runs a background check every N minutes for new PDFs
in registered watch folders. Lightweight — just a
'list files' API call per folder, costs nothing.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

_scheduler_running = False
_scheduler_thread = None
_poll_interval_minutes = 30  # Default: check every 30 minutes


def start_drive_scheduler(poll_minutes: int = 30):
    """Start the background Drive folder polling scheduler."""
    global _scheduler_running, _scheduler_thread, _poll_interval_minutes
    _poll_interval_minutes = poll_minutes

    if _scheduler_running:
        logger.info("[drive_scheduler] Already running")
        return

    _scheduler_running = True
    _scheduler_thread = threading.Thread(
        target=_poll_loop, daemon=True, name="drive-watcher"
    )
    _scheduler_thread.start()
    logger.info("[drive_scheduler] Started (every %d minutes)", poll_minutes)


def stop_drive_scheduler():
    """Stop the background scheduler."""
    global _scheduler_running
    _scheduler_running = False
    logger.info("[drive_scheduler] Stopped")


def get_scheduler_status() -> dict:
    """Get current scheduler status."""
    return {
        "running": _scheduler_running,
        "poll_interval_minutes": _poll_interval_minutes,
        "thread_alive": _scheduler_thread.is_alive() if _scheduler_thread else False,
    }


def _poll_loop():
    """Main polling loop — runs in background thread."""
    global _scheduler_running

    while _scheduler_running:
        try:
            _run_scan()
        except Exception as e:
            logger.error("[drive_scheduler] Scan error: %s", e)

        # Sleep in small increments so we can stop quickly
        for _ in range(_poll_interval_minutes * 60):
            if not _scheduler_running:
                return
            import time
            time.sleep(1)


def _run_scan():
    """Execute a single scan of all watch folders."""
    from app.database import SessionLocal
    from app.finance.services.drive_watcher_service import scan_all_watch_folders
    from app.finance.services.drive_auth_service import get_drive_service

    # Skip if not authenticated
    service = get_drive_service()
    if not service:
        return

    db = SessionLocal()
    try:
        result = scan_all_watch_folders(db)
        if result.total_new_files > 0:
            logger.info(
                "[drive_scheduler] Found %d new files, imported %d transactions",
                result.total_new_files, result.total_transactions,
            )
            # TODO: Send notification to frontend via WebSocket
        else:
            logger.debug("[drive_scheduler] No new files")
    finally:
        db.close()

# FILE: app/drive/file_watcher.py
"""
Live filesystem watcher — keeps drive_file_manifest in sync.

Observes all user category paths (Documents, Pictures, Desktop, etc.)
and reacts to create / modify / move / delete events by calling
index_single_file() or removing rows.

Debounced per-path (1s settle window) so Office/editor atomic-save
patterns (temp file → rename) produce a single index call, not three.

Runs in a daemon thread. Safe to start once at backend startup.
Called from: main.py startup block.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
    FileDeletedEvent,
    DirDeletedEvent,
)
from watchdog.observers import Observer

from app.drive.file_utils import get_category_paths

logger = logging.getLogger(__name__)

# Settle window: wait this many seconds after the last event on a path
# before acting. Collapses atomic-save bursts into one index call.
DEBOUNCE_SECONDS = 1.0

# Skip the same stuff the boot scanner skips
SKIP_PREFIXES = (".", "~$", "Thumbs.db", "desktop.ini")
SKIP_DIR_NAMES = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    ".gradle", ".idea", ".vs", "$RECYCLE.BIN",
    "System Volume Information",
}


def _should_skip(path: str) -> bool:
    """Cheap filter — skip temp / hidden / build-artifact paths."""
    p = Path(path)
    if any(p.name.startswith(pref) for pref in SKIP_PREFIXES):
        return True
    # Skip if any parent dir is a build/venv/vcs dir
    for part in p.parts:
        if part in SKIP_DIR_NAMES:
            return True
    return False


def _index_path(path: str) -> None:
    """Add / update a single file in the manifest."""
    try:
        from app.drive.manifest_scanner import index_single_file
        index_single_file(path)
        logger.debug("[file_watcher] indexed: %s", path)
    except ImportError:
        # index_single_file may not exist yet — fall back to rescan helper
        try:
            from app.drive.manifest_rescan import rescan_path
            rescan_path(path)
        except Exception as e:
            logger.warning("[file_watcher] index fallback failed for %s: %s", path, e)
    except Exception as e:
        logger.warning("[file_watcher] index failed for %s: %s", path, e)


def _remove_path(path: str) -> None:
    """Remove a file row from the manifest."""
    try:
        from app.db import SessionLocal
        from app.drive.manifest_models import DriveFileManifest
        db = SessionLocal()
        try:
            db.query(DriveFileManifest).filter(
                DriveFileManifest.path == path
            ).delete(synchronize_session=False)
            db.commit()
            logger.debug("[file_watcher] removed from manifest: %s", path)
        finally:
            db.close()
    except Exception as e:
        logger.warning("[file_watcher] remove failed for %s: %s", path, e)


class _DebouncedHandler(FileSystemEventHandler):
    """Collapses rapid events on the same path into one action."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: Dict[str, float] = {}  # path -> deadline (monotonic)
        self._deleted: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._drain_loop,
            name="file_watcher_drain",
            daemon=True,
        )
        self._worker.start()

    # ── Event hooks ─────────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, deleted=False)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, deleted=False)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Source path is now gone; destination is the new file
        self._schedule(event.src_path, deleted=True)
        dest = getattr(event, "dest_path", "")
        if dest:
            self._schedule(dest, deleted=False)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
            # For dir deletions we don't recurse — boot scan / periodic
            # cleanup catches orphaned rows. File deletions handled here.
            if not event.is_directory:
                self._schedule(event.src_path, deleted=True)

    # ── Scheduling ──────────────────────────────────────────────

    def _schedule(self, path: str, deleted: bool) -> None:
        if _should_skip(path):
            return
        deadline = time.monotonic() + DEBOUNCE_SECONDS
        with self._lock:
            if deleted:
                self._deleted[path] = deadline
                # A later deletion supersedes a pending index
                self._pending.pop(path, None)
            else:
                self._pending[path] = deadline
                # A later create/modify supersedes a pending delete
                self._deleted.pop(path, None)
        self._wake.set()

    def _drain_loop(self) -> None:
        """Worker thread: fires actions once debounce windows expire."""
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()

            now = time.monotonic()
            due_index: list[str] = []
            due_remove: list[str] = []

            with self._lock:
                for path, deadline in list(self._pending.items()):
                    if deadline <= now:
                        due_index.append(path)
                        self._pending.pop(path, None)
                for path, deadline in list(self._deleted.items()):
                    if deadline <= now:
                        due_remove.append(path)
                        self._deleted.pop(path, None)

            for path in due_index:
                _index_path(path)
            for path in due_remove:
                _remove_path(path)

            # If anything still pending, wake again shortly
            with self._lock:
                if self._pending or self._deleted:
                    self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


# ─── Module-level singleton ─────────────────────────────────────

_observer: Optional[Observer] = None
_handler: Optional[_DebouncedHandler] = None


def start_file_watcher() -> dict:
    """
    Start the watchdog observer on all user category paths.

    Idempotent — safe to call multiple times; second call is a no-op.
    Returns a summary dict for startup logging.
    """
    global _observer, _handler

    if _observer is not None:
        return {"status": "already_running", "paths": []}

    try:
        paths = get_category_paths()
    except Exception as e:
        logger.error("[file_watcher] could not resolve category paths: %s", e)
        return {"status": "error", "error": str(e), "paths": []}

    handler = _DebouncedHandler()
    observer = Observer()
    watched: list[str] = []
    errors: list[str] = []

    for cat_id, cat_path in paths.items():
        # Skip non-user roots — those are ASTRA's own managed dirs
        # or external project dirs with their own write patterns.
        if cat_id in ("astra_output", "android_project"):
            continue
        if not cat_path.exists():
            logger.debug(
                "[file_watcher] skipping missing path %s=%s",
                cat_id, cat_path,
            )
            continue
        try:
            observer.schedule(handler, str(cat_path), recursive=True)
            watched.append(str(cat_path))
        except Exception as e:
            errors.append(f"{cat_id}: {e}")
            logger.warning(
                "[file_watcher] could not watch %s (%s): %s",
                cat_id, cat_path, e,
            )

    if not watched:
        logger.warning("[file_watcher] no paths watched; not starting observer")
        return {"status": "no_paths", "paths": [], "errors": errors}

    observer.daemon = True
    observer.start()

    _observer = observer
    _handler = handler

    logger.info(
        "[file_watcher] watching %d path(s): %s",
        len(watched), ", ".join(watched),
    )
    return {"status": "running", "paths": watched, "errors": errors}


def stop_file_watcher() -> None:
    """Stop the observer (used for clean shutdown / tests)."""
    global _observer, _handler
    if _observer is not None:
        try:
            _observer.stop()
            _observer.join(timeout=2.0)
        except Exception:
            pass
        _observer = None
    if _handler is not None:
        _handler.stop()
        _handler = None


def is_running() -> bool:
    return _observer is not None and _observer.is_alive()

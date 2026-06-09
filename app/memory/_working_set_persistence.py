# FILE: app/memory/_working_set_persistence.py
"""
Disk persistence for the project working set.

Survives backend restarts.  Without this, the in-memory store resets
every time the process bounces, which is the whole reason cross-model
file handoff breaks across sessions.

Storage layout:
  D:\\Orb\\data\\working_set\\project_{id}.json
  {
    "project_id": 215,
    "canonical_folder": "C:/Users/dizzi/OneDrive/Documents/Work",
    "files": [
      {
        "path": "...",
        "last_touched": 1716567890.123,
        "last_action": "wrote",
        "last_model": "gemini",
        "size_bytes": 8169,
        "mtime": 1716567880.0
      }
    ],
    "aliases": {"the dashboard": "..."}
  }

On load, each file is stat'd — if it no longer exists or mtime has
moved, content is re-read from disk before injection.  Stale entries
are dropped silently.

v1.0 (2026-05-24): Initial implementation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.memory.working_set import ProjectWorkingSet

logger = logging.getLogger(__name__)


# Storage location — sits alongside other ASTRA persistent state.
_STORE_DIR = Path(r"D:\Orb\data\working_set")
_lock = threading.RLock()


def _ensure_store_dir() -> None:
    """Make sure the directory exists.  Cheap, idempotent."""
    try:
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("[working_set_persist] Could not create %s: %s", _STORE_DIR, e)


def _path_for(project_id: int) -> Path:
    return _STORE_DIR / f"project_{project_id}.json"


def save(ws: "ProjectWorkingSet") -> None:
    """Persist a project's working set to disk.  Best-effort —
    failures are logged but never raise, because losing one save
    must not break a tool call."""
    if not ws or not ws.project_id:
        return
    _ensure_store_dir()
    payload = {
        "project_id": ws.project_id,
        "canonical_folder": ws.canonical_folder,
        "files": [
            {
                "path": wf.path,
                "last_touched": wf.last_touched,
                "last_action": wf.last_action,
                "last_model": wf.last_model,
                "size_bytes": wf.size_bytes,
                "mtime": wf.mtime,
            }
            for wf in ws.files.values()
        ],
        "aliases": dict(ws.aliases),
    }
    target = _path_for(ws.project_id)
    tmp = target.with_suffix(".json.tmp")
    try:
        with _lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            # Atomic rename so a crashed write never leaves a partial
            # file readable on next load.
            os.replace(tmp, target)
    except OSError as e:
        logger.warning(
            "[working_set_persist] Save failed for project %d: %s",
            ws.project_id, e,
        )


def load(project_id: int) -> Optional[dict]:
    """Load a project's persisted working set from disk.  Returns the
    raw dict for the caller (working_set.py) to rehydrate into the
    live ProjectWorkingSet — keeps the dataclass surface inside its
    own module without circular imports."""
    if not project_id:
        return None
    target = _path_for(project_id)
    if not target.exists():
        return None
    try:
        with _lock:
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "[working_set_persist] Load failed for project %d: %s",
            project_id, e,
        )
        return None


def delete(project_id: int) -> None:
    """Drop the persisted state for one project (e.g. on explicit reset)."""
    target = _path_for(project_id)
    try:
        if target.exists():
            target.unlink()
    except OSError as e:
        logger.warning(
            "[working_set_persist] Delete failed for project %d: %s",
            project_id, e,
        )

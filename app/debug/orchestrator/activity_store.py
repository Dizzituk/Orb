# FILE: app/debug/orchestrator/activity_store.py
"""
Activity log for Debug Projects.

Stores a timeline of what happened against each debug project:
  - user messages
  - orchestration runs (with phase, resolution, summary)
  - chat completions (when available)
  - file changes applied (files modified by executors)

Persisted to the same SQLite DB as debug_projects so it survives restarts.
Displayed in the Info tab as a timeline.

v1.0 (2026-04-13): initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "orb_memory.db",
)

TABLE = "debug_activity"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              TEXT PRIMARY KEY,
    debug_project_id TEXT NOT NULL,
    kind            TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    body            TEXT NOT NULL DEFAULT '',
    data_json       TEXT NOT NULL DEFAULT '{{}}',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_debug_activity_proj
    ON {TABLE}(debug_project_id, created_at DESC);
"""


def _connect():
    import sqlite3
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    # DDL has multiple statements, so executescript
    conn.executescript(DDL)
    return conn


def _now() -> float:
    return time.time()


def _row_to_dict(row) -> Dict[str, Any]:
    from datetime import datetime, timezone
    d = dict(row)
    if "created_at" in d and isinstance(d["created_at"], (int, float)):
        d["created_at_iso"] = datetime.fromtimestamp(d["created_at"], tz=timezone.utc).isoformat()
    try:
        d["data"] = json.loads(d.get("data_json") or "{}")
    except Exception:
        d["data"] = {}
    d.pop("data_json", None)
    return d


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def record(
    debug_project_id: str,
    kind: str,
    title: str = "",
    body: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Record a single activity entry. Returns entry id or None on failure."""
    if not debug_project_id:
        return None
    try:
        conn = _connect()
        try:
            entry_id = uuid.uuid4().hex[:16]
            conn.execute(
                f"INSERT INTO {TABLE} (id, debug_project_id, kind, title, body, data_json, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    str(debug_project_id),
                    str(kind)[:64],
                    str(title)[:500],
                    str(body)[:20000],
                    json.dumps(data or {})[:100000],
                    _now(),
                ),
            )
            conn.commit()
            return entry_id
        finally:
            conn.close()
    except Exception as e:
        logger.warning("[activity_store] record failed: %s", e)
        return None


def record_user_message(debug_project_id: str, message: str) -> Optional[str]:
    return record(
        debug_project_id=debug_project_id,
        kind="user_message",
        title="You said",
        body=message,
    )


def record_assistant_message(
    debug_project_id: str,
    message: str,
    provider: str = "",
    model: str = "",
) -> Optional[str]:
    return record(
        debug_project_id=debug_project_id,
        kind="assistant_message",
        title=f"{provider}/{model}".strip("/") or "Assistant",
        body=message,
    )


def record_orchestration(
    debug_project_id: str,
    resolution: Dict[str, Any],
) -> Optional[str]:
    # Build a compact body
    resolved = resolution.get("resolved")
    final_phase = resolution.get("final_phase") or ""
    iterations = resolution.get("iterations") or []
    total_tokens = resolution.get("total_tokens", 0)
    elapsed_ms = resolution.get("total_elapsed_ms", 0)
    files_modified: List[str] = []
    for it in iterations:
        for rep in (it.get("execution_reports") or []):
            for f in (rep.get("files_modified") or []):
                if f and f not in files_modified:
                    files_modified.append(f)

    summary_lines = [
        f"Result: {'RESOLVED' if resolved else final_phase}",
        f"Iterations: {len(iterations)}",
        f"Total tokens: {total_tokens:,}",
        f"Elapsed: {elapsed_ms // 1000}s",
    ]
    if files_modified:
        summary_lines.append(f"Files modified: {len(files_modified)}")
        for f in files_modified[:25]:
            summary_lines.append(f"  - {f}")
    if resolution.get("summary"):
        summary_lines.append("")
        summary_lines.append(resolution["summary"])

    return record(
        debug_project_id=debug_project_id,
        kind="orchestration",
        title=f"Orchestration run ({'resolved' if resolved else final_phase})",
        body="\n".join(summary_lines),
        data={
            "resolved": bool(resolved),
            "final_phase": final_phase,
            "iterations": len(iterations),
            "total_tokens": total_tokens,
            "elapsed_ms": elapsed_ms,
            "files_modified": files_modified,
            "unresolved_bugs": resolution.get("unresolved_bugs") or [],
            "contradictions": resolution.get("surfaced_contradictions") or [],
        },
    )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def list_activity(debug_project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Return activity entries newest-first."""
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM {TABLE} WHERE debug_project_id = ? "
                f"ORDER BY created_at DESC LIMIT ?",
                (str(debug_project_id), int(limit)),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.warning("[activity_store] list failed: %s", e)
        return []


def clear_activity(debug_project_id: str) -> int:
    try:
        conn = _connect()
        try:
            cur = conn.execute(
                f"DELETE FROM {TABLE} WHERE debug_project_id = ?",
                (str(debug_project_id),),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
    except Exception as e:
        logger.warning("[activity_store] clear failed: %s", e)
        return 0

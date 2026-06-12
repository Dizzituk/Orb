# FILE: app/debug/project_service.py
# Purpose: Debug Project Service — CRUD operations for debug projects.
# Called-by: app.debug.debug_chat, app.debug.orchestrator.endpoint, app.debug.project_router, app.pipeline_v2.orchestrator
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Debug Project Service — CRUD operations for debug projects.

Uses the existing astra.db SQLite database. Creates the table
on first use if it doesn't exist.

v1.0 (2026-03-07): Initial implementation for debug workspace.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# DB path — same SQLite database as the rest of ASTRA
_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "orb_memory.db")

TABLE = "debug_projects"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',
    error_summary TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{{}}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""


def _connect():
    """Get a sqlite3 connection with row_factory."""
    import sqlite3
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(DDL)
    return conn


def _now() -> float:
    return time.time()


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert sqlite3.Row to dict with ISO timestamps."""
    from datetime import datetime, timezone
    d = dict(row)
    for key in ("created_at", "updated_at"):
        if key in d and isinstance(d[key], (int, float)):
            d[key] = datetime.fromtimestamp(d[key], tz=timezone.utc).isoformat()
    return d


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_projects(status: Optional[str] = None) -> List[Dict]:
    """List all debug projects, optionally filtered by status."""
    conn = _connect()
    try:
        if status:
            rows = conn.execute(
                f"SELECT * FROM {TABLE} WHERE status = ? ORDER BY updated_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM {TABLE} ORDER BY updated_at DESC",
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_project(project_id: str) -> Optional[Dict]:
    """Get a single project by ID."""
    conn = _connect()
    try:
        row = conn.execute(
            f"SELECT * FROM {TABLE} WHERE id = ?", (project_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def create_project(title: str, description: str = "", error_summary: str = "", metadata_json: str = "{}") -> Dict:
    """Create a new debug project."""
    conn = _connect()
    try:
        now = _now()
        project_id = uuid.uuid4().hex[:12]
        conn.execute(
            f"INSERT INTO {TABLE} (id, title, description, status, error_summary, metadata_json, created_at, updated_at) "
            f"VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
            (project_id, title, description, error_summary, metadata_json, now, now),
        )
        conn.commit()
        return get_project(project_id)
    finally:
        conn.close()


def update_project(project_id: str, **fields) -> Optional[Dict]:
    """Update a debug project. Pass only the fields to change."""
    allowed = {"title", "description", "status", "error_summary"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_project(project_id)

    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [project_id]

    conn = _connect()
    try:
        conn.execute(
            f"UPDATE {TABLE} SET {set_clause} WHERE id = ?", values
        )
        conn.commit()
        return get_project(project_id)
    finally:
        conn.close()


def delete_project(project_id: str) -> bool:
    """Delete a debug project."""
    conn = _connect()
    try:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE id = ?", (project_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

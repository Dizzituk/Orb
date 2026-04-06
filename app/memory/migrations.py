# FILE: app/memory/migrations.py
"""
Schema migrations for the memory module.

Handles:
- Adding project_key, type, status columns to projects table
- Backfilling project_key for existing rows
- Seeding the astra-core project
- Adding project_id and domain columns to arch_code_chunks

All migrations are idempotent — safe to run multiple times.
"""

import logging
from datetime import datetime

from sqlalchemy import inspect, text

from app.db import engine

logger = logging.getLogger(__name__)


def _get_existing_columns(table_name: str) -> set[str]:
    """Get column names for a table."""
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def _add_column_if_missing(
    table_name: str,
    col_name: str,
    col_type: str,
    default: str | None = None,
) -> bool:
    """
    Add a column to a table if it doesn't exist.

    Returns True if the column was added, False if it already existed.
    """
    existing = _get_existing_columns(table_name)
    if col_name in existing:
        return False

    if default is not None:
        sql = f"ALTER TABLE [{table_name}] ADD COLUMN [{col_name}] {col_type} DEFAULT {default}"
    else:
        sql = f"ALTER TABLE [{table_name}] ADD COLUMN [{col_name}] {col_type}"

    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()

    logger.info(f"[migrations] Added column: {table_name}.{col_name}")
    return True


# =========================================================================
# Project table migrations
# =========================================================================

def migrate_project_columns() -> list[str]:
    """
    Add project_key, type, status columns to projects table.

    Returns list of columns that were added.
    """
    added = []

    if _add_column_if_missing("projects", "project_key", "TEXT"):
        added.append("project_key")

    if _add_column_if_missing("projects", "type", "TEXT", "'development'"):
        added.append("type")

    if _add_column_if_missing("projects", "status", "TEXT", "'active'"):
        added.append("status")

    if _add_column_if_missing("projects", "pinned", "BOOLEAN", "0"):
        added.append("pinned")

    if added:
        print(f"[migrations] projects table updated: added {added}")
    return added


def backfill_project_keys() -> int:
    """
    Set project_key = 'project-{id}' for any existing rows where it's NULL.

    This ensures the UNIQUE constraint on project_key won't block on NULLs
    when we later enforce it as NOT NULL.

    Returns number of rows updated.
    """
    with engine.connect() as conn:
        result = conn.execute(text(
            "UPDATE projects SET project_key = 'project-' || CAST(id AS TEXT) "
            "WHERE project_key IS NULL"
        ))
        conn.commit()
        count = result.rowcount

    if count > 0:
        logger.info(f"[migrations] Backfilled project_key for {count} existing rows")
    return count


def seed_astra_core() -> bool:
    """
    Insert the astra-core project if it doesn't exist.

    Returns True if inserted, False if already present.
    """
    with engine.connect() as conn:
        existing = conn.execute(text(
            "SELECT id FROM projects WHERE project_key = 'astra-core'"
        )).fetchone()

        if existing:
            logger.debug("[migrations] astra-core project already exists")
            return False

        now = datetime.utcnow().isoformat()
        conn.execute(text(
            "INSERT INTO projects (name, description, project_key, type, status, created_at, updated_at) "
            "VALUES (:name, :desc, :key, :type, :status, :created, :updated)"
        ), {
            "name": "ASTRA Core",
            "desc": "Core ASTRA system — autonomous AI software engineering platform",
            "key": "astra-core",
            "type": "development",
            "status": "active",
            "created": now,
            "updated": now,
        })
        conn.commit()

    print("[migrations] Seeded astra-core project")
    logger.info("[migrations] Seeded astra-core project")
    return True


# =========================================================================
# rag_entries table migrations
# =========================================================================

def migrate_rag_entries_package_role() -> bool:
    """
    Widen package_role from VARCHAR(50) to VARCHAR(500).

    v5.4: The ContextStore stores JSON metadata in package_role
    (TTL timestamps, entry types) which can exceed 50 chars.
    Truncation at 50 chars breaks GC — can't parse JSON → entries
    never expire → memory fills up.

    The Python model (rag_entries_model.py) already uses String(500).
    This migration fixes existing live databases.

    Returns True if migrated, False if not needed or table missing.
    """
    inspector = inspect(engine)
    if "rag_entries" not in inspector.get_table_names():
        return False

    columns = {c["name"]: c for c in inspector.get_columns("rag_entries")}
    pr_col = columns.get("package_role")
    if not pr_col:
        return False

    # Check current length — SQLite uses TEXT so this only matters for Postgres
    col_type_str = str(pr_col.get("type", "")).upper()
    if "TEXT" in col_type_str:
        return False  # TEXT is unlimited, no migration needed

    # For SQLite, ALTER COLUMN isn't supported but VARCHAR is TEXT anyway.
    # For Postgres-style DBs, widen the column.
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE rag_entries ALTER COLUMN package_role TYPE VARCHAR(500)"
            ))
            conn.commit()
        logger.info("[migrations] Widened rag_entries.package_role to VARCHAR(500)")
        print("[migrations] Widened rag_entries.package_role to VARCHAR(500)")
        return True
    except Exception as e:
        # SQLite will fail here — that's fine, SQLite treats all VARCHAR as TEXT
        logger.debug("[migrations] package_role migration skipped (likely SQLite): %s", e)
        return False


# =========================================================================
# Run all memory migrations
# =========================================================================

def run_memory_migrations() -> dict:
    """
    Run all memory module migrations in order.

    Called from app/db.py init_db().
    Returns summary of what was done.
    """
    summary = {}

    # Project table
    summary["project_columns_added"] = migrate_project_columns()
    summary["project_keys_backfilled"] = backfill_project_keys()
    summary["astra_core_seeded"] = seed_astra_core()

    # v5.4: rag_entries column widening
    summary["package_role_widened"] = migrate_rag_entries_package_role()

    return summary
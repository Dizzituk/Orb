# FILE: scripts/migrate_rag_lifecycle.py
"""
Migration: Add lifecycle tracking columns to existing architecture tables.

Adds to arch_code_chunks:
  - status: ACTIVE/QUARANTINED/PURGED (default ACTIVE)
  - source_monolith: original file path if this came from a refactor
  - refactor_job_id: which refactor job created this entry
  - package_role: role in package (init, core, models, utils, etc.)

Adds to architecture_file_index:
  - status: ACTIVE/QUARANTINED/PURGED (default ACTIVE)
  - source_monolith: original file path if this came from a refactor
  - refactor_job_id: which refactor job created this entry
  - quarantined_at: when this entry was quarantined

All existing rows get status='active' (they represent the current live codebase).

Safe to run multiple times - checks for column existence before adding.
"""

import sqlite3
import sys
from datetime import datetime

DB_PATH = r"D:\Orb\data\orb_memory.db"


def get_existing_columns(cursor, table_name):
    """Get list of column names for a table."""
    rows = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [r[1] for r in rows]


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    changes = 0
    
    # === arch_code_chunks ===
    cols = get_existing_columns(cursor, "arch_code_chunks")
    
    if "status" not in cols:
        cursor.execute("ALTER TABLE arch_code_chunks ADD COLUMN status VARCHAR(20) DEFAULT 'active' NOT NULL")
        print("[migrate] Added arch_code_chunks.status")
        changes += 1
    
    if "source_monolith" not in cols:
        cursor.execute("ALTER TABLE arch_code_chunks ADD COLUMN source_monolith VARCHAR(500)")
        print("[migrate] Added arch_code_chunks.source_monolith")
        changes += 1
    
    if "refactor_job_id" not in cols:
        cursor.execute("ALTER TABLE arch_code_chunks ADD COLUMN refactor_job_id VARCHAR(100)")
        print("[migrate] Added arch_code_chunks.refactor_job_id")
        changes += 1
    
    if "package_role" not in cols:
        cursor.execute("ALTER TABLE arch_code_chunks ADD COLUMN package_role VARCHAR(50)")
        print("[migrate] Added arch_code_chunks.package_role")
        changes += 1
    
    # === architecture_file_index ===
    cols_fi = get_existing_columns(cursor, "architecture_file_index")
    
    if "status" not in cols_fi:
        cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN status VARCHAR(20) DEFAULT 'active' NOT NULL")
        print("[migrate] Added architecture_file_index.status")
        changes += 1
    
    if "source_monolith" not in cols_fi:
        cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN source_monolith VARCHAR(500)")
        print("[migrate] Added architecture_file_index.source_monolith")
        changes += 1
    
    if "refactor_job_id" not in cols_fi:
        cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN refactor_job_id VARCHAR(100)")
        print("[migrate] Added architecture_file_index.refactor_job_id")
        changes += 1
    
    if "quarantined_at" not in cols_fi:
        cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN quarantined_at DATETIME")
        print("[migrate] Added architecture_file_index.quarantined_at")
        changes += 1
    
    # === Create indexes for the new status columns ===
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_chunks_status ON arch_code_chunks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_chunks_monolith ON arch_code_chunks(source_monolith)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_fi_status ON architecture_file_index(status)")
        print("[migrate] Created indexes")
    except Exception as e:
        print(f"[migrate] Index creation note: {e}")
    
    conn.commit()
    
    # Verify
    final_cols = get_existing_columns(cursor, "arch_code_chunks")
    final_fi = get_existing_columns(cursor, "architecture_file_index")
    
    print(f"\n[migrate] Complete. {changes} columns added.")
    print(f"[migrate] arch_code_chunks columns: {final_cols}")
    print(f"[migrate] architecture_file_index columns: {final_fi}")
    
    # Quick stats
    total_chunks = cursor.execute("SELECT COUNT(*) FROM arch_code_chunks WHERE status='active'").fetchone()[0]
    total_files = cursor.execute("SELECT COUNT(*) FROM architecture_file_index WHERE status='active'").fetchone()[0]
    print(f"[migrate] Active chunks: {total_chunks}, Active files: {total_files}")
    
    conn.close()


if __name__ == "__main__":
    migrate()

# scripts/add_model_column.py
"""
Migration: Add `model` column to messages table.

v0.12.1 FIX: The messages table was missing the `model` column that tracks
which specific LLM model generated each assistant response.

This migration adds the column to existing databases.

Usage:
    cd D:\Orb
    .\.venv\Scripts\Activate.ps1
    python scripts/add_model_column.py
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path("data/orb_memory.db")


def check_column_exists(cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def main():
    if not DB_PATH.exists():
        print(f"[migration] Database not found at {DB_PATH}")
        print("[migration] Nothing to migrate - column will be created on first run")
        return
    
    print(f"[migration] Connecting to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if column already exists
        if check_column_exists(cursor, "messages", "model"):
            print("[migration] Column 'model' already exists in messages table")
            print("[migration] Nothing to do")
            return
        
        # Add the column
        print("[migration] Adding 'model' column to messages table...")
        cursor.execute("""
            ALTER TABLE messages
            ADD COLUMN model VARCHAR(100)
        """)
        
        conn.commit()
        print("[migration] Successfully added 'model' column")
        
        # Verify
        if check_column_exists(cursor, "messages", "model"):
            print("[migration] Verified: column exists")
        else:
            print("[migration] ERROR: Column was not added!")
            return
        
        # Show current schema
        print("\n[migration] Current messages table schema:")
        cursor.execute("PRAGMA table_info(messages)")
        for row in cursor.fetchall():
            print(f"  {row[1]}: {row[2]} {'(nullable)' if row[3] == 0 else '(not null)'}")
        
        # Count existing messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        print(f"\n[migration] Existing messages: {count}")
        print("[migration] Note: Existing messages will have model=NULL")
        print("[migration] New messages will have the model field populated")
        
    finally:
        conn.close()
    
    print("\n[migration] Done!")


if __name__ == "__main__":
    main()
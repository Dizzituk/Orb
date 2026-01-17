#!/usr/bin/env python3
"""
Migration for orb_memory.db - the ACTUAL database Orb uses.
Stop Orb first, then run:
    python fix_orb_memory_db.py
"""
import sqlite3
import os

DB_PATH = r"D:\Orb\data\orb_memory.db"

if not os.path.exists(DB_PATH):
    print(f"ERROR: Database not found: {DB_PATH}")
    exit(1)

print(f"Using database: {DB_PATH}")
print(f"Size: {os.path.getsize(DB_PATH):,} bytes")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get existing columns
cursor.execute("PRAGMA table_info(architecture_file_index)")
columns = [row[1] for row in cursor.fetchall()]
print(f"\nExisting columns: {columns}")

# Add missing columns
if 'line_count' not in columns:
    print("\nAdding line_count column...")
    cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN line_count INTEGER")
    print("  Done!")
else:
    print("\nline_count column already exists")

if 'language' not in columns:
    print("Adding language column...")
    cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN language VARCHAR(50)")
    print("  Done!")
else:
    print("language column already exists")

# Create index
print("\nCreating language index...")
cursor.execute("""
    CREATE INDEX IF NOT EXISTS ix_arch_file_scan_lang 
    ON architecture_file_index(scan_id, language)
""")

# Check if architecture_file_content exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='architecture_file_content'")
if not cursor.fetchone():
    print("\nCreating architecture_file_content table...")
    cursor.execute("""
        CREATE TABLE architecture_file_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_index_id INTEGER NOT NULL UNIQUE,
            content_text TEXT NOT NULL,
            content_hash VARCHAR(64),
            FOREIGN KEY (file_index_id) REFERENCES architecture_file_index(id) ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_arch_content_file_id 
        ON architecture_file_content(file_index_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_arch_content_hash 
        ON architecture_file_content(content_hash)
    """)
    print("  Done!")
else:
    print("\narchitecture_file_content table already exists")

conn.commit()

# Verify
cursor.execute("PRAGMA table_info(architecture_file_index)")
columns = [row[1] for row in cursor.fetchall()]
print(f"\nFinal columns: {columns}")

conn.close()

print("\n" + "="*50)
print("MIGRATION COMPLETE!")
print("Now restart Orb.")
print("="*50)

#!/usr/bin/env python3
"""
Quick migration script for architecture content columns.
Run from D:\Orb directory:
    python migrate_arch_content.py
"""
import sqlite3
import os

# Find the database
DB_PATH = os.environ.get("ORB_DB_PATH", "orb.db")

if not os.path.exists(DB_PATH):
    # Try common locations
    for path in ["orb.db", "data/orb.db", "app/orb.db"]:
        if os.path.exists(path):
            DB_PATH = path
            break

print(f"Using database: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if columns already exist
cursor.execute("PRAGMA table_info(architecture_file_index)")
columns = [row[1] for row in cursor.fetchall()]

if 'line_count' not in columns:
    print("Adding line_count column...")
    cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN line_count INTEGER")
else:
    print("line_count column already exists")

if 'language' not in columns:
    print("Adding language column...")
    cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN language VARCHAR(50)")
else:
    print("language column already exists")

# Create index
print("Creating language index...")
cursor.execute("""
    CREATE INDEX IF NOT EXISTS ix_arch_file_scan_lang 
    ON architecture_file_index(scan_id, language)
""")

# Create content table
print("Creating architecture_file_content table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS architecture_file_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_index_id INTEGER NOT NULL UNIQUE,
        content_text TEXT NOT NULL,
        content_hash VARCHAR(64),
        FOREIGN KEY (file_index_id) REFERENCES architecture_file_index(id) ON DELETE CASCADE
    )
""")

# Create indexes for content table
cursor.execute("""
    CREATE INDEX IF NOT EXISTS ix_arch_file_content_file_index_id 
    ON architecture_file_content(file_index_id)
""")
cursor.execute("""
    CREATE INDEX IF NOT EXISTS ix_arch_file_content_hash 
    ON architecture_file_content(content_hash)
""")

conn.commit()
conn.close()

print("Migration complete!")

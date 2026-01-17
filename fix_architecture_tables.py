#!/usr/bin/env python3
"""
Migration script for architecture tables - handles existing tables.
Stop Orb first, then run:
    python fix_architecture_tables.py
"""
import sqlite3
import os

DB_PATH = "orb.db"
if not os.path.exists(DB_PATH):
    for path in ["orb.db", "data/orb.db"]:
        if os.path.exists(path):
            DB_PATH = path
            break

print(f"Using database: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check what tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Existing tables: {tables}")

# Check if architecture_file_index exists
if 'architecture_file_index' in tables:
    # Get existing columns
    cursor.execute("PRAGMA table_info(architecture_file_index)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns in architecture_file_index: {columns}")
    
    # Add missing columns
    if 'line_count' not in columns:
        print("Adding line_count column...")
        cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN line_count INTEGER")
    
    if 'language' not in columns:
        print("Adding language column...")
        cursor.execute("ALTER TABLE architecture_file_index ADD COLUMN language VARCHAR(50)")
    
    # Create index if not exists
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_arch_file_scan_lang 
        ON architecture_file_index(scan_id, language)
    """)
else:
    print("Creating architecture_file_index table...")
    cursor.execute("""
        CREATE TABLE architecture_file_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            path VARCHAR(1000) NOT NULL,
            name VARCHAR(255) NOT NULL,
            ext VARCHAR(20),
            size_bytes BIGINT,
            mtime VARCHAR(30),
            zone VARCHAR(50) NOT NULL,
            root VARCHAR(500),
            line_count INTEGER,
            language VARCHAR(50),
            FOREIGN KEY (scan_id) REFERENCES architecture_scan_runs(id) ON DELETE CASCADE
        )
    """)

# Create architecture_scan_runs if not exists
if 'architecture_scan_runs' not in tables:
    print("Creating architecture_scan_runs table...")
    cursor.execute("""
        CREATE TABLE architecture_scan_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'running',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            stats_json TEXT,
            error_message TEXT
        )
    """)

# Create architecture_file_content if not exists
if 'architecture_file_content' not in tables:
    print("Creating architecture_file_content table...")
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

conn.commit()
conn.close()

print("\nMigration complete! Now restart Orb.")

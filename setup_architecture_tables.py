#!/usr/bin/env python3
"""
Complete setup script for architecture scan tables.
Run from D:\Orb directory:
    python setup_architecture_tables.py
"""
import sqlite3
import os

# Find the database
DB_PATH = os.environ.get("ORB_DB_PATH", "orb.db")

if not os.path.exists(DB_PATH):
    for path in ["orb.db", "data/orb.db", "app/orb.db"]:
        if os.path.exists(path):
            DB_PATH = path
            break

print(f"Using database: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create architecture_scan_runs table
print("Creating architecture_scan_runs table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS architecture_scan_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope VARCHAR(50) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'running',
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
        finished_at TIMESTAMP,
        stats_json TEXT,
        error_message TEXT
    )
""")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_scan_scope ON architecture_scan_runs(scope)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_scan_status ON architecture_scan_runs(status)")

# Create architecture_file_index table
print("Creating architecture_file_index table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS architecture_file_index (
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
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_scan_id ON architecture_file_index(scan_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_name ON architecture_file_index(name)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_ext ON architecture_file_index(ext)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_zone ON architecture_file_index(zone)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_path_prefix ON architecture_file_index(path)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_scan_zone ON architecture_file_index(scan_id, zone)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_scan_ext ON architecture_file_index(scan_id, ext)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_file_scan_lang ON architecture_file_index(scan_id, language)")

# Create architecture_file_content table
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
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_content_file_id ON architecture_file_content(file_index_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS ix_arch_content_hash ON architecture_file_content(content_hash)")

conn.commit()
conn.close()

print("All architecture tables created successfully!")

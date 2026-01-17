#!/usr/bin/env python3
"""Debug script to find all databases and check architecture table schemas."""
import sqlite3
import os
from pathlib import Path

print("=" * 60)
print("DATABASE DEBUG")
print("=" * 60)

# Find all .db files
db_files = []
for root, dirs, files in os.walk("D:\\Orb"):
    # Skip venv and node_modules
    dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', 'node_modules', '__pycache__']]
    for f in files:
        if f.endswith('.db'):
            db_files.append(os.path.join(root, f))

print(f"\nFound {len(db_files)} database file(s):\n")

for db_path in db_files:
    size = os.path.getsize(db_path)
    print(f"\n{'='*60}")
    print(f"DATABASE: {db_path}")
    print(f"SIZE: {size:,} bytes ({size/1024/1024:.2f} MB)")
    print("="*60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nTables: {tables}")
        
        # Check for architecture tables
        if 'architecture_file_index' in tables:
            print("\n>>> HAS architecture_file_index <<<")
            cursor.execute("PRAGMA table_info(architecture_file_index)")
            columns = [(row[1], row[2]) for row in cursor.fetchall()]
            print(f"Columns: {columns}")
            
            # Check row count
            cursor.execute("SELECT COUNT(*) FROM architecture_file_index")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")
        else:
            print("\n(no architecture_file_index table)")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*60)
print("END DEBUG")
print("="*60)

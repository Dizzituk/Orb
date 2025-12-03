#!/usr/bin/env python3
"""
Encrypt existing plaintext data in the database.

This script is for the scenario where data was stored in plaintext
(because encryption wasn't properly wired up) and now needs to be encrypted.

Security Level 4: Uses master key from ORB_MASTER_KEY environment variable.

Usage:
    cd D:\Orb
    $env:ORB_MASTER_KEY = "your-43-char-key"
    python scripts/encrypt_existing_plaintext.py
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.crypto.encryption import (
    init_master_key_from_env,
    encrypt_string,
    is_encryption_ready,
)

# Database path
DB_PATH = Path("data/orb_memory.db")

# Fields to encrypt: (table, column)
FIELDS_TO_ENCRYPT = [
    ("messages", "content"),
    ("notes", "content"),
    ("tasks", "description"),
    ("document_contents", "raw_text"),
    ("document_contents", "summary"),
    ("document_contents", "structured_data"),
    ("files", "description"),
]


def is_already_encrypted(value: str) -> bool:
    """Check if a value is already encrypted."""
    return value is not None and value.startswith("ENC:")


def encrypt_field(conn: sqlite3.Connection, table: str, column: str) -> int:
    """
    Encrypt all plaintext values in a specific column.
    Returns the number of rows encrypted.
    """
    cursor = conn.cursor()
    
    # Get all rows with non-null values
    cursor.execute(f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL")
    rows = cursor.fetchall()
    
    encrypted_count = 0
    skipped_count = 0
    
    for row_id, value in rows:
        if is_already_encrypted(value):
            skipped_count += 1
            continue
        
        # Encrypt the value
        encrypted_value = encrypt_string(value)
        
        # Update the row
        cursor.execute(
            f"UPDATE {table} SET {column} = ? WHERE id = ?",
            (encrypted_value, row_id)
        )
        encrypted_count += 1
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} already-encrypted rows")
    
    return encrypted_count


def main():
    print("=" * 60)
    print("ENCRYPT EXISTING PLAINTEXT DATA")
    print("=" * 60)
    print()
    
    # Check for master key
    if not os.environ.get("ORB_MASTER_KEY"):
        print("[ERROR] ORB_MASTER_KEY environment variable is not set.")
        print()
        print("To get the master key:")
        print("  cd D:\\orb-desktop")
        print("  node -e \"require('keytar').getPassword('OrbMasterKey','default').then(k=>console.log(k))\"")
        print()
        print("Then set it:")
        print("  $env:ORB_MASTER_KEY = 'your-43-char-key'")
        sys.exit(1)
    
    # Initialize encryption
    if not init_master_key_from_env():
        print("[ERROR] Failed to initialize encryption from master key.")
        sys.exit(1)
    
    if not is_encryption_ready():
        print("[ERROR] Encryption is not ready after initialization.")
        sys.exit(1)
    
    print("[OK] Encryption initialized with master key")
    print()
    
    # Check database exists
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)
    
    # Create backup
    backup_path = DB_PATH.with_suffix(".db.pre_encryption_backup")
    if not backup_path.exists():
        import shutil
        shutil.copy(DB_PATH, backup_path)
        print(f"[OK] Created backup: {backup_path}")
    else:
        print(f"[INFO] Backup already exists: {backup_path}")
    print()
    
    # Confirm with user
    print("This script will encrypt the following fields:")
    for table, column in FIELDS_TO_ENCRYPT:
        print(f"  - {table}.{column}")
    print()
    print("WARNING: This operation cannot be undone without the backup.")
    print()
    response = input("Type 'ENCRYPT' to proceed: ")
    if response != "ENCRYPT":
        print("Aborted.")
        sys.exit(0)
    
    print()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    total_encrypted = 0
    
    try:
        for table, column in FIELDS_TO_ENCRYPT:
            print(f"[INFO] Encrypting {table}.{column}...")
            
            try:
                count = encrypt_field(conn, table, column)
                print(f"  Encrypted {count} rows")
                total_encrypted += count
            except sqlite3.OperationalError as e:
                if "no such column" in str(e) or "no such table" in str(e):
                    print(f"  Skipped (column/table doesn't exist)")
                else:
                    raise
        
        # Commit all changes
        conn.commit()
        print()
        print("=" * 60)
        print(f"ENCRYPTION COMPLETE: {total_encrypted} total rows encrypted")
        print("=" * 60)
        
    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] Encryption failed: {e}")
        print("All changes have been rolled back.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
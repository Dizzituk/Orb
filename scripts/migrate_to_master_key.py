#!/usr/bin/env python3
# scripts/migrate_to_master_key.py
"""
Migration Script: Password-derived encryption → Master Key encryption

This script migrates all encrypted data in the Orb database from the old
password-derived encryption (PBKDF2) to the new master key encryption
(Security Level 4).

Requirements:
- ORB_MASTER_KEY environment variable must be set
- You must know the old Orb password
- Backend should NOT be running during migration

Usage:
    cd D:\Orb
    set ORB_MASTER_KEY=<your-master-key>
    python scripts/migrate_to_master_key.py
"""

import os
import sys
import shutil
import getpass
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ============================================================================
# CONFIGURATION
# ============================================================================

DATABASE_PATH = Path("data/orb_memory.db")
BACKUP_PATH = Path("data/orb_memory_pre_master_key_backup.db")
METADATA_TABLE = "migration_metadata"
MIGRATION_VERSION = "master_key_v1"

# Tables and columns that contain encrypted data
ENCRYPTED_FIELDS = [
    ("messages", "content", "id"),
    ("notes", "content", "id"),
    ("tasks", "description", "id"),
    ("document_contents", "raw_text", "id"),
    ("document_contents", "summary", "id"),
    ("document_contents", "structured_data", "id"),
]


# ============================================================================
# ENCRYPTION HELPERS
# ============================================================================

def create_old_encryption_manager(password: str):
    """Create encryption manager using old password-derived key."""
    from app.crypto.encryption import create_legacy_encryption_manager
    return create_legacy_encryption_manager(password)


def create_new_encryption_manager(master_key: str):
    """Create encryption manager using new master key."""
    import base64
    from app.crypto.encryption import EncryptionManager
    
    # Decode master key (URL-safe base64)
    padded = master_key + '=' * (4 - len(master_key) % 4) if len(master_key) % 4 else master_key
    key_bytes = base64.urlsafe_b64decode(padded.encode('ascii'))
    
    if len(key_bytes) != 32:
        raise ValueError(f"Invalid master key length: expected 32 bytes, got {len(key_bytes)}")
    
    return EncryptionManager(key_bytes)


def is_encrypted(value: str) -> bool:
    """Check if a value is encrypted (has ENC: prefix)."""
    return value and isinstance(value, str) and value.startswith("ENC:")


# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_database_session():
    """Create a database session."""
    if not DATABASE_PATH.exists():
        print(f"[ERROR] Database not found: {DATABASE_PATH}")
        sys.exit(1)
    
    engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def check_migration_status(session) -> Tuple[bool, Optional[str]]:
    """
    Check if migration has already been run.
    Returns (already_migrated, version).
    """
    try:
        result = session.execute(text(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{METADATA_TABLE}'"
        )).fetchone()
        
        if not result:
            return False, None
        
        result = session.execute(text(
            f"SELECT value FROM {METADATA_TABLE} WHERE key='encryption_migration_version'"
        )).fetchone()
        
        if result and result[0] == MIGRATION_VERSION:
            return True, result[0]
        
        return False, result[0] if result else None
    except Exception:
        return False, None


def mark_migration_complete(session):
    """Mark migration as complete in metadata table."""
    try:
        # Create metadata table if it doesn't exist
        session.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """))
        
        # Insert or update migration version
        session.execute(text(f"""
            INSERT OR REPLACE INTO {METADATA_TABLE} (key, value, updated_at)
            VALUES ('encryption_migration_version', :version, :timestamp)
        """), {"version": MIGRATION_VERSION, "timestamp": datetime.now().isoformat()})
        
        session.commit()
        print(f"[INFO] Migration marked complete: {MIGRATION_VERSION}")
    except Exception as e:
        print(f"[WARNING] Could not mark migration complete: {e}")


def backup_database() -> bool:
    """Create a backup of the database. Returns True on success."""
    try:
        if BACKUP_PATH.exists():
            print(f"[WARNING] Backup already exists: {BACKUP_PATH}")
            response = input("Overwrite existing backup? (yes/no): ").strip().lower()
            if response != "yes":
                print("[ABORT] User declined to overwrite backup.")
                return False
        
        print(f"[INFO] Creating backup: {BACKUP_PATH}")
        shutil.copy2(DATABASE_PATH, BACKUP_PATH)
        
        # Verify backup
        if not BACKUP_PATH.exists():
            print("[ERROR] Backup file was not created.")
            return False
        
        original_size = DATABASE_PATH.stat().st_size
        backup_size = BACKUP_PATH.stat().st_size
        
        if original_size != backup_size:
            print(f"[ERROR] Backup size mismatch: original={original_size}, backup={backup_size}")
            return False
        
        print(f"[INFO] Backup created successfully ({backup_size:,} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Backup failed: {e}")
        return False


# ============================================================================
# MIGRATION LOGIC
# ============================================================================

def migrate_field(
    session,
    table: str,
    column: str,
    id_column: str,
    old_manager,
    new_manager,
) -> Tuple[int, int, List[int]]:
    """
    Migrate a single encrypted field.
    Returns (total_rows, migrated_count, failed_ids).
    """
    print(f"\n[INFO] Migrating {table}.{column}...")
    
    # Get all rows
    try:
        rows = session.execute(text(
            f"SELECT {id_column}, {column} FROM {table}"
        )).fetchall()
    except Exception as e:
        print(f"[ERROR] Failed to read {table}: {e}")
        return 0, 0, []
    
    total = len(rows)
    migrated = 0
    failed_ids = []
    
    for row_id, value in rows:
        # Skip NULL or empty values
        if not value:
            continue
        
        # Skip non-encrypted values
        if not is_encrypted(value):
            continue
        
        try:
            # Decrypt with old key
            plaintext = old_manager.decrypt(value)
            
            # Check if decryption actually worked
            if plaintext == value:
                # Decryption returned the same value - might already be migrated or corrupted
                print(f"[WARNING] {table}.{column} row {row_id}: decryption returned unchanged value")
                continue
            
            # Re-encrypt with new key
            new_ciphertext = new_manager.encrypt(plaintext)
            
            # Update in database
            session.execute(text(
                f"UPDATE {table} SET {column} = :value WHERE {id_column} = :id"
            ), {"value": new_ciphertext, "id": row_id})
            
            migrated += 1
            
            # Progress indicator
            if migrated % 100 == 0:
                print(f"[INFO] {table}.{column}: migrated {migrated} rows...")
                
        except Exception as e:
            print(f"[ERROR] {table}.{column} row {row_id}: {e}")
            failed_ids.append(row_id)
            # Stop on first error to preserve data integrity
            return total, migrated, failed_ids
    
    print(f"[INFO] {table}.{column}: {migrated}/{total} rows migrated")
    return total, migrated, failed_ids


def run_migration(old_password: str, master_key: str) -> bool:
    """
    Run the full migration.
    Returns True on success, False on failure.
    """
    print("\n" + "=" * 60)
    print("STARTING MIGRATION")
    print("=" * 60)
    
    # Create encryption managers
    print("\n[INFO] Initializing encryption managers...")
    try:
        old_manager = create_old_encryption_manager(old_password)
        print("[INFO] Old encryption manager ready (PBKDF2)")
    except Exception as e:
        print(f"[ERROR] Failed to create old encryption manager: {e}")
        return False
    
    try:
        new_manager = create_new_encryption_manager(master_key)
        print("[INFO] New encryption manager ready (master key)")
    except Exception as e:
        print(f"[ERROR] Failed to create new encryption manager: {e}")
        return False
    
    # Get database session
    session, engine = get_database_session()
    
    # Track overall stats
    total_migrated = 0
    total_failed = 0
    all_failed_ids = []
    
    try:
        for table, column, id_column in ENCRYPTED_FIELDS:
            total, migrated, failed_ids = migrate_field(
                session, table, column, id_column, old_manager, new_manager
            )
            
            total_migrated += migrated
            
            if failed_ids:
                total_failed += len(failed_ids)
                all_failed_ids.append((table, column, failed_ids))
                print(f"\n[ERROR] Migration stopped due to errors in {table}.{column}")
                print("[ERROR] Rolling back all changes...")
                session.rollback()
                return False
        
        # All successful - commit
        print("\n[INFO] All fields migrated successfully. Committing changes...")
        session.commit()
        
        # Mark migration complete
        mark_migration_complete(session)
        
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETE")
        print("=" * 60)
        print(f"Total rows migrated: {total_migrated}")
        print(f"Backup location: {BACKUP_PATH}")
        print("\nYou can now start Orb normally via Electron.")
        print("The backup can be deleted once you've verified everything works.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print("[ERROR] Rolling back all changes...")
        session.rollback()
        return False
    finally:
        session.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("ORB ENCRYPTION MIGRATION")
    print("Password-derived key → Master key (Security Level 4)")
    print("=" * 60)
    
    # Check for master key
    master_key = os.environ.get("ORB_MASTER_KEY")
    if not master_key:
        print("\n[ERROR] ORB_MASTER_KEY environment variable is not set.")
        print("\nTo set it:")
        print("  1. Start Electron once to create the master key")
        print("  2. The key is stored in Windows Credential Manager under 'OrbMasterKey'")
        print("  3. You can retrieve it with PowerShell:")
        print("       [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(")
        print("         [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR(")
        print("           (Get-StoredCredential -Target OrbMasterKey).Password))")
        print("  4. Then: set ORB_MASTER_KEY=<the-key>")
        print("  5. Run this script again")
        sys.exit(1)
    
    print(f"\n[INFO] Master key found in environment (length: {len(master_key)} chars)")
    
    # Check database exists
    if not DATABASE_PATH.exists():
        print(f"\n[ERROR] Database not found: {DATABASE_PATH}")
        print("Make sure you're running from the D:\\Orb directory.")
        sys.exit(1)
    
    print(f"[INFO] Database found: {DATABASE_PATH}")
    
    # Check if already migrated
    session, _ = get_database_session()
    already_migrated, version = check_migration_status(session)
    session.close()
    
    if already_migrated:
        print(f"\n[WARNING] Migration has already been run (version: {version})")
        print("If you need to re-run, manually delete the migration_metadata table.")
        sys.exit(0)
    
    if version:
        print(f"\n[INFO] Previous migration detected: {version}")
    
    # Get old password
    print("\n" + "-" * 40)
    print("Enter your CURRENT Orb password.")
    print("This is the password you used before this migration.")
    print("-" * 40)
    
    old_password = getpass.getpass("Current Orb password: ")
    
    if not old_password:
        print("[ERROR] Password cannot be empty.")
        sys.exit(1)
    
    # Backup
    print("\n" + "-" * 40)
    print("BACKUP")
    print("-" * 40)
    
    if not backup_database():
        print("\n[ABORT] Backup failed. Migration cancelled.")
        sys.exit(1)
    
    # Confirmation
    print("\n" + "-" * 40)
    print("WARNING: DESTRUCTIVE OPERATION")
    print("-" * 40)
    print("""
This migration will:
1. Decrypt ALL encrypted data using your old password
2. Re-encrypt ALL data using the new master key
3. Update the database in place

Tables affected:
- messages (content)
- notes (content)
- tasks (description)
- document_contents (raw_text, summary, structured_data)

A backup has been created at:
  {backup}

If ANYTHING goes wrong:
1. The migration will stop and rollback
2. Your backup remains intact
3. You can restore with: copy "{backup}" "{db}"
""".format(backup=BACKUP_PATH, db=DATABASE_PATH))
    
    print("-" * 40)
    confirmation = input("Type YES (uppercase) to proceed: ").strip()
    
    if confirmation != "YES":
        print("\n[ABORT] User did not confirm. Migration cancelled.")
        sys.exit(0)
    
    # Run migration
    success = run_migration(old_password, master_key)
    
    if not success:
        print("\n[FAILED] Migration failed. Your backup is intact at:")
        print(f"  {BACKUP_PATH}")
        print("\nTo restore:")
        print(f'  copy "{BACKUP_PATH}" "{DATABASE_PATH}"')
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
# scripts/encrypt_existing_data.py
"""
Migration script to encrypt existing database data.
Run this ONCE after setting up encryption to encrypt existing plaintext data.

Usage:
    cd D:\Orb
    py -3.13 scripts/encrypt_existing_data.py

IMPORTANT: 
    - Back up your database before running!
    - This will encrypt: messages, notes, document content
    - Data encrypted with this password can only be decrypted with the same password
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from getpass import getpass
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Import encryption
from app.crypto import set_encryption_key, encrypt_string, is_encryption_ready

DATABASE_PATH = Path("data/orb_memory.db")
BACKUP_PATH = Path("data/orb_memory_backup_pre_encryption.db")


def backup_database():
    """Create a backup of the database."""
    if DATABASE_PATH.exists():
        import shutil
        shutil.copy(DATABASE_PATH, BACKUP_PATH)
        print(f"✓ Database backed up to: {BACKUP_PATH}")
        return True
    else:
        print("✗ Database not found!")
        return False


def encrypt_table_column(session, table: str, id_col: str, column: str, count_only: bool = False):
    """
    Encrypt a specific column in a table.
    Only encrypts rows that are not already encrypted (don't start with 'ENC:').
    """
    # Count unencrypted rows
    count_query = text(f"""
        SELECT COUNT(*) FROM {table} 
        WHERE {column} IS NOT NULL 
        AND {column} != '' 
        AND {column} NOT LIKE 'ENC:%'
    """)
    count = session.execute(count_query).scalar()
    
    if count_only:
        return count
    
    if count == 0:
        print(f"  {table}.{column}: Already encrypted or empty")
        return 0
    
    print(f"  {table}.{column}: Encrypting {count} rows...")
    
    # Fetch unencrypted rows
    select_query = text(f"""
        SELECT {id_col}, {column} FROM {table} 
        WHERE {column} IS NOT NULL 
        AND {column} != '' 
        AND {column} NOT LIKE 'ENC:%'
    """)
    rows = session.execute(select_query).fetchall()
    
    # Encrypt each row
    encrypted_count = 0
    for row in rows:
        row_id = row[0]
        plaintext = row[1]
        
        if plaintext:
            ciphertext = encrypt_string(plaintext)
            update_query = text(f"""
                UPDATE {table} SET {column} = :ciphertext WHERE {id_col} = :id
            """)
            session.execute(update_query, {"ciphertext": ciphertext, "id": row_id})
            encrypted_count += 1
    
    session.commit()
    print(f"    ✓ Encrypted {encrypted_count} rows")
    return encrypted_count


def main():
    print("=" * 60)
    print("Orb Database Encryption Migration")
    print("=" * 60)
    print()
    print("This script will encrypt existing plaintext data in your database.")
    print("Make sure you have a backup before proceeding!")
    print()
    
    # Check database exists
    if not DATABASE_PATH.exists():
        print(f"✗ Database not found at {DATABASE_PATH}")
        print("  Run the app first to create the database.")
        return
    
    # Get password
    password = getpass("Enter your Orb password: ")
    if len(password) < 4:
        print("✗ Invalid password (too short)")
        return
    
    # Initialize encryption
    print()
    print("Initializing encryption...")
    set_encryption_key(password)
    
    if not is_encryption_ready():
        print("✗ Failed to initialize encryption")
        return
    print("✓ Encryption initialized")
    
    # Create backup
    print()
    print("Creating database backup...")
    if not backup_database():
        return
    
    # Connect to database
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Preview what will be encrypted
        print()
        print("Scanning for unencrypted data...")
        
        tables_to_encrypt = [
            ("messages", "id", "content"),
            ("notes", "id", "content"),
            ("document_contents", "id", "raw_text"),
            ("document_contents", "id", "summary"),
            ("document_contents", "id", "structured_data"),
        ]
        
        total_rows = 0
        for table, id_col, column in tables_to_encrypt:
            try:
                count = encrypt_table_column(session, table, id_col, column, count_only=True)
                if count > 0:
                    print(f"  {table}.{column}: {count} rows to encrypt")
                    total_rows += count
            except Exception as e:
                print(f"  {table}.{column}: Table/column not found (skipping)")
        
        if total_rows == 0:
            print()
            print("✓ All data is already encrypted or empty. Nothing to do.")
            return
        
        # Confirm
        print()
        print(f"Total: {total_rows} rows will be encrypted")
        confirm = input("Proceed with encryption? (yes/no): ")
        
        if confirm.lower() != "yes":
            print("Cancelled.")
            return
        
        # Encrypt
        print()
        print("Encrypting data...")
        
        for table, id_col, column in tables_to_encrypt:
            try:
                encrypt_table_column(session, table, id_col, column)
            except Exception as e:
                print(f"  {table}.{column}: Error - {e}")
        
        print()
        print("=" * 60)
        print("✓ Encryption complete!")
        print()
        print("IMPORTANT:")
        print("  - Keep your password safe - you need it to decrypt data")
        print("  - Backup saved at:", BACKUP_PATH)
        print("  - If you change your password, run this script again")
        print("=" * 60)
        
    finally:
        session.close()


if __name__ == "__main__":
    main()

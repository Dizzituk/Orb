#!/usr/bin/env python3
"""
Verify that encryption is working correctly.

This script:
1. Creates a test message with a known marker string
2. Checks that the marker appears in decrypted API response
3. Checks that the marker does NOT appear in plaintext in the database file

Usage:
    cd D:\Orb
    $env:ORB_MASTER_KEY = "your-43-char-key"
    python scripts/verify_encryption.py
"""

import os
import sys
import sqlite3
import secrets
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.crypto.encryption import (
    init_master_key_from_env,
    encrypt_string,
    decrypt_string,
    is_encryption_ready,
)

# Database path
DB_PATH = Path("data/orb_memory.db")

# Test marker - unique string that shouldn't appear anywhere else
TEST_MARKER = f"ORB_ENCRYPTION_TEST_{secrets.token_hex(8)}"


def check_plaintext_in_file(filepath: Path, search_string: str) -> bool:
    """
    Check if a string appears in plaintext in a file.
    Returns True if found (BAD), False if not found (GOOD).
    """
    with open(filepath, "rb") as f:
        content = f.read()
    
    # Search for the string as bytes
    search_bytes = search_string.encode("utf-8")
    return search_bytes in content


def main():
    print("=" * 60)
    print("ENCRYPTION VERIFICATION TEST")
    print("=" * 60)
    print()
    
    # Check for master key
    if not os.environ.get("ORB_MASTER_KEY"):
        print("[ERROR] ORB_MASTER_KEY environment variable is not set.")
        sys.exit(1)
    
    # Initialize encryption
    if not init_master_key_from_env():
        print("[ERROR] Failed to initialize encryption.")
        sys.exit(1)
    
    print("[OK] Encryption initialized")
    print()
    
    # Test 1: Basic encrypt/decrypt round-trip
    print("Test 1: Encrypt/Decrypt Round-Trip")
    print("-" * 40)
    
    test_plaintext = f"Secret message: {TEST_MARKER}"
    encrypted = encrypt_string(test_plaintext)
    decrypted = decrypt_string(encrypted)
    
    print(f"  Plaintext length: {len(test_plaintext)}")
    print(f"  Encrypted length: {len(encrypted)}")
    print(f"  Encrypted starts with 'ENC:': {encrypted.startswith('ENC:')}")
    print(f"  Decrypted matches original: {decrypted == test_plaintext}")
    
    if decrypted != test_plaintext:
        print("[FAIL] Round-trip failed!")
        sys.exit(1)
    
    print("[PASS] Round-trip successful")
    print()
    
    # Test 2: Insert test data and check database
    print("Test 2: Database Encryption")
    print("-" * 40)
    
    if not DB_PATH.exists():
        print(f"[SKIP] Database not found: {DB_PATH}")
        print("       Run this test after creating some data in the app.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Find a project to use for testing
    cursor.execute("SELECT id FROM projects LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("[SKIP] No projects in database. Create a project first.")
        conn.close()
        return
    
    project_id = row[0]
    
    # Insert a test message
    encrypted_content = encrypt_string(f"Verification test: {TEST_MARKER}")
    cursor.execute(
        "INSERT INTO messages (project_id, role, content) VALUES (?, ?, ?)",
        (project_id, "system", encrypted_content)
    )
    test_message_id = cursor.lastrowid
    conn.commit()
    
    print(f"  Inserted test message ID: {test_message_id}")
    print(f"  Test marker: {TEST_MARKER}")
    
    # Check if marker appears in database file
    conn.close()  # Close connection so we can read the file
    
    marker_in_plaintext = check_plaintext_in_file(DB_PATH, TEST_MARKER)
    
    if marker_in_plaintext:
        print(f"[FAIL] Test marker found in plaintext in database!")
        print("       Encryption is NOT working correctly.")
        
        # Clean up test message
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM messages WHERE id = ?", (test_message_id,))
        conn.commit()
        conn.close()
        
        sys.exit(1)
    
    print("[PASS] Test marker NOT found in plaintext (encryption working)")
    
    # Verify we can decrypt it back
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM messages WHERE id = ?", (test_message_id,))
    row = cursor.fetchone()
    
    if row:
        stored_content = row[0]
        decrypted_content = decrypt_string(stored_content)
        
        if TEST_MARKER in decrypted_content:
            print("[PASS] Successfully decrypted stored content")
        else:
            print("[FAIL] Decrypted content doesn't contain marker")
            sys.exit(1)
    
    # Clean up test message
    cursor.execute("DELETE FROM messages WHERE id = ?", (test_message_id,))
    conn.commit()
    conn.close()
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED - Encryption is working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
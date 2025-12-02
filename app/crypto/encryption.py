# app/crypto/encryption.py
"""
Field-level encryption using Fernet (symmetric encryption).

Security Level 4: Master key from ORB_MASTER_KEY environment variable.
The master key is managed by Electron and stored in Windows Credential Manager.

Legacy PBKDF2 key derivation is retained only for migration purposes.
"""

import os
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Salt storage path (used only for legacy migration)
SALT_PATH = Path("data/encryption_salt.bin")

# Global encryption manager instance
_encryption_manager: Optional["EncryptionManager"] = None
_master_key_initialized: bool = False


class EncryptionManager:
    """
    Manages encryption/decryption of sensitive data.
    Uses Fernet symmetric encryption.
    """
    
    def __init__(self, key: bytes):
        """Initialize with a 32-byte encryption key."""
        # Fernet requires URL-safe base64 encoded key
        self._fernet = Fernet(base64.urlsafe_b64encode(key))
        self._key = key
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.
        Returns base64-encoded ciphertext prefixed with 'ENC:'.
        """
        if not plaintext:
            return plaintext
        
        ciphertext = self._fernet.encrypt(plaintext.encode('utf-8'))
        return f"ENC:{base64.urlsafe_b64encode(ciphertext).decode('ascii')}"
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string.
        Expects base64-encoded ciphertext prefixed with 'ENC:'.
        Returns original plaintext, or ciphertext if decryption fails.
        """
        if not ciphertext:
            return ciphertext
        
        # Not encrypted
        if not ciphertext.startswith("ENC:"):
            return ciphertext
        
        try:
            encoded = ciphertext[4:]  # Remove 'ENC:' prefix
            encrypted_bytes = base64.urlsafe_b64decode(encoded.encode('ascii'))
            plaintext = self._fernet.decrypt(encrypted_bytes)
            return plaintext.decode('utf-8')
        except (InvalidToken, ValueError, UnicodeDecodeError) as e:
            print(f"[crypto] Decryption failed: {e}")
            # Return as-is if decryption fails (might be corrupted or wrong key)
            return ciphertext
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted."""
        return value and value.startswith("ENC:")


# ============================================================================
# MASTER KEY INITIALIZATION (Security Level 4)
# ============================================================================

def _decode_master_key(key_string: str) -> Optional[bytes]:
    """
    Decode and validate a master key string.
    Expects URL-safe base64 encoded 32-byte key.
    Returns the 32-byte key, or None if invalid.
    """
    try:
        # URL-safe base64 may or may not have padding
        # Add padding if needed
        padded = key_string + '=' * (4 - len(key_string) % 4) if len(key_string) % 4 else key_string
        
        # Decode from URL-safe base64
        key_bytes = base64.urlsafe_b64decode(padded.encode('ascii'))
        
        if len(key_bytes) != 32:
            print(f"[crypto] Invalid master key length: expected 32 bytes, got {len(key_bytes)}")
            return None
        
        return key_bytes
    except Exception as e:
        print(f"[crypto] Failed to decode master key: {e}")
        return None


def init_master_key_from_env() -> bool:
    """
    Initialize encryption using ORB_MASTER_KEY environment variable.
    
    This is called once at backend startup.
    Returns True if successful, False otherwise.
    
    If ORB_MASTER_KEY is not set or invalid, this function will:
    - Print a clear error message
    - Return False (caller should exit)
    """
    global _encryption_manager, _master_key_initialized
    
    key_string = os.environ.get("ORB_MASTER_KEY")
    
    if not key_string:
        print("[crypto] ERROR: ORB_MASTER_KEY environment variable is not set.")
        print("[crypto] The backend must be started via Electron, which provides the master key.")
        print("[crypto] If you need to run the backend manually for development:")
        print("[crypto]   1. Start Electron once to create the master key in Credential Manager")
        print("[crypto]   2. Export the key: set ORB_MASTER_KEY=<key>")
        print("[crypto]   3. Then start the backend")
        return False
    
    key_bytes = _decode_master_key(key_string)
    if not key_bytes:
        print("[crypto] ERROR: ORB_MASTER_KEY is invalid (could not decode or wrong length).")
        return False
    
    _encryption_manager = EncryptionManager(key_bytes)
    _master_key_initialized = True
    print("[crypto] Master key initialized from environment")
    return True


def require_master_key_or_exit() -> None:
    """
    Ensure master key is initialized, or exit the process.
    Call this at backend startup after loading environment.
    """
    if not init_master_key_from_env():
        print("[crypto] FATAL: Cannot start without valid master key. Exiting.")
        sys.exit(1)


def is_master_key_initialized() -> bool:
    """Check if master key has been initialized."""
    return _master_key_initialized


# ============================================================================
# LEGACY: Password-derived key (for migration only)
# ============================================================================

def _get_or_create_salt() -> bytes:
    """
    LEGACY: Get existing salt or create a new one.
    Used only for migration from password-derived encryption.
    """
    SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if SALT_PATH.exists():
        return SALT_PATH.read_bytes()
    
    # Generate new salt (16 bytes)
    salt = os.urandom(16)
    SALT_PATH.write_bytes(salt)
    print(f"[crypto] Generated new encryption salt")
    return salt


def derive_key_from_password(password: str) -> bytes:
    """
    LEGACY: Derive a 32-byte encryption key from password using PBKDF2.
    
    Used only by migration script to decrypt data encrypted with the old scheme.
    Do not use for new encryption operations.
    """
    salt = _get_or_create_salt()
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,  # OWASP recommended minimum
    )
    
    key = kdf.derive(password.encode('utf-8'))
    return key


def create_legacy_encryption_manager(password: str) -> EncryptionManager:
    """
    LEGACY: Create an EncryptionManager using password-derived key.
    
    Used only by migration script to decrypt old data.
    """
    key = derive_key_from_password(password)
    return EncryptionManager(key)


# ============================================================================
# DEPRECATED: Old password-based initialization
# These functions are kept for backwards compatibility but should not be used.
# ============================================================================

def set_encryption_key(password: str) -> None:
    """
    DEPRECATED: Initialize encryption with password-derived key.
    
    This function is deprecated. Encryption is now initialized via
    init_master_key_from_env() using the ORB_MASTER_KEY environment variable.
    
    Kept only for backwards compatibility during transition.
    """
    global _encryption_manager
    
    # If master key is already initialized, ignore password-based init
    if _master_key_initialized:
        print("[crypto] Master key already initialized, ignoring password-based init")
        return
    
    # Fallback to password-derived key (legacy behavior)
    print("[crypto] WARNING: Using deprecated password-derived encryption")
    key = derive_key_from_password(password)
    _encryption_manager = EncryptionManager(key)
    print("[crypto] Encryption initialized (legacy mode)")


def clear_encryption_key() -> None:
    """
    Clear the encryption key from memory.
    
    Note: With master key mode, encryption remains active for the lifetime
    of the backend process. This function is kept for API compatibility
    but only clears the key if using legacy password-based encryption.
    """
    global _encryption_manager, _master_key_initialized
    
    if _master_key_initialized:
        # In master key mode, don't clear - encryption should remain active
        print("[crypto] Master key mode active, encryption remains enabled")
        return
    
    _encryption_manager = None
    print("[crypto] Encryption key cleared")


# ============================================================================
# PUBLIC API
# ============================================================================

def get_encryption_manager() -> Optional[EncryptionManager]:
    """Get the current encryption manager, or None if not initialized."""
    return _encryption_manager


def is_encryption_ready() -> bool:
    """Check if encryption is initialized."""
    return _encryption_manager is not None


def encrypt_string(plaintext: str) -> str:
    """
    Encrypt a string using the current encryption key.
    Returns plaintext unchanged if encryption not initialized.
    """
    if not _encryption_manager:
        return plaintext
    return _encryption_manager.encrypt(plaintext)


def decrypt_string(ciphertext: str) -> str:
    """
    Decrypt a string using the current encryption key.
    Returns ciphertext unchanged if encryption not initialized.
    """
    if not _encryption_manager:
        return ciphertext
    return _encryption_manager.decrypt(ciphertext)
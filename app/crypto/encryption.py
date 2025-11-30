# app/crypto/encryption.py
"""
Field-level encryption using Fernet (symmetric encryption).
Key is derived from user's password using PBKDF2.
"""

import os
import base64
import hashlib
import json
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Salt storage path
SALT_PATH = Path("data/encryption_salt.bin")

# Global encryption manager instance
_encryption_manager: Optional["EncryptionManager"] = None


class EncryptionManager:
    """
    Manages encryption/decryption of sensitive data.
    Uses Fernet symmetric encryption with password-derived key.
    """
    
    def __init__(self, key: bytes):
        """Initialize with a 32-byte encryption key."""
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


def _get_or_create_salt() -> bytes:
    """Get existing salt or create a new one."""
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
    Derive a 32-byte encryption key from password using PBKDF2.
    Uses a stored salt for consistency.
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


def set_encryption_key(password: str) -> None:
    """
    Initialize encryption with password-derived key.
    Call this after successful login.
    """
    global _encryption_manager
    
    key = derive_key_from_password(password)
    _encryption_manager = EncryptionManager(key)
    print("[crypto] Encryption initialized")


def clear_encryption_key() -> None:
    """Clear the encryption key from memory. Call on logout."""
    global _encryption_manager
    _encryption_manager = None
    print("[crypto] Encryption key cleared")


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

# app/crypto/__init__.py
"""
Cryptography module for Orb.
Provides field-level encryption for sensitive database content.
"""

from .encryption import (
    EncryptionManager,
    get_encryption_manager,
    set_encryption_key,
    clear_encryption_key,
    encrypt_string,
    decrypt_string,
    is_encryption_ready,
)

from .types import (
    EncryptedText,
    EncryptedJSON,
)

__all__ = [
    # Encryption manager
    "EncryptionManager",
    "get_encryption_manager",
    "set_encryption_key",
    "clear_encryption_key",
    "encrypt_string",
    "decrypt_string",
    "is_encryption_ready",
    # SQLAlchemy types
    "EncryptedText",
    "EncryptedJSON",
]

# app/crypto/__init__.py
"""
Encryption module for Orb.

Security Level 4: Master key from ORB_MASTER_KEY environment variable.
The master key is managed by Electron and stored in Windows Credential Manager.

Exports:
    - EncryptedText, EncryptedJSON: SQLAlchemy column types for transparent encryption
    - encrypt_string, decrypt_string: Manual encryption functions
    - is_encryption_ready: Check if encryption is initialized
    - init_master_key_from_env, require_master_key_or_exit: Startup functions
    - derive_key_from_password, create_legacy_encryption_manager: Migration only
"""

# Core encryption functions
from .encryption import (
    # Master key (Security Level 4)
    init_master_key_from_env,
    require_master_key_or_exit,
    is_master_key_initialized,
    
    # Public API
    get_encryption_manager,
    is_encryption_ready,
    encrypt_string,
    decrypt_string,
    
    # Legacy (for migration only)
    derive_key_from_password,
    create_legacy_encryption_manager,
    
    # Deprecated (kept for API compatibility)
    set_encryption_key,
    clear_encryption_key,
)

# SQLAlchemy column types
from .types import (
    EncryptedText,
    EncryptedJSON,
)

__all__ = [
    # Master key
    "init_master_key_from_env",
    "require_master_key_or_exit",
    "is_master_key_initialized",
    
    # Public API
    "get_encryption_manager",
    "is_encryption_ready",
    "encrypt_string",
    "decrypt_string",
    
    # SQLAlchemy types
    "EncryptedText",
    "EncryptedJSON",
    
    # Legacy
    "derive_key_from_password",
    "create_legacy_encryption_manager",
    
    # Deprecated
    "set_encryption_key",
    "clear_encryption_key",
]
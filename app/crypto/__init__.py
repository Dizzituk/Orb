# app/crypto/__init__.py
"""
Encryption module exports.

Security Level 4: Master key encryption via ORB_MASTER_KEY environment variable.
"""

from .encryption import (
    # Primary API
    encrypt_string,
    decrypt_string,
    is_encryption_ready,
    get_encryption_manager,
    EncryptionManager,
    
    # Master key initialization (Security Level 4)
    init_master_key_from_env,
    require_master_key_or_exit,
    is_master_key_initialized,
    
    # Legacy (for migration only)
    derive_key_from_password,
    create_legacy_encryption_manager,
    
    # Deprecated (backwards compatibility)
    set_encryption_key,
    clear_encryption_key,
)

__all__ = [
    "encrypt_string",
    "decrypt_string",
    "is_encryption_ready",
    "get_encryption_manager",
    "EncryptionManager",
    "init_master_key_from_env",
    "require_master_key_or_exit",
    "is_master_key_initialized",
    "derive_key_from_password",
    "create_legacy_encryption_manager",
    "set_encryption_key",
    "clear_encryption_key",
]
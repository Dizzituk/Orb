# app/crypto/types.py
"""
SQLAlchemy custom types for encrypted columns.

Security Level 4: These types require encryption to be initialized.
If encryption is not ready, writes will fail (not silently store plaintext).
"""

from sqlalchemy import TypeDecorator, Text
from .encryption import encrypt_string, decrypt_string, is_encryption_ready


class EncryptionNotReadyError(Exception):
    """Raised when attempting to encrypt/decrypt without initialized encryption."""
    pass


class EncryptedText(TypeDecorator):
    """
    SQLAlchemy type that transparently encrypts/decrypts text.
    
    Usage in models:
        content = Column(EncryptedText, nullable=True)
    
    The encryption happens automatically on INSERT/UPDATE,
    and decryption happens automatically on SELECT.
    
    Security Level 4: Encryption MUST be initialized before use.
    Attempting to write without encryption will raise an error.
    """
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Encrypt before storing in database."""
        if value is None:
            return None
        
        if not is_encryption_ready():
            raise EncryptionNotReadyError(
                "Cannot store encrypted data: encryption not initialized. "
                "Ensure ORB_MASTER_KEY is set and require_master_key_or_exit() was called."
            )
        
        return encrypt_string(value)
    
    def process_result_value(self, value, dialect):
        """Decrypt when reading from database."""
        if value is None:
            return None
        
        # If value is not encrypted (legacy plaintext), return as-is
        if not value.startswith("ENC:"):
            return value
        
        if not is_encryption_ready():
            # Return the encrypted value with a marker - don't fail reads
            # This allows the app to start and show that data exists but can't be decrypted
            return f"[ENCRYPTED - key not available] {value[:20]}..."
        
        return decrypt_string(value)


class EncryptedJSON(TypeDecorator):
    """
    SQLAlchemy type that encrypts JSON data as a string.
    
    Usage in models:
        structured_data = Column(EncryptedJSON, nullable=True)
    
    Security Level 4: Encryption MUST be initialized before use.
    """
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Convert to JSON string and encrypt."""
        import json
        
        if value is None:
            return None
        
        if not is_encryption_ready():
            raise EncryptionNotReadyError(
                "Cannot store encrypted JSON: encryption not initialized. "
                "Ensure ORB_MASTER_KEY is set and require_master_key_or_exit() was called."
            )
        
        json_str = json.dumps(value, ensure_ascii=False)
        return encrypt_string(json_str)
    
    def process_result_value(self, value, dialect):
        """Decrypt and parse JSON."""
        import json
        
        if value is None:
            return None
        
        # If value is not encrypted (legacy plaintext), try to parse
        if not value.startswith("ENC:"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        
        if not is_encryption_ready():
            return {"_encrypted": True, "_error": "Encryption key not available"}
        
        decrypted = decrypt_string(value)
        
        # Handle case where decryption returns the ENC: prefix (wrong key)
        if decrypted.startswith("ENC:"):
            return {"_encrypted": True, "_error": "Decryption failed - wrong key"}
        
        try:
            return json.loads(decrypted)
        except json.JSONDecodeError:
            return decrypted  # Return as string if not valid JSON
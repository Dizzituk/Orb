# app/crypto/types.py
"""
SQLAlchemy custom types for encrypted columns.
"""

from sqlalchemy import TypeDecorator, Text
from .encryption import encrypt_string, decrypt_string, is_encryption_ready


class EncryptedText(TypeDecorator):
    """
    SQLAlchemy type that transparently encrypts/decrypts text.
    
    Usage in models:
        content = Column(EncryptedText, nullable=True)
    
    The encryption happens automatically on INSERT/UPDATE,
    and decryption happens automatically on SELECT.
    
    If encryption is not initialized (user not logged in),
    data is stored/returned as plaintext.
    """
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Encrypt before storing in database."""
        if value is None:
            return None
        
        if is_encryption_ready():
            return encrypt_string(value)
        return value
    
    def process_result_value(self, value, dialect):
        """Decrypt when reading from database."""
        if value is None:
            return None
        
        if is_encryption_ready():
            return decrypt_string(value)
        return value


class EncryptedJSON(TypeDecorator):
    """
    SQLAlchemy type that encrypts JSON data as a string.
    
    Usage in models:
        structured_data = Column(EncryptedJSON, nullable=True)
    """
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Convert to JSON string and encrypt."""
        import json
        
        if value is None:
            return None
        
        json_str = json.dumps(value, ensure_ascii=False)
        
        if is_encryption_ready():
            return encrypt_string(json_str)
        return json_str
    
    def process_result_value(self, value, dialect):
        """Decrypt and parse JSON."""
        import json
        
        if value is None:
            return None
        
        if is_encryption_ready():
            value = decrypt_string(value)
        
        # Handle case where decryption returns the ENC: prefix (wrong key)
        if value.startswith("ENC:"):
            return {"_encrypted": True, "_error": "Decryption failed"}
        
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value  # Return as string if not valid JSON

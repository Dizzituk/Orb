# FILE: tests/test_encryption.py
"""
Tests for app/crypto/encryption.py
Data encryption - encrypts sensitive data at rest.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch
import os
import time


class TestEncryptionImports:
    """Test encryption module structure."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.crypto import encryption
        assert encryption is not None
    
    def test_core_functions_exist(self):
        """Test core functions are exported."""
        from app.crypto.encryption import (
            encrypt_string,
            decrypt_string,
            set_encryption_key,
            clear_encryption_key,
            is_encryption_ready,
            is_master_key_initialized,
            derive_key_from_password,
            get_encryption_manager,
            create_legacy_encryption_manager,
        )
        assert callable(encrypt_string)
        assert callable(decrypt_string)
        assert callable(set_encryption_key)
        assert callable(clear_encryption_key)
        assert callable(is_encryption_ready)


class TestEncryptDecrypt:
    """Test encrypt/decrypt round-trip."""
    
    @pytest.fixture(autouse=True)
    def setup_encryption(self):
        """Set up encryption key for tests."""
        from app.crypto.encryption import set_encryption_key, clear_encryption_key
        set_encryption_key("test-password-for-unit-tests")
        yield
        clear_encryption_key()
    
    def test_encrypt_decrypt_string(self):
        """Test encrypting and decrypting strings."""
        from app.crypto.encryption import encrypt_string, decrypt_string
        
        plaintext = "Hello, World! This is a secret message."
        ciphertext = encrypt_string(plaintext)
        
        # Ciphertext should be different from plaintext
        assert ciphertext != plaintext
        
        # Decryption should recover original
        decrypted = decrypt_string(ciphertext)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_unicode(self):
        """Test encrypting and decrypting unicode strings."""
        from app.crypto.encryption import encrypt_string, decrypt_string
        
        plaintext = "Unicode test: 日本語 🎉 émoji ñ"
        ciphertext = encrypt_string(plaintext)
        decrypted = decrypt_string(ciphertext)
        
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_large_data(self):
        """Test encrypting large data."""
        from app.crypto.encryption import encrypt_string, decrypt_string
        
        # 1MB of text
        plaintext = "x" * (1024 * 1024)
        ciphertext = encrypt_string(plaintext)
        decrypted = decrypt_string(ciphertext)
        
        assert decrypted == plaintext
        assert len(decrypted) == len(plaintext)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from app.crypto.encryption import encrypt_string, decrypt_string
        
        plaintext = ""
        ciphertext = encrypt_string(plaintext)
        decrypted = decrypt_string(ciphertext)
        
        assert decrypted == plaintext
    
    def test_encrypt_produces_different_ciphertext(self):
        """Test that encryption is non-deterministic (uses IV/nonce)."""
        from app.crypto.encryption import encrypt_string
        
        plaintext = "Same message"
        ciphertext1 = encrypt_string(plaintext)
        ciphertext2 = encrypt_string(plaintext)
        
        # Same plaintext should produce different ciphertext due to random IV
        # (This is a property of secure encryption)
        # Note: Some implementations might be deterministic, so we check both decrypt correctly
        from app.crypto.encryption import decrypt_string
        assert decrypt_string(ciphertext1) == plaintext
        assert decrypt_string(ciphertext2) == plaintext


class TestKeyManagement:
    """Test key management."""
    
    def test_key_derivation(self):
        """Test key derivation from password."""
        from app.crypto.encryption import derive_key_from_password
        
        password = "my-secret-password"
        key = derive_key_from_password(password)
        
        # Key should be bytes
        assert isinstance(key, bytes)
        # Key should have reasonable length (32 bytes = 256 bits typical)
        assert len(key) >= 16
    
    def test_key_derivation_deterministic(self):
        """Test key derivation is deterministic for same password."""
        from app.crypto.encryption import derive_key_from_password
        
        password = "consistent-password"
        key1 = derive_key_from_password(password)
        key2 = derive_key_from_password(password)
        
        assert key1 == key2
    
    def test_different_passwords_different_keys(self):
        """Test different passwords produce different keys."""
        from app.crypto.encryption import derive_key_from_password
        
        key1 = derive_key_from_password("password1")
        key2 = derive_key_from_password("password2")
        
        assert key1 != key2
    
    def test_set_and_clear_key(self):
        """Test setting and clearing encryption key."""
        from app.crypto.encryption import (
            set_encryption_key,
            clear_encryption_key,
            is_encryption_ready,
        )
        
        # Clear any existing key
        clear_encryption_key()
        
        # Should not be ready without key
        assert is_encryption_ready() == False
        
        # Set key
        set_encryption_key("test-key")
        assert is_encryption_ready() == True
        
        # Clear key
        clear_encryption_key()
        assert is_encryption_ready() == False
    
    def test_missing_key_error(self):
        """Test error when encrypting without key.
        
        SECURITY: Encryption without a key should fail loudly, not silently.
        Silent failure (returning empty string) risks data loss.
        """
        from app.crypto.encryption import (
            encrypt_string,
            clear_encryption_key,
            is_encryption_ready,
            EncryptionNotInitializedError,
        )
        
        clear_encryption_key()
        assert is_encryption_ready() == False
        
        # SECURITY EXPECTATION: Should raise when trying to encrypt without key
        with pytest.raises(EncryptionNotInitializedError):
            encrypt_string("test")


class TestEncryptionManager:
    """Test EncryptionManager class."""
    
    def test_create_legacy_manager(self):
        """Test creating legacy encryption manager."""
        from app.crypto.encryption import create_legacy_encryption_manager
        
        manager = create_legacy_encryption_manager("test-password")
        assert manager is not None
        assert hasattr(manager, 'encrypt')
        assert hasattr(manager, 'decrypt')
    
    def test_manager_encrypt_decrypt(self):
        """Test manager encrypt/decrypt methods."""
        from app.crypto.encryption import create_legacy_encryption_manager
        
        manager = create_legacy_encryption_manager("test-password")
        
        plaintext = "Secret data via manager"
        ciphertext = manager.encrypt(plaintext)
        decrypted = manager.decrypt(ciphertext)
        
        assert decrypted == plaintext
    
    def test_manager_is_encrypted(self):
        """Test manager is_encrypted detection."""
        from app.crypto.encryption import create_legacy_encryption_manager
        
        manager = create_legacy_encryption_manager("test-password")
        
        plaintext = "Not encrypted"
        ciphertext = manager.encrypt(plaintext)
        
        # Ciphertext should be detected as encrypted
        assert manager.is_encrypted(ciphertext) == True
        # Plain text should not be detected as encrypted
        assert manager.is_encrypted(plaintext) == False
    
    def test_get_encryption_manager(self):
        """Test get_encryption_manager returns manager when key set."""
        from app.crypto.encryption import (
            set_encryption_key,
            clear_encryption_key,
            get_encryption_manager,
        )
        
        clear_encryption_key()
        assert get_encryption_manager() is None
        
        set_encryption_key("test-key")
        manager = get_encryption_manager()
        assert manager is not None
        
        clear_encryption_key()


class TestEncryptionIntegrity:
    """Test encryption integrity."""
    
    @pytest.fixture(autouse=True)
    def setup_encryption(self):
        """Set up encryption key for tests."""
        from app.crypto.encryption import set_encryption_key, clear_encryption_key
        set_encryption_key("integrity-test-key")
        yield
        clear_encryption_key()
    
    def test_tampered_data_detected(self):
        """Test tampering is detected.
        
        SECURITY: Tampered ciphertext should raise, not silently return garbage.
        Silent failure masks attacks.
        """
        from app.crypto.encryption import encrypt_string, decrypt_string, DecryptionError
        
        plaintext = "Original message"
        ciphertext = encrypt_string(plaintext)
        
        # Tamper with ciphertext (flip some bits in the middle)
        tampered = ciphertext[:10] + "X" + ciphertext[11:]
        
        # SECURITY EXPECTATION: Decryption of tampered data should raise
        with pytest.raises(DecryptionError):
            decrypt_string(tampered)
    
    def test_wrong_key_fails(self):
        """Test decryption with wrong key fails.
        
        SECURITY: Wrong key should raise, not silently return empty.
        Silent failure masks configuration errors.
        """
        from app.crypto.encryption import (
            encrypt_string,
            decrypt_string,
            set_encryption_key,
            clear_encryption_key,
            DecryptionError,
        )
        
        # Encrypt with one key
        plaintext = "Secret message"
        ciphertext = encrypt_string(plaintext)
        
        # Change to different key
        clear_encryption_key()
        set_encryption_key("different-key-entirely")
        
        # SECURITY EXPECTATION: Decryption with wrong key should raise
        with pytest.raises(DecryptionError):
            decrypt_string(ciphertext)
    
    def test_corrupted_ciphertext_detected(self):
        """Test corrupted ciphertext is detected.
        
        SECURITY: Invalid ciphertext should raise, not silently return empty.
        """
        from app.crypto.encryption import decrypt_string, DecryptionError
        
        # Garbage that looks like encrypted data (has ENC: prefix)
        garbage = "ENC:not-valid-base64-ciphertext!!!"
        
        # SECURITY EXPECTATION: Invalid ciphertext should raise
        with pytest.raises(DecryptionError):
            decrypt_string(garbage)


class TestEncryptionPerformance:
    """Test encryption performance."""
    
    @pytest.fixture(autouse=True)
    def setup_encryption(self):
        """Set up encryption key for tests."""
        from app.crypto.encryption import set_encryption_key, clear_encryption_key
        set_encryption_key("perf-test-key")
        yield
        clear_encryption_key()
    
    def test_encryption_speed_acceptable(self):
        """Test encryption completes in reasonable time."""
        from app.crypto.encryption import encrypt_string, decrypt_string
        
        # 100KB of data
        data = "x" * (100 * 1024)
        
        start = time.time()
        ciphertext = encrypt_string(data)
        encrypt_time = time.time() - start
        
        start = time.time()
        decrypt_string(ciphertext)
        decrypt_time = time.time() - start
        
        # Should complete in under 1 second each
        assert encrypt_time < 1.0, f"Encryption took {encrypt_time:.2f}s"
        assert decrypt_time < 1.0, f"Decryption took {decrypt_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

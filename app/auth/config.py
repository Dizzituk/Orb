# app/auth/config.py
"""
Authentication configuration and password management.
Supports password-based authentication with bcrypt hashing.
Integrates with crypto module for database encryption.
"""

import secrets
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Try to import bcrypt, fall back to hashlib if not available
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    print("[auth] bcrypt not installed, using SHA256 (less secure). Install with: pip install bcrypt")

# Import encryption module
try:
    from app.crypto import set_encryption_key, clear_encryption_key
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("[auth] crypto module not available, database encryption disabled")

AUTH_CONFIG_PATH = Path("data/auth.json")


def _ensure_data_dir():
    """Ensure the data directory exists."""
    AUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict:
    """Load auth configuration from disk."""
    if not AUTH_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(AUTH_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def _save_config(config: dict):
    """Save auth configuration to disk."""
    _ensure_data_dir()
    AUTH_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt (preferred) or SHA256 (fallback)."""
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    else:
        # Fallback to SHA256 with salt
        salt = secrets.token_hex(16)
        hash_val = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"sha256:{salt}:{hash_val}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against stored hash."""
    if HAS_BCRYPT and stored_hash.startswith("$2"):
        # bcrypt hash
        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except Exception:
            return False
    elif stored_hash.startswith("sha256:"):
        # SHA256 fallback
        parts = stored_hash.split(":")
        if len(parts) != 3:
            return False
        _, salt, hash_val = parts
        check_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return secrets.compare_digest(check_hash, hash_val)
    else:
        return False


def _generate_session_token() -> str:
    """Generate a secure session token."""
    return f"orb_session_{secrets.token_hex(32)}"


def _init_encryption(password: str):
    """Initialize database encryption with password-derived key."""
    if HAS_CRYPTO:
        try:
            set_encryption_key(password)
        except Exception as e:
            print(f"[auth] Failed to initialize encryption: {e}")


def _clear_encryption():
    """Clear database encryption key."""
    if HAS_CRYPTO:
        try:
            clear_encryption_key()
        except Exception as e:
            print(f"[auth] Failed to clear encryption: {e}")


# ============ PUBLIC API ============

def is_auth_configured() -> bool:
    """Check if authentication (password) is configured."""
    config = _load_config()
    return bool(config.get("password_hash"))


def is_legacy_api_key_auth() -> bool:
    """Check if using legacy API key auth (for migration)."""
    config = _load_config()
    return bool(config.get("api_key_hash")) and not config.get("password_hash")


def is_encryption_enabled() -> bool:
    """Check if database encryption is enabled."""
    config = _load_config()
    return config.get("encryption_enabled", False)


def setup_password(password: str, enable_encryption: bool = True) -> dict:
    """
    Set up password authentication.
    Optionally enables database encryption.
    Returns session info on success.
    """
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters")
    
    config = _load_config()
    
    # Hash and store password
    config["password_hash"] = _hash_password(password)
    config["created_at"] = datetime.now().isoformat()
    config["auth_type"] = "password"
    config["encryption_enabled"] = enable_encryption and HAS_CRYPTO
    
    # Remove legacy API key fields if present
    config.pop("api_key_hash", None)
    config.pop("api_key_plain", None)
    
    # Generate initial session
    session_token = _generate_session_token()
    config["current_session"] = {
        "token": session_token,
        "created_at": datetime.now().isoformat(),
    }
    
    _save_config(config)
    
    # Initialize encryption if enabled
    if config["encryption_enabled"]:
        _init_encryption(password)
    
    return {
        "session_token": session_token,
        "message": "Password configured successfully",
        "encryption_enabled": config["encryption_enabled"]
    }


def login(password: str) -> Optional[dict]:
    """
    Authenticate with password.
    Returns session info on success, None on failure.
    """
    config = _load_config()
    stored_hash = config.get("password_hash")
    
    if not stored_hash:
        return None
    
    if not _verify_password(password, stored_hash):
        return None
    
    # Generate new session token
    session_token = _generate_session_token()
    config["current_session"] = {
        "token": session_token,
        "created_at": datetime.now().isoformat(),
    }
    config["last_login"] = datetime.now().isoformat()
    
    _save_config(config)
    
    # Initialize encryption if enabled
    if config.get("encryption_enabled", False):
        _init_encryption(password)
    
    return {
        "session_token": session_token,
        "message": "Login successful",
        "encryption_enabled": config.get("encryption_enabled", False)
    }


def validate_session(token: str) -> bool:
    """Validate a session token."""
    if not token:
        return False
    
    config = _load_config()
    session = config.get("current_session", {})
    stored_token = session.get("token")
    
    if not stored_token:
        return False
    
    return secrets.compare_digest(token, stored_token)


def logout() -> bool:
    """Invalidate the current session and clear encryption key."""
    config = _load_config()
    config.pop("current_session", None)
    _save_config(config)
    
    # Clear encryption key from memory
    _clear_encryption()
    
    return True


def change_password(current_password: str, new_password: str) -> bool:
    """Change the password. Requires current password for verification."""
    config = _load_config()
    stored_hash = config.get("password_hash")
    
    if not stored_hash:
        return False
    
    if not _verify_password(current_password, stored_hash):
        return False
    
    if len(new_password) < 4:
        raise ValueError("Password must be at least 4 characters")
    
    config["password_hash"] = _hash_password(new_password)
    config["password_changed_at"] = datetime.now().isoformat()
    
    # Invalidate current session (force re-login)
    config.pop("current_session", None)
    
    _save_config(config)
    
    # Clear old encryption key - will re-init on next login
    _clear_encryption()
    
    # Note: Changing password means existing encrypted data needs re-encryption
    # This is handled by a separate migration process
    
    return True


def reset_auth() -> bool:
    """Reset all authentication (for recovery). Deletes auth.json."""
    if AUTH_CONFIG_PATH.exists():
        AUTH_CONFIG_PATH.unlink()
    _clear_encryption()
    return True


# ============ LEGACY API KEY SUPPORT (for migration) ============

def validate_api_key(provided_key: str) -> bool:
    """Validate legacy API key (for backwards compatibility during migration)."""
    config = _load_config()
    stored_hash = config.get("api_key_hash")
    
    if not stored_hash:
        return False
    
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return secrets.compare_digest(provided_hash, stored_hash)


def migrate_to_password(api_key: str, new_password: str) -> Optional[dict]:
    """
    Migrate from API key auth to password auth.
    Requires valid API key and new password.
    """
    if not validate_api_key(api_key):
        return None
    
    return setup_password(new_password)

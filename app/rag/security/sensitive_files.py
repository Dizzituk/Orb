"""
Sensitive file detection for RAG.

MUST run BEFORE any storage or embedding operations.
"""

import fnmatch
from typing import Tuple, Optional, Set


# =============================================================================
# EXACT FILENAMES (always blocked)
# =============================================================================

SENSITIVE_EXACT_NAMES: Set[str] = {
    # Environment files
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.staging", ".env.test",
    
    # SSH keys
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "id_rsa.pub", "id_dsa.pub", "id_ecdsa.pub", "id_ed25519.pub",
    
    # Credentials
    "credentials.json", "service_account.json",
    "secrets.json", "secrets.yaml", "secrets.yml",
    
    # Package auth
    ".npmrc", ".pypirc", ".netrc",
    
    # Other
    "htpasswd", ".htpasswd",
    "aws_credentials",
}


# =============================================================================
# GLOB PATTERNS (always blocked)
# =============================================================================

SENSITIVE_PATTERNS: Set[str] = {
    # Certificates and keys
    "*.pem", "*.key", "*.crt", "*.cer",
    "*.p12", "*.pfx", "*.jks", "*.keystore",
    
    # GPG
    "*.gpg", "*.asc",
    
    # SSH variants
    "*_rsa", "*_dsa", "*_ecdsa", "*_ed25519",
    
    # Environment variants
    ".env.*",
}


# =============================================================================
# PATH PATTERNS (blocked if in path)
# =============================================================================

SENSITIVE_PATH_PARTS: Set[str] = {
    ".ssh",
    ".gnupg",
    ".aws",
}


# =============================================================================
# SUBSTRINGS (user/sandbox only - too aggressive for repos)
# =============================================================================

SENSITIVE_SUBSTRINGS_USER_ONLY: Set[str] = {
    "secret",
    "token", 
    "password",
    "credential",
    "apikey",
    "api_key",
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def is_sensitive_file(
    filename: str,
    filepath: str,
    root_kind: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if file is sensitive.
    
    MUST call this BEFORE storing or embedding.
    
    Args:
        filename: Just the filename
        filepath: Full path
        root_kind: "repo", "user", or "sandbox"
        
    Returns:
        (is_sensitive, reason)
    """
    filename_lower = filename.lower()
    filepath_lower = filepath.lower().replace("\\", "/")
    
    # Exact name match
    if filename_lower in {n.lower() for n in SENSITIVE_EXACT_NAMES}:
        return True, f"Blocked filename: {filename}"
    
    # Glob pattern match
    for pattern in SENSITIVE_PATTERNS:
        if fnmatch.fnmatch(filename_lower, pattern.lower()):
            return True, f"Blocked pattern: {pattern}"
    
    # Path part match
    for part in SENSITIVE_PATH_PARTS:
        if f"/{part}/" in filepath_lower or filepath_lower.endswith(f"/{part}"):
            return True, f"Blocked path: {part}"
    
    # Substring match (user/sandbox only)
    if root_kind in ("user", "sandbox"):
        for substring in SENSITIVE_SUBSTRINGS_USER_ONLY:
            if substring in filename_lower:
                return True, f"Blocked substring: {substring}"
    
    return False, None


def should_skip_directory(dirname: str) -> bool:
    """Check if directory contains sensitive data."""
    dirname_lower = dirname.lower()
    return dirname_lower in {p.lower() for p in SENSITIVE_PATH_PARTS}

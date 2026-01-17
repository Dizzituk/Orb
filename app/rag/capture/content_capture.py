"""
Safe content capture with redaction.

Handles:
- Zone-aware size limits
- Binary detection
- Secret redaction
"""

import os
import re
import hashlib
from typing import Dict, Any, Set


# =============================================================================
# SIZE LIMITS BY ZONE
# =============================================================================

SIZE_LIMITS: Dict[str, int] = {
    "repo": 5 * 1024 * 1024,      # 5 MB
    "sandbox": 5 * 1024 * 1024,   # 5 MB
    "user": 100 * 1024,           # 100 KB (strict)
}

DEFAULT_SIZE_LIMIT = 100 * 1024


# =============================================================================
# CAPTURABLE EXTENSIONS
# =============================================================================

CAPTURABLE_EXTENSIONS: Set[str] = {
    # Code
    ".py", ".pyw", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".mts", ".cts", ".tsx",
    ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs",
    ".rb", ".php", ".swift", ".scala",
    ".sql", ".sh", ".bash", ".ps1", ".bat",
    
    # Config
    ".json", ".jsonc", ".yaml", ".yml",
    ".toml", ".xml", ".ini", ".cfg",
    
    # Web
    ".html", ".htm", ".css", ".scss", ".vue", ".svelte",
    
    # Docs
    ".md", ".markdown", ".txt", ".rst",
}

# Files without extension that are capturable
CAPTURABLE_NAMES: Set[str] = {
    "Dockerfile", "Makefile", "Rakefile", "Gemfile",
    "Procfile", ".gitignore", ".dockerignore",
}


# =============================================================================
# REDACTION PATTERNS
# =============================================================================

REDACTION_PATTERNS = [
    # API keys
    (r'(api[_-]?key\s*[=:]\s*["\']?)([a-zA-Z0-9_-]{20,})', r'\1[REDACTED]'),
    (r'(sk-[a-zA-Z0-9]{20,})', '[REDACTED_SK]'),
    (r'(pk-[a-zA-Z0-9]{20,})', '[REDACTED_PK]'),
    
    # AWS
    (r'(AKIA[A-Z0-9]{16})', '[REDACTED_AWS]'),
    (r'(aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?)([a-zA-Z0-9/+=]{30,})', r'\1[REDACTED]'),
    
    # Generic secrets
    (r'(secret[_-]?key\s*[=:]\s*["\']?)([a-zA-Z0-9_-]{16,})', r'\1[REDACTED]'),
    (r'(password\s*[=:]\s*["\']?)([^\s"\']{8,})', r'\1[REDACTED]'),
    (r'(token\s*[=:]\s*["\']?)([a-zA-Z0-9_.-]{20,})', r'\1[REDACTED]'),
    
    # Bearer tokens
    (r'(Bearer\s+)([a-zA-Z0-9_.-]{20,})', r'\1[REDACTED]'),
    
    # Private keys
    (r'-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----',
     '[REDACTED_PRIVATE_KEY]'),
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def is_binary_content(data: bytes, sample_size: int = 4096) -> bool:
    """Detect binary content via null bytes."""
    return b'\x00' in data[:sample_size]


def is_capturable_file(filename: str) -> bool:
    """Check if file type is capturable."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in CAPTURABLE_EXTENSIONS or filename in CAPTURABLE_NAMES


def capture_file_content(
    file_path: str,
    file_name: str,
    file_size: int,
    root_kind: str,
) -> Dict[str, Any]:
    """
    Safely capture file content.
    
    Args:
        file_path: Absolute path to file
        file_name: Filename
        file_size: Size in bytes
        root_kind: "repo", "user", or "sandbox"
        
    Returns:
        {
            success: bool,
            content: str | None,
            content_hash: str,
            line_count: int,
            skip_reason: str | None,
            redaction_count: int,
        }
    """
    result = {
        "success": False,
        "content": None,
        "content_hash": "",
        "line_count": 0,
        "skip_reason": None,
        "redaction_count": 0,
    }
    
    # Check extension
    if not is_capturable_file(file_name):
        ext = os.path.splitext(file_name)[1]
        result["skip_reason"] = f"Non-capturable: {ext}"
        return result
    
    # Check size
    limit = SIZE_LIMITS.get(root_kind, DEFAULT_SIZE_LIMIT)
    if file_size > limit:
        result["skip_reason"] = f"Too large: {file_size} > {limit}"
        return result
    
    # Read file
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
    except Exception as e:
        result["skip_reason"] = f"Read error: {e}"
        return result
    
    # Binary check
    if is_binary_content(raw):
        result["skip_reason"] = "Binary content"
        return result
    
    # Decode
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = raw.decode("latin-1")
        except Exception:
            result["skip_reason"] = "Encoding error"
            return result
    
    # Redact secrets
    redaction_count = 0
    for pattern, replacement in REDACTION_PATTERNS:
        content, count = re.subn(pattern, replacement, content, flags=re.IGNORECASE)
        redaction_count += count
    
    # Compute hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    # Count lines
    line_count = content.count("\n") + 1
    
    result["success"] = True
    result["content"] = content
    result["content_hash"] = content_hash
    result["line_count"] = line_count
    result["redaction_count"] = redaction_count
    
    return result

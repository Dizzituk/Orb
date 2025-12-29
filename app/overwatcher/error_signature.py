# FILE: app/overwatcher/error_signature.py
"""ErrorSignature computation for three-strike error handling.

Spec v2.3 §9.4: Define an error signature so 'same error' is machine-detectable.

ErrorSignature = sha256(exception_type + failing_test_name + top_N_stack_frames + module_path)

Strike rules:
- Strike 1: Internal knowledge only
- Strike 2 (same signature): Deep Research allowed
- Strike 3 (same signature): HARD STOP, quarantine

Reset rule: Different ErrorSignature resets strikes to 1.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ErrorSignature:
    """Unique identifier for an error pattern.
    
    Used to determine if consecutive failures are the "same error"
    for three-strike tracking.
    """
    
    exception_type: str
    failing_test_name: Optional[str]
    top_stack_frames: List[str]
    module_path: Optional[str]
    signature_hash: str = field(init=False)
    
    def __post_init__(self):
        """Compute the signature hash."""
        self.signature_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of error components."""
        components = [
            self.exception_type or "",
            self.failing_test_name or "",
            "|".join(self.top_stack_frames),
            self.module_path or "",
        ]
        combined = "\n".join(components)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
    
    def matches(self, other: "ErrorSignature") -> bool:
        """Check if this signature matches another (same error)."""
        return self.signature_hash == other.signature_hash
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "exception_type": self.exception_type,
            "failing_test_name": self.failing_test_name,
            "top_stack_frames": self.top_stack_frames,
            "module_path": self.module_path,
            "signature_hash": self.signature_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ErrorSignature":
        """Deserialize from dict."""
        return cls(
            exception_type=data.get("exception_type", ""),
            failing_test_name=data.get("failing_test_name"),
            top_stack_frames=data.get("top_stack_frames", []),
            module_path=data.get("module_path"),
        )


# =============================================================================
# Parsing Functions
# =============================================================================

# Number of stack frames to include in signature
TOP_N_FRAMES = 5


def extract_exception_type(error_output: str) -> str:
    """Extract exception type from error output.
    
    Looks for patterns like:
    - "ValueError: invalid literal"
    - "AssertionError"
    - "pytest.PytestUnraisableExceptionWarning"
    """
    # Common Python exception pattern
    patterns = [
        r"(\w+Error):\s",
        r"(\w+Exception):\s",
        r"(\w+Warning):\s",
        r"^E\s+(\w+):",  # pytest format
        r"raise (\w+)\(",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_output, re.MULTILINE)
        if match:
            return match.group(1)
    
    # Fallback: look for any capitalized word ending in Error/Exception
    match = re.search(r"\b([A-Z]\w*(?:Error|Exception))\b", error_output)
    if match:
        return match.group(1)
    
    return "UnknownError"


def extract_failing_test_name(error_output: str) -> Optional[str]:
    """Extract failing test name from pytest output.
    
    Looks for patterns like:
    - "FAILED tests/test_foo.py::test_bar"
    - "tests/test_foo.py::TestClass::test_method FAILED"
    """
    patterns = [
        r"FAILED\s+([\w/\\]+\.py::\w+(?:::\w+)?)",
        r"([\w/\\]+\.py::\w+(?:::\w+)?)\s+FAILED",
        r"ERROR\s+([\w/\\]+\.py::\w+(?:::\w+)?)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_output)
        if match:
            return match.group(1)
    
    return None


def extract_stack_frames(error_output: str, top_n: int = TOP_N_FRAMES) -> List[str]:
    """Extract top N stack frames from traceback.
    
    Looks for patterns like:
    - "  File \"path/to/file.py\", line 42, in function_name"
    
    Returns normalized frame strings for consistent hashing.
    """
    # Match Python traceback frames
    frame_pattern = r'File "([^"]+)", line (\d+), in (\w+)'
    matches = re.findall(frame_pattern, error_output)
    
    frames = []
    for filepath, line, func in matches[:top_n]:
        # Normalize: remove absolute path prefixes, keep relative
        normalized_path = normalize_filepath(filepath)
        frames.append(f"{normalized_path}:{func}")
    
    return frames


def extract_module_path(error_output: str) -> Optional[str]:
    """Extract the primary module path where error occurred.
    
    Returns the first file path from the traceback that looks like
    application code (not stdlib/site-packages).
    """
    frame_pattern = r'File "([^"]+)", line \d+'
    matches = re.findall(frame_pattern, error_output)
    
    for filepath in matches:
        normalized = normalize_filepath(filepath)
        # Skip stdlib and site-packages
        if "site-packages" in filepath.lower():
            continue
        if "python" in filepath.lower() and "lib" in filepath.lower():
            continue
        if normalized.startswith("app/") or normalized.startswith("tests/"):
            return normalized
    
    return matches[0] if matches else None


def normalize_filepath(filepath: str) -> str:
    """Normalize filepath for consistent comparison.
    
    - Convert backslashes to forward slashes
    - Remove common prefixes (D:\\Orb\\, /home/user/project/, etc.)
    - Keep relative path from project root
    """
    # Normalize slashes
    normalized = filepath.replace("\\", "/")
    
    # Remove common absolute prefixes
    prefixes_to_strip = [
        r"^[A-Za-z]:/[^/]+/",  # D:/Orb/, C:/Projects/
        r"^/home/[^/]+/[^/]+/",  # /home/user/project/
        r"^/app/",  # Docker /app/
    ]
    
    for prefix in prefixes_to_strip:
        normalized = re.sub(prefix, "", normalized)
    
    return normalized


# =============================================================================
# Main API
# =============================================================================

def compute_error_signature(
    error_output: str,
    exception_type: Optional[str] = None,
    failing_test: Optional[str] = None,
) -> ErrorSignature:
    """Compute ErrorSignature from error output.
    
    Args:
        error_output: Raw error/traceback output
        exception_type: Override extracted exception type
        failing_test: Override extracted test name
    
    Returns:
        ErrorSignature for strike tracking
    """
    return ErrorSignature(
        exception_type=exception_type or extract_exception_type(error_output),
        failing_test_name=failing_test or extract_failing_test_name(error_output),
        top_stack_frames=extract_stack_frames(error_output),
        module_path=extract_module_path(error_output),
    )


def signatures_match(sig1: ErrorSignature, sig2: ErrorSignature) -> bool:
    """Check if two signatures represent the same error."""
    return sig1.matches(sig2)


__all__ = [
    "ErrorSignature",
    "compute_error_signature",
    "signatures_match",
    "extract_exception_type",
    "extract_failing_test_name",
    "extract_stack_frames",
    "extract_module_path",
]

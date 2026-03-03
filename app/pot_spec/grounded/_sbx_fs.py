# FILE: app/pot_spec/grounded/_sbx_fs.py
"""
v9.0: Shared sandbox filesystem helpers for pot_spec/grounded/.

Previously _sbx_isfile / _sbx_isdir / _sbx_exists / _sbx_ls / _sbx_read
were referenced across 15+ files in pot_spec/grounded/ but never defined
or imported, causing NameError crashes in SpecGate.

v4.3 established that SpecGate uses the HOST filesystem (not sandbox),
so these are simple aliases for os / os.path functions.

Note: The orchestrator/ and overwatcher/ packages correctly import from
app.overwatcher.sandbox_client — this module is ONLY for pot_spec/grounded/.
"""

import os


def _sbx_isfile(path: str) -> bool:
    """Check if path is a file on host filesystem."""
    return os.path.isfile(path)


def _sbx_isdir(path: str) -> bool:
    """Check if path is a directory on host filesystem."""
    return os.path.isdir(path)


def _sbx_exists(path: str) -> bool:
    """Check if path exists on host filesystem."""
    return os.path.exists(path)


def _sbx_ls(path: str) -> list:
    """List directory contents on host filesystem."""
    try:
        return os.listdir(path)
    except (OSError, PermissionError):
        return []


def _sbx_read(path: str) -> str | None:
    """Read file text content from host filesystem."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except (OSError, PermissionError):
        return None


# Alias for consistency with sandbox_read_text naming
_sbx_read_text = _sbx_read

# FILE: app/pot_spec/grounded/_sbx_fs.py
"""
v10.0: Sandbox filesystem helpers for pot_spec/grounded/.

v9.0 established these as host aliases. v10.0 enforces the rule:
**the sandbox is the only source of truth for repo files.**

All 15+ files in pot_spec/grounded/ that import from this module
now read from the sandbox at http://192.168.250.2:8765 automatically.

No host fallbacks. If sandbox returns None, the file doesn't exist.
"""

from app.sandbox_fs import (
    sandbox_isfile,
    sandbox_isdir,
    sandbox_exists,
    sandbox_listdir,
    sandbox_read_text,
)


def _sbx_isfile(path: str) -> bool:
    """Check if path is a file in the sandbox."""
    return sandbox_isfile(path)


def _sbx_isdir(path: str) -> bool:
    """Check if path is a directory in the sandbox."""
    return sandbox_isdir(path)


def _sbx_exists(path: str) -> bool:
    """Check if path exists in the sandbox."""
    return sandbox_exists(path)


def _sbx_ls(path: str) -> list:
    """List directory contents from the sandbox.

    Returns a list of filenames (strings) for backward compatibility
    with existing call sites that expect os.listdir-style output.
    """
    entries = sandbox_listdir(path)
    return [e.get("name", "") for e in entries if e.get("name")]


def _sbx_read(path: str) -> str | None:
    """Read file text content from the sandbox. Returns None if unreadable."""
    return sandbox_read_text(path)


# Alias for consistency with sandbox_read_text naming
_sbx_read_text = _sbx_read

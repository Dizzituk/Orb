"""
Canonical path utilities.

Format: {root_kind}:{alias}/{relative_path}
Example: sandbox:d-drive/Orb/app/main.py
"""

import os
from typing import Optional, Tuple, Dict, Any

from app.rag.config.scan_roots import (
    get_root_alias,
    ROOT_ALIASES,
    is_scannable_path,
)


def canonicalize_path(abs_path: str) -> Tuple[str, str, str, str]:
    """
    Convert absolute to canonical path.
    
    Args:
        abs_path: Absolute filesystem path
        
    Returns:
        (canonical_path, alias, root_kind, zone)
        
    Raises:
        ValueError: If path not scannable or not under known root
    """
    abs_path = os.path.normpath(abs_path)
    
    # Check scannable first
    scannable, reason = is_scannable_path(abs_path)
    if not scannable:
        raise ValueError(f"Path not scannable: {abs_path} ({reason})")
    
    result = get_root_alias(abs_path)
    if result is None:
        raise ValueError(f"Path not under known root: {abs_path}")
    
    alias, root_kind, zone = result
    root_config = ROOT_ALIASES[alias]
    root_abs = os.path.normpath(root_config["abs_path"])
    
    # Calculate relative
    if abs_path.lower() == root_abs.lower():
        relative = ""
    else:
        relative = abs_path[len(root_abs):]
        if relative.startswith(os.sep):
            relative = relative[1:]
    
    # Forward slashes
    relative = relative.replace(os.sep, "/")
    
    if relative:
        canonical = f"{root_kind}:{alias}/{relative}"
    else:
        canonical = f"{root_kind}:{alias}"
    
    return canonical, alias, root_kind, zone


def parse_canonical_path(canonical: str) -> Dict[str, Any]:
    """
    Parse canonical path into components.
    
    Returns:
        Dict with root_kind, alias, relative_path, full_canonical
    """
    if ":" not in canonical:
        raise ValueError(f"Invalid canonical (no colon): {canonical}")
    
    root_kind, rest = canonical.split(":", 1)
    
    if "/" in rest:
        alias, relative_path = rest.split("/", 1)
    else:
        alias = rest
        relative_path = ""
    
    return {
        "root_kind": root_kind,
        "alias": alias,
        "relative_path": relative_path,
        "full_canonical": canonical,
    }


def canonical_to_absolute(canonical: str) -> str:
    """Convert canonical path back to absolute."""
    parsed = parse_canonical_path(canonical)
    alias = parsed["alias"]
    relative = parsed["relative_path"]
    
    if alias not in ROOT_ALIASES:
        raise ValueError(f"Unknown alias: {alias}")
    
    root_abs = ROOT_ALIASES[alias]["abs_path"]
    
    if relative:
        relative_native = relative.replace("/", os.sep)
        return os.path.join(root_abs, relative_native)
    return root_abs


def get_canonical_directory(canonical: str) -> str:
    """Get parent directory of canonical path."""
    parsed = parse_canonical_path(canonical)
    relative = parsed["relative_path"]
    
    if "/" in relative:
        parent_relative = "/".join(relative.split("/")[:-1])
    else:
        parent_relative = ""
    
    if parent_relative:
        return f"{parsed['root_kind']}:{parsed['alias']}/{parent_relative}"
    return f"{parsed['root_kind']}:{parsed['alias']}"


def is_under_canonical_prefix(canonical: str, prefix: str) -> bool:
    """Check if path is under prefix."""
    canonical = canonical.lower().rstrip("/")
    prefix = prefix.lower().rstrip("/")
    return canonical == prefix or canonical.startswith(prefix + "/")


def get_path_depth(canonical: str) -> int:
    """Get depth from root (0 = root itself)."""
    parsed = parse_canonical_path(canonical)
    relative = parsed["relative_path"]
    if not relative:
        return 0
    return relative.count("/") + 1

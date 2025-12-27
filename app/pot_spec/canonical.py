# FILE: app/pot_spec/canonical.py
"""Canonical JSON serialization for deterministic hashing.

RULES (must be universal across all stages):
1. Keys: sorted alphabetically at all nesting levels
2. Whitespace: none (compact separators)
3. Lists: stable ordering (application must ensure consistent order before hashing)
4. Encoding: UTF-8, no BOM
5. Numbers: no trailing zeros, no unnecessary precision

This module is the SINGLE SOURCE OF TRUTH for spec hashing.
All components must use compute_spec_hash() from here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Union


def _sort_nested(obj: Any) -> Any:
    """Recursively sort dict keys and list contents where applicable.
    
    For lists: sort if all elements are comparable (strings/numbers).
    For dicts: sort keys and recurse into values.
    """
    if isinstance(obj, dict):
        return {k: _sort_nested(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        sorted_list = [_sort_nested(item) for item in obj]
        # Only sort if all items are strings (common case for requirement lists)
        if sorted_list and all(isinstance(x, str) for x in sorted_list):
            return sorted(sorted_list)
        # For mixed types or dicts, preserve order but recurse
        return sorted_list
    else:
        return obj


def canonical_json_bytes(obj: Any, *, sort_lists: bool = True) -> bytes:
    """Serialize object to canonical JSON bytes for hashing.
    
    Args:
        obj: Object to serialize
        sort_lists: If True, sort string lists for stable ordering
    
    Returns:
        UTF-8 encoded bytes with:
        - Sorted keys at all levels
        - No whitespace (compact separators)
        - Sorted string lists (if sort_lists=True)
    """
    if sort_lists:
        obj = _sort_nested(obj)
    
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_string(obj: Any, *, sort_lists: bool = True) -> str:
    """Serialize object to canonical JSON string."""
    return canonical_json_bytes(obj, sort_lists=sort_lists).decode("utf-8")


def compute_spec_hash(spec_dict: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of spec in canonical form.
    
    This is THE canonical hash function. All stages must use this.
    
    Args:
        spec_dict: The spec dictionary to hash
    
    Returns:
        64-character hex string (SHA-256)
    """
    raw = canonical_json_bytes(spec_dict, sort_lists=True)
    return hashlib.sha256(raw).hexdigest()


def verify_hash(spec_dict: Dict[str, Any], expected_hash: str) -> bool:
    """Verify spec hash matches expected value.
    
    Args:
        spec_dict: The spec dictionary
        expected_hash: Expected SHA-256 hex string
    
    Returns:
        True if hashes match
    """
    actual = compute_spec_hash(spec_dict)
    return actual == expected_hash


__all__ = [
    "canonical_json_bytes",
    "canonical_json_string",
    "compute_spec_hash",
    "verify_hash",
]

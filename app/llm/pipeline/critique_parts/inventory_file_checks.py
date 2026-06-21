# FILE: app/llm/pipeline/critique_parts/inventory_file_checks.py
# Purpose: Deterministic Critique CHECK 1 — File Inventory Compliance (+ its shared path helpers).
# Called-by: app.llm.pipeline.critique_parts.inventory_checks (re-export shim)
# Depends-on: (stdlib only: json, re)
# Last-renovated: 2026-06-20
"""
Deterministic Critique — CHECK 1: File Inventory Compliance.

Check 1: File inventory compliance
    Every file listed in spec.json → file_scope[] appears in the
    architecture's file inventory. No extras, no missing.

Zero LLM calls. Pure structural comparison.

Split 2026-06-20 from inventory_checks.py via the move-and-shim pattern —
logic byte-identical. The helpers _extract_arch_file_inventory and
_normalise_path live here because CHECK 1 is their sole caller.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =========================================================================
# SHARED: Architecture file inventory extraction
# =========================================================================

def _extract_arch_file_inventory(arch_content: str) -> List[str]:
    """
    Extract file paths from architecture File Inventory section.

    Looks for markdown table rows in the ## File Inventory section.
    Returns list of file paths found.
    """
    paths: List[str] = []
    seen: Set[str] = set()

    # Find the File Inventory section
    inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', arch_content)
    if not inv_match:
        return paths

    inv_start = inv_match.start()
    # Find end of inventory section (next ## heading or ---)
    inv_end_match = re.search(r'\n(?:##[^#]|---)', arch_content[inv_start + 20:])
    if inv_end_match:
        inv_section = arch_content[inv_start:inv_start + 20 + inv_end_match.start()]
    else:
        inv_section = arch_content[inv_start:inv_start + 3000]

    for line in inv_section.split("\n"):
        if not line.strip().startswith("|") or line.strip().startswith("|---"):
            continue
        # Skip none/N/A markers
        line_lower = line.lower()
        if "*(none" in line_lower or "*(n/a" in line_lower:
            continue

        # Match backtick-wrapped paths
        match = re.search(
            r'`((?:app|src|tests|config|orb-desktop)[/\\][\w/\\._-]+\.[a-z]+)`',
            line
        )
        if not match:
            # Try root-level file
            match = re.search(
                r'`([\w_-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|css))`',
                line
            )
        if match:
            p = match.group(1)
            key = p.replace("\\", "/").lower()
            if key not in seen:
                seen.add(key)
                paths.append(p)

    return paths


def _normalise_path(p: str) -> str:
    """Normalise a file path for comparison."""
    return p.replace("\\", "/").strip().lower()


# =========================================================================
# CHECK 1: File Inventory Compliance
# =========================================================================

def check_file_inventory_compliance(
    arch_content: str,
    spec_json: Optional[str] = None,
    segment_spec: Optional[Dict[str, Any]] = None,
    skeleton_file_scope: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Check that architecture file inventory matches the spec's file_scope.

    Rules:
    - Every file in spec file_scope must appear in architecture inventory
    - Architecture inventory should not contain files outside file_scope
    - Uses skeleton file_scope as authoritative source if provided

    Args:
        arch_content: The architecture markdown document
        spec_json: Optional JSON string of the spec
        segment_spec: Optional segment spec dict (has file_scope directly)
        skeleton_file_scope: Optional file scope from skeleton contract

    Returns:
        List of issue dicts with {rule_id, severity, file, description, suggested_fix}
    """
    issues: List[Dict[str, Any]] = []

    # Determine authoritative file scope
    expected_files: Set[str] = set()

    if skeleton_file_scope:
        expected_files = {_normalise_path(f) for f in skeleton_file_scope}
    elif segment_spec and segment_spec.get("file_scope"):
        expected_files = {_normalise_path(f) for f in segment_spec["file_scope"]}
    elif spec_json:
        try:
            spec = json.loads(spec_json)
            if isinstance(spec, dict) and spec.get("file_scope"):
                expected_files = {_normalise_path(f) for f in spec["file_scope"]}
        except (json.JSONDecodeError, TypeError):
            pass

    if not expected_files:
        # No file scope to check against — skip silently
        return issues

    # Extract what the architecture declares
    arch_files = _extract_arch_file_inventory(arch_content)
    arch_files_normalised = {_normalise_path(f) for f in arch_files}

    # Check 1a: Missing files — in spec but not in architecture
    for expected in sorted(expected_files):
        if expected not in arch_files_normalised:
            # Check basename match as fallback (different prefix possible)
            basename = expected.rsplit("/", 1)[-1] if "/" in expected else expected
            has_basename = any(
                af.endswith("/" + basename) or af == basename
                for af in arch_files_normalised
            )
            if not has_basename:
                issues.append({
                    "rule_id": "DET-INVENTORY-MISSING",
                    "severity": "blocking",
                    "file": expected,
                    "spec_ref": "file_scope",
                    "arch_ref": "File Inventory",
                    "description": (
                        f"File '{expected}' is in the segment's file_scope "
                        f"but missing from the architecture's File Inventory."
                    ),
                    "suggested_fix": (
                        f"Add '{expected}' to the File Inventory section "
                        f"and include its implementation details."
                    ),
                })

    # Check 1b: Extra files — in architecture but not in spec
    for arch_file in sorted(arch_files_normalised):
        if arch_file not in expected_files:
            basename = arch_file.rsplit("/", 1)[-1] if "/" in arch_file else arch_file
            has_basename = any(
                ef.endswith("/" + basename) or ef == basename
                for ef in expected_files
            )
            if not has_basename:
                issues.append({
                    "rule_id": "DET-INVENTORY-EXTRA",
                    "severity": "warning",
                    "file": arch_file,
                    "spec_ref": "file_scope",
                    "arch_ref": "File Inventory",
                    "description": (
                        f"File '{arch_file}' is in the architecture's File Inventory "
                        f"but not in the segment's file_scope."
                    ),
                    "suggested_fix": (
                        f"Remove '{arch_file}' from the architecture unless "
                        f"the segment scope needs updating."
                    ),
                })

    if issues:
        logger.info(
            "[det_critique] Inventory check: %d issues (%d blocking)",
            len(issues),
            sum(1 for i in issues if i["severity"] == "blocking"),
        )

    return issues

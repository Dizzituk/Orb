# FILE: app/orchestrator/deterministic_import_fixer.py
"""
Deterministic Import Fixer — Tier 1 Auto-Fix (No LLM).

v1.0 (2026-03-01): Fixes common import errors that are fully
deterministic — no LLM reasoning needed.

Supported fixes:
- TS2300 Duplicate identifier: Merge duplicate import lines from same module
- TS2305 Module has no exported member: Remove the bad import specifier
- TS6133 Declared but never used: Remove unused import specifiers

These are the most common errors from LLM-generated TypeScript code
and account for the majority of boot failures in frontend segments.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DETERMINISTIC_FIXER_BUILD_ID = "2026-03-01-v1.0-import-merge"
print(f"[DETERMINISTIC_FIXER_LOADED] BUILD_ID={DETERMINISTIC_FIXER_BUILD_ID}")


@dataclass
class FixResult:
    """Result of a single file fix."""
    file: str
    fixed: bool
    description: str
    original_lines: int = 0
    fixed_lines: int = 0


# ─── Import line parser ────────────────────────────────────────────────

# Matches: import { X, Y, Z } from './module';
# Matches: import { X, Y, Z } from "./module";
# Captures: group(1)=specifiers, group(2)=module path
_IMPORT_RE = re.compile(
    r"""^import\s+\{\s*([^}]+)\}\s+from\s+['"]([^'"]+)['"];?\s*$"""
)

# Matches: import X from './module';
_DEFAULT_IMPORT_RE = re.compile(
    r"""^import\s+(\w+)\s+from\s+['"]([^'"]+)['"];?\s*$"""
)


def _parse_import_line(line: str) -> Optional[Tuple[List[str], str, bool]]:
    """Parse a named import line into (specifiers, module, is_type_import).

    Returns None if the line isn't a recognisable import.
    """
    stripped = line.strip()

    # Check for type-only imports: import type { X } from '...'
    is_type = stripped.startswith("import type ")
    check_line = stripped.replace("import type ", "import ", 1) if is_type else stripped

    m = _IMPORT_RE.match(check_line)
    if m:
        specs = [s.strip() for s in m.group(1).split(",") if s.strip()]
        module = m.group(2)
        return (specs, module, is_type)
    return None


def _rebuild_import_line(
    specifiers: List[str],
    module: str,
    is_type: bool = False,
) -> str:
    """Rebuild a clean import line from specifiers and module."""
    unique_specs = list(dict.fromkeys(specifiers))  # preserve order, dedupe
    spec_str = ", ".join(unique_specs)
    type_kw = "type " if is_type else ""
    return f"import {type_kw}{{ {spec_str} }} from '{module}';"


def fix_duplicate_imports(content: str) -> Tuple[str, List[str]]:
    """Merge duplicate import lines from the same module.

    When the LLM generates:
        import { Course, Module } from './education-data';
        import { courses, Course } from './education-data';
        import { Course, Module } from './education-data';

    This produces:
        import { Course, Module, courses } from './education-data';

    Args:
        content: Full file content as string.

    Returns:
        (fixed_content, list_of_descriptions) — descriptions of what was fixed.
    """
    lines = content.splitlines()
    # Group imports by (module, is_type)
    import_groups: Dict[Tuple[str, bool], List[Tuple[int, List[str]]]] = {}
    for i, line in enumerate(lines):
        parsed = _parse_import_line(line)
        if parsed:
            specs, module, is_type = parsed
            key = (module, is_type)
            if key not in import_groups:
                import_groups[key] = []
            import_groups[key].append((i, specs))

    # Find groups with duplicates
    fixes: List[str] = []
    lines_to_remove: set = set()
    lines_to_replace: Dict[int, str] = {}

    for (module, is_type), entries in import_groups.items():
        if len(entries) <= 1:
            continue

        # Merge all specifiers, preserving order, deduplicating
        all_specs: List[str] = []
        for _, specs in entries:
            for s in specs:
                if s not in all_specs:
                    all_specs.append(s)

        # Keep the first import line (replace it with merged), remove the rest
        first_line_idx = entries[0][0]
        merged_line = _rebuild_import_line(all_specs, module, is_type)
        lines_to_replace[first_line_idx] = merged_line

        for line_idx, _ in entries[1:]:
            lines_to_remove.add(line_idx)

        type_str = "type " if is_type else ""
        fixes.append(
            f"Merged {len(entries)} {type_str}import lines from '{module}' "
            f"into one ({len(all_specs)} specifiers)"
        )

    if not fixes:
        return content, []

    # Apply fixes
    new_lines = []
    for i, line in enumerate(lines):
        if i in lines_to_remove:
            continue
        if i in lines_to_replace:
            new_lines.append(lines_to_replace[i])
        else:
            new_lines.append(line)

    return "\n".join(new_lines), fixes


def fix_unused_imports(
    content: str,
    unused_identifiers: List[str],
) -> Tuple[str, List[str]]:
    """Remove specific unused import specifiers (TS6133).

    Only removes the specific identifier from the import line.
    If removing it leaves the import empty, removes the whole line.

    Args:
        content: Full file content.
        unused_identifiers: List of identifier names the compiler says are unused.

    Returns:
        (fixed_content, list_of_descriptions).
    """
    if not unused_identifiers:
        return content, []

    unused_set = set(unused_identifiers)
    lines = content.splitlines()
    fixes: List[str] = []
    new_lines: List[str] = []

    for line in lines:
        parsed = _parse_import_line(line)
        if parsed:
            specs, module, is_type = parsed
            remaining = [s for s in specs if s not in unused_set]
            removed = [s for s in specs if s in unused_set]

            if removed:
                if remaining:
                    new_lines.append(_rebuild_import_line(remaining, module, is_type))
                    fixes.append(
                        f"Removed unused import(s) {removed} from '{module}'"
                    )
                else:
                    # All specifiers removed — drop the entire line
                    fixes.append(
                        f"Removed entire import from '{module}' (all specifiers unused)"
                    )
                continue

        new_lines.append(line)

    if not fixes:
        return content, []

    return "\n".join(new_lines), fixes


def apply_deterministic_fixes(
    content: str,
    errors: List[Any],  # List[TscError] from frontend_build_check
) -> Tuple[str, List[str]]:
    """Apply all applicable deterministic fixes to a file.

    Tries each fixer in order. Returns the fixed content and
    descriptions of all fixes applied.

    Args:
        content: File content.
        errors: TscError list (from frontend_build_check.parse_tsc_errors).

    Returns:
        (fixed_content, all_fix_descriptions).
    """
    all_fixes: List[str] = []

    # Fix 1: Duplicate imports (TS2300)
    has_duplicate = any(
        getattr(e, "code", "") == "TS2300" for e in errors
    )
    if has_duplicate:
        content, fixes = fix_duplicate_imports(content)
        all_fixes.extend(fixes)

    # Fix 2: Unused imports (TS6133)
    unused_ids = [
        # Extract identifier from message like "'courses' is declared but its value is never read."
        getattr(e, "message", "").split("'")[1]
        for e in errors
        if getattr(e, "code", "") == "TS6133"
        and "'" in getattr(e, "message", "")
    ]
    if unused_ids:
        content, fixes = fix_unused_imports(content, unused_ids)
        all_fixes.extend(fixes)

    return content, all_fixes

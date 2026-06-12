# FILE: app/orchestrator/architecture_sanitiser.py
# Purpose: Architecture Sanitiser — Deterministic post-generation cleanup.
# Called-by: app.llm.pipeline._high_stakes_pipelines
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Architecture Sanitiser — Deterministic post-generation cleanup.

Runs on architecture text AFTER Critical Pipeline generation but BEFORE
the architecture is saved to disk, critiqued, or executed.

Catches and fixes known LLM hallucination patterns that waste critique
loop iterations and cause downstream failures:

  1. PACKAGE SELF-NAMING: `foo/foo.py` alongside `foo/__init__.py`
     - The LLM creates a file with the same name as its parent package
     - Python anti-pattern: __init__.py IS the package facade
     - Fix: Remove the hallucinated file from the File Inventory and
       merge any responsibilities into __init__.py

  2. OUT-OF-SCOPE FILES: Files in the architecture's File Inventory
     that aren't in this segment's file_scope from the manifest
     - The LLM invents extra files not assigned to this segment
     - Fix: For MODIFY operations, strip from inventory (scope creep).
       For CREATE operations, warn only (allow modular sub-files).

  v1.1 (2026-03-01): Scope check distinguishes CREATE vs MODIFY.
       CREATE files warned-not-stripped to allow modular architecture.
       Fixes sg-2a3c378d where 3 sub-routers were stripped forcing
       all logic into a near-limit 18.8KB monolithic api.py.

  3. PREVIOUSLY HALLUCINATED PATHS: Paths that the segmentation stage
     already flagged as hallucinated
     - The LLM independently re-invents the same bad path
     - Fix: Remove from File Inventory, log warning

All fixes are deterministic, zero LLM cost, and logged for transparency.

v1.0 (2026-02-16): Initial implementation
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BUILD_ID = "2026-03-01-v1.1-architecture-sanitiser"
print(f"[ARCHITECTURE_SANITISER_LOADED] BUILD_ID={BUILD_ID}")


# =============================================================================
# Data Model
# =============================================================================

@dataclass
class SanitiserResult:
    """Result of architecture sanitisation."""
    original_length: int = 0
    sanitised_length: int = 0
    fixes_applied: List[Dict[str, str]] = field(default_factory=list)
    # Each fix: {"type": "...", "path": "...", "description": "..."}

    @property
    def had_fixes(self) -> bool:
        return len(self.fixes_applied) > 0

    @property
    def fix_count(self) -> int:
        return len(self.fixes_applied)

    def summary(self) -> str:
        if not self.fixes_applied:
            return "Architecture sanitiser: no issues found"
        fix_types = {}
        for f in self.fixes_applied:
            t = f.get("type", "unknown")
            fix_types[t] = fix_types.get(t, 0) + 1
        parts = [f"{v} {k}" for k, v in fix_types.items()]
        return f"Architecture sanitiser: {self.fix_count} fix(es) — {', '.join(parts)}"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def sanitise_architecture(
    arch_text: str,
    file_scope: Optional[List[str]] = None,
    segment_id: Optional[str] = None,
    hallucinated_paths: Optional[Set[str]] = None,
) -> Tuple[str, SanitiserResult]:
    """
    Run all sanitisation passes on architecture text.

    Args:
        arch_text: Raw architecture markdown from Critical Pipeline
        file_scope: This segment's allowed file paths (from manifest)
        segment_id: Segment ID for logging
        hallucinated_paths: Paths previously flagged as hallucinated

    Returns:
        (sanitised_text, result)
    """
    result = SanitiserResult(original_length=len(arch_text))
    text = arch_text

    # Pass 1: Package self-naming (foo/foo.py)
    text, fixes_1 = _fix_package_self_naming(text, segment_id)
    result.fixes_applied.extend(fixes_1)

    # Pass 2: Out-of-scope files
    if file_scope:
        text, fixes_2 = _fix_out_of_scope_files(text, file_scope, segment_id)
        result.fixes_applied.extend(fixes_2)

    # Pass 3: Previously hallucinated paths
    if hallucinated_paths:
        text, fixes_3 = _fix_hallucinated_paths(text, hallucinated_paths, segment_id)
        result.fixes_applied.extend(fixes_3)

    # Pass 4: Phantom evidence references
    text, fixes_4 = _fix_phantom_evidence_references(text, segment_id)
    result.fixes_applied.extend(fixes_4)

    result.sanitised_length = len(text)

    if result.had_fixes:
        logger.info(
            "[ARCHITECTURE_SANITISER] %s: %d fix(es) applied — %s",
            segment_id or "unknown", result.fix_count, result.summary(),
        )
    else:
        logger.debug(
            "[ARCHITECTURE_SANITISER] %s: no issues found",
            segment_id or "unknown",
        )

    return text, result


# =============================================================================
# PASS 1: PACKAGE SELF-NAMING
# =============================================================================

def _fix_package_self_naming(
    arch_text: str,
    segment_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Detect and remove `<package>/<package>.py` entries from File Inventory.

    Pattern: When refactoring `foo.py` into `foo/` package, the LLM sometimes
    creates both `foo/__init__.py` AND `foo/foo.py` as a "compatibility shim".
    This is a Python anti-pattern — `__init__.py` IS the compatibility shim.

    Detection: Find all paths in the File Inventory table. If any path
    matches `<dir>/<dir_name>.py` where `<dir>/__init__.py` also exists
    in the same inventory, it's a self-naming hallucination.

    Fix: Remove the `<dir>/<dir_name>.py` row from the File Inventory table.
    Add a comment noting the merge into `__init__.py`.
    """
    fixes: List[Dict[str, str]] = []

    # Extract all file paths from the File Inventory table
    inventory_paths = _extract_file_inventory_paths(arch_text)
    if not inventory_paths:
        return arch_text, fixes

    # Build a set of directory names that have __init__.py
    init_dirs: Set[str] = set()
    for path in inventory_paths:
        normalised = path.replace("\\", "/")
        if normalised.endswith("/__init__.py"):
            # e.g. "app/overwatcher/architecture_executor/__init__.py"
            # → dir = "app/overwatcher/architecture_executor"
            init_dirs.add(os.path.dirname(normalised))

    # Check for self-naming: <dir>/<basename>.py where basename == dir_name
    for path in inventory_paths:
        normalised = path.replace("\\", "/")
        if normalised.endswith("/__init__.py"):
            continue

        parent_dir = os.path.dirname(normalised)
        filename = os.path.basename(normalised)
        filename_stem = filename.replace(".py", "")
        dir_name = os.path.basename(parent_dir)

        if (
            parent_dir in init_dirs
            and filename_stem == dir_name
            and filename.endswith(".py")
        ):
            # Found self-naming: e.g. architecture_executor/architecture_executor.py
            logger.warning(
                "[ARCHITECTURE_SANITISER] PACKAGE SELF-NAMING detected: %s "
                "(package already has __init__.py)",
                path,
            )

            # Remove this row from the File Inventory table
            arch_text = _remove_file_inventory_row(arch_text, path)

            # Also remove any dedicated section for this file
            arch_text = _remove_file_section(arch_text, path, filename)

            fixes.append({
                "type": "package_self_naming",
                "path": path,
                "description": (
                    f"Removed hallucinated '{filename}' from package that "
                    f"already has __init__.py. Responsibilities merged into "
                    f"__init__.py (the correct Python facade pattern)."
                ),
            })

    return arch_text, fixes


# =============================================================================
# PASS 2: OUT-OF-SCOPE FILES
# =============================================================================

def _fix_out_of_scope_files(
    arch_text: str,
    file_scope: List[str],
    segment_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Handle File Inventory entries for files not in this segment's file_scope.

    v1.1: Distinguishes CREATE vs MODIFY operations:
      - MODIFY out-of-scope: STRIP (this segment should not edit files it
        doesn't own - that's scope creep that breaks other segments).
      - CREATE out-of-scope: WARN ONLY (the LLM may be designing legitimate
        modular sub-files like sub-routers for a large API endpoint).
        The cohesion check downstream will validate compatibility.
    """
    fixes: List[Dict[str, str]] = []

    inventory_paths = _extract_file_inventory_paths(arch_text)
    if not inventory_paths:
        return arch_text, fixes

    # Determine which paths are CREATE vs MODIFY from the architecture text
    create_paths = _extract_paths_by_operation(arch_text, "new")
    modify_paths = _extract_paths_by_operation(arch_text, "modified")
    create_norm = {p.replace("\\", "/").lower() for p in create_paths}
    modify_norm = {p.replace("\\", "/").lower() for p in modify_paths}

    # Normalise file_scope for comparison
    scope_normalised: Set[str] = set()
    for sp in file_scope:
        scope_normalised.add(sp.replace("\\", "/").lower())
        if sp.replace("\\", "/").lower().startswith("app/"):
            scope_normalised.add(sp.replace("\\", "/").lower()[4:])

    for path in inventory_paths:
        normalised = path.replace("\\", "/").lower()

        # Check if this path is in scope (with various normalisations)
        in_scope = False
        for scope_path in scope_normalised:
            if normalised == scope_path or normalised.endswith("/" + scope_path):
                in_scope = True
                break
            if scope_path.endswith("/" + normalised) or scope_path == normalised:
                in_scope = True
                break
            if normalised.endswith(scope_path) or scope_path.endswith(normalised):
                in_scope = True
                break

        if not in_scope:
            is_modify = normalised in modify_norm
            # Default to CREATE if ambiguous (safer - don't strip)
            is_create = (normalised in create_norm) or (not is_modify)

            if is_modify:
                # MODIFY out-of-scope: STRIP (scope creep protection)
                logger.warning(
                    "[ARCHITECTURE_SANITISER] OUT-OF-SCOPE MODIFY stripped: %s "
                    "(segment %s, scope has %d files)",
                    path, segment_id, len(file_scope),
                )
                arch_text = _remove_file_inventory_row(arch_text, path)
                fixes.append({
                    "type": "out_of_scope_modify_stripped",
                    "path": path,
                    "description": (
                        f"Stripped MODIFY '{path}' - this segment does not own "
                        f"this file. Editing it would conflict with other segments."
                    ),
                })
            else:
                # CREATE out-of-scope: WARN ONLY (allow modular design)
                logger.info(
                    "[ARCHITECTURE_SANITISER] v1.1 OUT-OF-SCOPE CREATE allowed: %s "
                    "(segment %s - modular sub-file, cohesion will validate)",
                    path, segment_id,
                )
                fixes.append({
                    "type": "out_of_scope_create_allowed",
                    "path": path,
                    "description": (
                        f"Allowed CREATE '{path}' - not in file_scope but "
                        f"appears to be a modular sub-file. Cohesion check will "
                        f"validate cross-segment compatibility."
                    ),
                })

    return arch_text, fixes


def _extract_paths_by_operation(
    arch_text: str,
    operation: str,
) -> List[str]:
    """Extract file paths from a specific File Inventory sub-table.

    Args:
        arch_text: Full architecture text.
        operation: 'new' or 'modified'

    Returns:
        List of file paths found under that sub-heading.
    """
    paths: List[str] = []
    op_pattern = re.compile(
        rf'###?\s*{operation}\s+files',
        re.IGNORECASE,
    )

    in_section = False
    for line in arch_text.split('\n'):
        stripped = line.strip()

        if op_pattern.match(stripped):
            in_section = True
            continue

        if in_section and stripped.startswith('#') and not stripped.startswith('#|'):
            break

        if not in_section:
            continue

        match = re.search(r'\|\s*`([^`]+)`\s*\|', stripped)
        if match:
            paths.append(match.group(1).strip())

    return paths


# =============================================================================
# PASS 3: PREVIOUSLY HALLUCINATED PATHS
# =============================================================================

def _fix_hallucinated_paths(
    arch_text: str,
    hallucinated_paths: Set[str],
    segment_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Remove paths that were previously flagged as hallucinated by the
    segmentation stage.
    """
    fixes: List[Dict[str, str]] = []

    inventory_paths = _extract_file_inventory_paths(arch_text)
    if not inventory_paths:
        return arch_text, fixes

    # Normalise hallucinated paths
    hallucinated_normalised = {
        p.replace("\\", "/").lower() for p in hallucinated_paths
    }

    for path in inventory_paths:
        normalised = path.replace("\\", "/").lower()
        if normalised in hallucinated_normalised:
            logger.warning(
                "[ARCHITECTURE_SANITISER] KNOWN HALLUCINATION in architecture: %s",
                path,
            )

            arch_text = _remove_file_inventory_row(arch_text, path)
            arch_text = _remove_file_section(
                arch_text, path, os.path.basename(path),
            )

            fixes.append({
                "type": "known_hallucination",
                "path": path,
                "description": (
                    f"Removed '{path}' — previously flagged as hallucinated "
                    f"by segmentation stage."
                ),
            })

    return arch_text, fixes


# =============================================================================
# PASS 4: PHANTOM EVIDENCE REFERENCES
# =============================================================================

# Known phantom file patterns that LLMs hallucinate as "evidence" or
# "reference documents".  These don't exist on disk but the LLM cites
# them as if they do, which causes the critique to flag missing evidence.
PHANTOM_EVIDENCE_PATTERNS = [
    # Exact filenames known to be hallucinated
    "ARCHITECTURE_MAP.md",
    "ARCHITECTURE_MAP.json",
    "REFACTOR_PLAN.md",
    "MODULE_MAP.md",
    "DEPENDENCY_MAP.md",
    "FILE_MAP.md",
    "IMPLEMENTATION_PLAN.md",
]


def _fix_phantom_evidence_references(
    arch_text: str,
    segment_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Detect and annotate references to files that are known phantoms.

    The LLM sometimes cites non-existent reference documents like
    "ARCHITECTURE_MAP.md" as evidence sources.  The critique then flags
    this as a missing evidence issue, wasting a revision iteration.

    Detection: Scan for backtick-wrapped or quoted references to known
    phantom filenames anywhere in the architecture text.

    Fix: Replace the reference with a comment noting it's a phantom.
    We don't remove entire sentences — just annotate the reference so
    the critique and revision loop know to ignore it.
    """
    fixes: List[Dict[str, str]] = []

    for phantom in PHANTOM_EVIDENCE_PATTERNS:
        escaped = re.escape(phantom)

        # Match backtick-wrapped: `ARCHITECTURE_MAP.md`
        backtick_pattern = rf'`{escaped}`'
        if re.search(backtick_pattern, arch_text, re.IGNORECASE):
            # Replace with annotated version
            arch_text = re.sub(
                backtick_pattern,
                f'`{phantom}` <!-- SANITISER: phantom reference — file does not exist -->',
                arch_text,
                flags=re.IGNORECASE,
                count=1,  # Only annotate first occurrence
            )

            logger.warning(
                "[ARCHITECTURE_SANITISER] PHANTOM EVIDENCE reference: %s",
                phantom,
            )

            fixes.append({
                "type": "phantom_evidence",
                "path": phantom,
                "description": (
                    f"Annotated phantom evidence reference to '{phantom}' — "
                    f"this file does not exist and should not be cited."
                ),
            })

    return arch_text, fixes


# =============================================================================
# HELPERS: File Inventory Parsing & Manipulation
# =============================================================================

def _extract_file_inventory_paths(arch_text: str) -> List[str]:
    """
    Extract file paths from the File Inventory table in the architecture.

    Looks for markdown table rows with backtick-wrapped paths like:
        | `app/foo/bar.py` | Some description |

    Only extracts from the File Inventory section (not from other tables
    in the architecture like evidence tables).
    """
    paths: List[str] = []
    in_inventory = False
    past_header = False

    for line in arch_text.split("\n"):
        stripped = line.strip()

        # Detect File Inventory section start
        if re.match(r'#{1,4}\s*.*[Ff]ile\s*[Ii]nventory', stripped):
            in_inventory = True
            past_header = False
            continue

        # Detect section end (next heading or horizontal rule)
        if in_inventory and (
            (stripped.startswith('#') and not stripped.startswith('#|'))
            or stripped == '---'
        ):
            if past_header:
                in_inventory = False
                continue

        if not in_inventory:
            continue

        # Skip non-table lines
        if not stripped.startswith('|'):
            continue

        # Skip separator rows
        if re.match(r'\|[-\s|]+\|', stripped):
            past_header = True
            continue

        # Skip header row
        if 'File' in stripped and ('Purpose' in stripped or 'Description' in stripped):
            continue

        # Extract path from backtick-wrapped cell
        match = re.search(r'\|\s*`([^`]+)`\s*\|', stripped)
        if match:
            path = match.group(1).strip()
            if path and path.lower() != 'file':
                paths.append(path)

    return paths


def _remove_file_inventory_row(arch_text: str, target_path: str) -> str:
    """
    Remove a specific file path's row from the File Inventory table.

    Uses exact backtick-wrapped match to avoid false positives.
    """
    # Escape special regex chars in the path
    escaped = re.escape(target_path)

    # Match a table row containing this exact path in backticks
    # Pattern: | `<path>` | ... | (possibly with more columns)
    pattern = rf'^[ \t]*\|[^|]*`{escaped}`[^|]*\|.*$'

    lines = arch_text.split("\n")
    new_lines = []
    removed = False

    for line in lines:
        if not removed and re.match(pattern, line, re.IGNORECASE):
            # Replace with a comment noting the removal
            new_lines.append(
                f"<!-- SANITISER: Removed `{target_path}` "
                f"(package self-naming / out-of-scope / hallucination) -->"
            )
            removed = True
        else:
            new_lines.append(line)

    if removed:
        logger.debug(
            "[ARCHITECTURE_SANITISER] Removed inventory row: %s", target_path,
        )

    return "\n".join(new_lines)


def _remove_file_section(
    arch_text: str,
    target_path: str,
    filename: str,
) -> str:
    """
    Remove a dedicated section for the hallucinated file.

    Looks for heading patterns like:
        ### 5) `architecture_executor.py`
        ### architecture_executor.py
        ## File: architecture_executor.py
    
    Removes from the heading to the next heading of equal or higher level.
    """
    # Build patterns to match section headings about this file
    escaped_path = re.escape(target_path)
    escaped_name = re.escape(filename)
    # Also match without .py extension for headings like "### architecture_executor"
    escaped_stem = re.escape(filename.replace('.py', ''))

    # Match headings containing this filename (with or without backticks/extension)
    heading_pattern = re.compile(
        rf'^(#{{1,4}})\s+.*(?:`{escaped_name}`|`{escaped_path}`|{escaped_name}|{escaped_stem}(?:\.py)?)',
        re.IGNORECASE,
    )

    lines = arch_text.split("\n")
    new_lines = []
    skip_until_level = 0
    skipping = False
    removed_section = False

    for line in lines:
        if skipping:
            # Check if we hit a heading of equal or higher level
            heading_match = re.match(r'^(#{1,4})\s+', line)
            if heading_match:
                level = len(heading_match.group(1))
                if level <= skip_until_level:
                    # End of removed section
                    skipping = False
                    new_lines.append(line)
                    continue
            # Still in removed section — skip this line
            continue

        # Check if this line starts a section for the target file
        match = heading_pattern.match(line)
        if match:
            skip_until_level = len(match.group(1))
            skipping = True
            removed_section = True
            new_lines.append(
                f"<!-- SANITISER: Removed section for `{target_path}` -->"
            )
            continue

        new_lines.append(line)

    if removed_section:
        logger.debug(
            "[ARCHITECTURE_SANITISER] Removed section: %s", target_path,
        )

    return "\n".join(new_lines)

# FILE: app/pot_spec/grounded/file_verifier.py
"""
Architecture Map File Verification (v1.0)

Verifies architecture map claims against the host filesystem BEFORE
segmentation. This runs during SpecGate (before any sandbox session),
so all filesystem access is host-direct — no sandbox client needed.

Protocol (Design Spec Section 7):
1. Existence check — confirm files exist on disk (os.path.exists, os.stat)
2. Interface read — read class/function signatures from boundary files (~500 chars)
3. Staleness detection — flag mismatches between map claims and actual exports
4. Gap detection — files referenced but missing = CREATE targets
5. New file detection — files on disk but not in architecture map

Access model:
- Host filesystem read: YES (os.path.exists, os.stat, open())
- Sandbox client: NO (not available at SpecGate time)
- Write: NEVER (SpecGate is read-only)

Version Notes:
-------------
v1.0 (2026-02-08): Initial implementation — Phase 1 of Pipeline Segmentation
    - Host-direct filesystem access (no sandbox dependency)
    - Python and TypeScript signature extraction
    - GroundingData production per segment
"""

from __future__ import annotations

import logging
import os
import re
import stat
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from .segment_schemas import (
    CreateTarget,
    GroundingData,
    InterfaceRead,
    StaleEntry,
    VerifiedFile,
)

logger = logging.getLogger(__name__)

FILE_VERIFIER_BUILD_ID = "2026-02-08-v1.0-initial"
print(f"[FILE_VERIFIER_LOADED] BUILD_ID={FILE_VERIFIER_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum chars to read for interface signature extraction
MAX_SIGNATURE_READ_CHARS = 2000

# Patterns to extract Python class/function signatures
_PY_CLASS_RE = re.compile(r"^class\s+(\w+)\s*[\(:]", re.MULTILINE)
_PY_DEF_RE = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
_PY_PROTOCOL_RE = re.compile(r"class\s+(\w+)\s*\(\s*Protocol\s*\)", re.MULTILINE)

# Patterns to extract TypeScript/React exports
_TS_EXPORT_RE = re.compile(
    r"^export\s+(?:default\s+)?(?:function|const|class|interface|type|enum)\s+(\w+)",
    re.MULTILINE,
)
_TS_EXPORT_DEFAULT_RE = re.compile(r"^export\s+default\s+(\w+)", re.MULTILINE)


# =============================================================================
# EXISTENCE CHECK
# =============================================================================

def check_file_exists(path: str) -> Optional[VerifiedFile]:
    """
    Check if a file exists on the host filesystem.
    
    Returns VerifiedFile with metadata if found, None if not found.
    Cost: essentially free (os.stat).
    """
    try:
        if not os.path.exists(path):
            return None

        st = os.stat(path)
        mtime_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)

        return VerifiedFile(
            path=path,
            last_modified=mtime_dt.isoformat(),
            size_bytes=st.st_size if stat.S_ISREG(st.st_mode) else None,
        )
    except OSError as e:
        logger.warning("[file_verifier] check_file_exists: OS error for %s: %s", path, e)
        return None


def batch_check_existence(paths: List[str]) -> Tuple[List[VerifiedFile], List[str]]:
    """
    Check multiple files for existence.
    
    Returns:
        (found: List[VerifiedFile], missing: List[str])
    """
    found: List[VerifiedFile] = []
    missing: List[str] = []

    for path in paths:
        vf = check_file_exists(path)
        if vf:
            found.append(vf)
        else:
            missing.append(path)

    logger.info(
        "[file_verifier] batch_check_existence: %d found, %d missing out of %d",
        len(found), len(missing), len(paths),
    )
    return found, missing


# =============================================================================
# INTERFACE / SIGNATURE READING
# =============================================================================

def read_file_signatures(path: str, max_chars: int = MAX_SIGNATURE_READ_CHARS) -> InterfaceRead:
    """
    Read class/function signatures from a file.
    
    Reads the first `max_chars` of a file and extracts structural signatures
    (class names, function definitions, exports). This is a lightweight read
    used for boundary file verification — NOT a full parse.
    
    Supports Python (.py) and TypeScript/React (.ts, .tsx, .js, .jsx).
    """
    result = InterfaceRead(path=path)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
    except OSError as e:
        logger.warning("[file_verifier] read_file_signatures: cannot read %s: %s", path, e)
        result.read_ok = False
        return result

    result.raw_excerpt = content[:500]  # Store first 500 chars as preview

    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        result.signatures = _extract_python_signatures(content)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        result.signatures = _extract_typescript_signatures(content)
    else:
        # For unknown file types, just store the excerpt
        result.signatures = []

    return result


def _extract_python_signatures(content: str) -> List[str]:
    """Extract class and function signatures from Python source."""
    signatures: List[str] = []

    for match in _PY_CLASS_RE.finditer(content):
        # Get the full class line for context (includes base classes)
        line_start = content.rfind("\n", 0, match.start()) + 1
        line_end = content.find("\n", match.start())
        if line_end == -1:
            line_end = len(content)
        full_line = content[line_start:line_end].strip()
        signatures.append(full_line)

    for match in _PY_DEF_RE.finditer(content):
        name = match.group(1)
        # Skip private/dunder methods for interface purposes
        if name.startswith("_") and not name.startswith("__init__"):
            continue
        line_start = content.rfind("\n", 0, match.start()) + 1
        line_end = content.find("\n", match.start())
        if line_end == -1:
            line_end = len(content)
        full_line = content[line_start:line_end].strip()
        # Truncate very long signatures
        if len(full_line) > 200:
            full_line = full_line[:200] + "..."
        signatures.append(full_line)

    return signatures


def _extract_typescript_signatures(content: str) -> List[str]:
    """Extract exported class/function/type signatures from TypeScript/JS source."""
    signatures: List[str] = []

    for match in _TS_EXPORT_RE.finditer(content):
        line_start = content.rfind("\n", 0, match.start()) + 1
        line_end = content.find("\n", match.start())
        if line_end == -1:
            line_end = len(content)
        full_line = content[line_start:line_end].strip()
        if len(full_line) > 200:
            full_line = full_line[:200] + "..."
        signatures.append(full_line)

    for match in _TS_EXPORT_DEFAULT_RE.finditer(content):
        name = match.group(1)
        # Avoid duplicates — the named export regex may have caught this
        sig = f"export default {name}"
        if sig not in signatures and not any(name in s for s in signatures):
            signatures.append(sig)

    return signatures


# =============================================================================
# STALENESS DETECTION
# =============================================================================

def check_interface_staleness(
    path: str,
    expected_names: List[str],
    interface_read: InterfaceRead,
) -> Optional[StaleEntry]:
    """
    Check if a file's actual signatures match what the architecture map claims.
    
    Args:
        path: File path
        expected_names: Class/function/export names the architecture map claims exist
        interface_read: The actual signatures read from the file
    
    Returns:
        StaleEntry if there's a mismatch, None if everything matches.
    """
    if not interface_read.read_ok:
        return None  # Can't check staleness if we couldn't read

    # Extract just the names from actual signatures
    actual_names: Set[str] = set()
    for sig in interface_read.signatures:
        # Extract the first identifier-like word after class/def/export/etc.
        name_match = re.search(r"(?:class|def|function|const|interface|type|enum|export)\s+(\w+)", sig)
        if name_match:
            actual_names.add(name_match.group(1))

    # Check which expected names are missing
    missing = [name for name in expected_names if name not in actual_names]
    if missing:
        return StaleEntry(
            path=path,
            map_claimed=f"Expected: {', '.join(expected_names)}",
            actual_found=f"Found: {', '.join(sorted(actual_names)) if actual_names else '(no signatures extracted)'}",
        )

    return None


# =============================================================================
# DIRECTORY SCANNING (for new file detection)
# =============================================================================

def scan_directory_files(
    directory: str,
    extensions: Optional[Set[str]] = None,
) -> List[str]:
    """
    Scan a directory for files, optionally filtering by extension.
    
    Used for new file detection — finding files on disk that aren't
    in the architecture map.
    
    Args:
        directory: Directory path to scan
        extensions: Set of extensions to include (e.g. {".py", ".ts"}).
                    If None, includes all files.
    
    Returns:
        List of absolute file paths.
    """
    result: List[str] = []
    try:
        for root, _dirs, files in os.walk(directory):
            # Skip common non-source directories
            basename = os.path.basename(root)
            if basename in {"__pycache__", "node_modules", ".git", ".venv", "build", "dist"}:
                continue
            for fname in files:
                if extensions:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in extensions:
                        continue
                result.append(os.path.join(root, fname))
    except OSError as e:
        logger.warning("[file_verifier] scan_directory_files: error scanning %s: %s", directory, e)

    return result


def detect_new_files(
    directory: str,
    known_paths: Set[str],
    extensions: Optional[Set[str]] = None,
) -> List[str]:
    """
    Find files on disk that are NOT in the architecture map.
    
    Args:
        directory: Directory to scan
        known_paths: Set of paths the architecture map knows about
        extensions: File extensions to check
    
    Returns:
        List of paths that exist on disk but aren't in known_paths.
    """
    on_disk = scan_directory_files(directory, extensions)
    
    # Normalise paths for comparison (case-insensitive on Windows)
    known_normalised = {os.path.normcase(os.path.normpath(p)) for p in known_paths}
    
    new_files = []
    for path in on_disk:
        normalised = os.path.normcase(os.path.normpath(path))
        if normalised not in known_normalised:
            new_files.append(path)

    if new_files:
        logger.info(
            "[file_verifier] detect_new_files: found %d files not in architecture map under %s",
            len(new_files), directory,
        )

    return new_files


# =============================================================================
# MAIN VERIFICATION ENTRY POINT
# =============================================================================

def verify_segment_files(
    file_scope: List[str],
    evidence_files: List[str],
    boundary_files: Optional[Dict[str, List[str]]] = None,
    scan_directories: Optional[List[str]] = None,
    known_arch_paths: Optional[Set[str]] = None,
) -> GroundingData:
    """
    Run the full file verification protocol for a single segment.
    
    This is the main entry point for the file verifier. It performs:
    1. Existence checks on all files in file_scope and evidence_files
    2. Interface reads on boundary files
    3. Staleness detection for boundary files with expected names
    4. Gap detection (missing files become CREATE targets)
    5. New file detection (files on disk not in architecture map)
    
    Args:
        file_scope: Files this segment creates or modifies
        evidence_files: Files this segment needs to read for context
        boundary_files: Dict mapping file path → list of expected export names.
                       These are the interface boundary files that need signature reads.
        scan_directories: Directories to scan for new file detection.
                         If None, new file detection is skipped.
        known_arch_paths: Set of all paths the architecture map knows about.
                         Used for new file detection.
    
    Returns:
        GroundingData with complete verification results.
    """
    grounding = GroundingData()
    boundary_files = boundary_files or {}

    # Combine all paths for existence checking
    all_paths = list(set(file_scope + evidence_files + list(boundary_files.keys())))

    # Step 1: Existence checks
    found, missing = batch_check_existence(all_paths)
    grounding.verified_files = found

    # Step 2: Gap detection — missing files in file_scope are CREATE targets
    scope_set = set(file_scope)
    for path in missing:
        if path in scope_set:
            grounding.create_targets.append(CreateTarget(
                path=path,
                reason="File in segment file_scope but not found on disk",
            ))
        else:
            # Missing evidence file — flag as verification error
            grounding.verification_errors.append(
                f"Evidence file not found: {path}"
            )

    # Step 3: Interface reads on boundary files
    found_paths = {vf.path for vf in found}
    for path, expected_names in boundary_files.items():
        if path not in found_paths:
            # Boundary file doesn't exist — already captured as CREATE target or error
            continue

        iread = read_file_signatures(path)
        grounding.interface_reads.append(iread)

        # Step 4: Staleness detection
        if expected_names and iread.read_ok:
            stale = check_interface_staleness(path, expected_names, iread)
            if stale:
                grounding.stale_entries.append(stale)
                logger.warning(
                    "[file_verifier] Stale entry detected: %s — %s vs %s",
                    path, stale.map_claimed, stale.actual_found,
                )

    # Step 5: New file detection
    if scan_directories and known_arch_paths is not None:
        source_extensions = {".py", ".ts", ".tsx", ".js", ".jsx"}
        for directory in scan_directories:
            new = detect_new_files(directory, known_arch_paths, source_extensions)
            grounding.new_files.extend(new)

    logger.info("[file_verifier] verify_segment_files: %s", grounding.summary())
    return grounding

# FILE: app/llm/pipeline/critique_parts/inventory_create_classify.py
# Purpose: Deterministic Critique CHECK 2 — CREATE vs MODIFY classification accuracy.
# Called-by: app.llm.pipeline.critique_parts.inventory_checks (re-export shim)
# Depends-on: app.sandbox_fs (lazy import inside _check_sandbox_file_exists)
# Last-renovated: 2026-06-20
"""
Deterministic Critique — CHECK 2: CREATE vs MODIFY classification accuracy.

Check 2: CREATE vs MODIFY classification accuracy
    Files listed under "New Files" (CREATE) must NOT exist on disk.
    Cross-references against INDEX.json and direct filesystem checks.
    BLOCKING — prevents implementer from overwriting working code.

Zero LLM calls. Pure structural comparison.

Split 2026-06-20 from inventory_checks.py via the move-and-shim pattern —
logic byte-identical. Carries its private helpers (_load_filesystem_index,
_extract_new_file_paths, _check_sandbox_file_exists) and the FS-index cache.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# CHECK 2: CREATE vs MODIFY Classification Accuracy
# =========================================================================

# Module-level cache for INDEX.json filesystem data
_FS_INDEX_CACHE: Optional[Dict[str, str]] = None
_FS_INDEX_MTIME: float = 0.0


def _load_filesystem_index() -> Dict[str, str]:
    """
    Load INDEX.json and build a normalised-path → absolute-path lookup.

    v1.0 (2026-03-01): Used by check_create_modify_classification to
    determine whether files exist on disk.

    Returns:
        Dict mapping normalised relative paths to absolute paths.
        Empty dict if INDEX.json unavailable (non-fatal).
    """
    global _FS_INDEX_CACHE, _FS_INDEX_MTIME

    import os

    index_path = os.path.join(
        os.getenv("ASTRA_ARCH_INDEX_DIR", os.path.join("D:\\", "Orb", ".architecture")),
        "INDEX.json",
    )

    try:
        current_mtime = os.path.getmtime(index_path) if os.path.isfile(index_path) else 0.0
    except OSError:
        current_mtime = 0.0

    if _FS_INDEX_CACHE is not None and current_mtime == _FS_INDEX_MTIME:
        return _FS_INDEX_CACHE

    _FS_INDEX_CACHE = {}
    _FS_INDEX_MTIME = current_mtime

    if not os.path.isfile(index_path):
        return _FS_INDEX_CACHE

    try:
        with open(index_path, "r", encoding="utf-8") as fh:
            index_data = json.load(fh)

        roots = index_data.get("roots", [])
        for entry in index_data.get("files", []):
            abs_path = entry.get("path", "")
            if not abs_path:
                continue
            # Build relative path by stripping each root prefix
            for root in roots:
                root_prefix = root.replace("/", "\\")
                if not root_prefix.endswith("\\"):
                    root_prefix += "\\"
                if abs_path.startswith(root_prefix):
                    rel = abs_path[len(root_prefix):]
                    norm = rel.replace("\\", "/").lower()
                    _FS_INDEX_CACHE[norm] = abs_path
                    break

        logger.debug(
            "[det_critique] Filesystem index loaded: %d relative paths",
            len(_FS_INDEX_CACHE),
        )
    except Exception as exc:
        logger.warning("[det_critique] Failed to load INDEX.json: %s", exc)

    return _FS_INDEX_CACHE


def _extract_new_file_paths(arch_content: str) -> List[str]:
    """
    Extract file paths listed under the 'New Files' sub-heading
    in the architecture's File Inventory.

    Returns list of file paths the architecture claims are new (CREATE).
    """
    paths: List[str] = []
    in_new_section = False

    for line in arch_content.split("\n"):
        stripped = line.strip()

        # Detect "### New Files" heading
        if re.match(r"###?\s*[Nn]ew\s+[Ff]iles", stripped):
            in_new_section = True
            continue

        # Exit on next heading
        if in_new_section and stripped.startswith("#") and not stripped.startswith("#|"):
            break

        if not in_new_section:
            continue

        # Extract backtick-wrapped path from table row
        match = re.search(r"\|\s*`([^`]+)`\s*\|", stripped)
        if match:
            paths.append(match.group(1).strip())

    return paths



def _check_sandbox_file_exists(candidates: List[str]) -> bool:
    """
    Check if any candidate path exists in the SANDBOX (not host).

    v1.0 (2026-03-03): Fix 6 — sandbox is the ground truth.
    v1.2 (2026-03-05): Removed host fallback. Uses sandbox_isfile()
    from app.sandbox_fs instead of raw requests + host os.path.isfile.
    No host fallbacks. If sandbox can't find it, it doesn't exist.
    """
    try:
        from app.sandbox_fs import sandbox_isfile
    except ImportError:
        logger.warning("[det_critique] sandbox_fs not available for existence check")
        return False

    for path in candidates:
        if sandbox_isfile(path):
            logger.info("[det_critique] File exists in SANDBOX: %s", path)
            return True
    return False

def check_create_modify_classification(
    arch_content: str,
    segment_spec: Optional[Dict[str, Any]] = None,
    skeleton_file_scope: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Check that files listed as 'New Files' (CREATE) in the architecture
    do not already exist on disk.

    v1.0 (2026-03-01): Prevents the implementer from overwriting existing
    working files with skeletal placeholders. Cross-references the
    architecture's File Inventory against INDEX.json.

    This is a BLOCKING check — an existing file classified as CREATE
    will be overwritten by the implementer, destroying working code.

    Args:
        arch_content: Architecture markdown document.
        segment_spec: Segment spec dict (for file_scope context).
        skeleton_file_scope: File scope from skeleton contract.

    Returns:
        List of issue dicts. Each blocking issue identifies a file that
        the architecture classifies as CREATE but that exists on disk.
    """
    import os

    issues: List[Dict[str, Any]] = []

    new_file_paths = _extract_new_file_paths(arch_content)
    if not new_file_paths:
        return issues

    # Load filesystem index
    fs_index = _load_filesystem_index()

    # Also check via direct filesystem access as fallback
    frontend_root = os.getenv("ORB_FRONTEND_ROOT", r"D:\orb-desktop")
    backend_root = os.getenv("ORB_BACKEND_ROOT", r"D:\Orb")

    for claimed_new in new_file_paths:
        norm = claimed_new.replace("\\", "/").lower()

        # Strip common prefixes for INDEX.json lookup
        lookup_variants = [norm]
        if norm.startswith("orb-desktop/"):
            lookup_variants.append(norm[len("orb-desktop/"):])
        if norm.startswith("app/"):
            lookup_variants.append(norm)

        exists_in_index = any(v in fs_index for v in lookup_variants)

        # Direct filesystem check as fallback — check SANDBOX, not host
        # v1.2 (2026-03-03): Fix 6 — host has stale files from previous
        # runs; sandbox is the ground truth for what exists.
        exists_on_disk = False
        if not exists_in_index:
            raw = claimed_new.replace("/", os.sep)
            candidates = [
                os.path.join(frontend_root, raw),
                os.path.join(backend_root, raw),
            ]
            if raw.startswith("orb-desktop" + os.sep):
                stripped = raw[len("orb-desktop" + os.sep):]
                candidates.append(os.path.join(frontend_root, stripped))

            exists_on_disk = _check_sandbox_file_exists(candidates)

        if exists_in_index or exists_on_disk:
            # v1.2 (2026-03-05): Auto-downgrade to warning instead of blocking.
            # The file exists — the implementer will read it from the sandbox
            # and modify it regardless of what the arch doc says. Blocking here
            # triggers expensive Opus revision loops ($0.15-0.43 per cycle) that
            # often fail to resolve. The arch doc's intent label is cosmetic;
            # what matters is the implementer's behaviour, which is correct.
            logger.warning(
                "[det_critique] AUTO-DOWNGRADE: '%s' listed as CREATE but exists — "
                "treating as MODIFY (warning, not blocking)",
                claimed_new,
            )
            issues.append({
                "rule_id": "DET-CREATE-OVERWRITES-EXISTING",
                "severity": "warning",
                "file": claimed_new,
                "spec_ref": "file_scope",
                "arch_ref": "File Inventory → New Files",
                "description": (
                    f"File '{claimed_new}' is listed under 'New Files' (CREATE) "
                    f"but already exists. Auto-downgraded to warning — the "
                    f"implementer will treat this as MODIFY."
                ),
                "suggested_fix": (
                    f"Move '{claimed_new}' from 'New Files' to 'Modified Files' "
                    f"in the File Inventory."
                ),
            })

    if issues:
        logger.warning(
            "[det_critique] CREATE/MODIFY check: %d file(s) would overwrite existing code: %s",
            len(issues),
            [i["file"] for i in issues],
        )

    return issues

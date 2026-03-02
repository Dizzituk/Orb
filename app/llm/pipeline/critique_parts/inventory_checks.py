# FILE: app/llm/pipeline/critique_parts/inventory_checks.py
"""
Deterministic Critique — File Inventory, Classification & Import Checks.

Check 1: File inventory compliance
    Every file listed in spec.json → file_scope[] appears in the
    architecture's file inventory. No extras, no missing.

Check 2: CREATE vs MODIFY classification accuracy
    Files listed under "New Files" (CREATE) must NOT exist on disk.
    Cross-references against INDEX.json and direct filesystem checks.
    BLOCKING — prevents implementer from overwriting working code.

Check 3: Import contract validation
    All imports declared in the architecture match the skeleton
    contract's consumes bindings. Every consumed symbol has a
    provider segment that exposes it.

Zero LLM calls. Pure structural comparison.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
v1.1 (2026-03-01): Added CHECK 2 — CREATE/MODIFY classification
accuracy via INDEX.json and filesystem cross-reference.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

INVENTORY_CHECKS_BUILD_ID = "2026-03-01-v1.1-inventory-import-and-classification-checks"


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

        # Direct filesystem check as fallback
        exists_on_disk = False
        if not exists_in_index:
            raw = claimed_new.replace("/", os.sep)
            candidates = [
                os.path.join(frontend_root, raw),
                os.path.join(backend_root, raw),
            ]
            # Also try stripping orb-desktop/ prefix
            if raw.startswith("orb-desktop" + os.sep):
                stripped = raw[len("orb-desktop" + os.sep):]
                candidates.append(os.path.join(frontend_root, stripped))

            exists_on_disk = any(os.path.isfile(c) for c in candidates)

        if exists_in_index or exists_on_disk:
            issues.append({
                "rule_id": "DET-CREATE-OVERWRITES-EXISTING",
                "severity": "blocking",
                "file": claimed_new,
                "spec_ref": "file_scope",
                "arch_ref": "File Inventory → New Files",
                "description": (
                    f"File '{claimed_new}' is listed under 'New Files' (CREATE) "
                    f"but ALREADY EXISTS on disk. Sending this to the implementer "
                    f"would overwrite the existing file with a skeletal placeholder. "
                    f"This file should be under 'Modified Files' (MODIFY) instead."
                ),
                "suggested_fix": (
                    f"Move '{claimed_new}' from 'New Files' to 'Modified Files' "
                    f"in the File Inventory, and update its architecture section "
                    f"to describe modifications rather than creating from scratch."
                ),
            })

    if issues:
        logger.warning(
            "[det_critique] CREATE/MODIFY check: %d file(s) would overwrite existing code: %s",
            len(issues),
            [i["file"] for i in issues],
        )

    return issues


# =========================================================================
# CHECK 3: Import Contract Validation
# =========================================================================

def _extract_imports_from_arch(arch_content: str) -> List[Dict[str, str]]:
    """
    Extract import statements from architecture code blocks and prose.

    Returns list of {module, names, raw_line} dicts.
    """
    imports: List[Dict[str, str]] = []

    # Pattern: from .module import name1, name2
    for m in re.finditer(
        r'from\s+\.(\w+)\s+import\s+([^(\n]+)',
        arch_content,
    ):
        module = m.group(1).strip()
        names_str = m.group(2).strip().rstrip("\\").strip("`")
        names = [
            n.strip().strip("`").split(" as ")[0].strip()
            for n in names_str.split(",")
            if n.strip()
        ]
        # Filter garbage
        names = [n for n in names if re.match(r'^[a-zA-Z_]\w*$', n)]
        if module and names:
            imports.append({
                "module": module,
                "names": ", ".join(names),
                "raw_line": m.group(0).strip(),
            })

    return imports


def check_import_contracts(
    arch_content: str,
    segment_id: str,
    skeleton_contract: Optional[Dict[str, Any]] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Check that architecture imports align with skeleton contract bindings.

    Rules:
    - Imports from sibling segments must reference files in the
      skeleton's consumes/imports_from bindings
    - Imported symbols should exist in the providing segment's
      enrichment export list

    Args:
        arch_content: Architecture markdown document
        segment_id: This segment's ID
        skeleton_contract: Full skeleton_contract.json dict
        enrichment_data: Enrichment data for all segments
        manifest_dict: Full manifest dict

    Returns:
        List of issue dicts
    """
    issues: List[Dict[str, Any]] = []

    if not skeleton_contract and not manifest_dict:
        return issues

    # Build module-to-segment mapping from manifest
    module_to_segment: Dict[str, str] = {}
    if manifest_dict:
        for seg in manifest_dict.get("segments", []):
            sid = seg.get("segment_id", "")
            for fp in seg.get("file_scope", []):
                # Key by basename without extension
                basename = fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
                module_to_segment[basename] = sid

    # Build set of declared dependencies for this segment
    declared_deps: Set[str] = set()
    if skeleton_contract:
        for skel in skeleton_contract.get("skeletons", []):
            if skel.get("segment_id") == segment_id:
                declared_deps = set(skel.get("dependencies", []))
                # Also include imports_from keys
                declared_deps.update(skel.get("imports_from", {}).keys())
                declared_deps.update(skel.get("peer_imports_from", {}).keys())
                break
    elif manifest_dict:
        for seg in manifest_dict.get("segments", []):
            if seg.get("segment_id") == segment_id:
                declared_deps = set(seg.get("dependencies", []))
                break

    # Build own modules set
    own_modules: Set[str] = set()
    if manifest_dict:
        for seg in manifest_dict.get("segments", []):
            if seg.get("segment_id") == segment_id:
                for fp in seg.get("file_scope", []):
                    basename = fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
                    own_modules.add(basename)
                break

    # Build enrichment export map
    enrichment_exports: Dict[str, Set[str]] = {}  # segment_id -> set of symbol names
    if enrichment_data:
        for sid, edata in enrichment_data.items():
            if isinstance(edata, dict):
                symbols: Set[str] = set()
                for func in edata.get("functions", []):
                    if isinstance(func, dict) and func.get("name"):
                        symbols.add(func["name"])
                    elif isinstance(func, str):
                        symbols.add(func)
                for exp in edata.get("exports", []):
                    if isinstance(exp, str):
                        symbols.add(exp)
                enrichment_exports[sid] = symbols

    # Extract imports from architecture
    arch_imports = _extract_imports_from_arch(arch_content)

    for imp in arch_imports:
        target_module = imp["module"].lower()

        # Skip own modules
        if target_module in own_modules:
            continue

        # Find which segment owns this module
        owner_segment = module_to_segment.get(target_module)
        if not owner_segment:
            continue  # Can't validate — unknown module

        if owner_segment == segment_id:
            continue  # Same segment, fine

        # Check 3a: Is the owning segment a declared dependency?
        if owner_segment not in declared_deps:
            issues.append({
                "rule_id": "DET-IMPORT-UNDECLARED",
                "severity": "warning",
                "file": f".{imp['module']}",
                "spec_ref": "skeleton_contract.dependencies",
                "arch_ref": imp["raw_line"],
                "description": (
                    f"Architecture imports from '.{imp['module']}' (owned by "
                    f"{owner_segment}) but {owner_segment} is not a declared "
                    f"dependency of {segment_id}."
                ),
                "suggested_fix": (
                    f"Add {owner_segment} to {segment_id}'s dependencies, "
                    f"or remove this import."
                ),
            })

        # Check 3b: Do imported symbols exist in the provider's enrichment?
        if owner_segment in enrichment_exports:
            available = enrichment_exports[owner_segment]
            for name in imp["names"].split(", "):
                name = name.strip()
                if name and name not in available:
                    issues.append({
                        "rule_id": "DET-IMPORT-PHANTOM",
                        "severity": "blocking",
                        "file": f".{imp['module']}",
                        "spec_ref": f"enrichment.{owner_segment}.exports",
                        "arch_ref": imp["raw_line"],
                        "description": (
                            f"Architecture imports '{name}' from '.{imp['module']}' "
                            f"(owned by {owner_segment}) but '{name}' is not in "
                            f"{owner_segment}'s known exports. Available: "
                            f"{sorted(list(available))[:8]}"
                        ),
                        "suggested_fix": (
                            f"Use one of the available symbols from {owner_segment}, "
                            f"or add '{name}' to {owner_segment}'s architecture."
                        ),
                    })

    if issues:
        logger.info(
            "[det_critique] Import contract check: %d issues (%d blocking)",
            len(issues),
            sum(1 for i in issues if i["severity"] == "blocking"),
        )

    return issues


__all__ = [
    "check_file_inventory_compliance",
    "check_create_modify_classification",
    "check_import_contracts",
    "INVENTORY_CHECKS_BUILD_ID",
]

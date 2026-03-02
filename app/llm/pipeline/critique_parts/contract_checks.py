# FILE: app/llm/pipeline/critique_parts/contract_checks.py
"""
Deterministic contract validation checks for the critique engine.

Check 1: File inventory compliance — spec file_scope vs architecture inventory.
Check 3: Import contract validation — architecture imports vs skeleton consumes.

These checks validate the architecture document against the segment's
structural contracts (spec, skeleton, enrichment) with zero LLM cost.

v1.0 (2026-02-27): Initial implementation — deterministic verification migration.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from app.llm.pipeline.critique_schemas import CritiqueIssue

logger = logging.getLogger(__name__)


# =========================================================================
# Helpers: Load contract data from disk
# =========================================================================

def _load_skeleton_contract(job_dir: str) -> Optional[Dict[str, Any]]:
    """Load skeleton_contract.json from the job's segments directory."""
    skeleton_path = os.path.join(job_dir, "segments", "skeleton_contract.json")
    if not os.path.isfile(skeleton_path):
        logger.debug("[contract_checks] No skeleton contract at %s", skeleton_path)
        return None
    try:
        with open(skeleton_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[contract_checks] Failed to load skeleton contract: %s", e)
        return None


def _load_manifest(job_dir: str) -> Optional[Dict[str, Any]]:
    """Load manifest.json from the job's segments directory."""
    manifest_path = os.path.join(job_dir, "segments", "manifest.json")
    if not os.path.isfile(manifest_path):
        logger.debug("[contract_checks] No manifest at %s", manifest_path)
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[contract_checks] Failed to load manifest: %s", e)
        return None


def _extract_arch_file_inventory(arch_content: str) -> List[str]:
    """
    Extract file paths from the architecture's File Inventory section.

    Looks for markdown table rows with backtick-wrapped file paths
    within a "File Inventory" section heading.
    """
    paths: List[str] = []
    seen: Set[str] = set()

    inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', arch_content)
    if not inv_match:
        return paths

    inv_start = inv_match.start()
    inv_end_match = re.search(r'\n(?:##[^#]|---)', arch_content[inv_start + 20:])
    if inv_end_match:
        inv_section = arch_content[inv_start:inv_start + 20 + inv_end_match.start()]
    else:
        inv_section = arch_content[inv_start:inv_start + 3000]

    for line in inv_section.split("\n"):
        if not line.strip().startswith("|") or line.strip().startswith("|---"):
            continue
        line_lower = line.lower()
        if "*(none" in line_lower or "_(none" in line_lower:
            continue

        match = re.search(
            r'`((?:app|src|tests|config|orb-desktop)[/\\][\w/\\._-]+\.[a-z]+)`',
            line,
        )
        if not match:
            match = re.search(
                r'`([\w_-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|css))`',
                line,
            )
        if match:
            p = match.group(1)
            key = p.replace("\\", "/").lower()
            if key not in seen:
                seen.add(key)
                paths.append(p)

    return paths


# =========================================================================
# Check 1: File inventory compliance
# =========================================================================

def check_file_inventory_compliance(
    arch_content: str,
    spec_json: Optional[str] = None,
    segment_id: Optional[str] = None,
    job_dir: Optional[str] = None,
) -> List[CritiqueIssue]:
    """
    Verify every file in the segment's spec file_scope appears in the
    architecture's file inventory, and no unexpected files are present.

    Data sources:
        - spec_json: segment spec with file_scope[]
        - skeleton contract: file_scope per segment (loaded from disk)
        - architecture: File Inventory table

    Returns list of CritiqueIssue for any mismatches.
    """
    issues: List[CritiqueIssue] = []

    # Parse spec file_scope
    spec_files: Set[str] = set()
    if spec_json:
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
            raw_scope = spec_data.get("file_scope", [])
            spec_files = {f.replace("\\", "/").lower() for f in raw_scope}
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    if not spec_files:
        # Nothing to check — no file_scope in spec
        return issues

    # Extract architecture file inventory
    arch_files_raw = _extract_arch_file_inventory(arch_content)
    arch_files = {f.replace("\\", "/").lower() for f in arch_files_raw}

    if not arch_files:
        # Architecture has no parseable file inventory — warn but don't block
        issues.append(CritiqueIssue(
            id="DET-INVENTORY-001",
            spec_ref="file_scope",
            arch_ref="File Inventory",
            category="file_inventory",
            severity="warning",
            description=(
                "Architecture does not contain a parseable File Inventory section. "
                "Cannot verify file_scope compliance."
            ),
            fix_suggestion="Add a File Inventory table with backtick-wrapped file paths.",
        ))
        return issues

    # Check for missing files (in spec but not in architecture)
    missing = spec_files - arch_files
    # Relaxed match: check basename match for paths that might differ in prefix
    still_missing: Set[str] = set()
    for m in missing:
        basename = m.rsplit("/", 1)[-1] if "/" in m else m
        if not any(a.endswith("/" + basename) or a == basename for a in arch_files):
            still_missing.add(m)

    for miss in sorted(still_missing):
        issues.append(CritiqueIssue(
            id="DET-INVENTORY-002",
            spec_ref="file_scope",
            arch_ref="File Inventory",
            category="file_inventory",
            severity="blocking",
            description=(
                f"Spec requires file '{miss}' but it does not appear in the "
                f"architecture's File Inventory."
            ),
            fix_suggestion=f"Add '{miss}' to the File Inventory section.",
        ))

    # Check for extra files (in architecture but not in spec)
    extra = arch_files - spec_files
    still_extra: Set[str] = set()
    for e in extra:
        basename = e.rsplit("/", 1)[-1] if "/" in e else e
        if not any(s.endswith("/" + basename) or s == basename for s in spec_files):
            still_extra.add(e)

    for ext in sorted(still_extra):
        issues.append(CritiqueIssue(
            id="DET-INVENTORY-003",
            spec_ref="file_scope",
            arch_ref="File Inventory",
            category="file_inventory",
            severity="warning",
            description=(
                f"Architecture includes file '{ext}' which is not in the "
                f"segment's file_scope. This may indicate scope creep."
            ),
            fix_suggestion=(
                f"Remove '{ext}' from the architecture or verify it belongs "
                f"to this segment's scope."
            ),
        ))

    if issues:
        logger.info(
            "[contract_checks] File inventory: %d issue(s) for %s",
            len(issues), segment_id or "unknown",
        )

    return issues


# =========================================================================
# Check 3: Import contract validation
# =========================================================================

def check_import_contracts(
    arch_content: str,
    segment_id: Optional[str] = None,
    job_dir: Optional[str] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
) -> List[CritiqueIssue]:
    """
    Verify all cross-segment imports in the architecture match the
    skeleton contract's consumes bindings.

    For every import in the architecture that references a module owned
    by another segment, checks that:
    1. The owning segment is declared as a dependency
    2. The imported symbols exist in the provider segment's exports

    Uses enrichment data and skeleton contract loaded from disk.
    """
    issues: List[CritiqueIssue] = []

    if not job_dir or not segment_id:
        return issues

    skeleton = _load_skeleton_contract(job_dir)
    manifest = _load_manifest(job_dir)

    if not skeleton or not manifest:
        return issues

    # Build maps from manifest
    file_to_segment: Dict[str, str] = {}
    seg_dependencies: Dict[str, Set[str]] = {}

    for seg in manifest.get("segments", []):
        sid = seg.get("segment_id", "")
        seg_dependencies[sid] = set(seg.get("dependencies", []))
        for fp in seg.get("file_scope", []):
            normalised = fp.replace("\\", "/").lower()
            file_to_segment[normalised] = sid
            # Also map just the basename for relative import matching
            basename = normalised.rsplit("/", 1)[-1].replace(".py", "")
            file_to_segment[basename] = sid

    # Build export map from enrichment (if available) or skeleton
    seg_exports: Dict[str, Set[str]] = {}  # segment_id -> set of exported symbols
    if enrichment_data:
        for sid_key, enr in enrichment_data.items():
            if isinstance(enr, dict):
                funcs = enr.get("functions", [])
                exports = enr.get("exports", [])
                symbols: Set[str] = set()
                for f in funcs:
                    if isinstance(f, dict):
                        symbols.add(f.get("name", ""))
                    elif isinstance(f, str):
                        symbols.add(f)
                for e in exports:
                    if isinstance(e, str):
                        symbols.add(e)
                symbols.discard("")
                seg_exports[sid_key] = symbols

    # Parse imports from architecture
    # Matches: from .module import name1, name2
    import_pattern = re.compile(
        r'from\s+\.(\w+)\s+import\s+([^(\n]+)', re.MULTILINE,
    )

    my_deps = seg_dependencies.get(segment_id, set())

    for match in import_pattern.finditer(arch_content):
        target_module = match.group(1).lower()
        imports_str = match.group(2).strip().rstrip("\\").strip("`")
        imported_names = [
            n.strip().strip("`").split(" as ")[0].strip()
            for n in imports_str.split(",")
            if n.strip()
        ]

        # Find which segment owns this module
        owner_seg = file_to_segment.get(target_module)
        if not owner_seg or owner_seg == segment_id:
            continue  # Same segment or unknown — skip

        # Check 1: Is the owner declared as a dependency?
        if owner_seg not in my_deps:
            issues.append(CritiqueIssue(
                id="DET-IMPORT-001",
                spec_ref="dependencies",
                arch_ref=f"import from .{target_module}",
                category="import_contract",
                severity="warning",
                description=(
                    f"Architecture imports from '.{target_module}' (owned by "
                    f"{owner_seg}) but {segment_id} does not declare {owner_seg} "
                    f"as a dependency."
                ),
                fix_suggestion=(
                    f"Add {owner_seg} to this segment's dependencies, or "
                    f"remove the import."
                ),
            ))

        # Check 2: Do the imported symbols exist in the provider's exports?
        available = seg_exports.get(owner_seg, set())
        if not available:
            continue  # Can't verify without export data

        for name in imported_names:
            if not name or not re.match(r'^[a-zA-Z_]\w*$', name):
                continue
            if name not in available:
                issues.append(CritiqueIssue(
                    id="DET-IMPORT-002",
                    spec_ref=f"exports from {owner_seg}",
                    arch_ref=f"import {name} from .{target_module}",
                    category="import_contract",
                    severity="blocking",
                    description=(
                        f"Architecture imports '{name}' from '.{target_module}' "
                        f"(owned by {owner_seg}), but '{name}' is not in "
                        f"{owner_seg}'s known exports. Available: "
                        f"{sorted(list(available))[:10]}"
                    ),
                    fix_suggestion=(
                        f"Use one of the available exports from {owner_seg}, "
                        f"or add '{name}' to {owner_seg}'s architecture."
                    ),
                ))

    if issues:
        logger.info(
            "[contract_checks] Import contracts: %d issue(s) for %s",
            len(issues), segment_id,
        )

    return issues


__all__ = [
    "check_file_inventory_compliance",
    "check_import_contracts",
]

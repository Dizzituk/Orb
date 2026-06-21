# FILE: app/llm/pipeline/critique_parts/inventory_import_contracts.py
# Purpose: Deterministic Critique CHECK 3 — Import contract validation.
# Called-by: app.llm.pipeline.critique_parts.inventory_checks (re-export shim)
# Depends-on: (stdlib only: re)
# Last-renovated: 2026-06-20
"""
Deterministic Critique — CHECK 3: Import Contract Validation.

Check 3: Import contract validation
    All imports declared in the architecture match the skeleton
    contract's consumes bindings. Every consumed symbol has a
    provider segment that exposes it.

Zero LLM calls. Pure structural comparison.

Split 2026-06-20 from inventory_checks.py via the move-and-shim pattern —
logic byte-identical. Carries its private helper _extract_imports_from_arch.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


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

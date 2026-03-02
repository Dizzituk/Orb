# FILE: app/llm/pipeline/critique_parts/ordering_checks.py
"""
Deterministic Critique — Dependency Order Verification.

Check 5: Dependency order verification
    Architecture doesn't reference symbols from segments that haven't
    been built yet (based on execution order from manifest).

Zero LLM calls. Builds ordered segment list from manifest and checks
cross-segment references against build order.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

ORDERING_CHECKS_BUILD_ID = "2026-02-27-v1.0-dependency-order"


# =========================================================================
# CHECK 5: Dependency Order Verification
# =========================================================================

def _build_execution_order(manifest_dict: Dict[str, Any]) -> List[str]:
    """
    Determine segment execution order from manifest.

    Segments are built in dependency order: segments with no dependencies
    first, then segments whose dependencies are all already built.
    Falls back to manifest order if topology is unclear.
    """
    segments = manifest_dict.get("segments", [])
    if not segments:
        return []

    # Build dependency map
    deps: Dict[str, Set[str]] = {}
    all_ids: List[str] = []
    for seg in segments:
        sid = seg.get("segment_id", "")
        all_ids.append(sid)
        deps[sid] = set(seg.get("dependencies", []))

    # Topological sort (Kahn's algorithm)
    order: List[str] = []
    remaining = set(all_ids)
    built: Set[str] = set()

    max_iterations = len(all_ids) + 1
    for _ in range(max_iterations):
        if not remaining:
            break
        # Find segments whose dependencies are all built
        ready = [
            sid for sid in remaining
            if deps.get(sid, set()).issubset(built)
        ]
        if not ready:
            # Cycle or unresolvable deps — add remaining in manifest order
            for sid in all_ids:
                if sid in remaining:
                    order.append(sid)
            break
        # Sort ready list for determinism
        ready.sort()
        for sid in ready:
            order.append(sid)
            built.add(sid)
            remaining.discard(sid)

    return order


def check_dependency_order(
    arch_content: str,
    segment_id: str,
    manifest_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Check that architecture doesn't reference symbols from segments
    that haven't been built yet.

    If this segment is at position N in the build order, it can only
    reference segments at positions 0..N-1.

    Args:
        arch_content: Architecture markdown document
        segment_id: This segment's ID
        manifest_dict: Full manifest dict (needed for build order)

    Returns:
        List of issue dicts
    """
    issues: List[Dict[str, Any]] = []

    if not manifest_dict:
        return issues

    execution_order = _build_execution_order(manifest_dict)
    if not execution_order:
        return issues

    # Find this segment's position in build order
    try:
        my_position = execution_order.index(segment_id)
    except ValueError:
        return issues  # Segment not in manifest — can't check

    # Segments that are built before this one
    already_built = set(execution_order[:my_position])

    # Build module-to-segment map
    module_to_segment: Dict[str, str] = {}
    for seg in manifest_dict.get("segments", []):
        sid = seg.get("segment_id", "")
        for fp in seg.get("file_scope", []):
            basename = fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
            module_to_segment[basename] = sid

    # Find own modules
    own_modules: Set[str] = set()
    for seg in manifest_dict.get("segments", []):
        if seg.get("segment_id") == segment_id:
            for fp in seg.get("file_scope", []):
                basename = fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
                own_modules.add(basename)
            break

    # Extract cross-segment imports from architecture
    for m in re.finditer(r'from\s+\.(\w+)\s+import', arch_content):
        target_module = m.group(1).lower()

        if target_module in own_modules:
            continue  # Own module, fine

        owner = module_to_segment.get(target_module)
        if not owner:
            continue  # Unknown module

        if owner == segment_id:
            continue  # Same segment

        if owner not in already_built:
            # This segment references a module from a segment that
            # hasn't been built yet
            position = execution_order.index(owner) if owner in execution_order else -1
            issues.append({
                "rule_id": "DET-ORDER-FUTURE-REF",
                "severity": "warning",
                "file": f".{m.group(1)}",
                "spec_ref": "manifest.execution_order",
                "arch_ref": m.group(0),
                "description": (
                    f"Architecture imports from '.{m.group(1)}' (owned by "
                    f"{owner}, build position {position}) but {segment_id} "
                    f"is at position {my_position}. The referenced segment "
                    f"hasn't been built yet at this point."
                ),
                "suggested_fix": (
                    f"This is a build-order dependency. Ensure {owner} is "
                    f"listed as a dependency of {segment_id} so it builds first."
                ),
            })

    if issues:
        logger.info(
            "[det_critique] Order check: %d future references for %s",
            len(issues), segment_id,
        )

    return issues


__all__ = [
    "check_dependency_order",
    "ORDERING_CHECKS_BUILD_ID",
]

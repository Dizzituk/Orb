"""
Cohesion Check — Cross-Segment Architecture Verification.

Two-layer verification:
  Layer 1: Deterministic skeleton compliance (free, instant)
  Layer 2: LLM-based cross-segment cohesion (Opus 4.6, deep analysis)

v5.0 (2026-02): Extracted checks 4-8 to _cohesion_skeleton_checks.py
v2.0 (2026-02-12): Added deterministic skeleton compliance (Layer 1)
v1.0 (2026-02-10): Initial LLM-based cohesion check
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.orchestrator._cohesion_check_utils_8 import (
    COHESION_CHECK_BUILD_ID,
    _build_cohesion_prompt,
    _classify_fix_tier,
    _extract_arch_file_paths,
    _extract_import_replacements,
    _extract_segment_references,
    _inject_logging_import,
    _save_patched_architecture,
)
from app.orchestrator._cohesion_check_utils_9 import (
    _apply_tier1_fix,
    _apply_tier2_fix,
    _parse_cohesion_response,
    load_cohesion_result,
    save_cohesion_result,
)
from app.orchestrator._cohesion_check_utils_10 import (
    CohesionIssue,
    CohesionResult,
    load_segment_architectures,
    run_cohesion_check,
)
from app.orchestrator._cohesion_check_utils_11 import attempt_auto_fixes
from app.orchestrator._cohesion_skeleton_checks import (
    check_undeclared_dependencies,
    check_missing_stdlib_imports,
    check_cross_segment_symbols,
    check_duplicate_functions,
    check_phantom_symbols,
)

logger = logging.getLogger(__name__)
print(f"[COHESION_CHECK_LOADED] BUILD_ID={COHESION_CHECK_BUILD_ID}")


# =============================================================================
# LAYER 1: DETERMINISTIC SKELETON COMPLIANCE
# =============================================================================

def run_skeleton_compliance(
    architectures: Dict[str, str],
    skeleton_json: Optional[str] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
) -> List[CohesionIssue]:
    """
    Deterministic skeleton compliance check.

    Checks 1-3 run inline (scope, segment refs, exports).
    Checks 4-8 delegated to _cohesion_skeleton_checks.py.
    """
    issues: List[CohesionIssue] = []
    issue_counter = 0

    if not skeleton_json:
        return issues

    try:
        skeleton = json.loads(skeleton_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[cohesion_check] Failed to parse skeleton JSON")
        return issues

    all_segment_ids = set()
    scope_by_segment: Dict[str, set] = {}
    exports_by_segment: Dict[str, List[Dict]] = {}

    for skel in skeleton.get("skeletons", []):
        seg_id = skel.get("segment_id", "")
        all_segment_ids.add(seg_id)
        scope_by_segment[seg_id] = {
            p.replace("\\", "/").lower()
            for p in skel.get("file_scope", [])
        }
        exports_by_segment[seg_id] = skel.get("exports", [])

    # --- Checks 1-3: Inline (scope, refs, exports) ---
    for seg_id, arch_content in architectures.items():
        seg_scope = scope_by_segment.get(seg_id, set())

        # Check 1: File inventory within scope
        arch_files = _extract_arch_file_paths(arch_content)
        for arch_file in arch_files:
            normalised = arch_file.replace("\\", "/").lower()
            if normalised not in seg_scope:
                basename = normalised.rsplit("/", 1)[-1] if "/" in normalised else normalised
                partial_match = any(
                    s.endswith("/" + basename) or s == basename for s in seg_scope
                )
                if not partial_match:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"SKEL-{issue_counter:03d}",
                        severity="blocking",
                        category="scope_violation",
                        description=(
                            f"Architecture for {seg_id} includes file "
                            f"'{arch_file}' which is outside its skeleton scope"
                        ),
                        source_segment=seg_id,
                        file_path=arch_file,
                        expected=f"Files in scope: {', '.join(sorted(seg_scope))}",
                        suggested_fix="Remove this file or update the manifest scope",
                    ))

        # Check 2: Segment references must exist
        seg_refs = _extract_segment_references(arch_content)
        for ref_num in seg_refs:
            ref_found = any(
                f"seg-{ref_num:02d}" in vid or f"seg-{ref_num}" in vid
                for vid in all_segment_ids
            )
            if not ref_found:
                issue_counter += 1
                issues.append(CohesionIssue(
                    issue_id=f"SKEL-{issue_counter:03d}",
                    severity="blocking",
                    category="phantom_segment",
                    description=(
                        f"Architecture for {seg_id} references segment {ref_num} "
                        f"which doesn't exist"
                    ),
                    source_segment=seg_id,
                    suggested_fix="Remove reference to non-existent segment",
                ))

        # Check 3: Required exports present
        seg_exports = exports_by_segment.get(seg_id, [])
        for export in seg_exports:
            export_path = export.get("file_path", "").replace("\\", "/").lower()
            if export_path and export_path not in {
                f.replace("\\", "/").lower() for f in arch_files
            }:
                consumed_by = export.get("consumed_by", [])
                if consumed_by:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"SKEL-{issue_counter:03d}",
                        severity="warning",
                        category="missing_export",
                        description=(
                            f"Segment {seg_id} should export '{export_path}' "
                            f"(consumed by {', '.join(consumed_by)})"
                        ),
                        source_segment=seg_id,
                        related_segment=consumed_by[0] if consumed_by else "",
                        file_path=export_path,
                        suggested_fix="Add this file to the architecture",
                    ))

    # --- Checks 4-8: Delegated to _cohesion_skeleton_checks ---
    if manifest_dict:
        new_issues, issue_counter = check_undeclared_dependencies(
            architectures, manifest_dict, issue_counter,
        )
        issues.extend(new_issues)

    if architectures:
        new_issues, issue_counter = check_missing_stdlib_imports(
            architectures, issue_counter,
        )
        issues.extend(new_issues)

    if manifest_dict and len(architectures) > 1:
        new_issues, issue_counter = check_cross_segment_symbols(
            architectures, manifest_dict, issue_counter,
        )
        issues.extend(new_issues)

    if len(architectures) > 1:
        new_issues, issue_counter = check_duplicate_functions(
            architectures, manifest_dict, issue_counter,
        )
        issues.extend(new_issues)

    if manifest_dict and len(architectures) > 1:
        issues, issue_counter = check_phantom_symbols(
            architectures, manifest_dict, issues, issue_counter,
        )

    return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CohesionIssue",
    "CohesionResult",
    "run_skeleton_compliance",
    "run_cohesion_check",
    "attempt_auto_fixes",
    "load_segment_architectures",
    "save_cohesion_result",
    "load_cohesion_result",
    "COHESION_CHECK_BUILD_ID",
]

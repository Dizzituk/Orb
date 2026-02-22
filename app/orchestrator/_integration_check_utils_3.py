from __future__ import annotations
import os
import re
from app.orchestrator.ast_helpers import get_all_defined_names
from app.orchestrator.integration_check import IntegrationIssue
from app.orchestrator.segment_state import JobState
from app.pot_spec.grounded.segment_schemas import SegmentManifest, SegmentSpec, SegmentStatus
from typing import Dict, List, Set, Tuple


INTEGRATION_CHECK_BUILD_ID = "2026-02-08-v1.0-phase3"

def _collect_segment_outputs(
    state: JobState,
    manifest: SegmentManifest,
    job_dir: str,
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Collect output files from COMPLETE segments.

    Returns:
        (segment_outputs, checked_segment_ids, skipped_segment_ids)
    """
    segment_outputs: Dict[str, List[str]] = {}
    checked: List[str] = []
    skipped: List[str] = []

    for seg_spec in manifest.segments:
        seg_id = seg_spec.segment_id
        seg_state = state.segments.get(seg_id)

        if seg_state is None or seg_state.status != SegmentStatus.COMPLETE.value:
            skipped.append(seg_id)
            continue

        checked.append(seg_id)
        files = list(seg_state.output_files) if seg_state.output_files else []

        output_dir = os.path.join(job_dir, "segments", seg_id, "output")
        if os.path.isdir(output_dir):
            for root, _dirs, filenames in os.walk(output_dir):
                for fname in filenames:
                    full = os.path.normpath(os.path.join(root, fname))
                    if full not in files:
                        files.append(full)

        segment_outputs[seg_id] = files

    return segment_outputs, checked, skipped

_DEFAULT_PROJECT_ROOTS = [
    r"D:\Orb",
    r"D:\orb-desktop",
]

def _get_project_roots(job_dir: str) -> List[str]:
    """Determine project roots for import resolution."""
    roots = list(_DEFAULT_PROJECT_ROOTS)
    segments_dir = os.path.join(job_dir, "segments")
    if os.path.isdir(segments_dir):
        roots.append(segments_dir)
    return [r for r in roots if os.path.isdir(r)]

def _verify_exposes(
    seg_spec: SegmentSpec,
    output_files: List[str],
    project_roots: List[str],
) -> List[IntegrationIssue]:
    """Verify that a segment's output files actually define what it promises to expose."""
    issues: List[IntegrationIssue] = []
    seg_id = seg_spec.segment_id
    exposes = seg_spec.exposes
    if not exposes:
        return issues

    all_defined: Set[str] = set()
    for f in output_files:
        if os.path.isfile(f):
            all_defined.update(get_all_defined_names(f))

    for class_name in exposes.class_names:
        if class_name not in all_defined:
            issues.append(IntegrationIssue(
                severity="error",
                check_type="interface_contract",
                segment_a=seg_id,
                segment_b="N/A",
                file_a=", ".join(output_files[:3]),
                file_b="N/A",
                expected=f"Class '{class_name}' should be defined (exposes contract)",
                actual="Not found in segment output files",
                message=(
                    f"Segment {seg_id} promises to expose class '{class_name}' "
                    f"but it is not defined in any output file."
                ),
            ))

    for export_name in exposes.export_names:
        if export_name not in all_defined:
            issues.append(IntegrationIssue(
                severity="error",
                check_type="interface_contract",
                segment_a=seg_id,
                segment_b="N/A",
                file_a=", ".join(output_files[:3]),
                file_b="N/A",
                expected=f"Export '{export_name}' should be defined (exposes contract)",
                actual="Not found in segment output files",
                message=(
                    f"Segment {seg_id} promises to expose '{export_name}' "
                    f"but it is not defined in any output file."
                ),
            ))

    return issues

_SQL_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?['\"]?(\w+)['\"]?",
    re.IGNORECASE,
)

_PY_ROUTE_RE = re.compile(
    r"@(?:app|router|bp)\.\s*(?:get|post|put|patch|delete|route)\s*\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

_TS_ROUTE_RE = re.compile(
    r"(?:app|router)\.\s*(?:get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

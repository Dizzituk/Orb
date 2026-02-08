# FILE: app/orchestrator/integration_check.py
"""
Cross-Segment Integration Check (Phase 3).

Verifies that segments produced by the orchestrator segment loop actually
work together. Runs AFTER all segments complete, BEFORE the final summary.

Two tiers:
    Tier 1 - Deterministic (no LLM): AST parsing, regex, filesystem checks.
        1. Import resolution: cross-segment imports resolve to real definitions
        2. Interface contracts: exposes/consumes match actual file contents
        3. File references: cross-segment path references are correct
        4. Duplicate definitions: no conflicting table/route/export names

    Tier 2 - Lightweight LLM (optional, advisory): single LLM call for
        semantic compatibility, naming consistency, integration completeness.
        Produces warnings only, never errors.

Design:
    - READ-ONLY: inspects output files, never modifies them
    - Host-direct filesystem access (same pattern as file_verifier.py)
    - Dataclass-based results with to_dict()/from_dict() (matching segment_schemas.py)
    - All logging uses [INTEGRATION_CHECK] prefix
    - Crash-safe: exceptions caught and reported, never crash the segment loop

Phase 3 of Pipeline Segmentation.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

INTEGRATION_CHECK_BUILD_ID = "2026-02-08-v1.0-phase3"
print(f"[INTEGRATION_CHECK_LOADED] BUILD_ID={INTEGRATION_CHECK_BUILD_ID}")

# --- Internal imports ---
from app.pot_spec.grounded.segment_schemas import (
    InterfaceContract,
    SegmentManifest,
    SegmentSpec,
    SegmentStatus,
)
from app.orchestrator.segment_state import JobState, SegmentState
from app.orchestrator.ast_helpers import (
    extract_python_definitions,
    extract_typescript_exports,
    resolve_python_import,
    resolve_typescript_import,
    get_all_defined_names,
    get_all_imports,
)

# Type alias
ProgressCallback = Optional[Callable[[str], None]]


# =============================================================================
# RESULT MODELS
# =============================================================================


@dataclass
class IntegrationIssue:
    """A single cross-segment integration issue."""

    severity: str       # "error" | "warning" | "info"
    check_type: str     # "import_resolution" | "interface_contract" | "file_reference" | "duplicate_definition" | "llm_review"
    segment_a: str      # segment_id of the producer
    segment_b: str      # segment_id of the consumer (or "N/A" for duplicates)
    file_a: str         # file in segment A
    file_b: str         # file in segment B
    expected: str       # what the contract/import says
    actual: str         # what was actually found (or "missing")
    message: str        # human-readable description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "check_type": self.check_type,
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "file_a": self.file_a,
            "file_b": self.file_b,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationIssue":
        return cls(
            severity=data.get("severity", "error"),
            check_type=data.get("check_type", "unknown"),
            segment_a=data.get("segment_a", ""),
            segment_b=data.get("segment_b", ""),
            file_a=data.get("file_a", ""),
            file_b=data.get("file_b", ""),
            expected=data.get("expected", ""),
            actual=data.get("actual", ""),
            message=data.get("message", ""),
        )


@dataclass
class IntegrationCheckResult:
    """Aggregated result of the cross-segment integration check."""

    status: str                             # "pass" | "warn" | "fail" | "error" | "skipped"
    tier1_issues: List[IntegrationIssue] = field(default_factory=list)
    tier2_issues: List[IntegrationIssue] = field(default_factory=list)
    segments_checked: List[str] = field(default_factory=list)
    segments_skipped: List[str] = field(default_factory=list)
    checked_at: str = ""
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "tier1_issues": [i.to_dict() for i in self.tier1_issues],
            "tier2_issues": [i.to_dict() for i in self.tier2_issues],
            "segments_checked": self.segments_checked,
            "segments_skipped": self.segments_skipped,
            "checked_at": self.checked_at,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationCheckResult":
        return cls(
            status=data.get("status", "error"),
            tier1_issues=[IntegrationIssue.from_dict(i) for i in data.get("tier1_issues", [])],
            tier2_issues=[IntegrationIssue.from_dict(i) for i in data.get("tier2_issues", [])],
            segments_checked=data.get("segments_checked", []),
            segments_skipped=data.get("segments_skipped", []),
            checked_at=data.get("checked_at", ""),
            error_message=data.get("error_message"),
        )

    @property
    def all_issues(self) -> List[IntegrationIssue]:
        return self.tier1_issues + self.tier2_issues

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == "warning")

    def summary(self) -> str:
        return (
            f"IntegrationCheck({self.status}: "
            f"{self.error_count} errors, {self.warning_count} warnings, "
            f"{len(self.segments_checked)} checked, {len(self.segments_skipped)} skipped)"
        )

# =============================================================================
# HELPERS
# =============================================================================


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


def _build_file_to_segment_map(
    segment_outputs: Dict[str, List[str]],
) -> Dict[str, str]:
    """Build a reverse map: normalised_file_path -> segment_id."""
    file_to_seg: Dict[str, str] = {}
    for seg_id, files in segment_outputs.items():
        for f in files:
            normalised = os.path.normpath(f).lower().replace("\\", "/")
            file_to_seg[normalised] = seg_id
    return file_to_seg


def _normalise_path(path: str) -> str:
    return os.path.normpath(path).lower().replace("\\", "/")


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


def _looks_like_project_import(module: str) -> bool:
    """Heuristic: does this import path look project-internal?"""
    project_prefixes = ("app.", "src.", "lib.", "utils.", "services.", "components.")
    return any(module.startswith(p) for p in project_prefixes)


def _module_to_expected_path(module: str, project_roots: List[str]) -> Optional[str]:
    """Convert a module path to the expected file path (without checking existence)."""
    parts = module.split(".")
    relative = os.path.join(*parts) + ".py"
    for root in project_roots:
        return os.path.normpath(os.path.join(root, relative))
    return None

# =============================================================================
# TIER 1: IMPORT RESOLUTION
# =============================================================================


def _check_import_resolution(
    segment_outputs: Dict[str, List[str]],
    manifest: SegmentManifest,
    project_roots: List[str],
) -> List[IntegrationIssue]:
    """
    For each segment's output files, find cross-segment imports.
    Verify the imported names exist in the target file.
    """
    issues: List[IntegrationIssue] = []
    file_to_seg = _build_file_to_segment_map(segment_outputs)

    for seg_id, files in segment_outputs.items():
        for file_path in files:
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".py":
                issues.extend(_check_python_cross_imports(
                    file_path, seg_id, file_to_seg, project_roots, segment_outputs,
                ))
            elif ext in (".ts", ".tsx", ".js", ".jsx"):
                issues.extend(_check_typescript_cross_imports(
                    file_path, seg_id, file_to_seg, project_roots, segment_outputs,
                ))

    return issues


def _check_python_cross_imports(
    file_path: str,
    owning_seg: str,
    file_to_seg: Dict[str, str],
    project_roots: List[str],
    segment_outputs: Dict[str, List[str]],
) -> List[IntegrationIssue]:
    """Check a Python file's imports for cross-segment reference issues."""
    issues: List[IntegrationIssue] = []
    defs = extract_python_definitions(file_path)

    for imp in defs.get("imports_from", []):
        module = imp["module"]
        imported_names = imp["names"]

        resolved = resolve_python_import(module, project_roots)
        if resolved is None:
            if _looks_like_project_import(module):
                expected_path = _module_to_expected_path(module, project_roots)
                if expected_path:
                    target_seg = file_to_seg.get(_normalise_path(expected_path))
                    if target_seg and target_seg != owning_seg:
                        issues.append(IntegrationIssue(
                            severity="error",
                            check_type="import_resolution",
                            segment_a=target_seg,
                            segment_b=owning_seg,
                            file_a="(missing)",
                            file_b=file_path,
                            expected=f"Module '{module}' should exist",
                            actual="File not found on disk",
                            message=(
                                f"File '{file_path}' (seg {owning_seg}) imports from "
                                f"'{module}' but the target file does not exist. "
                                f"Expected to be created by segment {target_seg}."
                            ),
                        ))
            continue

        target_seg = file_to_seg.get(_normalise_path(resolved))
        if target_seg is None or target_seg == owning_seg:
            continue

        target_names = get_all_defined_names(resolved)
        for name in imported_names:
            if name == "*":
                continue
            if name not in target_names:
                issues.append(IntegrationIssue(
                    severity="error",
                    check_type="import_resolution",
                    segment_a=target_seg,
                    segment_b=owning_seg,
                    file_a=resolved,
                    file_b=file_path,
                    expected=f"Name '{name}' should be defined in '{resolved}'",
                    actual=f"Defined names: {sorted(target_names)[:10]}",
                    message=(
                        f"Cross-segment import failure: '{file_path}' (seg {owning_seg}) "
                        f"imports '{name}' from '{module}', but '{name}' is not defined "
                        f"in '{resolved}' (seg {target_seg})."
                    ),
                ))

    return issues


def _check_typescript_cross_imports(
    file_path: str,
    owning_seg: str,
    file_to_seg: Dict[str, str],
    project_roots: List[str],
    segment_outputs: Dict[str, List[str]],
) -> List[IntegrationIssue]:
    """Check a TypeScript file's imports for cross-segment reference issues."""
    issues: List[IntegrationIssue] = []
    defs = extract_typescript_exports(file_path)

    for imp in defs.get("imports_from", []):
        module = imp["module"]
        imported_names = imp["names"]

        if not module.startswith(".") and not module.startswith("@/"):
            continue

        resolved = resolve_typescript_import(module, file_path, project_roots)
        if resolved is None:
            continue

        target_seg = file_to_seg.get(_normalise_path(resolved))
        if target_seg is None or target_seg == owning_seg:
            continue

        target_defs = extract_typescript_exports(resolved)
        target_exports = set(target_defs.get("exports", []))
        target_default = target_defs.get("default_export")

        for name in imported_names:
            if len(imported_names) == 1 and target_default and name != target_default:
                continue
            if name not in target_exports:
                issues.append(IntegrationIssue(
                    severity="error",
                    check_type="import_resolution",
                    segment_a=target_seg,
                    segment_b=owning_seg,
                    file_a=resolved,
                    file_b=file_path,
                    expected=f"Export '{name}' should exist in '{resolved}'",
                    actual=f"Available exports: {sorted(target_exports)[:10]}",
                    message=(
                        f"Cross-segment import failure: '{file_path}' (seg {owning_seg}) "
                        f"imports '{name}' from '{module}', but '{name}' is not exported "
                        f"from '{resolved}' (seg {target_seg})."
                    ),
                ))

    return issues

# =============================================================================
# TIER 1: INTERFACE CONTRACT VERIFICATION
# =============================================================================


def _check_interface_contracts(
    manifest: SegmentManifest,
    segment_outputs: Dict[str, List[str]],
    project_roots: List[str],
) -> List[IntegrationIssue]:
    """
    For each InterfaceContract in the manifest:
    - Verify exposes: the source segment's files define the declared names
    - Verify consumes: the consuming segment references names that exist in exposes
    """
    issues: List[IntegrationIssue] = []
    checked_segments = set(segment_outputs.keys())

    for seg_spec in manifest.segments:
        seg_id = seg_spec.segment_id
        if seg_id not in checked_segments:
            continue

        if seg_spec.exposes and not seg_spec.exposes.is_empty():
            issues.extend(_verify_exposes(
                seg_spec, segment_outputs.get(seg_id, []), project_roots,
            ))

        if seg_spec.consumes and not seg_spec.consumes.is_empty():
            issues.extend(_verify_consumes(
                seg_spec, manifest, segment_outputs, checked_segments,
            ))

    return issues


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


def _verify_consumes(
    seg_spec: SegmentSpec,
    manifest: SegmentManifest,
    segment_outputs: Dict[str, List[str]],
    checked_segments: Set[str],
) -> List[IntegrationIssue]:
    """Verify that consumed names actually exist in upstream segment exposes."""
    issues: List[IntegrationIssue] = []
    seg_id = seg_spec.segment_id
    consumes = seg_spec.consumes
    if not consumes:
        return issues

    upstream_exposed: Dict[str, str] = {}
    for dep_id in seg_spec.dependencies:
        dep_spec = manifest.get_segment(dep_id)
        if dep_spec is None or dep_id not in checked_segments:
            continue
        if dep_spec.exposes:
            for name in dep_spec.exposes.class_names:
                upstream_exposed[name] = dep_id
            for name in dep_spec.exposes.export_names:
                upstream_exposed[name] = dep_id

    for class_name in consumes.class_names:
        if class_name not in upstream_exposed:
            issues.append(IntegrationIssue(
                severity="warning",
                check_type="interface_contract",
                segment_a="N/A",
                segment_b=seg_id,
                file_a="N/A",
                file_b="N/A",
                expected=f"Class '{class_name}' should be exposed by an upstream segment",
                actual="Not found in any upstream exposes contract",
                message=(
                    f"Segment {seg_id} declares it consumes class '{class_name}' "
                    f"but no upstream segment (deps: {seg_spec.dependencies}) exposes it."
                ),
            ))

    for export_name in consumes.export_names:
        if export_name not in upstream_exposed:
            issues.append(IntegrationIssue(
                severity="warning",
                check_type="interface_contract",
                segment_a="N/A",
                segment_b=seg_id,
                file_a="N/A",
                file_b="N/A",
                expected=f"Export '{export_name}' should be exposed by an upstream segment",
                actual="Not found in any upstream exposes contract",
                message=(
                    f"Segment {seg_id} declares it consumes '{export_name}' "
                    f"but no upstream segment (deps: {seg_spec.dependencies}) exposes it."
                ),
            ))

    return issues

# =============================================================================
# TIER 1: FILE REFERENCE CONSISTENCY
# =============================================================================


def _check_file_references(
    segment_outputs: Dict[str, List[str]],
    manifest: SegmentManifest,
    project_roots: List[str],
) -> List[IntegrationIssue]:
    """
    Check cross-segment file path references are correct.
    Catches: segment 2 imports from 'app/services/transcription_service.py'
    but segment 1 created 'app/services/transcription.py'.
    """
    issues: List[IntegrationIssue] = []
    file_to_seg = _build_file_to_segment_map(segment_outputs)

    for seg_id, files in segment_outputs.items():
        for file_path in files:
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(file_path)[1].lower()
            imports = get_all_imports(file_path)

            for imp in imports:
                module = imp["module"]

                if ext == ".py":
                    if not _looks_like_project_import(module):
                        continue
                    resolved = resolve_python_import(module, project_roots)
                elif ext in (".ts", ".tsx", ".js", ".jsx"):
                    if not module.startswith(".") and not module.startswith("@/"):
                        continue
                    resolved = resolve_typescript_import(module, file_path, project_roots)
                else:
                    continue

                if resolved is None:
                    if ext == ".py":
                        expected = _module_to_expected_path(module, project_roots)
                    else:
                        expected = module

                    issues.append(IntegrationIssue(
                        severity="error",
                        check_type="file_reference",
                        segment_a="unknown",
                        segment_b=seg_id,
                        file_a=expected or module,
                        file_b=file_path,
                        expected=f"Import target '{module}' should resolve to a file",
                        actual="File not found on disk",
                        message=(
                            f"Broken file reference: '{file_path}' (seg {seg_id}) "
                            f"imports from '{module}' but the target file does not exist."
                        ),
                    ))

    return issues


# =============================================================================
# TIER 1: DUPLICATE DEFINITION DETECTION
# =============================================================================

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


def _check_duplicate_definitions(
    segment_outputs: Dict[str, List[str]],
) -> List[IntegrationIssue]:
    """
    Check for conflicting definitions across segments:
    - Duplicate table names in migration files
    - Duplicate route paths in router files
    """
    issues: List[IntegrationIssue] = []

    # --- Duplicate table names ---
    table_defs: Dict[str, List[Tuple[str, str]]] = {}
    for seg_id, files in segment_outputs.items():
        for f in files:
            if not os.path.isfile(f):
                continue
            ext = os.path.splitext(f)[1].lower()
            basename = os.path.basename(f).lower()
            if not any(kw in basename for kw in ("migration", "model", "schema", "table", "alembic")):
                if ext != ".sql":
                    continue
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                for match in _SQL_TABLE_RE.finditer(content):
                    table_name = match.group(1).lower()
                    table_defs.setdefault(table_name, []).append((f, seg_id))
            except OSError:
                continue

    for table_name, locations in table_defs.items():
        seg_ids = set(loc[1] for loc in locations)
        if len(seg_ids) > 1:
            files_str = ", ".join(f"{loc[0]} (seg {loc[1]})" for loc in locations)
            issues.append(IntegrationIssue(
                severity="error",
                check_type="duplicate_definition",
                segment_a=locations[0][1],
                segment_b=locations[1][1],
                file_a=locations[0][0],
                file_b=locations[1][0],
                expected=f"Table '{table_name}' should be defined in one segment only",
                actual=f"Defined in segments: {sorted(seg_ids)}",
                message=f"Duplicate table definition: '{table_name}' in multiple segments: {files_str}",
            ))

    # --- Duplicate route paths ---
    route_defs: Dict[str, List[Tuple[str, str]]] = {}
    for seg_id, files in segment_outputs.items():
        for f in files:
            if not os.path.isfile(f):
                continue
            ext = os.path.splitext(f)[1].lower()
            basename = os.path.basename(f).lower()
            if not any(kw in basename for kw in ("route", "router", "endpoint", "api", "view")):
                continue
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                pattern = _PY_ROUTE_RE if ext == ".py" else _TS_ROUTE_RE
                for match in pattern.finditer(content):
                    route_path = match.group(1).lower()
                    route_defs.setdefault(route_path, []).append((f, seg_id))
            except OSError:
                continue

    for route_path, locations in route_defs.items():
        seg_ids = set(loc[1] for loc in locations)
        if len(seg_ids) > 1:
            files_str = ", ".join(f"{loc[0]} (seg {loc[1]})" for loc in locations)
            issues.append(IntegrationIssue(
                severity="error",
                check_type="duplicate_definition",
                segment_a=locations[0][1],
                segment_b=locations[1][1],
                file_a=locations[0][0],
                file_b=locations[1][0],
                expected=f"Route '{route_path}' should be defined in one segment only",
                actual=f"Defined in segments: {sorted(seg_ids)}",
                message=f"Duplicate route definition: '{route_path}' in multiple segments: {files_str}",
            ))

    return issues

# =============================================================================
# TIER 2: LIGHTWEIGHT LLM REVIEW
# =============================================================================


def _run_llm_integration_review(
    manifest: SegmentManifest,
    extracted_interfaces: Dict[str, Dict[str, Any]],
    tier1_issues: List[IntegrationIssue],
    llm_call: Callable,
    provider: str,
    model: str,
) -> List[IntegrationIssue]:
    """
    Single LLM call for semantic validation.
    Advisory only - produces warnings, not errors.
    """
    issues: List[IntegrationIssue] = []

    prompt_parts = [
        "You are reviewing a segmented software project for cross-segment integration issues.",
        "Each segment was built independently. Check for:",
        "1. Semantic mismatches (function signatures that don't make sense together)",
        "2. Naming inconsistencies across segments",
        "3. Missing connections between segments",
        "",
        "Respond with a JSON array of issues. Each issue:",
        '{"segment_a": "...", "segment_b": "...", "message": "...", "severity": "warning"|"info"}',
        "If no issues found, respond with: []",
        "",
        "=== SEGMENT CONTRACTS ===",
    ]

    for seg_spec in manifest.segments:
        seg_id = seg_spec.segment_id
        prompt_parts.append(f"\n--- {seg_id}: {seg_spec.title} ---")
        if seg_spec.exposes and not seg_spec.exposes.is_empty():
            prompt_parts.append(f"  Exposes: {seg_spec.exposes.to_dict()}")
        if seg_spec.consumes and not seg_spec.consumes.is_empty():
            prompt_parts.append(f"  Consumes: {seg_spec.consumes.to_dict()}")
        if seg_id in extracted_interfaces:
            for file_path, defs in extracted_interfaces[seg_id].items():
                basename = os.path.basename(file_path)
                names = defs.get("classes", []) + defs.get("functions", []) + defs.get("exports", [])
                if names:
                    prompt_parts.append(f"  {basename} defines: {names[:15]}")

    if tier1_issues:
        prompt_parts.append("\n=== TIER 1 ISSUES FOUND ===")
        for issue in tier1_issues[:10]:
            prompt_parts.append(f"  [{issue.severity}] {issue.message}")

    prompt = "\n".join(prompt_parts)

    try:
        import asyncio

        messages = [
            {"role": "system", "content": "You are a code integration reviewer. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ]

        result = llm_call(
            provider_id=provider,
            model_id=model,
            messages=messages,
            max_tokens=2000,
        )

        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("[INTEGRATION_CHECK] Cannot run async LLM call from sync context")
                return issues
            result = loop.run_until_complete(result)

        if result and hasattr(result, "content") and result.content:
            content = result.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)

            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("message"):
                            issues.append(IntegrationIssue(
                                severity=item.get("severity", "warning"),
                                check_type="llm_review",
                                segment_a=item.get("segment_a", "N/A"),
                                segment_b=item.get("segment_b", "N/A"),
                                file_a="N/A",
                                file_b="N/A",
                                expected="",
                                actual="",
                                message=item["message"],
                            ))
            except json.JSONDecodeError:
                logger.warning("[INTEGRATION_CHECK] LLM response was not valid JSON")

    except Exception as e:
        logger.warning("[INTEGRATION_CHECK] Tier 2 LLM call failed: %s", e)

    return issues

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def run_integration_check(
    manifest: SegmentManifest,
    state: JobState,
    job_dir: str,
    *,
    llm_call: Callable = None,
    provider: str = None,
    model: str = None,
    on_progress: ProgressCallback = None,
) -> IntegrationCheckResult:
    """
    Run cross-segment integration verification.

    1. Collect all output files from COMPLETE segments
    2. Run Tier 1 deterministic checks
    3. If llm_call provided, run Tier 2
    4. Return aggregated results

    This function catches all exceptions internally -- it will never
    crash the segment loop.
    """
    _emit = on_progress or (lambda msg: None)

    try:
        logger.info("[INTEGRATION_CHECK] Starting cross-segment integration check")
        _emit("[INTEGRATION_CHECK] Starting cross-segment integration check...")

        # --- Step 1: Collect output files ---
        segment_outputs, checked, skipped = _collect_segment_outputs(
            state, manifest, job_dir,
        )

        if not checked:
            logger.info("[INTEGRATION_CHECK] No COMPLETE segments -- skipping")
            _emit("[INTEGRATION_CHECK] No COMPLETE segments to check -- skipping")
            return IntegrationCheckResult(
                status="skipped",
                segments_checked=[],
                segments_skipped=skipped,
            )

        _emit(
            f"[INTEGRATION_CHECK] Checking {len(checked)} segment(s), "
            f"skipping {len(skipped)}"
        )

        project_roots = _get_project_roots(job_dir)

        # --- Step 2: Tier 1 deterministic checks ---
        tier1_issues: List[IntegrationIssue] = []

        _emit("[INTEGRATION_CHECK] Tier 1: Checking import resolution...")
        import_issues = _check_import_resolution(
            segment_outputs, manifest, project_roots,
        )
        tier1_issues.extend(import_issues)
        if import_issues:
            _emit(f"[INTEGRATION_CHECK]   Import resolution: {len(import_issues)} issue(s)")
        else:
            _emit("[INTEGRATION_CHECK]   Import resolution: PASS")

        _emit("[INTEGRATION_CHECK] Tier 1: Checking interface contracts...")
        contract_issues = _check_interface_contracts(
            manifest, segment_outputs, project_roots,
        )
        tier1_issues.extend(contract_issues)
        if contract_issues:
            _emit(f"[INTEGRATION_CHECK]   Interface contracts: {len(contract_issues)} issue(s)")
        else:
            _emit("[INTEGRATION_CHECK]   Interface contracts: PASS")

        _emit("[INTEGRATION_CHECK] Tier 1: Checking file references...")
        file_ref_issues = _check_file_references(
            segment_outputs, manifest, project_roots,
        )
        tier1_issues.extend(file_ref_issues)
        if file_ref_issues:
            _emit(f"[INTEGRATION_CHECK]   File references: {len(file_ref_issues)} issue(s)")
        else:
            _emit("[INTEGRATION_CHECK]   File references: PASS")

        _emit("[INTEGRATION_CHECK] Tier 1: Checking for duplicate definitions...")
        dup_issues = _check_duplicate_definitions(segment_outputs)
        tier1_issues.extend(dup_issues)
        if dup_issues:
            _emit(f"[INTEGRATION_CHECK]   Duplicate definitions: {len(dup_issues)} issue(s)")
        else:
            _emit("[INTEGRATION_CHECK]   Duplicate definitions: PASS")

        # --- Step 3: Tier 2 LLM review (optional) ---
        tier2_issues: List[IntegrationIssue] = []
        if llm_call and provider and model:
            _emit("[INTEGRATION_CHECK] Tier 2: Running LLM integration review...")
            try:
                extracted: Dict[str, Dict[str, Any]] = {}
                for seg_id, files in segment_outputs.items():
                    extracted[seg_id] = {}
                    for f in files:
                        if not os.path.isfile(f):
                            continue
                        ext = os.path.splitext(f)[1].lower()
                        if ext == ".py":
                            extracted[seg_id][f] = extract_python_definitions(f)
                        elif ext in (".ts", ".tsx", ".js", ".jsx"):
                            extracted[seg_id][f] = extract_typescript_exports(f)

                tier2_issues = _run_llm_integration_review(
                    manifest=manifest,
                    extracted_interfaces=extracted,
                    tier1_issues=tier1_issues,
                    llm_call=llm_call,
                    provider=provider,
                    model=model,
                )
                if tier2_issues:
                    _emit(f"[INTEGRATION_CHECK]   LLM review: {len(tier2_issues)} advisory issue(s)")
                else:
                    _emit("[INTEGRATION_CHECK]   LLM review: PASS")
            except Exception as e:
                logger.warning("[INTEGRATION_CHECK] Tier 2 failed: %s", e)
                _emit(f"[INTEGRATION_CHECK]   LLM review: skipped (error: {e})")
        else:
            _emit("[INTEGRATION_CHECK] Tier 2: Skipped (no LLM configured)")

        # --- Step 4: Determine overall status ---
        has_errors = any(i.severity == "error" for i in tier1_issues)
        has_warnings = (
            any(i.severity == "warning" for i in tier1_issues)
            or any(i.severity == "warning" for i in tier2_issues)
        )

        if has_errors:
            status = "fail"
        elif has_warnings:
            status = "warn"
        else:
            status = "pass"

        result = IntegrationCheckResult(
            status=status,
            tier1_issues=tier1_issues,
            tier2_issues=tier2_issues,
            segments_checked=checked,
            segments_skipped=skipped,
        )

        logger.info("[INTEGRATION_CHECK] Complete: %s", result.summary())
        _emit(f"[INTEGRATION_CHECK] {result.summary()}")

        return result

    except Exception as e:
        logger.exception("[INTEGRATION_CHECK] Integration check crashed: %s", e)
        _emit(f"[INTEGRATION_CHECK] Integration check error: {e}")
        return IntegrationCheckResult(
            status="error",
            segments_checked=[],
            segments_skipped=[],
            error_message=str(e),
        )


__all__ = [
    "IntegrationIssue",
    "IntegrationCheckResult",
    "run_integration_check",
]
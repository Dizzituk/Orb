from __future__ import annotations
import json
import logging
import os
import re
from app.orchestrator._integration_check_utils_6 import _PY_ROUTE_RE, _SQL_TABLE_RE, _TS_ROUTE_RE, _verify_exposes
from app.orchestrator.ast_helpers import extract_typescript_exports, resolve_typescript_import
from app.pot_spec.grounded.segment_schemas import SegmentManifest, SegmentSpec
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
try:
    from app.sandbox_fs import (
        sandbox_isfile as _sbx_isfile,
        sandbox_isdir as _sbx_isdir,
        sandbox_exists as _sbx_exists,
        sandbox_read_text as _sbx_read_text,
    )
    _SBX_FS_OK = True
except ImportError:
    _SBX_FS_OK = False
logger = logging.getLogger(__name__)


def _normalise_path(path: str) -> str:
    return os.path.normpath(path).lower().replace("\\", "/")

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

def _check_typescript_cross_imports(
    file_path: str,
    owning_seg: str,
    file_to_seg: Dict[str, str],
    project_roots: List[str],
    segment_outputs: Dict[str, List[str]],
) -> List[IntegrationIssue]:
    """Check a TypeScript file's imports for cross-segment reference issues."""
    from .integration_check import IntegrationIssue
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
    from .integration_check import IntegrationIssue
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

def _verify_consumes(
    seg_spec: SegmentSpec,
    manifest: SegmentManifest,
    segment_outputs: Dict[str, List[str]],
    checked_segments: Set[str],
) -> List[IntegrationIssue]:
    """Verify that consumed names actually exist in upstream segment exposes."""
    from .integration_check import IntegrationIssue
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

def _check_duplicate_definitions(
    segment_outputs: Dict[str, List[str]],
) -> List[IntegrationIssue]:
    """
    Check for conflicting definitions across segments:
    - Duplicate table names in migration files
    - Duplicate route paths in router files
    """
    from .integration_check import IntegrationIssue
    issues: List[IntegrationIssue] = []

    # --- Duplicate table names ---
    table_defs: Dict[str, List[Tuple[str, str]]] = {}
    for seg_id, files in segment_outputs.items():
        for f in files:
            if not (_sbx_isfile(f) if _SBX_FS_OK else os.path.isfile(f)):
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
            if not (_sbx_isfile(f) if _SBX_FS_OK else os.path.isfile(f)):
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
    from .integration_check import IntegrationIssue
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

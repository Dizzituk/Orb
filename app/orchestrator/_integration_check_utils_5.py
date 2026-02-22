from __future__ import annotations
import logging
import os
from app.orchestrator._integration_check_utils import _check_duplicate_definitions, _check_interface_contracts, _check_typescript_cross_imports, _looks_like_project_import, _module_to_expected_path, _normalise_path, _run_llm_integration_review
from app.orchestrator._integration_check_utils import _collect_segment_outputs, _get_project_roots
from app.orchestrator.ast_helpers import extract_python_definitions, extract_typescript_exports, get_all_defined_names, get_all_imports, resolve_python_import, resolve_typescript_import
from app.orchestrator.integration_check import IntegrationIssue, ProgressCallback, logger
from app.orchestrator.segment_state import JobState
from app.pot_spec.grounded.segment_schemas import SegmentManifest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
ProgressCallback = Optional[Callable[[str], None]]


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

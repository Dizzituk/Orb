# FILE: app/agentic_pipeline/checks/check_runner.py
"""
Confidence-Gated Check Runner for the Agentic Loop.

Wraps existing deterministic checks with confidence classification.
Callable as a tool within the agentic loop.

Confidence levels:
  HARD_BLOCK — 100% certain error. Zero false-positive tolerance.
  WARNING    — High confidence issue. Model should fix or justify.
  INFO       — Awareness only. No action required.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Confidence(str, Enum):
    HARD_BLOCK = "HARD_BLOCK"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class CheckResult:
    check_id: str
    confidence: Confidence
    file_path: Optional[str] = None
    message: str = ""
    suggested_fix: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckReport:
    results: List[CheckResult] = field(default_factory=list)
    cycle_number: int = 0
    passed: bool = True

    @property
    def hard_blocks(self) -> List[CheckResult]:
        return [r for r in self.results if r.confidence == Confidence.HARD_BLOCK]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if r.confidence == Confidence.WARNING]

    @property
    def infos(self) -> List[CheckResult]:
        return [r for r in self.results if r.confidence == Confidence.INFO]

    @property
    def has_blockers(self) -> bool:
        return len(self.hard_blocks) > 0


def _check_python_ast_parse(file_path: str, code: str) -> Optional[CheckResult]:
    """HARD BLOCK: Python code must parse without SyntaxError."""
    if not file_path.endswith(".py"):
        return None
    import ast
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return CheckResult(
            check_id="AST_PARSE_FAILURE", confidence=Confidence.HARD_BLOCK,
            file_path=file_path,
            message=f"Python syntax error: {e.msg} (line {e.lineno})",
            suggested_fix=f"Fix syntax error at line {e.lineno}: {e.msg}",
        )


def _check_create_on_existing(file_path: str, declared_action: str, preflight_exists: bool) -> Optional[CheckResult]:
    """WARNING: File listed as CREATE but already exists in sandbox."""
    if declared_action.upper() == "CREATE" and preflight_exists:
        return CheckResult(
            check_id="CREATE_ON_EXISTING", confidence=Confidence.WARNING,
            file_path=file_path,
            message=f"File '{file_path}' listed as CREATE but already exists. Should be MODIFY?",
            suggested_fix=f"Change '{file_path}' from CREATE to MODIFY.",
        )
    return None


def _check_export_exists(importing_file: str, imported_symbol: str, provider_file: str, provider_exports: List[str]) -> Optional[CheckResult]:
    """HARD BLOCK: Imported symbol must exist in provider's export list."""
    if imported_symbol in provider_exports:
        return None
    close = [e for e in provider_exports if _fuzzy_similar(imported_symbol, e)]
    if close:
        return CheckResult(
            check_id="EXPORT_FUZZY_MISMATCH", confidence=Confidence.WARNING,
            file_path=importing_file,
            message=f"'{importing_file}' imports '{imported_symbol}' from '{provider_file}', closest: {close}",
            suggested_fix=f"Use one of {close} instead of '{imported_symbol}'.",
        )
    return CheckResult(
        check_id="EXPORT_NOT_FOUND", confidence=Confidence.HARD_BLOCK,
        file_path=importing_file,
        message=f"'{importing_file}' imports '{imported_symbol}' from '{provider_file}', not in exports: {provider_exports[:10]}",
        suggested_fix=f"Add '{imported_symbol}' to '{provider_file}' or fix the import.",
    )


def _check_file_path_valid(file_path: str, declared_action: str, preflight_exists: bool) -> Optional[CheckResult]:
    """HARD BLOCK: MODIFY on non-existent file."""
    if declared_action.upper() == "MODIFY" and not preflight_exists:
        return CheckResult(
            check_id="MODIFY_ON_MISSING", confidence=Confidence.HARD_BLOCK,
            file_path=file_path,
            message=f"File '{file_path}' listed as MODIFY but does not exist in sandbox.",
            suggested_fix=f"Change '{file_path}' to CREATE, or fix the path.",
        )
    return None


def _check_file_size(file_path: str, code: str) -> Optional[CheckResult]:
    """INFO/WARNING: Flag files exceeding size targets."""
    size_kb = len(code.encode("utf-8", errors="replace")) / 1024
    if size_kb > 30:
        return CheckResult(
            check_id="FILE_SIZE_OVER_HARD_LIMIT", confidence=Confidence.WARNING,
            file_path=file_path,
            message=f"File is {size_kb:.1f}KB, exceeding 30KB hard limit.",
            suggested_fix="Split into smaller cooperating modules.",
        )
    elif size_kb > 20:
        return CheckResult(
            check_id="FILE_SIZE_OVER_TARGET", confidence=Confidence.INFO,
            file_path=file_path, message=f"File is {size_kb:.1f}KB, exceeding 20KB target.",
        )
    return None


def _check_multiple_segment_modify(file_path: str, modifying_segments: List[str]) -> Optional[CheckResult]:
    """INFO: Multiple segments modifying the same file."""
    if len(modifying_segments) > 1:
        return CheckResult(
            check_id="MULTI_SEGMENT_MODIFY", confidence=Confidence.INFO,
            file_path=file_path,
            message=f"File '{file_path}' modified by {modifying_segments}. Ensure compatible.",
        )
    return None


def _fuzzy_similar(a: str, b: str) -> bool:
    a_l, b_l = a.lower(), b.lower()
    if a_l == b_l:
        return True
    if a_l.startswith(b_l) or b_l.startswith(a_l):
        return True
    if len(a) < 30 and len(b) < 30 and abs(len(a) - len(b)) <= 2:
        return sum(1 for ca, cb in zip(a_l, b_l) if ca != cb) <= 2
    return False


def run_deterministic_checks(
    arch_docs: Dict[str, str],
    preflight: Any,
    segment_file_scopes: Optional[Dict[str, List[str]]] = None,
    cycle_number: int = 1,
) -> CheckReport:
    """Run all deterministic checks on architecture output."""
    report = CheckReport(cycle_number=cycle_number)
    if not arch_docs:
        return report

    file_segments: Dict[str, List[str]] = {}
    if segment_file_scopes:
        for seg_id, files in segment_file_scopes.items():
            for f in files:
                file_segments.setdefault(f.replace("\\", "/"), []).append(seg_id)

    for seg_id, arch_content in arch_docs.items():
        file_scope = (segment_file_scopes or {}).get(seg_id, [])
        _check_segment(report, seg_id, arch_content, file_scope, preflight, file_segments)

    report.passed = not report.has_blockers
    logger.info("[check_runner] Cycle %d: %d checks, %d HARD_BLOCK, %d WARNING, %d INFO. Passed: %s",
                cycle_number, len(report.results), len(report.hard_blocks),
                len(report.warnings), len(report.infos), report.passed)
    return report


def _check_segment(report, seg_id, arch_content, file_scope, preflight, file_segments):
    from app.agentic_pipeline.loop_parser import extract_code_blocks_from_arch
    code_blocks = extract_code_blocks_from_arch(arch_content)

    for file_path, code in code_blocks.items():
        norm = file_path.replace("\\", "/")
        facts = preflight.get_facts(norm) if preflight else None
        exists = facts.exists if facts else False
        action = _infer_action(arch_content, file_path, exists)

        for check_fn, args in [
            (_check_python_ast_parse, (norm, code)),
            (_check_create_on_existing, (norm, action, exists)),
            (_check_file_path_valid, (norm, action, exists)),
            (_check_file_size, (norm, code)),
            (_check_multiple_segment_modify, (norm, file_segments.get(norm, []))),
        ]:
            result = check_fn(*args)
            if result:
                report.results.append(result)


def _infer_action(arch_content: str, file_path: str, exists_in_sandbox: bool) -> str:
    import re
    norm = file_path.replace("\\", "/")
    basename = norm.rsplit("/", 1)[-1] if "/" in norm else norm
    new_sec = re.search(r"###?\s*New\s+Files(.*?)(?=###?\s|$)", arch_content, re.DOTALL | re.IGNORECASE)
    if new_sec and basename in new_sec.group(1):
        return "CREATE"
    mod_sec = re.search(r"###?\s*Modified\s+Files(.*?)(?=###?\s|$)", arch_content, re.DOTALL | re.IGNORECASE)
    if mod_sec and basename in mod_sec.group(1):
        return "MODIFY"
    return "MODIFY" if exists_in_sandbox else "CREATE"


def format_check_report_for_prompt(report: CheckReport) -> str:
    """Format check results for injection into the agentic loop context."""
    if not report.results:
        return f"## DETERMINISTIC CHECK #{report.cycle_number} — ALL CLEAR\n\nNo issues found.\n"

    lines = [f"## DETERMINISTIC CHECK #{report.cycle_number} RESULTS", ""]

    if report.hard_blocks:
        lines.append(f"### HARD BLOCKS ({len(report.hard_blocks)}) — MUST FIX")
        lines.append("")
        for r in report.hard_blocks:
            lines.append(f"- **[{r.check_id}]** `{r.file_path or 'N/A'}`")
            lines.append(f"  {r.message}")
            if r.suggested_fix:
                lines.append(f"  Fix: {r.suggested_fix}")
        lines.append("")

    if report.warnings:
        lines.append(f"### WARNINGS ({len(report.warnings)}) — Review and fix or justify")
        lines.append("")
        for r in report.warnings:
            lines.append(f"- **[{r.check_id}]** `{r.file_path or 'N/A'}`")
            lines.append(f"  {r.message}")
            if r.suggested_fix:
                lines.append(f"  Suggested: {r.suggested_fix}")
        lines.append("")

    if report.infos:
        lines.append(f"### INFO ({len(report.infos)}) — No action needed")
        lines.append("")
        for r in report.infos:
            lines.append(f"- **[{r.check_id}]** `{r.file_path or 'N/A'}`: {r.message}")
        lines.append("")

    lines.append(f"**Summary**: {len(report.hard_blocks)} blockers, {len(report.warnings)} warnings, {len(report.infos)} info. "
                 f"{'MUST FIX BLOCKERS.' if report.hard_blocks else 'No blockers.'}")
    lines.append("")
    return "\n".join(lines)

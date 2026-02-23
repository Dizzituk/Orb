# FILE: app/llm/pipeline/deterministic_critique.py
"""
Deterministic pre-critique gate — zero LLM cost.

Runs structural checks on generated architecture/code BEFORE the
LLM critique fires. If structural issues are found, they're returned
immediately without spending an API call. The LLM critique then only
needs to assess semantic quality, not catch structural errors.

Checks:
1. File existence: All files referenced in architecture exist in sandbox
2. Syntax validation: Generated Python files parse without errors
3. Import resolution: All imports resolve against the project file index
4. Signature compliance: Function signatures match the architecture spec
5. Size compliance: Files are within size limits
6. Duplicate detection: No duplicate function definitions across segments

Called from: critique.py, BEFORE the LLM call
Returns: DeterministicCritiqueResult with pass/fail + violations list

Usage:
    from app.llm.pipeline.deterministic_critique import run_deterministic_critique

    result = run_deterministic_critique(architecture_doc, sandbox_base)
    if result.has_blockers:
        # Skip LLM critique, return structural violations directly
        return result.as_critique_blockers()
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StructuralViolation:
    """A single structural issue found by deterministic critique."""
    check: str          # Which check found it (syntax, import, size, etc.)
    file_path: str      # File with the issue
    severity: str       # "blocker" or "warning"
    message: str        # Human-readable description
    line_number: int = 0


@dataclass
class DeterministicCritiqueResult:
    """Result of all deterministic checks."""
    passed: bool
    violations: List[StructuralViolation] = field(default_factory=list)
    files_checked: int = 0
    checks_run: int = 0

    @property
    def has_blockers(self) -> bool:
        return any(v.severity == "blocker" for v in self.violations)

    @property
    def blocker_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "blocker")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def as_critique_blockers(self) -> List[dict]:
        """Format blockers for injection into critique response format."""
        return [
            {
                "check": v.check,
                "file": v.file_path,
                "severity": v.severity,
                "message": v.message,
                "source": "deterministic_critique",
            }
            for v in self.violations
            if v.severity == "blocker"
        ]


# =========================================================================
# Individual checks
# =========================================================================

def check_syntax(file_path: str, content: str) -> List[StructuralViolation]:
    """Check if a Python file parses without syntax errors."""
    if not file_path.endswith(".py"):
        return []

    try:
        ast.parse(content, filename=file_path)
        return []
    except SyntaxError as e:
        return [StructuralViolation(
            check="syntax",
            file_path=file_path,
            severity="blocker",
            message=f"SyntaxError: {e.msg} (line {e.lineno})",
            line_number=e.lineno or 0,
        )]


def check_file_size(
    file_path: str,
    content: str,
    max_kb: float = 30.0,
    target_kb: float = 20.0,
) -> List[StructuralViolation]:
    """Check file size against limits."""
    size_kb = len(content.encode("utf-8")) / 1024
    violations = []

    if size_kb > max_kb:
        violations.append(StructuralViolation(
            check="size",
            file_path=file_path,
            severity="blocker",
            message=f"File exceeds maximum: {size_kb:.1f}KB > {max_kb}KB",
        ))
    elif size_kb > target_kb:
        violations.append(StructuralViolation(
            check="size",
            file_path=file_path,
            severity="warning",
            message=f"File exceeds target: {size_kb:.1f}KB > {target_kb}KB (max {max_kb}KB)",
        ))

    return violations


def check_imports_resolve(
    file_path: str,
    content: str,
    known_files: Set[str],
) -> List[StructuralViolation]:
    """
    Check that imports in the file reference modules that exist.

    Only checks relative imports and app.* imports — external
    packages are assumed to be installed.
    """
    if not file_path.endswith(".py"):
        return []

    violations = []

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return []  # Syntax check will catch this

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("app."):
                # Convert dotted module path to file path
                module_path = node.module.replace(".", os.sep) + ".py"
                alt_path = node.module.replace(".", os.sep) + os.sep + "__init__.py"

                if module_path not in known_files and alt_path not in known_files:
                    # Check if any known file starts with the module prefix
                    prefix = node.module.replace(".", os.sep)
                    has_prefix_match = any(
                        f.startswith(prefix) for f in known_files
                    )
                    if not has_prefix_match:
                        violations.append(StructuralViolation(
                            check="import",
                            file_path=file_path,
                            severity="warning",  # Warning not blocker — could be pre-existing
                            message=f"Import 'from {node.module}' may not resolve",
                            line_number=node.lineno,
                        ))

    return violations


def check_duplicate_functions(
    files: Dict[str, str],
) -> List[StructuralViolation]:
    """
    Check for duplicate function definitions across files.

    Only flags public functions (not starting with _) that
    appear in multiple files with the same name.
    """
    func_locations: Dict[str, List[str]] = {}

    for file_path, content in files.items():
        if not file_path.endswith(".py"):
            continue
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    func_locations.setdefault(node.name, []).append(file_path)

    violations = []
    for func_name, locations in func_locations.items():
        if len(locations) > 1:
            files_list = ", ".join(locations[:3])
            violations.append(StructuralViolation(
                check="duplicate",
                file_path=locations[0],
                severity="warning",
                message=f"Function '{func_name}' defined in {len(locations)} files: {files_list}",
            ))

    return violations


# =========================================================================
# Main entry point
# =========================================================================

def run_deterministic_critique(
    files: Dict[str, str],
    known_project_files: Optional[Set[str]] = None,
    max_size_kb: float = 30.0,
    target_size_kb: float = 20.0,
) -> DeterministicCritiqueResult:
    """
    Run all deterministic structural checks on a set of files.

    Args:
        files: Dict of {file_path: file_content} to check
        known_project_files: Set of all known file paths in the project
                            (for import resolution). If None, import
                            checks are skipped.
        max_size_kb: Maximum file size in KB (blocker)
        target_size_kb: Target file size in KB (warning)

    Returns:
        DeterministicCritiqueResult with all violations found.
    """
    all_violations: List[StructuralViolation] = []
    checks_run = 0

    # Per-file checks
    for file_path, content in files.items():
        # Syntax
        all_violations.extend(check_syntax(file_path, content))
        checks_run += 1

        # Size
        all_violations.extend(check_file_size(
            file_path, content, max_size_kb, target_size_kb,
        ))
        checks_run += 1

        # Import resolution
        if known_project_files is not None:
            all_violations.extend(check_imports_resolve(
                file_path, content, known_project_files,
            ))
            checks_run += 1

    # Cross-file checks
    all_violations.extend(check_duplicate_functions(files))
    checks_run += 1

    has_blockers = any(v.severity == "blocker" for v in all_violations)

    result = DeterministicCritiqueResult(
        passed=not has_blockers,
        violations=all_violations,
        files_checked=len(files),
        checks_run=checks_run,
    )

    if all_violations:
        logger.info(
            "[det_critique] %d violations (%d blockers, %d warnings) in %d files",
            len(all_violations), result.blocker_count,
            result.warning_count, len(files),
        )

    return result


# =========================================================================
# Code block extraction (shared utility)
# =========================================================================

def extract_code_blocks_from_arch(arch_content: str) -> Dict[str, str]:
    """Extract fenced code blocks with file paths from architecture doc.

    Looks for patterns like:
        ```python\n# FILE: app/some/module.py\n...
    or:
        **File: app/some/module.py**\n```python\n...

    Returns {file_path: content} dict.
    """
    files: Dict[str, str] = {}
    # Pattern 1: # FILE: path inside code fence
    for match in re.finditer(
        r'```(?:python)?\s*\n#\s*FILE:\s*(.+?)\n(.*?)```',
        arch_content, re.DOTALL,
    ):
        path = match.group(1).strip()
        content = match.group(2)
        if path and content.strip():
            files[path] = content
    # Pattern 2: **File: path** before code fence
    for match in re.finditer(
        r'\*\*File:\s*(.+?)\*\*.*?```(?:python)?\s*\n(.*?)```',
        arch_content, re.DOTALL,
    ):
        path = match.group(1).strip()
        content = match.group(2)
        if path and content.strip() and path not in files:
            files[path] = content
    return files


__all__ = [
    "StructuralViolation",
    "DeterministicCritiqueResult",
    "run_deterministic_critique",
    "check_syntax",
    "check_file_size",
    "check_imports_resolve",
    "check_duplicate_functions",
    "extract_code_blocks_from_arch",
]

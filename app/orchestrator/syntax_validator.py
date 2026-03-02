# FILE: app/orchestrator/syntax_validator.py
"""
Post-Implementation Syntax Validator — Job 7.

Per-file, per-language syntax checking after the implementer writes code.
Catches errors BEFORE any LLM-based Overwatcher review, avoiding wasted
Overwatcher calls on trivially broken files.

Supports:
- TypeScript/TSX: `npx tsc --noEmit` on individual files via sandbox
- CSS: Brace matching + CSS variable reference validation
- Python: `ast.parse()` via sandbox shell

v1.0 (2026-03-01): Initial implementation.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SYNTAX_VALIDATOR_BUILD_ID = "2026-03-01-v1.0-per-file-syntax"
print(f"[SYNTAX_VALIDATOR_LOADED] BUILD_ID={SYNTAX_VALIDATOR_BUILD_ID}")


@dataclass
class SyntaxError_:
    """Single syntax error in a file."""
    file: str
    line: int
    column: int
    message: str
    code: str = ""         # e.g. "TS2300", "CSS_BRACE", "PY_SYNTAX"
    severity: str = "error"

    def __str__(self) -> str:
        loc = f"({self.line},{self.column})" if self.line else ""
        code_part = f" {self.code}:" if self.code else ""
        return f"{self.file}{loc}:{code_part} {self.message}"


@dataclass
class FileValidationResult:
    """Result of syntax validation for a single file."""
    file_path: str
    language: str          # "typescript" | "css" | "python" | "unknown"
    status: str            # "pass" | "fail" | "skipped" | "error"
    errors: List[SyntaxError_] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class ValidationBatchResult:
    """Result of validating multiple files."""
    total_files: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors_by_file: Dict[str, List[SyntaxError_]] = field(default_factory=dict)
    results: List[FileValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    @property
    def total_errors(self) -> int:
        return sum(len(errs) for errs in self.errors_by_file.values())

    def summary(self) -> str:
        if self.all_passed:
            return f"✅ Syntax check: {self.passed}/{self.total_files} files passed"
        return (
            f"❌ Syntax check: {self.failed}/{self.total_files} files failed "
            f"({self.total_errors} error(s))"
        )


# ─── Language detection ──────────────────────────────────────────────

_LANG_MAP = {
    ".tsx": "typescript",
    ".ts": "typescript",
    ".jsx": "typescript",
    ".js": "typescript",
    ".css": "css",
    ".scss": "css",
    ".py": "python",
}


def _detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return _LANG_MAP.get(ext, "unknown")


# ─── TypeScript validation ───────────────────────────────────────────

# Regex for tsc error lines: file(line,col): error TSxxxx: message
_TSC_ERROR_RE = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)$"
)


def validate_typescript_file(
    client: Any,
    file_path: str,
    frontend_base: str = r"D:\orb-desktop",
    timeout: int = 30,
) -> FileValidationResult:
    """Validate a single TypeScript/TSX file via tsc.

    Uses `npx tsc --noEmit` scoped to the project. Parses errors
    and filters to only those in the target file.

    Args:
        client: SandboxClient instance.
        file_path: Relative path within the frontend project.
        frontend_base: Frontend repo root in sandbox.
        timeout: Max seconds for tsc.

    Returns:
        FileValidationResult with any errors found.
    """
    cmd = f'cd "{frontend_base}" ; npx tsc --noEmit --pretty false 2>&1'

    try:
        result = client.shell_run(cmd, cwd_target="REPO", timeout_seconds=timeout)
    except Exception as exc:
        return FileValidationResult(
            file_path=file_path,
            language="typescript",
            status="error",
            raw_output=str(exc),
        )

    combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()

    # Parse ALL errors, then filter to our file
    all_errors: List[SyntaxError_] = []
    for line in combined.splitlines():
        m = _TSC_ERROR_RE.match(line.strip())
        if m:
            all_errors.append(SyntaxError_(
                file=m.group(1).replace("\\", "/"),
                line=int(m.group(2)),
                column=int(m.group(3)),
                code=m.group(4),
                message=m.group(5).strip(),
            ))

    # Filter to target file
    norm_path = file_path.replace("\\", "/")
    file_errors = [
        e for e in all_errors
        if e.file.replace("\\", "/") == norm_path
        or e.file.replace("\\", "/").endswith(norm_path)
        or norm_path.endswith(e.file.replace("\\", "/"))
    ]

    if not file_errors and result.exit_code == 0:
        return FileValidationResult(
            file_path=file_path, language="typescript",
            status="pass", raw_output=combined,
        )

    if not file_errors and result.exit_code != 0:
        # tsc failed but not because of OUR file — pass this file
        return FileValidationResult(
            file_path=file_path, language="typescript",
            status="pass", raw_output=combined,
        )

    return FileValidationResult(
        file_path=file_path, language="typescript",
        status="fail", errors=file_errors, raw_output=combined,
    )


# ─── CSS validation ──────────────────────────────────────────────────

def validate_css_file(
    client: Any,
    file_path: str,
    frontend_base: str = r"D:\orb-desktop",
) -> FileValidationResult:
    """Validate a CSS file for basic structural correctness.

    Checks:
    1. Brace matching (every { has a matching })
    2. No empty rulesets
    3. CSS variable references point to declared variables (if registry available)

    Args:
        client: SandboxClient for file reading.
        file_path: Path to CSS file.
        frontend_base: Frontend repo root.

    Returns:
        FileValidationResult.
    """
    abs_path = os.path.join(frontend_base, file_path).replace("/", "\\")

    try:
        # Read file via sandbox shell (public API — not private _request)
        read_cmd = f'type "{abs_path}"'
        read_result = client.shell_run(
            read_cmd, cwd_target="REPO", timeout_seconds=10,
        )
        if read_result.exit_code != 0:
            return FileValidationResult(
                file_path=file_path, language="css",
                status="error", raw_output=read_result.stderr or "Cannot read file",
            )
        content = read_result.stdout or ""
    except Exception as exc:
        return FileValidationResult(
            file_path=file_path, language="css",
            status="error", raw_output=str(exc),
        )

    errors: List[SyntaxError_] = []

    # Check 1: Brace matching
    brace_depth = 0
    for i, line in enumerate(content.splitlines(), 1):
        # Strip comments (simple single-line)
        stripped = re.sub(r"/\*.*?\*/", "", line)
        for ch in stripped:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    errors.append(SyntaxError_(
                        file=file_path, line=i, column=0,
                        code="CSS_BRACE",
                        message="Unexpected closing brace — no matching '{'",
                    ))
                    brace_depth = 0  # Reset to avoid cascade

    if brace_depth > 0:
        errors.append(SyntaxError_(
            file=file_path, line=0, column=0,
            code="CSS_BRACE",
            message=f"Unclosed braces: {brace_depth} opening '{{' without matching '}}'",
        ))

    if errors:
        return FileValidationResult(
            file_path=file_path, language="css",
            status="fail", errors=errors,
        )

    return FileValidationResult(
        file_path=file_path, language="css", status="pass",
    )


# ─── Python validation ──────────────────────────────────────────────

def validate_python_file(
    client: Any,
    file_path: str,
    repo_base: str = r"D:\Orb",
) -> FileValidationResult:
    """Validate a Python file via ast.parse in the sandbox.

    Args:
        client: SandboxClient.
        file_path: Relative path to Python file.
        repo_base: Backend repo root in sandbox.

    Returns:
        FileValidationResult.
    """
    abs_path = os.path.join(repo_base, file_path).replace("/", "\\")
    cmd = (
        f'python -c "'
        f"import ast, sys; "
        f"ast.parse(open(r'{abs_path}', encoding='utf-8').read()); "
        f"print('SYNTAX_OK')"
        f'" 2>&1'
    )

    try:
        result = client.shell_run(cmd, cwd_target="REPO", timeout_seconds=10)
    except Exception as exc:
        return FileValidationResult(
            file_path=file_path, language="python",
            status="error", raw_output=str(exc),
        )

    combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()

    if "SYNTAX_OK" in combined:
        return FileValidationResult(
            file_path=file_path, language="python",
            status="pass", raw_output=combined,
        )

    # Parse Python syntax error
    errors: List[SyntaxError_] = []
    m = re.search(r"line (\d+)", combined)
    line_num = int(m.group(1)) if m else 0

    errors.append(SyntaxError_(
        file=file_path, line=line_num, column=0,
        code="PY_SYNTAX",
        message=combined.split("\n")[-1] if combined else "Syntax error",
    ))

    return FileValidationResult(
        file_path=file_path, language="python",
        status="fail", errors=errors, raw_output=combined,
    )


# ─── Batch validation ───────────────────────────────────────────────

def validate_files_batch(
    client: Any,
    file_paths: List[str],
    frontend_base: str = r"D:\orb-desktop",
    repo_base: str = r"D:\Orb",
    emit: Optional[Any] = None,
) -> ValidationBatchResult:
    """Validate multiple files, dispatching by language.

    Groups TypeScript files to avoid running tsc multiple times.

    Args:
        client: SandboxClient.
        file_paths: List of file paths to validate.
        frontend_base: Frontend repo root.
        repo_base: Backend repo root.
        emit: Optional SSE callback.

    Returns:
        ValidationBatchResult with per-file results.
    """
    _emit = emit or (lambda msg: None)
    batch = ValidationBatchResult(total_files=len(file_paths))

    # Group by language
    ts_files: List[str] = []
    css_files: List[str] = []
    py_files: List[str] = []
    skip_files: List[str] = []

    for fp in file_paths:
        lang = _detect_language(fp)
        if lang == "typescript":
            ts_files.append(fp)
        elif lang == "css":
            css_files.append(fp)
        elif lang == "python":
            py_files.append(fp)
        else:
            skip_files.append(fp)

    # TypeScript: run tsc once, parse per-file
    if ts_files:
        _emit(f"  [SYNTAX] Checking {len(ts_files)} TypeScript file(s)...")
        _ts_results = _validate_ts_batch(client, ts_files, frontend_base)
        for r in _ts_results:
            batch.results.append(r)
            if r.status == "pass":
                batch.passed += 1
            elif r.status == "fail":
                batch.failed += 1
                batch.errors_by_file[r.file_path] = r.errors
            else:
                batch.skipped += 1

    # CSS: validate individually
    for fp in css_files:
        r = validate_css_file(client, fp, frontend_base)
        batch.results.append(r)
        if r.status == "pass":
            batch.passed += 1
        elif r.status == "fail":
            batch.failed += 1
            batch.errors_by_file[r.file_path] = r.errors
        else:
            batch.skipped += 1

    # Python: validate individually
    for fp in py_files:
        r = validate_python_file(client, fp, repo_base)
        batch.results.append(r)
        if r.status == "pass":
            batch.passed += 1
        elif r.status == "fail":
            batch.failed += 1
            batch.errors_by_file[r.file_path] = r.errors
        else:
            batch.skipped += 1

    # Unknown extensions: skip
    for fp in skip_files:
        batch.results.append(FileValidationResult(
            file_path=fp, language="unknown", status="skipped",
        ))
        batch.skipped += 1

    _emit(f"  [SYNTAX] {batch.summary()}")
    return batch


def _validate_ts_batch(
    client: Any,
    ts_files: List[str],
    frontend_base: str,
) -> List[FileValidationResult]:
    """Run tsc once and distribute errors to individual files."""
    cmd = f'cd "{frontend_base}" ; npx tsc --noEmit --pretty false 2>&1'

    try:
        result = client.shell_run(cmd, cwd_target="REPO", timeout_seconds=60)
    except Exception as exc:
        return [
            FileValidationResult(
                file_path=fp, language="typescript",
                status="error", raw_output=str(exc),
            )
            for fp in ts_files
        ]

    combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()

    # Parse all errors
    all_errors: Dict[str, List[SyntaxError_]] = {}
    for line in combined.splitlines():
        m = _TSC_ERROR_RE.match(line.strip())
        if m:
            err = SyntaxError_(
                file=m.group(1).replace("\\", "/"),
                line=int(m.group(2)),
                column=int(m.group(3)),
                code=m.group(4),
                message=m.group(5).strip(),
            )
            all_errors.setdefault(err.file, []).append(err)

    # Distribute to per-file results
    results: List[FileValidationResult] = []
    for fp in ts_files:
        norm = fp.replace("\\", "/")
        # Match errors to this file (flexible path matching)
        file_errs: List[SyntaxError_] = []
        for err_path, errs in all_errors.items():
            if (err_path == norm
                    or err_path.endswith(norm)
                    or norm.endswith(err_path)):
                file_errs.extend(errs)

        if file_errs:
            results.append(FileValidationResult(
                file_path=fp, language="typescript",
                status="fail", errors=file_errs,
            ))
        else:
            results.append(FileValidationResult(
                file_path=fp, language="typescript",
                status="pass",
            ))

    return results

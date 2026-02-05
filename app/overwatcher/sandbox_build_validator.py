"""Sandbox Build Validator: Post-POT build verification for sandbox projects.

Detects project types in the Windows Sandbox, runs the appropriate build
commands, parses error output, and returns structured results that the
Overwatcher can reason about for diagnostic/retry loops.

Supported project types:
    - vite_react: Vite + React + TypeScript (orb-desktop)
    - python_backend: Python/FastAPI backend (Orb)

v1.1 (2026-02-04): Path inference from actual file paths
    - Build commands now run in the directory where files were actually written
      (infers project root from POT file paths instead of hardcoded constants)
    - Fixes path mismatch: POT specs may use host paths (D:\\orb-desktop)
      while sandbox defaults are C:\\Orb\\orb-desktop
    - Inferred paths passed through fix execution chain
    - Diagnostic prompt no longer hardcodes C:\\Orb paths
    - Added BOM corruption hint to diagnostic system prompt
v1.0 (2026-02-03): Initial implementation
    - Project type detection from modified file paths
    - Build command execution via sandbox_client.shell_run()
    - Error output parsing (file paths, error types, structured summaries)
    - Multi-project validation (detects which projects were affected)
    - Diagnostic reasoning with LLM-powered fix generation
    - Bounded retry loop (max 3 attempts, configurable)

SAFETY INVARIANT:
    - All I/O goes through Windows Sandbox (sandbox_client)
    - NO direct host filesystem writes
    - If sandbox unavailable → FAIL (no local fallback)
    - Fix types constrained to known safe operations
    - No execution of arbitrary LLM-generated system commands
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    ShellResult,
    get_sandbox_client,
)

logger = logging.getLogger(__name__)

# Build verification
BUILD_VALIDATOR_BUILD_ID = "2026-02-04-v1.1-path-inference"
print(f"[BUILD_VALIDATOR_LOADED] BUILD_ID={BUILD_VALIDATOR_BUILD_ID}")


# =============================================================================
# Configuration (env-driven, following existing patterns)
# =============================================================================

MAX_BUILD_FIX_ATTEMPTS = int(os.getenv("OVERWATCHER_MAX_BUILD_FIX_ATTEMPTS", "3"))
BUILD_VALIDATION_TIMEOUT = int(os.getenv("OVERWATCHER_BUILD_VALIDATION_TIMEOUT", "120"))
BUILD_VALIDATION_ENABLED = os.getenv(
    "OVERWATCHER_BUILD_VALIDATION_ENABLED", "1"
).lower() in {"1", "true", "yes"}

# Truncation limits for LLM prompts (follows evidence.py cost guardrails)
MAX_BUILD_OUTPUT_CHARS = 5000
MAX_DIAGNOSTIC_PROMPT_CHARS = 15000

# Sandbox project paths (matches sandbox/manager.py paths)
SANDBOX_FRONTEND_PATH = r"C:\Orb\orb-desktop"
SANDBOX_BACKEND_PATH = r"C:\Orb\Orb"

# Project type identifiers
PROJECT_VITE_REACT = "vite_react"
PROJECT_PYTHON_BACKEND = "python_backend"
PROJECT_UNKNOWN = "unknown"

# Allowed fix command prefixes (safety constraint — Section 11.5 of job spec)
ALLOWED_FIX_COMMANDS = [
    "npm install",
    "npm ci",
    "npx tsc",
    "npx vite build",
    "python -m py_compile",
    "pip install",
]


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class BuildValidationResult:
    """Structured result from a build validation run."""

    passed: bool
    project_type: str                    # "vite_react", "python_backend", "unknown"
    project_path: str                    # Sandbox path that was validated
    build_command: str                   # What was run
    exit_code: int
    stdout: str                          # Truncated build output
    stderr: str                          # Truncated error output
    error_summary: Optional[str] = None  # Parsed error message
    error_type: Optional[str] = None     # "SyntaxError", "JSONParseError", etc.
    affected_files: List[str] = field(default_factory=list)
    duration_ms: int = 0
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "project_type": self.project_type,
            "project_path": self.project_path,
            "build_command": self.build_command,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:1000],  # Further truncate for dict serialization
            "stderr": self.stderr[:1000],
            "error_summary": self.error_summary,
            "error_type": self.error_type,
            "affected_files": self.affected_files,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }

    @property
    def error_evidence(self) -> str:
        """Format error output for LLM diagnostic prompt.

        Combines stderr and stdout, truncated to MAX_BUILD_OUTPUT_CHARS.
        """
        parts = []
        if self.error_summary:
            parts.append(f"Error Summary: {self.error_summary}")
        if self.error_type:
            parts.append(f"Error Type: {self.error_type}")
        if self.affected_files:
            parts.append(f"Affected Files: {', '.join(self.affected_files)}")
        if self.stderr and self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr}")
        if self.stdout and self.stdout.strip():
            parts.append(f"STDOUT:\n{self.stdout}")
        if self.timed_out:
            parts.append(f"BUILD TIMED OUT after {self.duration_ms}ms")

        combined = "\n\n".join(parts)
        if len(combined) > MAX_BUILD_OUTPUT_CHARS:
            # Keep head and tail
            half = MAX_BUILD_OUTPUT_CHARS // 2
            combined = (
                combined[:half]
                + f"\n\n... [{len(combined) - MAX_BUILD_OUTPUT_CHARS} chars truncated] ...\n\n"
                + combined[-half:]
            )
        return combined


@dataclass
class BuildFixAction:
    """A single corrective action generated by diagnostic reasoning.

    Fix types are constrained to known safe operations:
        - rewrite_file: Re-write a specific file with corrected content
        - run_command: Execute a safe sandbox command (npm install, etc.)
        - revert_file: Restore original content (placeholder for future)
    """

    fix_type: str          # "rewrite_file" | "run_command" | "revert_file"
    file_path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    diagnosis: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fix_type": self.fix_type,
            "file_path": self.file_path,
            "content_length": len(self.content) if self.content else 0,
            "command": self.command,
            "diagnosis": self.diagnosis,
            "rationale": self.rationale,
        }


@dataclass
class DiagnosticResult:
    """Result from LLM diagnostic reasoning about a build failure."""

    diagnosis: str
    root_cause: str
    fixes: List[BuildFixAction] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "root_cause": self.root_cause,
            "fixes": [f.to_dict() for f in self.fixes],
            "confidence": self.confidence,
        }


# =============================================================================
# Project Type Detection
# =============================================================================

def detect_project_from_path(file_path: str) -> Optional[str]:
    """Determine which sandbox project a file belongs to based on its path.

    Args:
        file_path: Sandbox file path (e.g. C:\\Orb\\orb-desktop\\src\\App.tsx)

    Returns:
        PROJECT_VITE_REACT, PROJECT_PYTHON_BACKEND, or None
    """
    # Normalize path separators for comparison
    normalized = file_path.replace("/", "\\").lower()

    if "orb-desktop" in normalized or "orb\\orb-desktop" in normalized:
        return PROJECT_VITE_REACT
    elif "orb\\orb\\" in normalized or normalized.endswith("orb\\orb"):
        return PROJECT_PYTHON_BACKEND
    # Also match paths like C:\Orb\Orb\app\...
    elif re.search(r"c:\\orb\\orb\\", normalized):
        return PROJECT_PYTHON_BACKEND
    return None


def detect_affected_projects(modified_files: List[str]) -> Dict[str, List[str]]:
    """Group modified files by the project they belong to.

    Args:
        modified_files: List of sandbox file paths from POT execution

    Returns:
        Dict mapping project_type -> list of file paths
    """
    projects: Dict[str, List[str]] = {}

    for fpath in modified_files:
        project_type = detect_project_from_path(fpath)
        if project_type:
            projects.setdefault(project_type, []).append(fpath)
        else:
            logger.warning(
                "[build_validator] Could not detect project for path: %s", fpath
            )

    return projects


def _infer_project_path(project_type: str, file_paths: List[str]) -> str:
    """Infer the sandbox project root directory from actual file paths.

    POT specs reference host paths (D:\\orb-desktop) which may differ from
    the sandbox defaults (C:\\Orb\\orb-desktop). The build command must
    run in the directory where files were actually written.

    Falls back to hardcoded SANDBOX_*_PATH defaults if inference fails.
    """
    for fpath in file_paths:
        normalized = fpath.replace("/", "\\")
        lower = normalized.lower()

        if project_type == PROJECT_VITE_REACT:
            # Find "orb-desktop" in path and return everything up to it
            idx = lower.find("orb-desktop")
            if idx >= 0:
                inferred = normalized[:idx + len("orb-desktop")]
                logger.info(
                    "[build_validator] Inferred frontend path: %s (from %s)",
                    inferred, fpath,
                )
                return inferred

        elif project_type == PROJECT_PYTHON_BACKEND:
            # Match pattern like ...\Orb\Orb or ...\Orb\app\...
            match = re.search(
                r"(.*?[\\]Orb[\\]Orb)(?:[\\]|$)", normalized, re.IGNORECASE
            )
            if match:
                inferred = match.group(1)
                logger.info(
                    "[build_validator] Inferred backend path: %s (from %s)",
                    inferred, fpath,
                )
                return inferred

    # Fall back to defaults
    defaults = {
        PROJECT_VITE_REACT: SANDBOX_FRONTEND_PATH,
        PROJECT_PYTHON_BACKEND: SANDBOX_BACKEND_PATH,
    }
    fallback = defaults.get(project_type, "")
    logger.info(
        "[build_validator] Could not infer path for %s, using default: %s",
        project_type, fallback,
    )
    return fallback


async def detect_project_type_from_sandbox(
    client: SandboxClient,
    project_path: str,
) -> str:
    """Detect project type by probing for marker files in the sandbox.

    Args:
        client: SandboxClient instance
        project_path: Sandbox path to check

    Returns:
        Project type string
    """
    try:
        # Check for package.json → Node/Vite project
        cmd = f'Test-Path "{project_path}\\package.json"'
        result = client.shell_run(cmd, timeout_seconds=10)
        if result.stdout and "True" in result.stdout:
            return PROJECT_VITE_REACT

        # Check for pyproject.toml or setup.py → Python project
        cmd = f'(Test-Path "{project_path}\\pyproject.toml") -or (Test-Path "{project_path}\\setup.py") -or (Test-Path "{project_path}\\main.py")'
        result = client.shell_run(cmd, timeout_seconds=10)
        if result.stdout and "True" in result.stdout:
            return PROJECT_PYTHON_BACKEND

    except SandboxError as e:
        logger.warning("[build_validator] Project detection failed: %s", e)

    return PROJECT_UNKNOWN


# =============================================================================
# Error Parsing
# =============================================================================

# Patterns for extracting file paths from build error output
FILE_PATH_PATTERNS = [
    # Vite/Node: absolute Windows paths
    re.compile(r"([A-Z]:\\[^\s:]+\.\w+)", re.IGNORECASE),
    # Vite/Node: relative paths with line numbers
    re.compile(r"([\w./\\-]+\.\w+):(\d+):(\d+)"),
    # Python: File "path", line N
    re.compile(r'File "([^"]+)"', re.IGNORECASE),
]

# Patterns for extracting error types
ERROR_TYPE_PATTERNS = [
    (re.compile(r"SyntaxError", re.IGNORECASE), "SyntaxError"),
    (re.compile(r"TypeError", re.IGNORECASE), "TypeError"),
    (re.compile(r"ReferenceError", re.IGNORECASE), "ReferenceError"),
    (re.compile(r"ModuleNotFoundError", re.IGNORECASE), "ModuleNotFoundError"),
    (re.compile(r"ImportError", re.IGNORECASE), "ImportError"),
    (re.compile(r"Cannot find module", re.IGNORECASE), "ModuleNotFound"),
    (re.compile(r"Failed to load PostCSS config", re.IGNORECASE), "PostCSSConfigError"),
    (re.compile(r"is not valid JSON", re.IGNORECASE), "JSONParseError"),
    (re.compile(r"Unexpected token", re.IGNORECASE), "JSONParseError"),
    (re.compile(r"ERR_MODULE_NOT_FOUND", re.IGNORECASE), "ModuleNotFound"),
    (re.compile(r"TS\d{4}", re.IGNORECASE), "TypeScriptError"),
    (re.compile(r"ENOENT", re.IGNORECASE), "FileNotFound"),
    (re.compile(r"Cannot resolve", re.IGNORECASE), "ResolutionError"),
]


def parse_build_error_output(
    stdout: str,
    stderr: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Parse build error output to extract structured error information.

    Args:
        stdout: Build command stdout
        stderr: Build command stderr

    Returns:
        Tuple of (error_summary, error_type, affected_files)
    """
    combined = f"{stderr}\n{stdout}"
    if not combined.strip():
        return None, None, []

    # Extract error type
    error_type = None
    for pattern, etype in ERROR_TYPE_PATTERNS:
        if pattern.search(combined):
            error_type = etype
            break

    # Extract affected files
    affected_files: List[str] = []
    seen_paths: set = set()
    for pattern in FILE_PATH_PATTERNS:
        for match in pattern.finditer(combined):
            fpath = match.group(1) if match.lastindex else match.group(0)
            # Normalize and deduplicate
            fpath_normalized = fpath.replace("/", "\\").strip()
            if fpath_normalized not in seen_paths and len(fpath_normalized) > 3:
                # Filter out common false positives
                if not fpath_normalized.startswith("http") and "node_modules" not in fpath_normalized:
                    seen_paths.add(fpath_normalized)
                    affected_files.append(fpath_normalized)

    # Extract error summary (first meaningful error line)
    error_summary = None
    for line in combined.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Look for lines that contain error indicators
        if any(
            kw in line.lower()
            for kw in ["error", "failed", "syntaxerror", "typeerror", "cannot"]
        ):
            error_summary = line[:300]  # Truncate long lines
            break

    return error_summary, error_type, affected_files


def _truncate_output(text: str, max_chars: int = MAX_BUILD_OUTPUT_CHARS) -> str:
    """Truncate text, keeping head and tail for diagnostic value."""
    if not text or len(text) <= max_chars:
        return text or ""
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n"
        + text[-half:]
    )


# =============================================================================
# Build Validation
# =============================================================================

async def validate_build(
    client: SandboxClient,
    project_path: str,
    project_type: Optional[str] = None,
    timeout_seconds: int = BUILD_VALIDATION_TIMEOUT,
) -> BuildValidationResult:
    """Run build validation in the sandbox and return a structured result.

    Args:
        client: SandboxClient instance
        project_path: Sandbox project path (e.g. C:\\Orb\\orb-desktop)
        project_type: Project type (auto-detects if None)
        timeout_seconds: Build command timeout

    Returns:
        BuildValidationResult with pass/fail and error details
    """
    start_time = time.time()

    # Auto-detect project type if not provided
    if project_type is None:
        project_type = await detect_project_type_from_sandbox(client, project_path)

    logger.info(
        "[build_validator] Validating build: type=%s, path=%s, timeout=%ds",
        project_type, project_path, timeout_seconds,
    )
    print(
        f"[BUILD_VALIDATOR] Validating: {project_type} at {project_path}"
    )

    # Determine build command
    if project_type == PROJECT_VITE_REACT:
        # Use npx vite build for speed (catches config/import/JSON errors)
        # Redirect stderr to stdout with 2>&1 for unified capture
        build_command = f'cd "{project_path}" ; npx vite build 2>&1'
    elif project_type == PROJECT_PYTHON_BACKEND:
        # Python syntax check on main entry point
        build_command = f'cd "{project_path}" ; python -m py_compile main.py 2>&1'
    else:
        # Unknown project type — run a basic check
        logger.warning(
            "[build_validator] Unknown project type '%s' at %s — skipping",
            project_type, project_path,
        )
        return BuildValidationResult(
            passed=True,  # Fail-safe: unknown projects don't block (Section 11.7)
            project_type=project_type,
            project_path=project_path,
            build_command="(skipped — unknown project type)",
            exit_code=0,
            stdout="",
            stderr="",
            error_summary="Unknown project type — build validation skipped (warning)",
            duration_ms=0,
        )

    # Execute build command in sandbox
    try:
        shell_result: ShellResult = client.shell_run(
            build_command,
            cwd_target="REPO",  # cwd_target doesn't matter since we cd explicitly
            timeout_seconds=timeout_seconds,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Parse error output
        error_summary, error_type, affected_files = parse_build_error_output(
            shell_result.stdout, shell_result.stderr
        )

        passed = shell_result.exit_code == 0 and shell_result.ok

        result = BuildValidationResult(
            passed=passed,
            project_type=project_type,
            project_path=project_path,
            build_command=build_command,
            exit_code=shell_result.exit_code,
            stdout=_truncate_output(shell_result.stdout),
            stderr=_truncate_output(shell_result.stderr),
            error_summary=error_summary,
            error_type=error_type,
            affected_files=affected_files,
            duration_ms=elapsed_ms,
        )

        if passed:
            logger.info(
                "[build_validator] ✓ BUILD PASSED: %s (%dms)",
                project_type, elapsed_ms,
            )
            print(f"[BUILD_VALIDATOR] ✓ PASSED: {project_type} ({elapsed_ms}ms)")
        else:
            logger.warning(
                "[build_validator] ✗ BUILD FAILED: %s, exit=%d, error=%s (%dms)",
                project_type, shell_result.exit_code, error_type, elapsed_ms,
            )
            print(
                f"[BUILD_VALIDATOR] ✗ FAILED: {project_type} "
                f"(exit={shell_result.exit_code}, {error_type}, {elapsed_ms}ms)"
            )

        return result

    except SandboxError as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "[build_validator] Sandbox error during build validation: %s", e
        )
        return BuildValidationResult(
            passed=False,
            project_type=project_type,
            project_path=project_path,
            build_command=build_command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            error_summary=f"Sandbox communication error: {e}",
            error_type="SandboxError",
            duration_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.exception(
            "[build_validator] Unexpected error during build validation: %s", e
        )
        return BuildValidationResult(
            passed=False,
            project_type=project_type,
            project_path=project_path,
            build_command=build_command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            error_summary=f"Unexpected validation error: {e}",
            error_type="InternalError",
            duration_ms=elapsed_ms,
            timed_out="timeout" in str(e).lower(),
        )


async def validate_all_affected_projects(
    client: SandboxClient,
    modified_files: List[str],
) -> List[BuildValidationResult]:
    """Determine which projects were affected by POT execution and validate each.

    Args:
        client: SandboxClient instance
        modified_files: List of file paths written by the POT executor

    Returns:
        List of BuildValidationResult (one per affected project)
    """
    if not modified_files:
        logger.info("[build_validator] No modified files — skipping validation")
        return []

    # Detect affected projects from file paths
    affected = detect_affected_projects(modified_files)

    if not affected:
        logger.warning(
            "[build_validator] Could not determine projects from paths: %s",
            modified_files,
        )
        return []

    logger.info(
        "[build_validator] Affected projects: %s",
        {k: len(v) for k, v in affected.items()},
    )

    results: List[BuildValidationResult] = []

    for project_type, files in affected.items():
        # Infer project path from actual file paths (handles host vs sandbox paths)
        project_path = _infer_project_path(project_type, files)
        if not project_path:
            logger.warning(
                "[build_validator] No sandbox path for project type: %s", project_type
            )
            continue

        result = await validate_build(
            client=client,
            project_path=project_path,
            project_type=project_type,
        )
        results.append(result)

    return results


# =============================================================================
# Diagnostic Reasoning (LLM-powered)
# =============================================================================

DIAGNOSTIC_SYSTEM_PROMPT = """You are a build error diagnostic expert for a Vite + React + Electron frontend and a Python FastAPI backend.

You are given:
1. The original spec (what was intended)
2. The POT execution results (what files were changed)
3. The build error output (what went wrong)

YOUR TASK: Diagnose the root cause and generate a fix.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{{
  "diagnosis": "One sentence describing what went wrong",
  "root_cause": "encoding|syntax|import|config|dependency|type_error|other",
  "confidence": 0.0-1.0,
  "fixes": [
    {{
      "fix_type": "rewrite_file|run_command|revert_file",
      "file_path": "<use same absolute path from error output>",
      "content": "Full corrected file content (for rewrite_file only)",
      "command": "npm install (for run_command only)",
      "rationale": "Why this fix addresses the root cause"
    }}
  ]
}}

RULES:
1. fix_type must be one of: rewrite_file, run_command, revert_file
2. For rewrite_file: provide the COMPLETE corrected file content (not a diff)
3. For run_command: ONLY these commands are allowed: npm install, npm ci, npx tsc, npx vite build, python -m py_compile, pip install
4. For revert_file: provide the file_path to revert (content from POT executor backup)
5. All file paths must use the SAME absolute paths shown in the error output and modified files list (do NOT change drive letters or path prefixes)
6. Do NOT suggest commands that delete files, modify system config, or access the network beyond npm
7. Focus on the MINIMAL fix needed — do not rewrite files that weren't part of the error
8. If the error mentions missing node_modules, suggest run_command with "npm install"
9. Output ONLY JSON — no markdown, no explanations outside the JSON
10. If the error is a UTF-8 BOM corruption (unexpected token at start of JSON), rewrite the affected file with clean UTF-8 content (no BOM)
"""

DIAGNOSTIC_USER_PROMPT = """## Build Error Diagnostic

### Spec Intent
{spec_summary}

### Files Modified by POT Execution
{modified_files_summary}

### Build Error Output
```
{build_error_output}
```

### Build Details
- Project Type: {project_type}
- Build Command: {build_command}
- Exit Code: {exit_code}
- Error Type: {error_type}
- Affected Files: {affected_files}

### Fix Attempt
This is fix attempt {attempt} of {max_attempts}.
{previous_fix_summary}

Diagnose the root cause and provide the minimal fix."""


async def diagnose_build_failure(
    *,
    llm_call_fn: Callable,
    spec_content: str,
    pot_result: Dict[str, Any],
    build_results: List[BuildValidationResult],
    attempt: int,
    max_attempts: int = MAX_BUILD_FIX_ATTEMPTS,
    previous_fixes: Optional[List[Dict]] = None,
) -> DiagnosticResult:
    """Ask the LLM to diagnose a build failure and generate a fix.

    Args:
        llm_call_fn: Async LLM call function (from overwatcher_stream)
        spec_content: Original spec content (what was intended)
        pot_result: POT execution results dict
        build_results: Failed build validation results
        attempt: Current fix attempt number (1-based)
        max_attempts: Maximum fix attempts
        previous_fixes: List of previously attempted fixes (for context)

    Returns:
        DiagnosticResult with diagnosis and fix actions
    """
    # Assemble evidence for the LLM
    modified_files = pot_result.get("artifacts_written", [])
    modified_files_summary = "\n".join(f"- {f}" for f in modified_files) or "None"

    # Combine build error evidence from all failed builds
    build_error_parts = []
    for br in build_results:
        if not br.passed:
            build_error_parts.append(
                f"[{br.project_type} at {br.project_path}]\n{br.error_evidence}"
            )
    build_error_output = "\n\n---\n\n".join(build_error_parts)

    # Truncate for cost control
    spec_summary = spec_content[:2000] if spec_content else "N/A"
    build_error_output = _truncate_output(build_error_output, MAX_BUILD_OUTPUT_CHARS)

    # Previous fix context
    previous_fix_summary = ""
    if previous_fixes:
        fix_descriptions = []
        for pf in previous_fixes:
            fix_descriptions.append(
                f"  - Attempt {pf.get('attempt', '?')}: {pf.get('diagnosis', 'N/A')} "
                f"(fix_type={pf.get('fix_type', 'N/A')})"
            )
        previous_fix_summary = (
            "Previous fixes attempted (did NOT resolve the issue):\n"
            + "\n".join(fix_descriptions)
        )

    # Use first failed build for details
    first_failed = next((br for br in build_results if not br.passed), None)

    user_prompt = DIAGNOSTIC_USER_PROMPT.format(
        spec_summary=spec_summary,
        modified_files_summary=modified_files_summary,
        build_error_output=build_error_output,
        project_type=first_failed.project_type if first_failed else "unknown",
        build_command=first_failed.build_command if first_failed else "N/A",
        exit_code=first_failed.exit_code if first_failed else -1,
        error_type=first_failed.error_type if first_failed else "unknown",
        affected_files=", ".join(first_failed.affected_files) if first_failed else "N/A",
        attempt=attempt,
        max_attempts=max_attempts,
        previous_fix_summary=previous_fix_summary,
    )

    logger.info(
        "[build_validator] Calling LLM for diagnostic (attempt %d/%d, prompt ~%d chars)",
        attempt, max_attempts, len(user_prompt),
    )

    # Call LLM — use the same pattern as overwatcher.py
    try:
        # Import stage_models for provider/model config
        from app.llm.stage_models import get_overwatcher_config

        config = get_overwatcher_config()

        messages = [
            {"role": "system", "content": DIAGNOSTIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_result = await llm_call_fn(
            provider_id=config.provider,
            model_id=config.model,
            messages=messages,
            max_tokens=config.max_output_tokens,
        )

        raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)

    except Exception as e:
        logger.error("[build_validator] LLM call failed for diagnostic: %s", e)
        return DiagnosticResult(
            diagnosis=f"LLM diagnostic call failed: {e}",
            root_cause="llm_error",
            confidence=0.0,
            raw_response="",
        )

    # Parse LLM response
    return _parse_diagnostic_response(raw_text)


def _parse_diagnostic_response(raw_text: str) -> DiagnosticResult:
    """Parse the LLM's diagnostic response into structured data.

    Handles: raw JSON, JSON in code fences, partial/malformed JSON.
    """
    if not raw_text:
        return DiagnosticResult(
            diagnosis="Empty response from diagnostic LLM",
            root_cause="llm_error",
        )

    text = raw_text.strip()

    # Try to extract JSON from code fence
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object with brace matching
        start = text.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i, char in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

    if not data or not isinstance(data, dict):
        logger.warning(
            "[build_validator] Could not parse diagnostic response: %s",
            raw_text[:300],
        )
        return DiagnosticResult(
            diagnosis=f"Could not parse diagnostic response: {raw_text[:200]}",
            root_cause="parse_error",
            raw_response=raw_text,
        )

    # Build fix actions
    fixes: List[BuildFixAction] = []
    for fix_data in data.get("fixes", []):
        fix_type = fix_data.get("fix_type", "")

        # Validate fix type
        if fix_type not in ("rewrite_file", "run_command", "revert_file"):
            logger.warning(
                "[build_validator] Unknown fix_type '%s' — skipping", fix_type
            )
            continue

        # Validate run_command safety
        if fix_type == "run_command":
            command = fix_data.get("command", "")
            if not _is_safe_command(command):
                logger.warning(
                    "[build_validator] Unsafe command rejected: '%s'", command
                )
                continue

        fixes.append(
            BuildFixAction(
                fix_type=fix_type,
                file_path=fix_data.get("file_path"),
                content=fix_data.get("content"),
                command=fix_data.get("command"),
                diagnosis=data.get("diagnosis", ""),
                rationale=fix_data.get("rationale", ""),
            )
        )

    return DiagnosticResult(
        diagnosis=data.get("diagnosis", "No diagnosis provided"),
        root_cause=data.get("root_cause", "unknown"),
        fixes=fixes,
        confidence=float(data.get("confidence", 0.0)),
        raw_response=raw_text,
    )


def _is_safe_command(command: str) -> bool:
    """Check if a command is in the allowed list (safety constraint).

    Only permits known safe operations like npm install, pip install, etc.
    """
    if not command:
        return False
    cmd_lower = command.strip().lower()
    return any(cmd_lower.startswith(prefix.lower()) for prefix in ALLOWED_FIX_COMMANDS)


# =============================================================================
# Fix Execution
# =============================================================================

async def execute_build_fix(
    client: SandboxClient,
    fix_action: BuildFixAction,
    inferred_frontend_path: str = SANDBOX_FRONTEND_PATH,
    inferred_backend_path: str = SANDBOX_BACKEND_PATH,
) -> Dict[str, Any]:
    """Execute a single build fix action in the sandbox.

    Args:
        client: SandboxClient instance
        fix_action: The fix action to execute
        inferred_frontend_path: Actual frontend project path (from file path inference)
        inferred_backend_path: Actual backend project path (from file path inference)

    Returns:
        Dict with execution result: {"success": bool, "details": str}

    SAFETY INVARIANT:
        - All writes go through sandbox_client
        - Commands are validated against ALLOWED_FIX_COMMANDS
        - No host filesystem access
    """
    logger.info(
        "[build_validator] Executing fix: type=%s, file=%s, cmd=%s",
        fix_action.fix_type,
        fix_action.file_path,
        fix_action.command,
    )

    try:
        if fix_action.fix_type == "rewrite_file":
            if not fix_action.file_path or not fix_action.content:
                return {
                    "success": False,
                    "details": "rewrite_file requires file_path and content",
                }

            # Use the same BOM-safe write method as pot_spec_executor
            import base64

            encoded = base64.b64encode(
                fix_action.content.encode("utf-8")
            ).decode("ascii")
            cmd = (
                f'$bytes = [System.Convert]::FromBase64String("{encoded}"); '
                f'[System.IO.File]::WriteAllBytes("{fix_action.file_path}", $bytes)'
            )
            result = client.shell_run(cmd, timeout_seconds=60)

            if result.stderr and result.stderr.strip():
                return {
                    "success": False,
                    "details": f"Write stderr: {result.stderr[:300]}",
                }

            # Verify the write
            verify_cmd = f'(Get-Item "{fix_action.file_path}").Length'
            verify_result = client.shell_run(verify_cmd, timeout_seconds=10)

            logger.info(
                "[build_validator] ✓ File rewritten: %s (%s bytes)",
                fix_action.file_path,
                verify_result.stdout.strip() if verify_result.stdout else "?",
            )
            return {
                "success": True,
                "details": f"Rewrote {fix_action.file_path}",
            }

        elif fix_action.fix_type == "run_command":
            if not fix_action.command:
                return {
                    "success": False,
                    "details": "run_command requires command",
                }

            if not _is_safe_command(fix_action.command):
                return {
                    "success": False,
                    "details": f"Command rejected (not in allowed list): {fix_action.command}",
                }

            # Determine project path from context
            # Commands like "npm install" need to run in the right directory
            project_path = inferred_frontend_path
            if "python" in fix_action.command or "pip" in fix_action.command:
                project_path = inferred_backend_path

            full_command = f'cd "{project_path}" ; {fix_action.command} 2>&1'
            result = client.shell_run(full_command, timeout_seconds=BUILD_VALIDATION_TIMEOUT)

            success = result.exit_code == 0
            logger.info(
                "[build_validator] Command result: exit=%d, ok=%s",
                result.exit_code, result.ok,
            )
            return {
                "success": success,
                "details": f"exit_code={result.exit_code}, stdout={result.stdout[:200]}",
            }

        elif fix_action.fix_type == "revert_file":
            # Placeholder: revert would need backup content from POT executor
            logger.warning(
                "[build_validator] revert_file not yet implemented — skipping"
            )
            return {
                "success": False,
                "details": "revert_file not yet implemented",
            }

        else:
            return {
                "success": False,
                "details": f"Unknown fix_type: {fix_action.fix_type}",
            }

    except SandboxError as e:
        logger.error("[build_validator] Fix execution sandbox error: %s", e)
        return {"success": False, "details": f"Sandbox error: {e}"}
    except Exception as e:
        logger.exception("[build_validator] Fix execution error: %s", e)
        return {"success": False, "details": f"Error: {e}"}


async def execute_all_fixes(
    client: SandboxClient,
    diagnostic: DiagnosticResult,
    inferred_frontend_path: str = SANDBOX_FRONTEND_PATH,
    inferred_backend_path: str = SANDBOX_BACKEND_PATH,
) -> List[Dict[str, Any]]:
    """Execute all fix actions from a diagnostic result.

    Args:
        client: SandboxClient instance
        diagnostic: DiagnosticResult containing fix actions
        inferred_frontend_path: Actual frontend project path
        inferred_backend_path: Actual backend project path

    Returns:
        List of execution result dicts
    """
    results: List[Dict[str, Any]] = []

    for fix in diagnostic.fixes:
        result = await execute_build_fix(
            client, fix,
            inferred_frontend_path=inferred_frontend_path,
            inferred_backend_path=inferred_backend_path,
        )
        results.append({
            "fix_type": fix.fix_type,
            "file_path": fix.file_path,
            "command": fix.command,
            **result,
        })

        if not result["success"]:
            logger.warning(
                "[build_validator] Fix failed: %s — %s",
                fix.fix_type, result["details"],
            )
            # Continue trying other fixes (they may be independent)

    return results


# =============================================================================
# Full Validation + Retry Loop (called from overwatcher_command.py)
# =============================================================================

async def run_build_validation_loop(
    *,
    client: SandboxClient,
    modified_files: List[str],
    spec_content: str,
    pot_result: Dict[str, Any],
    llm_call_fn: Callable,
    add_trace: Callable,
) -> Tuple[bool, List[BuildValidationResult], List[Dict[str, Any]]]:
    """Run the full build validation + diagnostic/retry loop.

    This is the main entry point called from overwatcher_command.py
    after POT execution succeeds.

    Args:
        client: SandboxClient instance
        modified_files: Files written by POT executor
        spec_content: Original spec content
        pot_result: POT execution result dict
        llm_call_fn: Async LLM call function
        add_trace: Trace logging callback

    Returns:
        Tuple of:
            - passed: bool (final result)
            - build_results: List of final BuildValidationResults
            - fix_history: List of fix attempt dicts (for evidence)
    """
    if not BUILD_VALIDATION_ENABLED:
        logger.info("[build_validator] Build validation disabled via env — skipping")
        add_trace("BUILD_VALIDATION", "disabled", {
            "reason": "OVERWATCHER_BUILD_VALIDATION_ENABLED=0",
        })
        return True, [], []

    # Initial build validation
    logger.info(
        "[build_validator] Starting build validation for %d modified files",
        len(modified_files),
    )
    add_trace("BUILD_VALIDATION", "started", {
        "modified_files_count": len(modified_files),
        "modified_files": modified_files[:20],  # Truncate for trace
    })

    build_results = await validate_all_affected_projects(client, modified_files)

    if not build_results:
        # No projects detected — treat as warning, not failure
        logger.warning(
            "[build_validator] No affected projects detected — treating as pass (warning)"
        )
        add_trace("BUILD_VALIDATION", "warning_no_projects", {
            "modified_files": modified_files[:10],
        })
        return True, [], []

    # Pre-compute inferred project paths for fix execution later
    affected = detect_affected_projects(modified_files)
    inferred_frontend = _infer_project_path(
        PROJECT_VITE_REACT,
        affected.get(PROJECT_VITE_REACT, []),
    ) if PROJECT_VITE_REACT in affected else SANDBOX_FRONTEND_PATH
    inferred_backend = _infer_project_path(
        PROJECT_PYTHON_BACKEND,
        affected.get(PROJECT_PYTHON_BACKEND, []),
    ) if PROJECT_PYTHON_BACKEND in affected else SANDBOX_BACKEND_PATH

    # Check if all builds passed
    all_passed = all(r.passed for r in build_results)
    if all_passed:
        add_trace("BUILD_VALIDATION", "passed", {
            "projects_validated": [
                {"type": r.project_type, "duration_ms": r.duration_ms}
                for r in build_results
            ],
        })
        logger.info("[build_validator] ✓ All builds passed on first try")
        return True, build_results, []

    # Build failed — enter diagnostic/retry loop
    logger.warning(
        "[build_validator] Build failed — entering diagnostic loop (max %d attempts)",
        MAX_BUILD_FIX_ATTEMPTS,
    )
    add_trace("BUILD_VALIDATION", "failed_entering_retry", {
        "failed_projects": [
            r.to_dict() for r in build_results if not r.passed
        ],
    })

    fix_history: List[Dict[str, Any]] = []
    previous_fixes: List[Dict] = []

    for attempt in range(1, MAX_BUILD_FIX_ATTEMPTS + 1):
        logger.info(
            "[build_validator] === Fix attempt %d/%d ===",
            attempt, MAX_BUILD_FIX_ATTEMPTS,
        )
        add_trace("BUILD_FIX_ATTEMPT", "started", {"attempt": attempt})

        # Diagnose
        diagnostic = await diagnose_build_failure(
            llm_call_fn=llm_call_fn,
            spec_content=spec_content,
            pot_result=pot_result,
            build_results=build_results,
            attempt=attempt,
            previous_fixes=previous_fixes,
        )

        logger.info(
            "[build_validator] Diagnosis: %s (root_cause=%s, confidence=%.2f, fixes=%d)",
            diagnostic.diagnosis,
            diagnostic.root_cause,
            diagnostic.confidence,
            len(diagnostic.fixes),
        )

        add_trace("BUILD_FIX_DIAGNOSIS", "complete", {
            "attempt": attempt,
            "diagnosis": diagnostic.diagnosis,
            "root_cause": diagnostic.root_cause,
            "confidence": diagnostic.confidence,
            "fix_count": len(diagnostic.fixes),
        })

        if not diagnostic.fixes:
            logger.warning(
                "[build_validator] No fixes suggested by LLM — attempt %d failed",
                attempt,
            )
            add_trace("BUILD_FIX_ATTEMPT", "no_fixes", {"attempt": attempt})
            fix_history.append({
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
                "root_cause": diagnostic.root_cause,
                "fixes": [],
                "result": "no_fixes_suggested",
            })
            previous_fixes.append({
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
                "fix_type": "none",
            })
            continue

        # Execute fixes (using inferred project paths for correct directory)
        fix_results = await execute_all_fixes(
            client, diagnostic,
            inferred_frontend_path=inferred_frontend,
            inferred_backend_path=inferred_backend,
        )

        add_trace("BUILD_FIX_EXECUTED", "complete", {
            "attempt": attempt,
            "fixes_executed": len(fix_results),
            "fixes_succeeded": sum(1 for r in fix_results if r.get("success")),
        })

        fix_history.append({
            "attempt": attempt,
            "diagnosis": diagnostic.diagnosis,
            "root_cause": diagnostic.root_cause,
            "fixes": [f.to_dict() for f in diagnostic.fixes],
            "fix_results": fix_results,
        })

        for fix_data in diagnostic.fixes:
            previous_fixes.append({
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
                "fix_type": fix_data.fix_type,
            })

        # Re-validate build
        build_results = await validate_all_affected_projects(client, modified_files)
        all_passed = all(r.passed for r in build_results)

        if all_passed:
            logger.info(
                "[build_validator] ✓ Build passed after fix attempt %d", attempt
            )
            add_trace("BUILD_FIX_SUCCESS", "passed", {
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
            })
            return True, build_results, fix_history

        logger.warning(
            "[build_validator] Build still failing after attempt %d", attempt
        )
        add_trace("BUILD_FIX_ATTEMPT", "still_failing", {
            "attempt": attempt,
            "remaining_errors": [
                r.error_summary for r in build_results if not r.passed
            ],
        })

    # Exhausted all retries
    logger.error(
        "[build_validator] ✗ Build validation FAILED after %d fix attempts",
        MAX_BUILD_FIX_ATTEMPTS,
    )
    add_trace("BUILD_FIX_EXHAUSTED", "failed", {
        "total_attempts": MAX_BUILD_FIX_ATTEMPTS,
        "final_errors": [r.to_dict() for r in build_results if not r.passed],
        "fix_history": fix_history,
    })

    return False, build_results, fix_history


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "MAX_BUILD_FIX_ATTEMPTS",
    "BUILD_VALIDATION_TIMEOUT",
    "BUILD_VALIDATION_ENABLED",
    "SANDBOX_FRONTEND_PATH",
    "SANDBOX_BACKEND_PATH",
    # Data classes
    "BuildValidationResult",
    "BuildFixAction",
    "DiagnosticResult",
    # Detection
    "detect_project_from_path",
    "detect_affected_projects",
    "detect_project_type_from_sandbox",
    "_infer_project_path",
    # Parsing
    "parse_build_error_output",
    # Validation
    "validate_build",
    "validate_all_affected_projects",
    # Diagnostic
    "diagnose_build_failure",
    "execute_build_fix",
    "execute_all_fixes",
    # Main entry point
    "run_build_validation_loop",
]

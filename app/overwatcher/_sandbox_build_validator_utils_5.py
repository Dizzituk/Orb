from __future__ import annotations
import logging
import re
import time
from app.overwatcher._sandbox_build_validator_utils_3 import MAX_BUILD_OUTPUT_CHARS, _is_safe_command, _truncate_output, parse_build_error_output
from app.overwatcher._sandbox_build_validator_utils_4 import BUILD_VALIDATION_TIMEOUT, BuildFixAction, detect_project_type_from_sandbox
from app.overwatcher.sandbox_client import SandboxClient, SandboxError, ShellResult
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
SANDBOX_FRONTEND_PATH = r"C:\Orb\orb-desktop"
SANDBOX_BACKEND_PATH = r"C:\Orb\Orb"


PROJECT_VITE_REACT = "vite_react"

PROJECT_PYTHON_BACKEND = "python_backend"

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

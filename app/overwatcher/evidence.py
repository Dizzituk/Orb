# FILE: app/overwatcher/evidence.py
"""Evidence bundle builder for Overwatcher.

Spec v2.3 §9.3: To control cost and avoid drift, the Overwatcher reads
an evidence bundle rather than the whole repo.

Evidence bundle contents:
- Changed file list + intent per file (short)
- Relevant diffs or small excerpts for touched areas only
- Test results (unit/integration) and failing test names
- Lint/typecheck results (if enabled)
- Full stack traces + environment context
- Spec/lock identifiers and stage_run_id

Spec v2.3 §9.6 Cost Guardrails:
- Overwatcher max input tokens: 120,000
- Never send large data/ or scan dumps
- Prefer diffs + test logs over full-file re-prints
- Do not re-send unchanged context
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.overwatcher.error_signature import ErrorSignature


# =============================================================================
# Configuration
# =============================================================================

# Spec §9.6: Hard limit for Overwatcher input
MAX_INPUT_TOKENS = 120_000
CHARS_PER_TOKEN = 4  # Rough estimate

# Limits for individual components
MAX_DIFF_LINES = 200
MAX_TEST_OUTPUT_LINES = 100
MAX_LINT_OUTPUT_LINES = 50
MAX_STACK_TRACE_LINES = 50
MAX_FILE_EXCERPT_LINES = 30


@dataclass
class FileChange:
    """A changed file with intent description."""
    
    path: str
    action: str  # "add" | "modify" | "delete"
    intent: str  # Short description of what changed
    diff_excerpt: Optional[str] = None  # Truncated diff if available
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "action": self.action,
            "intent": self.intent,
            "diff_excerpt": self.diff_excerpt,
        }


@dataclass
class TestResult:
    """Test execution result."""
    
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failing_tests: List[str] = field(default_factory=list)
    output_excerpt: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0
    
    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "failing_tests": self.failing_tests,
            "output_excerpt": self.output_excerpt,
        }


@dataclass
class LintResult:
    """Lint/typecheck result."""
    
    tool: str  # "ruff" | "mypy" | "pyright"
    errors: int = 0
    warnings: int = 0
    output_excerpt: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.errors == 0
    
    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "errors": self.errors,
            "warnings": self.warnings,
            "output_excerpt": self.output_excerpt,
        }


@dataclass
class EnvironmentContext:
    """Runtime environment information."""
    
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    os_name: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.release())
    cwd: str = field(default_factory=os.getcwd)
    
    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "cwd": self.cwd,
        }


@dataclass
class EvidenceBundle:
    """Complete evidence bundle for Overwatcher.
    
    This is the ONLY input the Overwatcher should receive.
    Never send full repo contents.
    """
    
    # Identity
    job_id: str
    chunk_id: str
    stage_run_id: str
    spec_id: str
    spec_hash: str
    
    # Strike context
    strike_number: int
    previous_error_signature: Optional[ErrorSignature] = None
    
    # Changes
    file_changes: List[FileChange] = field(default_factory=list)
    
    # Results
    test_result: Optional[TestResult] = None
    lint_results: List[LintResult] = field(default_factory=list)
    
    # Error details
    error_output: Optional[str] = None
    stack_trace: Optional[str] = None
    current_error_signature: Optional[ErrorSignature] = None
    
    # Context
    environment: EnvironmentContext = field(default_factory=EnvironmentContext)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Chunk metadata
    chunk_title: Optional[str] = None
    chunk_objective: Optional[str] = None
    verification_commands: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "chunk_id": self.chunk_id,
            "stage_run_id": self.stage_run_id,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "strike_number": self.strike_number,
            "previous_error_signature": self.previous_error_signature.to_dict() if self.previous_error_signature else None,
            "file_changes": [fc.to_dict() for fc in self.file_changes],
            "test_result": self.test_result.to_dict() if self.test_result else None,
            "lint_results": [lr.to_dict() for lr in self.lint_results],
            "error_output": self.error_output,
            "stack_trace": self.stack_trace,
            "current_error_signature": self.current_error_signature.to_dict() if self.current_error_signature else None,
            "environment": self.environment.to_dict(),
            "timestamp": self.timestamp,
            "chunk_title": self.chunk_title,
            "chunk_objective": self.chunk_objective,
            "verification_commands": self.verification_commands,
        }
    
    def to_prompt_text(self) -> str:
        """Convert bundle to text for Overwatcher prompt.
        
        Optimized for token efficiency.
        """
        lines = [
            "# EVIDENCE BUNDLE",
            "",
            "## Identity",
            f"- Job: {self.job_id}",
            f"- Chunk: {self.chunk_id}",
            f"- Stage Run: {self.stage_run_id}",
            f"- Spec: {self.spec_id} (hash: {self.spec_hash[:16]}...)",
            f"- Strike: {self.strike_number}/3",
            "",
        ]
        
        if self.chunk_title:
            lines.extend([
                "## Chunk Context",
                f"- Title: {self.chunk_title}",
                f"- Objective: {self.chunk_objective or 'N/A'}",
                "",
            ])
        
        if self.file_changes:
            lines.extend([
                "## File Changes",
            ])
            for fc in self.file_changes:
                lines.append(f"- [{fc.action}] {fc.path}: {fc.intent}")
                if fc.diff_excerpt:
                    lines.append(f"  ```diff\n{fc.diff_excerpt}\n  ```")
            lines.append("")
        
        if self.test_result:
            tr = self.test_result
            lines.extend([
                "## Test Results",
                f"- Passed: {tr.passed}, Failed: {tr.failed}, Errors: {tr.errors}, Skipped: {tr.skipped}",
            ])
            if tr.failing_tests:
                lines.append(f"- Failing: {', '.join(tr.failing_tests[:10])}")
            if tr.output_excerpt:
                lines.append(f"```\n{tr.output_excerpt}\n```")
            lines.append("")
        
        if self.lint_results:
            lines.append("## Lint Results")
            for lr in self.lint_results:
                status = "PASS" if lr.success else "FAIL"
                lines.append(f"- {lr.tool}: {status} ({lr.errors} errors, {lr.warnings} warnings)")
                if lr.output_excerpt and not lr.success:
                    lines.append(f"  ```\n{lr.output_excerpt}\n  ```")
            lines.append("")
        
        if self.stack_trace:
            lines.extend([
                "## Stack Trace",
                f"```\n{self.stack_trace}\n```",
                "",
            ])
        
        if self.current_error_signature:
            sig = self.current_error_signature
            lines.extend([
                "## Error Signature",
                f"- Type: {sig.exception_type}",
                f"- Test: {sig.failing_test_name or 'N/A'}",
                f"- Module: {sig.module_path or 'N/A'}",
                f"- Hash: {sig.signature_hash}",
            ])
            if self.previous_error_signature:
                match = "SAME" if sig.matches(self.previous_error_signature) else "DIFFERENT"
                lines.append(f"- vs Previous: {match}")
            lines.append("")
        
        if self.verification_commands:
            lines.extend([
                "## Verification Commands",
            ])
            for cmd in self.verification_commands:
                lines.append(f"- `{cmd}`")
            lines.append("")
        
        lines.extend([
            "## Environment",
            f"- Python: {self.environment.python_version}",
            f"- OS: {self.environment.os_name} {self.environment.os_version}",
            f"- CWD: {self.environment.cwd}",
        ])
        
        return "\n".join(lines)
    
    def estimate_tokens(self) -> int:
        """Estimate token count for this bundle."""
        text = self.to_prompt_text()
        return len(text) // CHARS_PER_TOKEN


# =============================================================================
# Builder Functions
# =============================================================================

def truncate_output(output: str, max_lines: int) -> str:
    """Truncate output to max lines, keeping head and tail."""
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    
    head_lines = max_lines // 2
    tail_lines = max_lines - head_lines - 1
    
    truncated = lines[:head_lines]
    truncated.append(f"... [{len(lines) - max_lines} lines truncated] ...")
    truncated.extend(lines[-tail_lines:])
    
    return "\n".join(truncated)


def get_git_diff(repo_path: str, files: List[str], max_lines: int = MAX_DIFF_LINES) -> Dict[str, str]:
    """Get git diff for specific files.
    
    Returns dict of path -> diff excerpt.
    """
    diffs = {}
    
    for filepath in files:
        try:
            result = subprocess.run(
                ["git", "diff", "--", filepath],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.stdout:
                diffs[filepath] = truncate_output(result.stdout, max_lines // len(files))
        except Exception:
            pass
    
    return diffs


def run_tests(
    repo_path: str,
    test_commands: List[str],
    timeout: int = 120,
) -> TestResult:
    """Run test commands and parse results.
    
    Returns TestResult with parsed output.
    """
    result = TestResult()
    outputs = []
    
    for cmd in test_commands:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output = proc.stdout + proc.stderr
            outputs.append(output)
            
            # Parse pytest output
            if "passed" in output.lower():
                import re
                match = re.search(r"(\d+) passed", output)
                if match:
                    result.passed += int(match.group(1))
            
            if "failed" in output.lower():
                import re
                match = re.search(r"(\d+) failed", output)
                if match:
                    result.failed += int(match.group(1))
                
                # Extract failing test names
                failed_matches = re.findall(r"FAILED\s+([\w/\\]+\.py::\w+)", output)
                result.failing_tests.extend(failed_matches)
            
            if "error" in output.lower():
                import re
                match = re.search(r"(\d+) error", output)
                if match:
                    result.errors += int(match.group(1))
                    
        except subprocess.TimeoutExpired:
            outputs.append(f"[TIMEOUT after {timeout}s]")
            result.errors += 1
        except Exception as e:
            outputs.append(f"[ERROR: {e}]")
            result.errors += 1
    
    result.output_excerpt = truncate_output("\n".join(outputs), MAX_TEST_OUTPUT_LINES)
    return result


def run_lint(
    repo_path: str,
    tool: str = "ruff",
    files: Optional[List[str]] = None,
) -> LintResult:
    """Run lint tool and parse results."""
    result = LintResult(tool=tool)
    
    try:
        if tool == "ruff":
            cmd = ["ruff", "check"]
            if files:
                cmd.extend(files)
            else:
                cmd.append(".")
        elif tool == "mypy":
            cmd = ["mypy"]
            if files:
                cmd.extend(files)
            else:
                cmd.append(".")
        else:
            return result
        
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        output = proc.stdout + proc.stderr
        
        # Count errors/warnings
        import re
        error_count = len(re.findall(r"error:", output, re.IGNORECASE))
        warning_count = len(re.findall(r"warning:", output, re.IGNORECASE))
        
        result.errors = error_count or (1 if proc.returncode != 0 else 0)
        result.warnings = warning_count
        result.output_excerpt = truncate_output(output, MAX_LINT_OUTPUT_LINES)
        
    except Exception as e:
        result.output_excerpt = f"[ERROR running {tool}: {e}]"
        result.errors = 1
    
    return result


def build_evidence_bundle(
    *,
    job_id: str,
    chunk_id: str,
    stage_run_id: str,
    spec_id: str,
    spec_hash: str,
    strike_number: int,
    file_changes: List[FileChange],
    repo_path: str,
    verification_commands: List[str],
    previous_error_signature: Optional[ErrorSignature] = None,
    chunk_title: Optional[str] = None,
    chunk_objective: Optional[str] = None,
    run_verification: bool = True,
) -> EvidenceBundle:
    """Build a complete evidence bundle for Overwatcher.
    
    Args:
        job_id: Job UUID
        chunk_id: Chunk ID
        stage_run_id: Stage run UUID
        spec_id: Spec ID
        spec_hash: Spec hash
        strike_number: Current strike (1, 2, or 3)
        file_changes: List of changed files
        repo_path: Path to repository
        verification_commands: Commands to run for verification
        previous_error_signature: Signature from previous strike (if any)
        chunk_title: Chunk title for context
        chunk_objective: Chunk objective for context
        run_verification: Whether to run tests/lint
    
    Returns:
        EvidenceBundle ready for Overwatcher
    """
    from app.overwatcher.error_signature import compute_error_signature
    
    bundle = EvidenceBundle(
        job_id=job_id,
        chunk_id=chunk_id,
        stage_run_id=stage_run_id,
        spec_id=spec_id,
        spec_hash=spec_hash,
        strike_number=strike_number,
        previous_error_signature=previous_error_signature,
        file_changes=file_changes,
        chunk_title=chunk_title,
        chunk_objective=chunk_objective,
        verification_commands=verification_commands,
    )
    
    # Get diffs for changed files
    changed_paths = [fc.path for fc in file_changes if fc.action in ("add", "modify")]
    if changed_paths:
        diffs = get_git_diff(repo_path, changed_paths)
        for fc in file_changes:
            if fc.path in diffs:
                fc.diff_excerpt = diffs[fc.path]
    
    # Run verification if requested
    if run_verification and verification_commands:
        # Separate test commands from lint commands
        test_cmds = [c for c in verification_commands if "pytest" in c or "test" in c.lower()]
        lint_cmds = [c for c in verification_commands if "ruff" in c or "lint" in c.lower()]
        
        if test_cmds:
            bundle.test_result = run_tests(repo_path, test_cmds)
        
        # Run ruff by default
        bundle.lint_results.append(run_lint(repo_path, "ruff", changed_paths))
        
        # Compute error signature if failures
        if bundle.test_result and not bundle.test_result.success:
            error_output = bundle.test_result.output_excerpt or ""
            bundle.error_output = error_output
            bundle.stack_trace = truncate_output(error_output, MAX_STACK_TRACE_LINES)
            bundle.current_error_signature = compute_error_signature(error_output)
    
    return bundle


__all__ = [
    # Data classes
    "FileChange",
    "TestResult",
    "LintResult",
    "EnvironmentContext",
    "EvidenceBundle",
    # Builder functions
    "truncate_output",
    "get_git_diff",
    "run_tests",
    "run_lint",
    "build_evidence_bundle",
    # Constants
    "MAX_INPUT_TOKENS",
]

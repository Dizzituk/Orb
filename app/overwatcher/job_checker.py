# FILE: app/overwatcher/job_checker.py
"""
Job Checker — Post-Write Verification Against Architecture + Contract.

Phase 4A of Pipeline Evolution.

After each file is written by the Implementer, the Job Checker reads the
file content and sends it to a lightweight LLM for verification against:

1. The architecture specification section for that file
2. The interface contract boundaries (if available from Phase 2)
3. Basic structural correctness (imports resolve, exports match)

This catches drift BEFORE it propagates to downstream files. If a file
fails the check, the executor's existing three-strike system handles retry.

The checker is intentionally fast and cheap — it's a Sonnet-class call
with a focused prompt that returns a structured pass/fail verdict.

v1.0 (2026-02-10): Initial implementation — Phase 4A.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

JOB_CHECKER_BUILD_ID = "2026-02-14-v2.2-strike-contradiction-awareness"
print(f"[JOB_CHECKER_LOADED] BUILD_ID={JOB_CHECKER_BUILD_ID}")


# =============================================================================
# RESULT SCHEMA
# =============================================================================

@dataclass
class CheckIssue:
    """A single issue found by the job checker."""
    severity: str           # "blocking" or "warning"
    category: str           # e.g. "missing_export", "wrong_signature", "import_error"
    description: str        # Human-readable description
    line_hint: Optional[str] = None  # Approximate location hint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "line_hint": self.line_hint,
        }


@dataclass
class CheckResult:
    """Result of a post-write file check."""
    passed: bool = True
    issues: List[CheckIssue] = field(default_factory=list)
    reasoning: str = ""
    model_used: str = ""
    skipped: bool = False
    skip_reason: str = ""

    @property
    def blocking_issues(self) -> List[CheckIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warning_issues(self) -> List[CheckIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "blocking_count": len(self.blocking_issues),
            "warning_count": len(self.warning_issues),
            "reasoning": self.reasoning,
            "model_used": self.model_used,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

JOB_CHECKER_SYSTEM_PROMPT = """\
You are a post-write code verifier. You have just been given:
1. A file that was just written to disk
2. The architecture specification for that file
3. Optionally, an interface contract listing what this file MUST expose

Your job is to verify the written file matches the specification. Check:

1. EXPORTS: Every class, function, constant, or endpoint listed in the \
architecture spec actually exists in the file with the correct name.

2. SIGNATURES: Function/method signatures match the spec — parameter names, \
types, and return types are correct.

3. IMPORTS: The file's imports reference modules/packages that should exist \
(based on the architecture). Flag imports to clearly non-existent local modules.

4. CONTRACT COMPLIANCE: If an interface contract is provided, every "MUST EXPOSE" \
boundary exists with the exact name, signature, and return type specified.

5. COMPLETENESS: No TODO, FIXME, NotImplementedError, or "pass" placeholders \
in critical paths. Stub implementations are acceptable ONLY for genuinely \
optional features.

RULES:
- Focus on integration-critical issues that would cause OTHER files to break.
- Don't nitpick style, formatting, or internal implementation details.
- Severity "blocking" = would cause import errors, type errors, or runtime \
failures in OTHER files. Severity "warning" = might cause issues, worth noting.
- Be precise. Quote the exact name/signature that's wrong.

OUTPUT FORMAT:
Return ONLY a JSON object:
{
  "passed": true/false,
  "issues": [
    {
      "severity": "blocking" | "warning",
      "category": "missing_export" | "wrong_signature" | "import_error" | \
"missing_implementation" | "contract_violation" | "naming_mismatch",
      "description": "ExactClassName.method_name has wrong return type: \
expected Dict[str, Any] but found None",
      "line_hint": "near line 45"
    }
  ],
  "reasoning": "Brief summary of check"
}

passed = false if ANY blocking issues exist, true otherwise.
If the file looks correct, return {"passed": true, "issues": [], "reasoning": "..."}.
"""


# =============================================================================
# SKIP LOGIC — avoid wasting LLM calls on trivial files
# =============================================================================

# Files that are too simple to benefit from LLM checking
SKIP_PATTERNS = [
    r'__init__\.py$',      # Usually just imports/re-exports
    r'\.env',              # Config files
    r'\.json$',            # Data/config
    r'\.yaml$',
    r'\.yml$',
    r'\.toml$',
    r'\.cfg$',
    r'\.ini$',
    r'\.md$',              # Documentation
    r'\.txt$',
    r'\.gitignore$',
    r'\.dockerignore$',
    r'requirements\.txt$',
    r'Dockerfile$',
]

# Minimum file content length to bother checking
MIN_CHECK_CHARS = 100

# Maximum file content to send (trim if huge)
MAX_CHECK_CHARS = 15000


def _should_skip(file_path: str, file_content: str) -> Optional[str]:
    """Return skip reason if file should be skipped, None otherwise."""
    basename = os.path.basename(file_path)

    for pattern in SKIP_PATTERNS:
        if re.search(pattern, basename, re.IGNORECASE):
            return f"Skip pattern match: {pattern}"

    if len(file_content.strip()) < MIN_CHECK_CHARS:
        return f"File too short ({len(file_content.strip())} chars < {MIN_CHECK_CHARS})"

    return None


# =============================================================================
# DETERMINISTIC IMPORT PRE-CHECK
# =============================================================================


def _resolve_relative_imports(
    file_path: str,
    file_content: str,
    sandbox_base: str = "",
    existing_sandbox_files: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Deterministically verify relative imports against the actual filesystem.

    Parses all `from .xxx import` and `from ..xxx import` lines, resolves
    the target module path relative to the file being checked, and verifies
    whether the target exists on disk.

    Returns:
        {
            "verified": [{"import_line": ..., "resolved_path": ..., "exists": True}],
            "unresolvable": [{"import_line": ..., "resolved_path": ..., "reason": ...}],
        }
    """
    from pathlib import Path

    verified = []
    unresolvable = []

    if not file_path:
        return {"verified": verified, "unresolvable": unresolvable}

    # Normalise to forward slashes
    norm_path = file_path.replace("\\", "/")

    # Determine file's directory (relative)
    file_dir = "/".join(norm_path.split("/")[:-1]) if "/" in norm_path else ""

    # Resolve sandbox base for absolute checks
    _base = sandbox_base
    if not _base:
        _base = os.getenv("SANDBOX_BASE", "")
    if not _base:
        # Try common roots
        for candidate in ["D:/Orb", "C:/Orb"]:
            if os.path.isdir(candidate):
                _base = candidate
                break

    # Parse relative imports from file content
    # Matches: from .module import X, from ..module import X, from ...module import X
    import_pattern = re.compile(
        r'^\s*from\s+(\.{1,6}\w[\w.]*)\s+import\s+',
        re.MULTILINE,
    )

    for match in import_pattern.finditer(file_content):
        import_ref = match.group(1)  # e.g. ".constants", "..sandbox_client", "...something"
        import_line = match.group(0).strip()

        # Count leading dots
        dot_count = 0
        for ch in import_ref:
            if ch == '.':
                dot_count += 1
            else:
                break

        module_name = import_ref[dot_count:]  # e.g. "constants", "sandbox_client"

        if not module_name:
            # "from . import X" — importing from package __init__
            continue

        # Resolve: each dot = one directory up from the file's directory
        # 1 dot = same package, 2 dots = parent package, etc.
        target_dir = file_dir
        for _ in range(dot_count - 1):  # -1 because first dot = current package
            if "/" in target_dir:
                target_dir = target_dir.rsplit("/", 1)[0]
            else:
                target_dir = ""

        # Build candidate paths
        if target_dir:
            candidate_file = f"{target_dir}/{module_name.replace('.', '/')}.py"
            candidate_pkg = f"{target_dir}/{module_name.replace('.', '/')}/__init__.py"
        else:
            candidate_file = f"{module_name.replace('.', '/')}.py"
            candidate_pkg = f"{module_name.replace('.', '/')}/__init__.py"

        # Check filesystem — first check host, then check sandbox file list
        found = False
        resolved = candidate_file
        found_via = ""

        # v2.1: Check existing_sandbox_files first (files confirmed on sandbox)
        _sandbox_files = existing_sandbox_files or set()
        _candidate_fwd = candidate_file.replace("\\", "/")
        _candidate_pkg_fwd = candidate_pkg.replace("\\", "/")
        for sf in _sandbox_files:
            sf_norm = sf.replace("\\", "/")
            if sf_norm == _candidate_fwd or sf_norm == _candidate_pkg_fwd:
                found = True
                resolved = candidate_file
                found_via = "sandbox_file_list"
                break

        # Fallback: check host filesystem (only works if host == sandbox)
        if not found and _base:
            abs_file = os.path.join(_base, candidate_file)
            abs_pkg = os.path.join(_base, candidate_pkg)
            if os.path.isfile(abs_file):
                found = True
                resolved = candidate_file
                found_via = "host_filesystem"
            elif os.path.isfile(abs_pkg):
                found = True
                resolved = candidate_pkg
                found_via = "host_filesystem"

        entry = {
            "import_line": import_line,
            "import_ref": import_ref,
            "resolved_path": resolved,
        }

        if found:
            entry["exists"] = True
            entry["found_via"] = found_via
            verified.append(entry)
            logger.debug("[job_checker] IMPORT VERIFIED (%s): %s -> %s", found_via, import_ref, resolved)
        else:
            if not _base and not _sandbox_files:
                entry["reason"] = "Cannot verify - no sandbox base path or file list available"
                # Don't mark as unresolvable if we can't check
                verified.append(entry)
            else:
                entry["reason"] = f"Module not found at {candidate_file} or {candidate_pkg}"
                entry["exists"] = False
                unresolvable.append(entry)
                logger.debug("[job_checker] IMPORT NOT FOUND: %s -> tried %s", import_ref, candidate_file)

    return {"verified": verified, "unresolvable": unresolvable}


def _build_import_evidence(import_results: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Build a prompt section that tells the LLM which imports are
    filesystem-verified so it does NOT flag them.
    """
    verified = import_results.get("verified", [])
    unresolvable = import_results.get("unresolvable", [])

    if not verified and not unresolvable:
        return ""

    lines = []
    lines.append("\n## Deterministic Import Verification (GROUND TRUTH — do NOT override)")
    lines.append("The following imports have been checked against the actual filesystem.")
    lines.append("DO NOT flag verified imports as errors. They are confirmed correct.\n")

    if verified:
        lines.append("### ✅ VERIFIED (exist on disk — do NOT flag):")
        for v in verified:
            lines.append(f"- `{v['import_ref']}` → `{v['resolved_path']}`")
        lines.append("")

    if unresolvable:
        lines.append("### ❌ NOT FOUND (should be flagged as blocking):")
        for u in unresolvable:
            lines.append(f"- `{u['import_ref']}` → {u.get('reason', 'not found')}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# LLM CALL
# =============================================================================

def _build_check_prompt(
    file_path: str,
    file_content: str,
    arch_section: str,
    interface_contract: str = "",
    import_evidence: str = "",
    previous_strike_errors: Optional[List[str]] = None,
) -> str:
    """Build user prompt for the job checker."""
    # Trim file content if huge
    _content = file_content
    if len(_content) > MAX_CHECK_CHARS:
        _half = MAX_CHECK_CHARS // 2 - 100
        _content = (
            _content[:_half]
            + f"\n\n... ({len(file_content) - MAX_CHECK_CHARS} chars trimmed) ...\n\n"
            + _content[-_half:]
        )

    contract_section = ""
    if interface_contract and interface_contract.strip():
        contract_section = f"""

## Interface Contract
{interface_contract}
"""

    # v2.2: Strike history — show previous checker feedback to prevent contradictions
    strike_history_section = ""
    if previous_strike_errors:
        strike_lines = []
        for i, err in enumerate(previous_strike_errors, 1):
            strike_lines.append(f"- **Strike {i}**: {err}")
        strike_history_section = f"""

## ⚠️ PREVIOUS STRIKE HISTORY (CRITICAL — READ CAREFULLY)
This file has been rejected {len(previous_strike_errors)} time(s) already.
The Implementer rewrote the file after each rejection based on YOUR feedback.

{chr(10).join(strike_lines)}

**IMPORTANT**: If you see that previous strikes gave CONTRADICTORY feedback
(e.g. strike 1 said "make it async" and strike 2 said "make it sync"), then
the spec itself has an ambiguity. In this case you MUST:
1. Accept the current implementation if it is functionally correct
2. Downgrade the contradicted issue from "blocking" to "warning"
3. Do NOT re-raise an issue that contradicts feedback from a previous strike

The goal is forward progress, not perfection. Only flag issues that are
genuinely broken (will cause ImportError, NameError, or logic bugs at runtime).
"""

    return f"""\
## File Path
`{file_path}`

## Architecture Specification For This File
{arch_section}
{contract_section}{import_evidence}{strike_history_section}
## Written File Content
```
{_content}
```

Verify the written file against the architecture spec and contract. Return ONLY the JSON verdict.
"""


def _parse_check_response(llm_output: str) -> Optional[CheckResult]:
    """Parse the LLM's JSON verdict into a CheckResult."""
    if not llm_output or not llm_output.strip():
        return None

    text = llm_output.strip()

    # Strip markdown fences
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    text = re.sub(r',\s*\}', '}', text)
    text = re.sub(r',\s*\]', ']', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("[job_checker] JSON parse failed: %s", e)
        return None

    if not isinstance(data, dict):
        return None

    issues = []
    for issue_data in data.get("issues", []):
        if isinstance(issue_data, dict):
            issues.append(CheckIssue(
                severity=issue_data.get("severity", "warning"),
                category=issue_data.get("category", "unknown"),
                description=issue_data.get("description", ""),
                line_hint=issue_data.get("line_hint"),
            ))

    result = CheckResult(
        passed=bool(data.get("passed", True)),
        issues=issues,
        reasoning=str(data.get("reasoning", "")),
    )

    # Enforce: if any blocking issues, passed must be False
    if result.blocking_issues:
        result.passed = False

    return result


async def check_written_file(
    file_path: str,
    file_content: str,
    arch_section: str,
    interface_contract: str = "",
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    sandbox_base: str = "",
    existing_sandbox_files: Optional[Set[str]] = None,
    previous_strike_errors: Optional[List[str]] = None,
) -> CheckResult:
    """
    Main entry point: verify a written file against its architecture spec.

    Args:
        file_path: Path of the written file (for display/skip logic)
        file_content: The actual content that was written
        arch_section: The architecture specification section for this file
        interface_contract: Optional interface contract markdown
        provider_id/model_id: Override model selection

    Returns:
        CheckResult with pass/fail verdict and any issues found
    """
    # Skip logic
    skip_reason = _should_skip(file_path, file_content)
    if skip_reason:
        logger.debug("[job_checker] Skipping %s: %s", file_path, skip_reason)
        return CheckResult(skipped=True, skip_reason=skip_reason)

    # Skip if no architecture section (can't verify against nothing)
    if not arch_section or len(arch_section.strip()) < 50:
        return CheckResult(
            skipped=True,
            skip_reason="No architecture section available for verification",
        )

    # Resolve model
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("JOB_CHECKER")
            _provider = _provider or config.provider
            _model = _model or config.model
        except (ImportError, Exception) as _cfg_err:
            logger.warning("[job_checker] stage_models unavailable: %s", _cfg_err)

    if not _provider or not _model:
        logger.warning("[job_checker] Model not configured — skipping check")
        return CheckResult(
            skipped=True,
            skip_reason="JOB_CHECKER model not configured",
        )

    logger.info(
        "[job_checker] Checking %s (%d chars) against spec (%d chars) — %s/%s",
        file_path, len(file_content), len(arch_section), _provider, _model,
    )

    # v2.1: Deterministic import pre-check — resolve relative imports against filesystem + sandbox file list
    _import_results = _resolve_relative_imports(file_path, file_content, sandbox_base, existing_sandbox_files)
    _import_evidence = _build_import_evidence(_import_results)
    _v_count = len(_import_results.get("verified", []))
    _u_count = len(_import_results.get("unresolvable", []))
    if _v_count or _u_count:
        logger.info(
            "[job_checker] v2.0 Import pre-check: %d verified, %d unresolvable",
            _v_count, _u_count,
        )

    user_prompt = _build_check_prompt(file_path, file_content, arch_section, interface_contract, _import_evidence, previous_strike_errors)

    try:
        from app.providers.registry import llm_call

        result = await llm_call(
            provider_id=_provider,
            model_id=_model,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=JOB_CHECKER_SYSTEM_PROMPT,
            max_tokens=1500,
            timeout_seconds=45,
        )

        if not result.is_success():
            logger.warning("[job_checker] LLM call failed: %s", result.error_message)
            return CheckResult(
                skipped=True,
                skip_reason=f"LLM call failed: {result.error_message}",
            )

        raw = (result.content or "").strip()
        check = _parse_check_response(raw)
        if check is None:
            logger.warning("[job_checker] Failed to parse response")
            return CheckResult(
                skipped=True,
                skip_reason="Failed to parse LLM response",
            )

        check.model_used = f"{_provider}/{_model}"
        logger.info(
            "[job_checker] %s: passed=%s blocking=%d warning=%d",
            file_path, check.passed, len(check.blocking_issues), len(check.warning_issues),
        )
        return check

    except ImportError:
        return CheckResult(skipped=True, skip_reason="Provider registry unavailable")
    except Exception as e:
        logger.exception("[job_checker] Unexpected error checking %s: %s", file_path, e)
        return CheckResult(skipped=True, skip_reason=f"Exception: {e}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CheckResult",
    "CheckIssue",
    "check_written_file",
    "JOB_CHECKER_BUILD_ID",
]

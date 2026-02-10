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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JOB_CHECKER_BUILD_ID = "2026-02-10-v1.0-post-write-verification"
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
# LLM CALL
# =============================================================================

def _build_check_prompt(
    file_path: str,
    file_content: str,
    arch_section: str,
    interface_contract: str = "",
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

    return f"""\
## File Path
`{file_path}`

## Architecture Specification For This File
{arch_section}
{contract_section}
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

    user_prompt = _build_check_prompt(file_path, file_content, arch_section, interface_contract)

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

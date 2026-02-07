# FILE: app/llm/critical_pipeline/quickcheck_scan.py
"""
Scan-only quickcheck — fast deterministic validation (NO LLM calls).

Validates scan spec configuration before executing read-only
filesystem scan/search/enumerate operations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List

from app.llm.critical_pipeline.config import (
    SAFE_DEFAULT_SCAN_ROOTS,
    validate_scan_roots,
)

logger = logging.getLogger(__name__)


@dataclass
class ScanQuickcheckResult:
    """Result of scan-only quickcheck validation."""
    passed: bool
    issues: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


def scan_quickcheck(
    spec_data: Dict[str, Any],
    plan_text: str,
) -> ScanQuickcheckResult:
    """
    Fast deterministic validation for SCAN_ONLY jobs.
    NO LLM calls — pure tick-box checks.

    Checks:
    0. HARD SECURITY GATE — scan_roots within SAFE_DEFAULT_SCAN_ROOTS
    1. scan_roots exists and is non-empty
    2. scan_terms or scan_targets present
    3. output_mode is CHAT_ONLY
    4. No write operations in plan
    5. write_policy is READ_ONLY (if present)
    """
    issues: List[Dict[str, str]] = []
    scan_roots = spec_data.get("scan_roots", [])

    # === HARD SECURITY GATE ===
    if scan_roots:
        valid_roots, rejected_roots = validate_scan_roots(scan_roots)

        if rejected_roots:
            logger.error(
                "[scan_quickcheck] SECURITY GATE: Rejected %d root(s): %s",
                len(rejected_roots), rejected_roots,
            )
            issues.append({
                "id": "SCAN-SECURITY-001",
                "description": (
                    f"SECURITY VIOLATION: Rejected scan roots: {rejected_roots}. "
                    f"Scans can ONLY target: {SAFE_DEFAULT_SCAN_ROOTS}"
                ),
                "severity": "blocking",
            })

        # Belt-and-suspenders: bare drive letters
        for root in scan_roots:
            normalized = root.replace('/', '\\').rstrip('\\')
            if len(normalized) <= 3:
                logger.error(
                    "[scan_quickcheck] SECURITY GATE: Bare drive letter '%s' BLOCKED",
                    root,
                )
                if not any(i["id"] == "SCAN-SECURITY-002" for i in issues):
                    issues.append({
                        "id": "SCAN-SECURITY-002",
                        "description": (
                            f"SECURITY VIOLATION: Bare drive letter '{root}' not allowed. "
                            f"Use specific paths within {SAFE_DEFAULT_SCAN_ROOTS}."
                        ),
                        "severity": "blocking",
                    })

        logger.info(
            "[scan_quickcheck] SECURITY GATE: valid=%s, rejected=%s",
            valid_roots, rejected_roots,
        )

    # Check 1: scan_roots present
    if not scan_roots:
        issues.append({
            "id": "SCAN-CHECK-001",
            "description": "scan_roots not specified — no scan target defined",
            "severity": "blocking",
        })

    # Check 2: search criteria present
    scan_terms = spec_data.get("scan_terms", [])
    scan_targets = spec_data.get("scan_targets", [])
    if not scan_terms and not scan_targets:
        issues.append({
            "id": "SCAN-CHECK-002",
            "description": "No scan_terms or scan_targets — what are we scanning for?",
            "severity": "blocking",
        })

    # Check 3: output_mode
    output_mode = (
        spec_data.get("sandbox_output_mode")
        or spec_data.get("output_mode")
        or ""
    ).lower()
    if output_mode and output_mode != "chat_only":
        issues.append({
            "id": "SCAN-CHECK-003",
            "description": f"SCAN_ONLY jobs should use CHAT_ONLY (found: {output_mode})",
            "severity": "warning",
        })

    # Check 4: no write operations in plan
    plan_lower = plan_text.lower()
    write_patterns = [
        "write file", "create file", "save to", "output to",
        "modify file", "edit file", "update file",
        "sandbox_output_path", "sandbox_generated_reply",
    ]
    for pattern in write_patterns:
        if pattern in plan_lower:
            issues.append({
                "id": "SCAN-CHECK-004",
                "description": f"SCAN_ONLY plan should not contain write ops (found: '{pattern}')",
                "severity": "warning",
            })
            break

    # Check 5: write_policy
    write_policy = (spec_data.get("write_policy") or "").lower()
    if write_policy and write_policy != "read_only":
        issues.append({
            "id": "SCAN-CHECK-005",
            "description": f"SCAN_ONLY should have write_policy=READ_ONLY (found: {write_policy})",
            "severity": "warning",
        })

    # Build result
    blocking = sum(1 for i in issues if i.get("severity") == "blocking")
    passed = blocking == 0

    if passed:
        summary = (
            f"✅ Scan quickcheck passed with {len(issues)} warning(s)"
            if issues else "✅ All scan quickchecks passed"
        )
    else:
        summary = f"❌ {blocking} blocking issue(s) found"

    logger.info(
        "[scan_quickcheck] passed=%s, issues=%d, roots=%d, terms=%d",
        passed, len(issues),
        len(scan_roots), len(scan_terms),
    )

    return ScanQuickcheckResult(passed=passed, issues=issues, summary=summary)

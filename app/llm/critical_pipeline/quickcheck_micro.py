# FILE: app/llm/critical_pipeline/quickcheck_micro.py
"""
Micro-execution quickcheck — fast deterministic validation (NO LLM calls).

Validates spec/plan alignment with pure tick-box checks before handing
micro-execution jobs to the Overwatcher.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List

from app.llm.critical_pipeline.evidence import _find_multi_target_files

logger = logging.getLogger(__name__)


@dataclass
class MicroQuickcheckResult:
    """Result of micro-execution quickcheck validation."""
    passed: bool
    issues: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


def micro_quickcheck(
    spec_data: Dict[str, Any],
    plan_text: str,
) -> MicroQuickcheckResult:
    """
    Fast deterministic validation for micro-execution jobs.
    NO LLM calls — pure tick-box checks.

    Checks (mode-aware):
    1. sandbox_input_path exists in spec (OR multi_target_files for multi-read)
    2. sandbox_output_path validation (mode-dependent)
    3. Plan paths match spec paths (skip output check for CHAT_ONLY)
    4. Plan has only safe operations (no destructive commands)
    5. If plan says "write output" but no generated_reply exists → fail (skip for CHAT_ONLY)
    """
    issues: List[Dict[str, str]] = []

    # Diagnostic logging
    logger.info(
        "[micro_quickcheck] DIAGNOSTIC spec_data keys: %s",
        list(spec_data.keys())[:20],
    )

    # --- Extract output_mode ---
    output_mode = (spec_data.get("sandbox_output_mode") or "").lower()
    logger.info("[micro_quickcheck] output_mode=%s", output_mode)

    # --- Determine multi-target mode ---
    is_multi_target_read = spec_data.get("is_multi_target_read", False)
    if not is_multi_target_read:
        grounding_data_check = spec_data.get("grounding_data", {})
        is_multi_target_read = grounding_data_check.get("is_multi_target_read", False)
        if is_multi_target_read:
            logger.info("[micro_quickcheck] Found is_multi_target_read=True in grounding_data")

    # Locate multi_target_files across all possible locations
    multi_target_files, mtf_source = _find_multi_target_files(spec_data)

    # v2.8: FALLBACK — if files exist but flag missing, auto-detect
    if multi_target_files and not is_multi_target_read:
        logger.info(
            "[micro_quickcheck] v2.8 FALLBACK: multi_target_files has %d entries "
            "but is_multi_target_read=False — auto-enabling",
            len(multi_target_files),
        )
        is_multi_target_read = True

    logger.info(
        "[micro_quickcheck] is_multi_target_read=%s, multi_target_files=%d (from %s)",
        is_multi_target_read, len(multi_target_files), mtf_source,
    )

    # =========================================================================
    # Check 1: Input file/files exist
    # =========================================================================
    if is_multi_target_read:
        if not multi_target_files:
            issues.append({
                "id": "MICRO-CHECK-001",
                "description": (
                    "Multi-target read mode but no multi_target_files entries found. "
                    "SpecGate should have populated multi_target_files."
                ),
            })
        else:
            logger.info(
                "[micro_quickcheck] CHECK 1 PASSED: multi_target_files has %d entries",
                len(multi_target_files),
            )
    else:
        sandbox_input = spec_data.get("sandbox_input_path", "")
        if not sandbox_input:
            issues.append({
                "id": "MICRO-CHECK-001",
                "description": (
                    "No sandbox_input_path in spec. "
                    "SpecGate should resolve the input file path."
                ),
            })
        else:
            logger.info("[micro_quickcheck] CHECK 1 PASSED: sandbox_input_path=%s", sandbox_input)

    # =========================================================================
    # Check 2: Output path validation (mode-dependent)
    # =========================================================================
    sandbox_output = spec_data.get("sandbox_output_path", "")
    sandbox_input = spec_data.get("sandbox_input_path", "")

    if output_mode == "chat_only":
        logger.info("[micro_quickcheck] CHECK 2 SKIPPED: CHAT_ONLY mode")
    elif output_mode in ("rewrite_in_place", "append_in_place"):
        if sandbox_output and sandbox_input and sandbox_output != sandbox_input:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": (
                    f"Output mode is {output_mode.upper()} but output_path "
                    f"({sandbox_output}) != input_path ({sandbox_input}). "
                    f"For {output_mode}, they should be the same file."
                ),
            })
        elif not sandbox_output and sandbox_input:
            logger.info(
                "[micro_quickcheck] CHECK 2 OK: %s with no explicit output_path "
                "(will use input_path)",
                output_mode,
            )
        else:
            logger.info("[micro_quickcheck] CHECK 2 PASSED: %s mode paths match", output_mode)
    elif output_mode == "separate_reply_file":
        if not sandbox_output:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": (
                    "Output mode is SEPARATE_REPLY_FILE but no sandbox_output_path in spec."
                ),
            })
        else:
            logger.info(
                "[micro_quickcheck] CHECK 2 PASSED: separate_reply_file output=%s",
                sandbox_output,
            )
    else:
        # Unknown or missing mode — check output_path presence
        if not sandbox_output:
            # Only warn, not fail — might be CHAT_ONLY with missing mode tag
            logger.info(
                "[micro_quickcheck] CHECK 2 NOTE: No output_mode specified and no "
                "sandbox_output_path. Assuming CHAT_ONLY."
            )
        else:
            logger.info("[micro_quickcheck] CHECK 2 PASSED: output_path=%s", sandbox_output)

    # =========================================================================
    # Check 3: Plan paths match spec paths
    # =========================================================================
    plan_lower = plan_text.lower()

    if not is_multi_target_read:
        if sandbox_input and sandbox_input.lower() not in plan_lower:
            # Check if just the filename is in the plan
            input_basename = os.path.basename(sandbox_input).lower()
            if input_basename not in plan_lower:
                issues.append({
                    "id": "MICRO-CHECK-003",
                    "description": (
                        f"Plan doesn't reference the spec input path ({sandbox_input})."
                    ),
                })
            else:
                logger.info("[micro_quickcheck] CHECK 3 PASSED: input basename in plan")
        else:
            logger.info("[micro_quickcheck] CHECK 3 PASSED: input path in plan")
    else:
        logger.info("[micro_quickcheck] CHECK 3 SKIPPED: multi-target read")

    if output_mode != "chat_only":
        if sandbox_output and sandbox_output.lower() not in plan_lower:
            output_basename = os.path.basename(sandbox_output).lower()
            if output_basename not in plan_lower:
                issues.append({
                    "id": "MICRO-CHECK-004",
                    "description": (
                        f"Plan doesn't reference the spec output path ({sandbox_output})."
                    ),
                })

    # =========================================================================
    # Check 4: Safe operations only
    # =========================================================================
    DANGEROUS_COMMANDS = [
        "rm -rf", "rmdir", "del /f", "format ", "fdisk",
        "drop table", "drop database", "truncate table",
        "shutdown", "reboot", "halt",
    ]
    for cmd in DANGEROUS_COMMANDS:
        if cmd in plan_lower:
            issues.append({
                "id": "MICRO-CHECK-005",
                "description": f"Plan contains potentially dangerous command: '{cmd}'",
            })

    # =========================================================================
    # Check 5: Write output requires generated_reply
    # =========================================================================
    if output_mode != "chat_only":
        write_indicators = ["write output", "write to output", "save output", "create output"]
        plan_writes = any(ind in plan_lower for ind in write_indicators)
        has_reply = bool(spec_data.get("sandbox_generated_reply"))

        if plan_writes and not has_reply:
            issues.append({
                "id": "MICRO-CHECK-006",
                "description": (
                    "Plan includes writing output but spec has no sandbox_generated_reply."
                ),
            })

    # =========================================================================
    # Build result
    # =========================================================================
    passed = len(issues) == 0
    if passed:
        summary = "✅ **Quickcheck PASSED** — All checks clear. Ready for Overwatcher."
    else:
        summary = f"❌ **Quickcheck FAILED** — {len(issues)} issue(s) found."

    logger.info(
        "[micro_quickcheck] Result: passed=%s, issues=%d",
        passed, len(issues),
    )

    return MicroQuickcheckResult(passed=passed, issues=issues, summary=summary)

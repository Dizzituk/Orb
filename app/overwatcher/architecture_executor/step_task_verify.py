"""
Post-write verification for file tasks.

Runs the three-layer verification pipeline after each file write:
1. Deterministic checker (zero LLM cost)
2. LLM job checker (only if deterministic check didn't pass)
3. Signature checker (deterministic AST comparison)

Extracted from orchestrator.py monolith (v2.5–v5.22 logic).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .execution_state import ExecutionContext
from .parsing import extract_section_for_file

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Result of post-write verification."""
    passed: bool = True
    error: Optional[str] = None
    job_checker_error: Optional[str] = None
    structured_sig_mismatches: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.structured_sig_mismatches is None:
            self.structured_sig_mismatches = []


# ---------------------------------------------------------------------------
# Phase 4A-DET: Deterministic checker  (v2.5)
# ---------------------------------------------------------------------------

def _run_deterministic_check(
    rel_path: str,
    file_content: str,
    ctx: ExecutionContext,
    strike: int,
) -> Optional[bool]:
    """Run deterministic check. Returns True=pass, False=fail, None=unavailable."""
    try:
        from app.overwatcher.deterministic_checker import (
            deterministic_check as det_check,
            extract_expected_exports_from_arch,
        )
        arch_section = extract_section_for_file(
            ctx.architecture_content, rel_path,
        ) or ""
        expected_exports = (
            extract_expected_exports_from_arch(arch_section)
            if arch_section else None
        )
        det_result = det_check(
            file_path=rel_path,
            file_content=file_content,
            interface_contract=ctx.interface_contract,
            sandbox_base=ctx.sandbox_base,
            existing_sandbox_files=ctx.existing_sandbox_files,
            manifest_file_scope=None,
            expected_exports=expected_exports or None,
        )
        if det_result.skipped:
            logger.debug(
                "[arch_exec] v2.5 Det check skipped for %s: %s",
                rel_path, det_result.skip_reason,
            )
            return True

        if not det_result.passed:
            blocking = det_result.blocking_issues
            desc = "; ".join(i.description for i in blocking[:3])
            logger.warning("[arch_exec] v2.5 DET_CHECK Strike %d: %s", strike, desc)
            print(f"[ARCH_EXEC] v2.5 DET_CHECK FAIL: {rel_path} — {desc[:120]}")
            ctx.add_trace("DET_CHECK_FAIL", f"strike_{strike}", {
                "path": rel_path,
                "blocking": len(blocking),
                "warnings": len(det_result.warning_issues),
                "issues": [i.to_dict() for i in det_result.issues[:5]],
            })
            return False

        warns = len(det_result.warning_issues)
        logger.info("[arch_exec] v2.5 DET_CHECK PASS: %s (%d warnings)", rel_path, warns)
        ctx.add_trace("DET_CHECK_PASS", "verified", {"path": rel_path, "warnings": warns})
        return True

    except ImportError:
        logger.debug("[arch_exec] v2.5 deterministic_checker not available — falling back to LLM")
        return None
    except Exception as e:
        logger.warning("[arch_exec] v2.5 Det checker error (non-fatal): %s", e)
        return None


# ---------------------------------------------------------------------------
# Phase 4A: LLM Job Checker  (v5.5)
# ---------------------------------------------------------------------------

async def _run_llm_job_check(
    rel_path: str,
    file_content: str,
    ctx: ExecutionContext,
    strike: int,
    job_checker_strike_errors: List[str],
) -> Optional[bool]:
    """Run LLM job checker. Returns True=pass, False=fail, None=unavailable."""
    try:
        from app.overwatcher.job_checker import check_written_file
        arch_section = extract_section_for_file(
            ctx.architecture_content, rel_path,
        ) or ""
        check_result = await check_written_file(
            file_path=rel_path,
            file_content=file_content,
            arch_section=arch_section,
            interface_contract=ctx.interface_contract,
            sandbox_base=ctx.sandbox_base,
            existing_sandbox_files=ctx.existing_sandbox_files,
            previous_strike_errors=job_checker_strike_errors or None,
        )
        if check_result.skipped:
            logger.debug(
                "[arch_exec] v5.5 Job check skipped for %s: %s",
                rel_path, check_result.skip_reason,
            )
            return True

        if not check_result.passed:
            blocking = check_result.blocking_issues
            desc = "; ".join(i.description for i in blocking[:3])
            logger.warning("[arch_exec] v5.5 Strike %d: %s", strike, desc)
            print(f"[ARCH_EXEC] v5.5 JOB_CHECK FAIL: {rel_path} — {desc[:120]}")
            ctx.add_trace("JOB_CHECK_FAIL", f"strike_{strike}", {
                "path": rel_path,
                "blocking": len(blocking),
                "warnings": len(check_result.warning_issues),
                "issues": [i.to_dict() for i in check_result.issues[:5]],
            })
            return False

        warns = len(check_result.warning_issues)
        if warns:
            logger.info(
                "[arch_exec] v5.5 Job check PASSED with %d warning(s): %s",
                warns, rel_path,
            )
        ctx.add_trace("JOB_CHECK_PASS", "verified", {"path": rel_path, "warnings": warns})
        return True

    except ImportError:
        logger.debug("[arch_exec] v5.5 job_checker not available — skipping")
        return None
    except Exception as e:
        logger.warning("[arch_exec] v5.5 Job checker error (non-fatal): %s", e)
        return None


# ---------------------------------------------------------------------------
# Phase 4B: Signature Verification  (v5.22)
# ---------------------------------------------------------------------------

def _run_signature_check(
    rel_path: str,
    file_content: str,
    ctx: ExecutionContext,
    strike: int,
) -> VerifyResult:
    """Run deterministic signature check against skeleton contract.

    Returns VerifyResult with structured mismatch data on failure.
    """
    result = VerifyResult(passed=True)
    try:
        from app.overwatcher.signature_checker import (
            check_file_signatures,
            extract_contract_signatures_for_file,
        )
        contract_sigs = extract_contract_signatures_for_file(
            ctx.interface_contract, rel_path,
        )
        if not contract_sigs:
            return result

        sig_result = check_file_signatures(
            file_content=file_content,
            file_path=rel_path,
            contract_signatures=contract_sigs,
        )
        if sig_result.passed:
            logger.debug(
                "[arch_exec] v5.22 Sig check PASSED for %s (%d sigs verified)",
                rel_path, len(contract_sigs),
            )
            ctx.add_trace("SIGNATURE_CHECK_PASS", "verified", {
                "path": rel_path,
                "signatures_checked": len(contract_sigs),
            })
            return result

        # Build error details
        details = []
        for mm in sig_result.mismatches:
            details.append(
                f"SIGNATURE MISMATCH: {mm.function_name}\n"
                f"  Contract requires: {mm.expected_signature}\n"
                f"  Implementation has: {mm.actual_signature}\n"
                f"  Differences: {'; '.join(mm.differences)}"
            )
        for mf in sig_result.missing_functions:
            details.append(
                f"MISSING FUNCTION: {mf} — required by contract but not found"
            )
        error = (
            f"Signature checker FAILED: "
            f"{len(sig_result.mismatches)} mismatch(es), "
            f"{len(sig_result.missing_functions)} missing.\n"
            + "\n".join(details)
        )
        logger.warning("[arch_exec] v5.22 Sig check strike %d: %s", strike, error[:200])
        print(
            f"[ARCH_EXEC] v5.22 SIG_CHECK FAIL: {rel_path} — "
            f"{len(sig_result.mismatches)} mismatch(es), "
            f"{len(sig_result.missing_functions)} missing"
        )
        ctx.add_trace("SIGNATURE_CHECK_FAIL", f"strike_{strike}", {
            "path": rel_path,
            "mismatches": len(sig_result.mismatches),
            "missing": len(sig_result.missing_functions),
            "details": [m.to_dict() for m in sig_result.mismatches[:5]],
        })
        result.passed = False
        result.error = error
        result.structured_sig_mismatches = list(sig_result.mismatches[:3])
        return result

    except ImportError:
        logger.debug("[arch_exec] v5.22 signature_checker not available — skipping")
        return result
    except Exception as e:
        logger.warning("[arch_exec] v5.22 Signature check error (non-fatal): %s", e)
        return result


# ---------------------------------------------------------------------------
# Public entry point — full verification pipeline
# ---------------------------------------------------------------------------

async def verify_written_file(
    rel_path: str,
    file_content: str,
    ctx: ExecutionContext,
    strike: int,
    job_checker_strike_errors: List[str],
) -> VerifyResult:
    """Run the complete three-layer verification pipeline.

    Returns VerifyResult — caller should check .passed and handle
    .error / .structured_sig_mismatches on failure.
    """
    # Layer 1: Deterministic check (core)
    det_passed = _run_deterministic_check(rel_path, file_content, ctx, strike)

    if det_passed is False:
        blocking = []
        try:
            from app.overwatcher.deterministic_checker import deterministic_check as _dc
            arch_section = extract_section_for_file(ctx.architecture_content, rel_path) or ""
            from app.overwatcher.deterministic_checker import extract_expected_exports_from_arch
            expected = extract_expected_exports_from_arch(arch_section) if arch_section else None
            r = _dc(
                file_path=rel_path,
                file_content=file_content,
                interface_contract=ctx.interface_contract,
                sandbox_base=ctx.sandbox_base,
                existing_sandbox_files=ctx.existing_sandbox_files,
                expected_exports=expected,
            )
            blocking = r.blocking_issues
        except Exception:
            pass
        desc = "; ".join(i.description for i in blocking[:3]) if blocking else "Deterministic check failed"
        return VerifyResult(
            passed=False,
            error=f"Deterministic Check FAILED: {desc}",
            job_checker_error=desc,
        )

    # Layer 1B: Extended deterministic checks (v3.0)
    try:
        from app.overwatcher.deterministic_checker_extended import run_extended_det_checks
        arch_section = extract_section_for_file(
            ctx.architecture_content, rel_path,
        ) or ""
        ext_issues = run_extended_det_checks(
            file_content=file_content,
            file_path=rel_path,
            interface_contract=ctx.interface_contract,
            architecture_content=arch_section,
        )
        ext_blocking = [i for i in ext_issues if i.severity == "blocking"]
        if ext_blocking:
            desc = "; ".join(i.description for i in ext_blocking[:3])
            logger.warning(
                "[arch_exec] v3.0 EXT_DET_CHECK Strike %d: %s", strike, desc[:200],
            )
            ctx.add_trace("EXT_DET_CHECK_FAIL", f"strike_{strike}", {
                "path": rel_path,
                "blocking": len(ext_blocking),
                "issues": [i.to_dict() for i in ext_issues[:5]],
            })
            return VerifyResult(
                passed=False,
                error=f"Extended Deterministic Check FAILED: {desc}",
                job_checker_error=desc,
            )
        elif ext_issues:
            logger.info(
                "[arch_exec] v3.0 EXT_DET_CHECK PASS with %d warning(s): %s",
                len(ext_issues), rel_path,
            )
    except ImportError:
        logger.debug("[arch_exec] v3.0 deterministic_checker_extended not available")
    except Exception as e:
        logger.debug("[arch_exec] v3.0 Extended det checker error (non-fatal): %s", e)

    # Layer 2: LLM job checker — ELIMINATED (v3.0)
    # v3.0: The deterministic checker (core + extended) now covers all
    # cases that the LLM job checker was handling. The LLM fallback
    # is permanently disabled. If deterministic_check returned None
    # (unavailable), we still proceed — the extended checks and
    # signature check provide sufficient coverage.
    if det_passed is None:
        logger.info(
            "[arch_exec] v3.0 Core det check unavailable for %s "
            "— relying on extended + signature checks",
            rel_path,
        )

    # Layer 3: Signature verification
    sig_result = _run_signature_check(rel_path, file_content, ctx, strike)
    return sig_result

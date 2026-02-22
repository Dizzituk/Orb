# FILE: app/orchestrator/phase_checkout_checks.py
"""
Phase Checkout -- Individual Verification Checks.

Contains the implementation of each check run during Phase Checkout:
1. Output file size validation (informational -- only checks segment outputs)
2. Skeleton contract verification (informational -- checks via sandbox)
3. Application boot test with fix loop (pass/fail gate)

v2.0 (2026-02-15): Boot test is now a fix loop. When boot fails, the system:
    1. Parses the traceback to identify the failing file and error type
    2. Reads the broken file from the sandbox
    3. Sends the file + error to the LLM for a targeted fix
    4. Writes the fix back to the sandbox
    5. Retries boot (up to BOOT_MAX_FIX_ATTEMPTS times)
    Size check now only validates segment-produced output files, not
    pre-existing source files. Contract check resolves paths via sandbox.
v2.8 (2026-02-16): Four boot-fix fixes from sg-38070318 post-mortem:
    1. FIX: _investigate_boot_error now receives `state` parameter —
       previously crashed with NameError when searching segment outputs.
    2. FIX: Pre-existing error detection — when grep finds zero matches
       in segment outputs AND BOOT_CHECK_PASS was in stdout, treat as
       PASS (the error is pre-existing, not caused by this refactor).
    3. FIX: Grep file list now filters out quarantined/non-existent
       paths before passing to Select-String.
    4. FIX: Boot check token detection — checks stdout for BOOT_CHECK_PASS
       before treating stderr as failure.
v2.7 (2026-02-16): Three boot-fix hardening improvements:
    1. Silent import detection: scans BOTH stdout+stderr (not just stderr),
       fixes regex group mismatch for "cannot import name X from Y",
       journals silent failures for learning system.
    2. Grep error tolerance: -ErrorAction SilentlyContinue on Select-String
       so quarantined/missing files don't abort the entire search.
    3. Investigation priority: searches segment output files FIRST before
       wider codebase, and prefers segment files when multiple matches found.
       Prevents infrastructure files (phase_checkout_checks.py) being picked
       over the actual broken segment file (__init__.py).
v2.6 (2026-02-15): Relative import awareness. When boot fails with
    "cannot import name 'X' from 'a.b.c.module'", grep and investigation
    now also search for relative import forms like "from .module import X".
    Previously only searched for the full dotted absolute path, which missed
    __init__.py files and sibling modules using relative imports.
v2.5 (2026-02-15): Strike-based boot fix loop. Different errors = progress,
    keeps going. Same error repeating = strikes accumulate per error signature.
    3 strikes on same error = hard stop. Safety cap of 12 total attempts.
    Default raised from 3 to 12 max attempts.
v2.4 (2026-02-15): Investigation step before abort. When failing file
    not found in segment outputs, searches the wider sandbox codebase,
    reads the importing file, and hands it to the LLM to investigate
    and fix -- same as a human would do.
v1.0 (2026-02-14): Extracted from phase_checkout.py for cap compliance.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.pot_spec.grounded.size_models import (
    MAX_FILE_KB,
    MAX_FILE_LINES,
    MAX_FUNCTION_LINES,
)
from .phase_checkout_models import (
    BootTestResult,
    ContractCheckResult,
    ContractViolation,
    SizeValidationResult,
    SizeViolation,
)
from app.orchestrator._phase_checkout_checks_utils_2 import _KNOWN_PREEXISTING_FAILURES, _SILENT_IMPORT_PATTERNS, _build_fix_prompt, _discover_sandbox_base, _file_exists_in_sandbox, _find_largest_function, _try_deterministic_import_fix, map_file_to_segment
from app.orchestrator._phase_checkout_checks_utils_3 import BOOT_MAX_FIX_ATTEMPTS, BOOT_STRIKE_LIMIT, _check_stderr_for_silent_import_failures, _norm, _try_reconciliation_import_fix, _write_file_to_sandbox_abs, check_output_file_sizes, check_skeleton_contracts
from app.orchestrator._phase_checkout_checks_utils_4 import _grep_sandbox_for_bad_import, _pick_boot_fix_max_tokens, _pick_boot_fix_model, _pick_boot_fix_provider, _read_file_via_sandbox, _resolve_output_path, _run_single_boot, _try_quarantine_constant_recovery
from app.orchestrator._phase_checkout_checks_utils_5 import BOOT_FIX_TIMEOUT, _BOOT_FIX_SYSTEM_PROMPT, _extract_fix_content, _parse_boot_failure, _write_file_to_sandbox, run_boot_test_with_fix_loop
from app.orchestrator._phase_checkout_checks_utils_6 import _investigate_boot_error

logger = logging.getLogger(__name__)


# =============================================================================
# CHECK 1: OUTPUT FILE SIZE VALIDATION
# =============================================================================


# =============================================================================
# CHECK 2: SKELETON CONTRACT VERIFICATION
# =============================================================================


# =============================================================================
# CHECK 3: BOOT TEST WITH FIX LOOP
# =============================================================================


async def _attempt_boot_fix(
    client: Any,
    actual_base: str,
    failing_file: str,
    error_summary: str,
    full_stderr: str,
    state: Any,
    emit: Any,
) -> Optional[str]:
    """
    Attempt to fix a boot failure by reading the broken file, sending it
    to the LLM with the error context, and writing the fix back.

    Returns a short description of the fix applied, or None if fix failed.
    """
    _emit = emit or (lambda msg: None)

    # Determine error type for targeted fix
    err_lower = error_summary.lower()
    is_import_error = ("modulenotfounderror" in err_lower or "importerror" in err_lower
                       or "cannot import" in err_lower or "no module named" in err_lower)
    is_syntax_error = "syntaxerror" in err_lower
    is_attribute_error = "attributeerror" in err_lower
    is_name_error = "nameerror" in err_lower

    if not (is_import_error or is_syntax_error or is_attribute_error or is_name_error):
        _emit(f"    Error type not fixable by phase checkout: {error_summary[:100]}")
        return None

    # Read the broken file from sandbox
    broken_content = _read_file_via_sandbox(client, failing_file, actual_base)
    if not broken_content:
        _emit(f"    Cannot read broken file: {failing_file}")
        return None

    # --- v4.0 Fix 4: For SyntaxError, try deterministic sanitise+parse first ---
    if is_syntax_error and failing_file.endswith('.py'):
        try:
            from app.overwatcher.architecture_executor.helpers import (
                _sanitise_python_content, _check_python_syntax,
            )
            sanitised, warnings = _sanitise_python_content(broken_content, failing_file)
            if warnings:
                _emit(f"    [SYNTAX_FIX] Stripped non-Python preamble: {len(warnings)} warning(s)")
                for w in warnings[:3]:
                    _emit(f"      {w[:120]}")
                # Check if sanitised version passes syntax
                syntax_err = _check_python_syntax(sanitised, failing_file)
                if syntax_err is None:
                    # Sanitisation fixed it! Write back without LLM.
                    success = _write_file_to_sandbox(client, failing_file, sanitised, actual_base)
                    if success:
                        return f"Deterministic syntax fix: stripped markdown preamble ({len(warnings)} patterns)"
                else:
                    _emit(f"    [SYNTAX_FIX] Sanitisation wasn't enough: {syntax_err[:120]}")
        except ImportError:
            pass  # helpers not available, fall through to LLM

    # --- For import errors, try smart reconciliation first (no LLM needed) ---
    if is_import_error:
        # v2.1: Try post-execution reconciliation to find the correct import name
        # instead of just commenting out the broken import.
        recon_fix = _try_reconciliation_import_fix(
            client, actual_base, failing_file, broken_content, error_summary, state, _emit,
        )
        if recon_fix:
            fixed_content, fix_desc = recon_fix
            success = _write_file_to_sandbox(client, failing_file, fixed_content, actual_base)
            if success:
                return fix_desc

        # v2.7: If the error is a missing constant from a constants module,
        # try to recover the definition from the quarantined original file.
        # This handles the recurring pattern where the LLM generates constants.py
        # but omits some constants that existed in the original monolith.
        quarantine_fix = _try_quarantine_constant_recovery(
            client, actual_base, error_summary, state, _emit,
        )
        if quarantine_fix:
            return quarantine_fix

        # Fallback: comment out the broken import as a last resort
        deterministic_fix = _try_deterministic_import_fix(
            broken_content, error_summary, full_stderr
        )
        if deterministic_fix:
            fixed_content, fix_desc = deterministic_fix
            success = _write_file_to_sandbox(client, failing_file, fixed_content, actual_base)
            if success:
                return fix_desc

    # --- Fall back to LLM-based fix ---
    fix_prompt = _build_fix_prompt(
        failing_file=failing_file,
        broken_content=broken_content,
        error_summary=error_summary,
        full_stderr=full_stderr,
        error_type="import" if is_import_error else
                   "syntax" if is_syntax_error else
                   "attribute" if is_attribute_error else "name",
    )

    try:
        from app.providers.registry import get_provider_registry
        registry = get_provider_registry()

        # Use a capable model for the fix
        provider_id = _pick_boot_fix_provider()
        model_id = _pick_boot_fix_model()
        llm_result = await registry.llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[
                {"role": "system", "content": _BOOT_FIX_SYSTEM_PROMPT},
                {"role": "user", "content": fix_prompt},
            ],
            max_tokens=_pick_boot_fix_max_tokens(),
            timeout_seconds=BOOT_FIX_TIMEOUT,
        )

        fixed_content = _extract_fix_content(llm_result)
        if not fixed_content or len(fixed_content) < 20:
            _emit("    LLM fix produced empty/minimal content")
            return None

        # Validate the fix isn't destroying the file
        if len(fixed_content) < len(broken_content) * 0.5:
            _emit("    LLM fix removed too much content (>50% reduction) -- rejecting")
            return None

        success = _write_file_to_sandbox(client, failing_file, fixed_content, actual_base)
        if success:
            return f"LLM fix applied ({len(fixed_content)} chars)"
        return None

    except Exception as exc:
        _emit(f"    LLM fix failed: {exc}")
        logger.warning("[phase_checkout] LLM boot fix failed: %s", exc)
        return None


# =============================================================================
# INVESTIGATION STEP (v2.4)
# =============================================================================


# =============================================================================
# SILENT IMPORT FAILURE DETECTION (v2.3)
# =============================================================================

# Import patterns that indicate real failures (not just warnings)

# Known pre-existing import failures to ignore (not from our segments)


# =============================================================================
# BOOT FIX PROMPTS
# =============================================================================


# =============================================================================
# SANDBOX HELPERS
# =============================================================================


# =============================================================================
# PARSING HELPERS
# =============================================================================

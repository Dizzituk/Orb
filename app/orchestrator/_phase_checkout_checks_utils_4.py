from __future__ import annotations
import logging
import os
import re
from app.orchestrator._phase_checkout_checks_utils_3 import _check_stderr_for_silent_import_failures
from typing import Any, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _run_single_boot(
    client: Any,
    actual_base: str,
) -> Tuple[bool, str, str, str, Optional[str]]:
    """
    Run a single boot test. Returns (passed, stdout, stderr, error_summary, failing_file).
    """
    from .phase_checkout_checks import _parse_boot_failure
    venv = actual_base + r"\.venv\Scripts\python.exe"
    # v2.2: Force UTF-8 to prevent UnicodeEncodeError from emoji in generated code
    cmd = (
        f'$env:PYTHONIOENCODING="ascii:replace"; $env:PYTHONUTF8="0" ; '
        f'cd "{actual_base}" ; '
        f'& "{venv}" -c '
        f'"import sys; sys.path.insert(0, r\'{actual_base}\'); '
        f'from main import app; print(\'BOOT_CHECK_PASS\')"'
    )

    try:
        shell_result = client.shell_run(cmd, timeout_seconds=30)
    except Exception as exc:
        return (False, "", str(exc), f"Shell execution failed: {exc}", None)

    stdout = (shell_result.stdout or "").strip()
    stderr = (shell_result.stderr or "").strip()

    if "BOOT_CHECK_PASS" in stdout:
        # v2.7: Scan BOTH stdout and stderr for silent import failures.
        # The app may boot (caught in try/except) but still have broken imports.
        # These are real bugs that need fixing. Warnings appear in stdout (via
        # print) and/or stderr (via logger.warning).
        silent_import_error = _check_stderr_for_silent_import_failures(stderr, stdout)
        if silent_import_error:
            # v2.7: Journal the silent failure so the learning system captures it
            try:
                from app.experience.context import journal_emit
                journal_emit(
                    stage="phase_checkout",
                    event_type="silent_import_failure",
                    severity="error",
                    description=f"Boot passed but critical import failed silently: {silent_import_error[:200]}",
                    error_signature=silent_import_error[:100],
                    root_cause="LLM-generated code imports a symbol that does not exist in the target module",
                    resolution="Boot check now treats silent import failures as boot failures",
                )
            except Exception:
                pass
            return (False, stdout, stderr, silent_import_error, None)
        return (True, stdout, stderr, "", None)

    err_summary, failing_file = _parse_boot_failure(stdout, stderr)
    return (False, stdout, stderr, err_summary, failing_file)

async def _grep_sandbox_for_bad_import(
    client: Any,
    actual_base: str,
    error_summary: str,
    state: Any,
    emit: Any,
) -> Optional[str]:
    """
    v2.2: When _parse_boot_failure can't find a File path in the traceback,
    grep the sandbox for files that contain the offending import.

    Parses error_summary for patterns like:
      - "No module named 'app.logger'"
      - "cannot import name 'X' from 'Y'"

    Then runs Select-String on the sandbox to find which .py file has that import.
    Returns the relative file path, or None.
    """
    _emit = emit or (lambda msg: None)

    # Extract the bad module name
    no_module = re.search(r"No module named '([^']+)'", error_summary)
    cannot_import = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)

    search_pattern = None
    if no_module:
        bad_module = no_module.group(1)
        # Search for 'from app.logger' or 'import app.logger'
        search_pattern = bad_module.replace(".", r"\.")
    elif cannot_import:
        bad_name = cannot_import.group(1)
        bad_source = cannot_import.group(2)
        # v2.6: Also search for RELATIVE imports (e.g. "from .constants import X")
        # When bad_source is "app.overwatcher.architecture_executor.constants",
        # the file doing the import might use "from .constants import X" (relative).
        rel_module = bad_source.rsplit(".", 1)[-1]  # "constants"
        search_pattern = (
            f"{bad_name}.*{bad_source.replace('.', r'\.')}|"
            f"{bad_source.replace('.', r'\.')}.*{bad_name}|"
            f"from \\.{rel_module}.*import.*{bad_name}"
        )

    if not search_pattern:
        return None

    _emit(f"  [GREP] Searching sandbox for files referencing '{search_pattern}'...")

    # Grep across all segment output files on the sandbox
    # Build a list of paths to search from the state
    # v2.8: Validate each path exists on sandbox before including it.
    # Quarantined files (e.g. original monolith moved to .quarantined/)
    # no longer exist at their original path and cause Select-String errors.
    search_paths = []
    for seg_id, seg_state in state.segments.items():
        for rel_path in (seg_state.output_files or []):
            if rel_path.endswith(".py"):
                normed = rel_path.replace("/", "\\")
                if not (normed.startswith("C:") or normed.startswith("D:")):
                    abs_p = f"{actual_base}\\{normed}"
                else:
                    abs_p = normed
                # v2.8: Skip paths that contain .quarantined
                if ".quarantined" in abs_p:
                    continue
                search_paths.append(abs_p)

    if not search_paths:
        _emit("  [GREP] No output files to search")
        return None

    # v2.8: Validate paths exist on sandbox in one batch call
    # This prevents Select-String errors for quarantined/moved files
    try:
        test_paths_str = "; ".join(
            f'if (Test-Path "{p}") {{ "{p}" }}'
            for p in search_paths[:30]
        )
        validate_result = client.shell_run(test_paths_str, timeout_seconds=15)
        valid_stdout = (validate_result.stdout or "").strip()
        if valid_stdout:
            validated_paths = [p.strip() for p in valid_stdout.split("\n") if p.strip()]
            if validated_paths:
                search_paths = validated_paths
    except Exception:
        pass  # Fall through to use unvalidated paths with -ErrorAction SilentlyContinue

    # Run Select-String on the sandbox — batch the paths
    # Use a simple grep command
    paths_str = ",".join(f'"{p}"' for p in search_paths[:30])  # Cap at 30 files
    cmd = (
        f'Select-String -Path {paths_str} '
        f'-Pattern "{search_pattern}" -SimpleMatch -ErrorAction SilentlyContinue '
        f'| Select-Object -First 3 -Property Filename, LineNumber, Line '
        f'| Format-List'
    )

    try:
        result = client.shell_run(cmd, timeout_seconds=15)
        stdout = (result.stdout or "").strip()

        if not stdout:
            # SimpleMatch didn't find it — try without SimpleMatch for regex
            cmd2 = (
                f'Select-String -Path {paths_str} '
                f'-Pattern "{search_pattern}" -ErrorAction SilentlyContinue '
                f'| Select-Object -First 3 -Property Filename, LineNumber, Line '
                f'| Format-List'
            )
            result2 = client.shell_run(cmd2, timeout_seconds=15)
            stdout = (result2.stdout or "").strip()

        if not stdout:
            _emit("  [GREP] No matches found on sandbox")
            return None

        _emit(f"  [GREP] Found: {stdout[:200]}")

        # Extract filename from the result
        fn_match = re.search(r'Filename\s*:\s*(\S+)', stdout)
        if fn_match:
            filename = fn_match.group(1)
            # Find the full relative path from state
            for seg_id, seg_state in state.segments.items():
                for rel_path in (seg_state.output_files or []):
                    if rel_path.endswith(filename):
                        _emit(f"  [GREP] Identified offending file: {rel_path} (from {seg_id})")
                        return rel_path
            # Fallback: return just the filename
            _emit(f"  [GREP] Found file {filename} but can't map to segment")
            return filename

        return None

    except Exception as exc:
        _emit(f"  [GREP] Sandbox search failed: {exc}")
        return None

def _try_quarantine_constant_recovery(
    client: Any,
    actual_base: str,
    error_summary: str,
    state: Any,
    emit: Any,
) -> Optional[str]:
    """
    v2.7: When a constant is missing from a generated constants module,
    recover its definition from the quarantined original file.

    Pattern: "cannot import name 'VERIFY_READ_TIMEOUT' from
    'app.overwatcher.architecture_executor.constants'"

    The quarantined monolith (.quarantined/architecture_executor.py) contains
    the original constant definition. We extract it and append it to the
    generated constants.py.

    Returns fix description if successful, None otherwise.
    """
    from .phase_checkout_checks import _write_file_to_sandbox
    _emit = emit or (lambda msg: None)

    # Only handle "cannot import name" errors
    cannot_import = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)
    if not cannot_import:
        return None

    bad_name = cannot_import.group(1)
    bad_source = cannot_import.group(2)

    # Only handle constants (ALL_CAPS names) from constants-like modules
    if not re.match(r'^[A-Z][A-Z0-9_]+$', bad_name):
        return None

    _emit(f"    [QUARANTINE_RECOVERY] Missing constant '{bad_name}' — searching quarantined files...")

    # Find the quarantine directory — look for .quarantined/ in the parent
    # of the target module. E.g. for app.overwatcher.architecture_executor.constants,
    # the quarantine is at app/overwatcher/.quarantined/architecture_executor.py
    module_parts = bad_source.replace(".", "/")  # app/overwatcher/architecture_executor/constants
    parent_parts = "/".join(module_parts.split("/")[:-2])  # app/overwatcher

    # Search for .quarantined files in the parent directory
    quarantine_dir = f"{actual_base}\\{parent_parts.replace('/', chr(92))}\\.quarantined"
    try:
        result = client.shell_run(
            f'Get-ChildItem -Path "{quarantine_dir}" -Filter "*.py" -ErrorAction SilentlyContinue '
            f'| Select-Object -ExpandProperty Name',
            timeout_seconds=10,
        )
        quarantined_files = [f.strip() for f in (result.stdout or "").strip().split("\n") if f.strip()]
    except Exception:
        quarantined_files = []

    if not quarantined_files:
        _emit(f"    [QUARANTINE_RECOVERY] No quarantined files found in {quarantine_dir}")
        return None

    _emit(f"    [QUARANTINE_RECOVERY] Found quarantined file(s): {quarantined_files}")

    # Read the quarantined file and grep for the constant definition
    for qf in quarantined_files:
        qf_path = f"{quarantine_dir}\\{qf}"
        try:
            # Use Select-String to find the constant definition line(s)
            grep_result = client.shell_run(
                f'Select-String -Path "{qf_path}" -Pattern "^{bad_name}\\s*=" '
                f'-ErrorAction SilentlyContinue '
                f'| Select-Object -ExpandProperty Line',
                timeout_seconds=10,
            )
            found_lines = (grep_result.stdout or "").strip()
        except Exception:
            continue

        if not found_lines:
            continue

        # We found the constant definition — now append it to the target constants file
        # The target is the constants module: bad_source -> file path
        constants_rel = bad_source.replace(".", "/").replace("/", "\\") + ".py"
        if not constants_rel.startswith("D:") and not constants_rel.startswith("C:"):
            constants_abs = f"{actual_base}\\{constants_rel}"
        else:
            constants_abs = constants_rel

        _emit(f"    [QUARANTINE_RECOVERY] Found definition in quarantined file: {found_lines[:120]}")
        _emit(f"    [QUARANTINE_RECOVERY] Appending to {constants_rel}...")

        # Read current constants file
        try:
            current_result = client.shell_run(
                f'Get-Content -Path "{constants_abs}" -Raw -Encoding UTF8',
                timeout_seconds=10,
            )
            current_content = (current_result.stdout or "").strip()
        except Exception:
            _emit(f"    [QUARANTINE_RECOVERY] Cannot read {constants_abs}")
            return None

        if not current_content:
            _emit(f"    [QUARANTINE_RECOVERY] Constants file is empty")
            return None

        # Append the missing constant(s)
        # Take the first definition line (in case grep returned multiple)
        definition = found_lines.split("\n")[0].strip()
        patched = f"{current_content}\n\n# Recovered from quarantined original (boot fix v2.7)\n{definition}\n"

        success = _write_file_to_sandbox(client, constants_abs, patched, actual_base)
        if success:
            fix_desc = f"Quarantine recovery: added '{bad_name}' to {constants_rel} from original monolith"
            _emit(f"    [QUARANTINE_RECOVERY] ✓ {fix_desc}")

            # Journal the recovery for learning
            try:
                from app.experience.context import journal_emit
                journal_emit(
                    stage="boot_fix",
                    event_type="quarantine_constant_recovery",
                    severity="info",
                    description=fix_desc,
                    error_signature=f"missing_constant:{bad_name}",
                    root_cause="LLM omitted constant from generated constants.py",
                    resolution=f"Recovered definition '{definition[:80]}' from quarantined monolith",
                )
            except Exception:
                pass

            return fix_desc

    _emit(f"    [QUARANTINE_RECOVERY] Could not find '{bad_name}' in quarantined files")
    return None

def _pick_boot_fix_provider() -> str:
    """Pick the LLM provider for boot fix from env."""
    return os.environ.get("PHASE_CHECKOUT_PROVIDER", "anthropic")

def _pick_boot_fix_model() -> str:
    """Pick the LLM model for boot fix from env."""
    return os.environ.get("PHASE_CHECKOUT_MODEL", "claude-opus-4-6")

def _pick_boot_fix_max_tokens() -> int:
    """Pick the max output tokens for boot fix from env."""
    return int(os.environ.get("PHASE_CHECKOUT_MAX_OUTPUT_TOKENS", "8000"))

def _read_file_via_sandbox(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> Optional[str]:
    """Read a file from the sandbox filesystem."""
    # Normalise path separators
    normed = rel_path.replace("/", "\\")
    # Build absolute path if not already
    if not (normed.startswith("C:") or normed.startswith("D:")):
        abs_path = f"{sandbox_base}\\{normed}"
    else:
        abs_path = normed

    try:
        result = client.shell_run(
            f'Get-Content -Path "{abs_path}" -Raw -Encoding UTF8',
            timeout_seconds=15,
        )
        content = result.stdout
        if content is not None and len(content.strip()) > 0:
            return content
    except Exception as exc:
        logger.debug("[phase_checkout] Cannot read %s from sandbox: %s", abs_path, exc)

    return None

def _resolve_output_path(rel_path: str, sandbox_base: str) -> Optional[str]:
    """Resolve a relative file path to absolute using sandbox base."""
    normalised = rel_path.replace("/", os.sep).replace("\\", os.sep)
    return os.path.join(sandbox_base, normalised)

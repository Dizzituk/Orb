from __future__ import annotations
import logging
import os
import re
from app.orchestrator._phase_checkout_checks_utils_3 import _write_file_to_sandbox_abs
from app.orchestrator._phase_checkout_checks_utils_4 import _pick_boot_fix_max_tokens, _pick_boot_fix_model, _pick_boot_fix_provider
from app.orchestrator._phase_checkout_checks_utils_5 import BOOT_FIX_TIMEOUT, _BOOT_FIX_SYSTEM_PROMPT, _extract_fix_content
from typing import Any, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


async def _investigate_boot_error(
    client: Any,
    actual_base: str,
    error_summary: str,
    full_stderr: str,
    state: Any,
    emit: Any,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    v2.4: When the failing file is not in segment outputs, investigate the
    wider sandbox codebase. This is what a human would do: search the whole
    project for who imports the bad module, read that file, understand the
    context, and decide what to fix.

    Steps:
    1. Search the ENTIRE sandbox (not just segment outputs) for the bad import
    2. Read the file that imports it
    3. Look at the target -- does the module exist? Was it renamed?
    4. Hand everything to the LLM to decide the fix
    5. Apply the fix

    Returns (failing_file, fix_description) if fixed, (failing_file, None) if
    file found but not fixed, or None if investigation found nothing.
    """
    _emit = emit or (lambda msg: None)

    # Extract the bad module name
    no_module = re.search(r"No module named '([^']+)'", error_summary)
    cannot_import = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)

    if not no_module and not cannot_import:
        return None

    if no_module:
        bad_module = no_module.group(1)
        search_term = bad_module
    else:
        bad_name = cannot_import.group(1)
        bad_source = cannot_import.group(2)
        search_term = bad_source
        # v2.6: Also search for relative import form
        rel_module = bad_source.rsplit(".", 1)[-1]
        rel_search_term = f".{rel_module}"

    # Step 1: Search segment output files FIRST (most likely to contain the bug),
    # then fall back to the wider codebase. v2.7: Prioritise segment outputs.
    _emit(f"  [INVESTIGATE] Searching sandbox for imports of '{search_term}'...")
    grep_output = ""

    # v2.7: First search segment output files directly (highest priority)
    if hasattr(state, 'segments') and state.segments:
        seg_output_paths = []
        for _sid, _ss in state.segments.items():
            for _rp in (_ss.output_files or []):
                if _rp.endswith(".py"):
                    _normed = _rp.replace("/", "\\")
                    if not (_normed.startswith("C:") or _normed.startswith("D:")):
                        seg_output_paths.append(f"{actual_base}\\{_normed}")
                    else:
                        seg_output_paths.append(_normed)
        if seg_output_paths:
            seg_paths_str = ",".join(f'"{p}"' for p in seg_output_paths[:30])
            # Search for both absolute and relative import patterns
            for _sp in [search_term, rel_search_term if cannot_import else None]:
                if not _sp or grep_output:
                    continue
                try:
                    _cmd = (
                        f'Select-String -Path {seg_paths_str} '
                        f'-Pattern "{_sp}" -SimpleMatch -ErrorAction SilentlyContinue '
                        f'| Select-Object -First 5 -Property Path, LineNumber, Line '
                        f'| Format-List'
                    )
                    _res = client.shell_run(_cmd, timeout_seconds=15)
                    grep_output = (_res.stdout or "").strip()
                    if grep_output:
                        _emit(f"  [INVESTIGATE] Found in segment output files (pattern: '{_sp}')")
                except Exception:
                    pass

    # Fall back to wider codebase search if segment outputs didn't match
    if not grep_output:
        _emit(f"  [INVESTIGATE] Segment outputs clean -- searching wider codebase...")
        try:
            app_dir = f"{actual_base}\\app"
            grep_cmd = (
                f'Get-ChildItem -Path "{app_dir}" -Filter "*.py" -Recurse '
                f'-ErrorAction SilentlyContinue '
                f'| Select-String -Pattern "{search_term}" -SimpleMatch '
                f'| Select-Object -First 5 -Property Path, LineNumber, Line '
                f'| Format-List'
            )
            result = client.shell_run(grep_cmd, timeout_seconds=30)
            grep_output = (result.stdout or "").strip()

            # v2.6: If nothing found with absolute path, try relative import form
            if not grep_output and cannot_import:
                rel_grep_cmd = (
                    f'Get-ChildItem -Path "{app_dir}" -Filter "*.py" -Recurse '
                    f'-ErrorAction SilentlyContinue '
                    f'| Select-String -Pattern "{rel_search_term}" -SimpleMatch '
                    f'| Select-Object -First 5 -Property Path, LineNumber, Line '
                    f'| Format-List'
                )
                result_rel = client.shell_run(rel_grep_cmd, timeout_seconds=30)
                grep_output = (result_rel.stdout or "").strip()
                if grep_output:
                    _emit(f"  [INVESTIGATE] Found via relative import pattern '{rel_search_term}'")

            # If nothing in app/, also check main.py and top-level files
            if not grep_output:
                top_cmd = (
                    f'Select-String -Path "{actual_base}\\*.py" '
                    f'-Pattern "{search_term}" -SimpleMatch -ErrorAction SilentlyContinue '
                    f'| Select-Object -First 3 -Property Path, LineNumber, Line '
                    f'| Format-List'
                )
                result2 = client.shell_run(top_cmd, timeout_seconds=10)
                grep_output = (result2.stdout or "").strip()
        except Exception as exc:
            _emit(f"  [INVESTIGATE] Sandbox search failed: {exc}")
            return None

    if not grep_output:
        _emit("  [INVESTIGATE] No files in entire sandbox reference this module")
        return None

    _emit(f"  [INVESTIGATE] Found references:\n{grep_output[:500]}")

    # Step 2: Extract the file path(s) that import the bad module
    path_matches = re.findall(r'Path\s*:\s*(.+?\.py)', grep_output)
    if not path_matches:
        _emit("  [INVESTIGATE] Could not parse file paths from search results")
        return None

    # v2.7: Prefer segment output files over infrastructure files.
    # If we found multiple matches, prioritise files that are in segment outputs.
    seg_output_set = set()
    if hasattr(state, 'segments') and state.segments:
        for _sid, _ss in state.segments.items():
            for _rp in (_ss.output_files or []):
                seg_output_set.add(os.path.normpath(_rp))
                if not (_rp.startswith("C:") or _rp.startswith("D:")):
                    seg_output_set.add(os.path.normpath(f"{actual_base}\\{_rp}"))

    importing_file_abs = path_matches[0].strip()  # default: first match
    for _pm in path_matches:
        _pm_clean = _pm.strip()
        if os.path.normpath(_pm_clean) in seg_output_set:
            importing_file_abs = _pm_clean
            break
    _emit(f"  [INVESTIGATE] Importing file: {importing_file_abs}")

    # Read the importing file
    importing_content = None
    try:
        result = client.shell_run(
            f'Get-Content -Path "{importing_file_abs}" -Raw -Encoding UTF8',
            timeout_seconds=15,
        )
        importing_content = (result.stdout or "").strip()
    except Exception:
        pass

    if not importing_content:
        _emit(f"  [INVESTIGATE] Cannot read {importing_file_abs}")
        return None

    # Step 3: Check if the target module/file exists (maybe it was renamed)
    target_investigation = ""
    if no_module:
        # Convert module path to file path: app.logger_setup -> app/logger_setup.py
        module_as_path = bad_module.replace(".", "\\") + ".py"
        module_as_pkg = bad_module.replace(".", "\\") + "\\__init__.py"
        target_file_path = f"{actual_base}\\{module_as_path}"
        target_pkg_path = f"{actual_base}\\{module_as_pkg}"

        # Check if target exists
        try:
            r1 = client.shell_run(f'Test-Path -Path "{target_file_path}" -PathType Leaf', timeout_seconds=10)
            r2 = client.shell_run(f'Test-Path -Path "{target_pkg_path}" -PathType Leaf', timeout_seconds=10)
            target_exists_file = (r1.stdout or "").strip().lower() == "true"
            target_exists_pkg = (r2.stdout or "").strip().lower() == "true"
        except Exception:
            target_exists_file = False
            target_exists_pkg = False

        if target_exists_file or target_exists_pkg:
            target_investigation += f"Target module file EXISTS at {target_file_path if target_exists_file else target_pkg_path}. The import should work -- investigate why it fails.\n"
        else:
            target_investigation += f"Target module file does NOT exist at {target_file_path} or {target_pkg_path}.\n"
            # Look for similar files in the same directory
            parent_dir = os.path.dirname(target_file_path)
            try:
                ls_result = client.shell_run(
                    f'Get-ChildItem -Path "{parent_dir}" -Filter "*.py" -ErrorAction SilentlyContinue '
                    f'| Select-Object -ExpandProperty Name',
                    timeout_seconds=10,
                )
                nearby_files = (ls_result.stdout or "").strip()
                if nearby_files:
                    target_investigation += f"Files that DO exist in {parent_dir}:\n{nearby_files}\n"
            except Exception:
                pass

    # Step 4: Hand everything to the LLM for investigation and fix
    _emit("  [INVESTIGATE] Sending to LLM for analysis and fix...")

    investigation_prompt = (
        f"BOOT ERROR: {error_summary}\n\n"
        f"FULL STDERR (last 2000 chars):\n{full_stderr[-2000:]}\n\n"
        f"INVESTIGATION RESULTS:\n{target_investigation}\n"
        f"IMPORTING FILE ({importing_file_abs}):\n"
        f"```python\n{importing_content[:6000]}\n```\n\n"
        f"YOUR TASK:\n"
        f"1. Understand WHY this boot error is happening\n"
        f"2. Determine the minimal fix needed\n"
        f"3. Output the COMPLETE fixed file content\n\n"
        f"Common causes:\n"
        f"- Import references a module that was renamed or moved\n"
        f"- Import is for a module that doesn't exist (hallucinated by code generator)\n"
        f"- Import is wrapped in try/except but the except handler also fails\n"
        f"- Import path is wrong (e.g. 'app.logger_setup' should be 'app.logging_config')\n\n"
        f"Output ONLY the complete fixed file. No explanations, no markdown fences."
    )

    try:
        from app.providers.registry import get_provider_registry
        registry = get_provider_registry()

        provider_id = _pick_boot_fix_provider()
        model_id = _pick_boot_fix_model()
        llm_result = await registry.llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[
                {"role": "system", "content": _BOOT_FIX_SYSTEM_PROMPT},
                {"role": "user", "content": investigation_prompt},
            ],
            max_tokens=_pick_boot_fix_max_tokens(),
            timeout_seconds=BOOT_FIX_TIMEOUT + 60,  # Extra time for investigation
        )

        fixed_content = _extract_fix_content(llm_result)
        if not fixed_content or len(fixed_content) < 20:
            _emit("  [INVESTIGATE] LLM investigation produced empty/minimal content")
            # Return the file path so _attempt_boot_fix can try its own approach
            rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
            return (rel_path, None)

        # Sanity check: don't let LLM destroy the file
        if len(fixed_content) < len(importing_content) * 0.5:
            _emit("  [INVESTIGATE] LLM fix removed too much content (>50% reduction) -- rejecting")
            rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
            return (rel_path, None)

        # Write the fix to the sandbox
        success = _write_file_to_sandbox_abs(client, importing_file_abs, fixed_content)
        if success:
            rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
            _emit(f"  [INVESTIGATE] Fix written to {rel_path}")
            return (rel_path, f"Investigation fix: {error_summary[:80]}")
        else:
            _emit("  [INVESTIGATE] Failed to write fix to sandbox")
            return None

    except Exception as exc:
        _emit(f"  [INVESTIGATE] LLM investigation failed: {exc}")
        logger.warning("[phase_checkout] Investigation LLM call failed: %s", exc)
        rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
        return (rel_path, None)

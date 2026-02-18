"""
Main orchestrator for architecture execution.

Contains run_architecture_execution() — the primary entry point that
supervises the full architecture-to-code pipeline:
1. Parses architecture documents for file operations
2. Calls the Implementer LLM to generate file content
3. Delegates writes via run_implementer_task()
4. Verifies results independently via sandbox reads
5. Implements three-strike error handling per task

Extracted from the original architecture_executor.py monolith.
All utility functions are imported from sibling modules.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..spec_resolution import ResolvedSpec
from ..sandbox_client import SandboxClient, get_sandbox_client

from .constants import (
    ARCHITECTURE_EXECUTOR_BUILD_ID,
    MAX_STRIKES_PER_TASK,
    IMPLEMENTER_MAX_TOKENS,
    VERIFY_READ_TIMEOUT,
    SOURCE_CONTEXT_MAX_CHARS,
    MODIFY_EDIT_MODE_THRESHOLD,
    INTERFACE_SUMMARY_MAX_CHARS,
)
from .prompts import (
    IMPLEMENTER_NEW_FILE_SYSTEM,
    IMPLEMENTER_MODIFY_FILE_SYSTEM,
    IMPLEMENTER_MODIFY_EDIT_SYSTEM,
    _parse_edit_pairs,
)
from .parsing import (
    parse_file_inventory,
    extract_section_for_file,
    _extract_verbatim_code_from_architecture,
)
from .context import (
    _read_existing_file,
    _read_source_context,
    _format_job_context,
    _extract_file_interfaces,
    _extract_existing_imports,
    _extract_router_registrations,
    _build_resolved_endpoints,
)
from .helpers import _extract_llm_content, _strip_markdown_fences
from .sandbox_ops import _verify_file_via_sandbox, _resolve_sandbox_base
from .path_resolution import _resolve_multi_root_path, _ensure_python_init_files, _infer_lang_from_path
from .source_extraction import _detect_source_files_from_architecture
from .process_boot_strike_loop import BOOT_MAX_STRIKES, _run_boot_check, _parse_broken_file_from_traceback

logger = logging.getLogger(__name__)


async def run_architecture_execution(
    *,
    spec: ResolvedSpec,
    architecture_content: str,
    architecture_path: str,
    job_id: str,
    llm_call_fn: Optional[Callable] = None,
    artifact_root: str = "D:/Orb/jobs",
    interface_contract: str = "",
    skip_boot_check: bool = False,
) -> Dict[str, Any]:
    """Supervise architecture-level spec execution.
    
    The Overwatcher (this function) is the supervisor. It:
    1. Parses the architecture document to find file operations
    2. For each file, calls the Implementer LLM (Sonnet) to generate content
    3. Delegates each write to the Implementer via run_implementer_task()
    4. Reads back from sandbox to independently verify
    5. Implements three-strike error handling per task
    
    The Implementer LLM (Sonnet) generates the code.
    The Implementer module (implementer.py) writes it to the sandbox.
    The Overwatcher (this module) only reads for verification.
    """
    start_time = time.time()
    trace: List[Dict[str, Any]] = []
    artifacts_written: List[str] = []
    
    def elapsed_ms() -> int:
        return int((time.time() - start_time) * 1000)
    
    def add_trace(stage: str, status: str, details: Optional[Dict] = None):
        trace.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })
        # Emit to Build Journal (fire-and-forget, never crashes pipeline)
        try:
            from app.experience.journal_writer import emit_from_trace
            _job_dir = os.path.join(artifact_root, "jobs", job_id)
            emit_from_trace(
                job_id=job_id,
                job_dir=_job_dir,
                trace_stage=stage,
                trace_status=status,
                trace_details=details,
            )
        except Exception:
            pass  # Journal must never crash the pipeline
    
    add_trace("ARCHITECTURE_EXECUTION_START", "started", {
        "spec_id": spec.spec_id,
        "architecture_path": architecture_path,
        "architecture_chars": len(architecture_content),
        "job_id": job_id,
    })
    
    logger.info(
        "[arch_exec] v2.1 Starting architecture execution for spec %s (%d chars)",
        spec.spec_id, len(architecture_content),
    )
    print(f"[ARCH_EXEC] Starting: spec={spec.spec_id}, arch={len(architecture_content)} chars")
    
    # =========================================================================
    # Step 1: Parse file inventory
    # =========================================================================
    new_files, modified_files = parse_file_inventory(architecture_content)
    total_operations = len(new_files) + len(modified_files)
    
    logger.info("[arch_exec] Files: %d new, %d modified", len(new_files), len(modified_files))
    print(f"[ARCH_EXEC] Files: {len(new_files)} new, {len(modified_files)} modified")
    
    add_trace("ARCHITECTURE_PARSE", "success", {
        "new_files": [f["path"] for f in new_files],
        "modified_files": [f["path"] for f in modified_files],
        "total_operations": total_operations,
    })
    
    if total_operations == 0:
        error_msg = "No file operations found in architecture document."
        logger.error("[arch_exec] v3.1 HARD FAIL: %s (arch_length=%d chars)", error_msg, len(architecture_content or ""))
        print(f"[ARCH_EXEC] âŒ HARD FAIL: {error_msg} â€” parser found 0 operations in {len(architecture_content or '')} chars of architecture")
        add_trace("ARCHITECTURE_PARSE", "failed", {"error": error_msg, "arch_length": len(architecture_content or "")})
        return {"success": False, "decision": "FAIL", "error": error_msg, "trace": trace, "artifacts_written": []}
    
    # =========================================================================
    # Step 2: Validate prerequisites
    # =========================================================================
    if llm_call_fn is None:
        error_msg = "LLM function required for architecture execution"
        add_trace("ARCHITECTURE_EXECUTION", "failed", {"error": error_msg})
        return {"success": False, "decision": "FAIL", "error": error_msg, "trace": trace, "artifacts_written": []}
    
    # Get sandbox client (READ-ONLY for Overwatcher â€” verification only)
    client = get_sandbox_client()
    if not client.is_connected():
        error_msg = "SAFETY: Sandbox not available"
        add_trace("ARCHITECTURE_EXECUTION", "failed", {"error": error_msg})
        return {"success": False, "decision": "FAIL", "error": error_msg, "trace": trace, "artifacts_written": []}
    
    add_trace("SANDBOX_CONNECTED", "success")
    
    # Get Implementer LLM config
    try:
        from app.llm.stage_models import get_implementer_config
        impl_config = get_implementer_config()
        impl_provider = impl_config.provider
        impl_model = impl_config.model
        impl_max_tokens = impl_config.max_output_tokens or IMPLEMENTER_MAX_TOKENS
    except Exception as e:
        logger.warning("[arch_exec] Could not load implementer config: %s â€” using defaults", e)
        impl_provider = "anthropic"
        impl_model = "claude-sonnet-4-5-20250929"
        impl_max_tokens = IMPLEMENTER_MAX_TOKENS
    
    # =========================================================================
    # Step 3: Resolve sandbox base path (READ-ONLY check)
    # =========================================================================
    sandbox_base = _resolve_sandbox_base(client)
    logger.info("[arch_exec] Sandbox base: %s", sandbox_base)
    add_trace("SANDBOX_BASE_RESOLVED", "success", {"base_path": sandbox_base})
    
    # =========================================================================
    # Step 3b: v2.6 Auto-create __init__.py for new Python packages
    # =========================================================================
    try:
        init_files = _ensure_python_init_files(
            new_files, modified_files, sandbox_base, client
        )
        if init_files:
            # Prepend to new_files so they're created BEFORE the files that need them
            new_files = init_files + new_files
            total_operations = len(new_files) + len(modified_files)
            add_trace("AUTO_INIT_PY", "success", {
                "init_files_added": [f["path"] for f in init_files],
                "new_total_operations": total_operations,
            })
            logger.info(
                "[arch_exec] v2.6 Added %d __init__.py files, total ops now %d",
                len(init_files), total_operations,
            )
    except Exception as e:
        # Non-fatal â€” continue without auto-init if it fails
        logger.warning("[arch_exec] v2.6 _ensure_python_init_files failed: %s", e)
        add_trace("AUTO_INIT_PY", "failed", {"error": str(e)})
    
    # =========================================================================
    # Step 3b: Module shadowing pre-flight check (v2.8)
    # Prevents creating a directory/package that shadows an existing .py file.
    # e.g. creating stream_utils/__init__.py when stream_utils.py already exists
    # would break all existing imports of stream_utils.
    # =========================================================================
    shadowing_blocked = []
    shadowing_renamed = []  # v2.9: refactor-to-package auto-rename

    # v3.2: Detect file->package refactors. If the segment job is creating
    # an __init__.py inside a directory that shadows an existing .py module,
    # this is an intentional conversion (e.g. segment_loop.py -> segment_loop/).
    # The original .py is quarantined AFTER all submodules are written.
    # Skip the shadow check for all files in that package.
    # v3.3: Also check if __init__.py already exists on disk (created by a
    # prior segment in this job). The package dir already exists, so later
    # segments adding files to it should also skip the shadow check.
    _refactor_package_dirs: set = set()
    _all_new_paths = {f["path"].replace("\\", "/") for f in new_files}
    for _np in _all_new_paths:
        if _np.endswith("/__init__.py"):
            _pkg_dir = _np.rsplit("/", 1)[0]
            _refactor_package_dirs.add(_pkg_dir)
    # v3.3: Check on-disk __init__.py for packages that prior segments created
    if not _refactor_package_dirs:
        for file_info in new_files:
            _fp = file_info["path"].replace("\\", "/")
            _parts = _fp.split("/")
            for _depth in range(1, len(_parts)):
                _dir_seg = "/".join(_parts[:_depth])
                _init_path = _dir_seg + "/__init__.py"
                _shadow_py = _dir_seg + ".py"
                # If both __init__.py exists AND the shadow .py exists,
                # a prior segment already started the file->package conversion
                try:
                    _check_init = client.shell_run(
                        f'if (Test-Path -Path "{_resolve_multi_root_path(_init_path, sandbox_base)}") '
                        f'{{ "EXISTS" }} else {{ "NONE" }}',
                        timeout_seconds=10,
                    )
                    _check_shadow = client.shell_run(
                        f'if (Test-Path -Path "{_resolve_multi_root_path(_shadow_py, sandbox_base)}") '
                        f'{{ "EXISTS" }} else {{ "NONE" }}',
                        timeout_seconds=10,
                    )
                    if (_check_init.stdout and "EXISTS" in _check_init.stdout
                            and _check_shadow.stdout and "EXISTS" in _check_shadow.stdout):
                        _refactor_package_dirs.add(_dir_seg)
                except Exception:
                    pass
    if _refactor_package_dirs:
        logger.info(
            "[arch_exec] v3.2 File->package refactor detected: %s — shadow check skipped for package contents",
            _refactor_package_dirs,
        )

    for file_info in new_files:
        new_path = file_info["path"]
        # If the new file lives inside a directory, check if a .py file
        # with the same name as that directory already exists
        parts = new_path.replace("\\", "/").split("/")

        # v3.2: Skip shadow check if this file is inside a deliberate
        # file->package refactor (the package has an __init__.py planned)
        _new_path_norm = new_path.replace("\\", "/")
        _skip_shadow = any(
            _new_path_norm.startswith(pkg_dir + "/")
            for pkg_dir in _refactor_package_dirs
        )
        if _skip_shadow:
            continue

        for depth in range(1, len(parts)):
            dir_segment = "/".join(parts[:depth])
            existing_py = dir_segment + ".py"
            # Check via sandbox filesystem
            try:
                check_cmd = (
                    f'if (Test-Path -Path "{_resolve_multi_root_path(existing_py, sandbox_base)}") '
                    f'{{ "EXISTS" }} else {{ "NONE" }}'
                )
                check_result = client.shell_run(check_cmd, timeout_seconds=10)
                if check_result.stdout and "EXISTS" in check_result.stdout:
                    shadowing_blocked.append({
                        "new_path": new_path,
                        "shadows": existing_py,
                        "dir_segment": dir_segment,
                        "reason": (
                            f"Creating '{new_path}' would create a package directory "
                            f"that shadows existing module '{existing_py}'. "
                            f"Python resolves packages before modules, so all "
                            f"existing 'import {dir_segment.replace('/', '.')}' "
                            f"statements would break."
                        ),
                    })
            except Exception as e:
                logger.warning("[arch_exec] v2.8 Shadow check failed for %s: %s", new_path, e)

    if shadowing_blocked:
        # v2.9 DEPRECATED: The rename logic that was here is now handled by
        # package_quarantine.py at the segment_loop level BEFORE any segments execute.
        # The quarantine moves the .py file into a .quarantined/ folder, so by the
        # time we reach here the shadow should not exist.
        #
        # If we still see shadows at this point, it means quarantine failed or
        # wasn't run â€” log a clear error rather than attempting a second rename.
        for blocked in shadowing_blocked:
            dir_seg = blocked["dir_segment"]
            init_path = dir_seg + "/__init__.py"
            new_paths_set = {f["path"].replace("\\", "/") for f in new_files}
            if init_path in new_paths_set:
                logger.error(
                    "[arch_exec] v2.9 Shadow still exists after quarantine for %s â€” "
                    "package_quarantine may have failed. Check quarantine logs.",
                    blocked["shadows"],
                )
                print(
                    f"[ARCH_EXEC] âš  Shadow conflict: {blocked['shadows']} still exists. "
                    f"Expected package_quarantine to have moved it."
                )

        # Any remaining blocked entries are genuine conflicts (no __init__.py planned)
        for blocked in shadowing_blocked:
            logger.error(
                "[arch_exec] v2.8 MODULE SHADOW BLOCKED: %s shadows %s",
                blocked["new_path"], blocked["shadows"],
            )
            print(f"[ARCH_EXEC] \u2717 BLOCKED: {blocked['reason']}")
            add_trace("MODULE_SHADOW_BLOCKED", "fatal", blocked)

        # Remove remaining shadowing files from new_files so they don't get created
        shadow_paths = {b["new_path"] for b in shadowing_blocked}
        original_count = len(new_files)
        new_files = [f for f in new_files if f["path"] not in shadow_paths]
        if original_count != len(new_files):
            logger.info(
                "[arch_exec] v2.8 Removed %d shadowing files from task list",
                original_count - len(new_files),
            )

    # =========================================================================
    # Step 4: Process all file tasks
    # =========================================================================
    # Import the Implementer's atomic task interface
    from ..implementer import run_implementer_task, run_implementer_edit_task
    
    files_created = 0
    files_modified_count = 0
    files_failed = 0
    
    all_tasks = (
        [{"info": f, "action": "create"} for f in new_files] +
        [{"info": f, "action": "modify"} for f in modified_files]
    )
    
    # v2.3: Cross-file context accumulator
    # After each successful file operation, we capture key interfaces
    # and inject them as context for subsequent Implementer calls.
    job_context: Dict[str, str] = {}
    # v2.5: Router registration prefix tracker (from main.py include_router calls)
    router_registrations: Dict[str, str] = {}
    # v2.5: Track file contents for two-pass re-extraction
    created_file_contents: Dict[str, str] = {}  # rel_path -> content
    
    # v5.11: Build set of existing files on sandbox for import validation.
    # This allows the Job Checker to verify imports against files that were
    # created by previously completed segments, not just the host filesystem.
    _existing_sandbox_files: set = set()
    try:
        # Scan the package directory if we're working inside one
        _all_task_paths = [t["info"]["path"] for t in all_tasks]
        _pkg_dirs = set()
        for _tp in _all_task_paths:
            _tp_norm = _tp.replace("\\", "/")
            _parts = _tp_norm.rsplit("/", 1)
            if len(_parts) == 2:
                _pkg_dirs.add(_parts[0])
        for _pkg_dir in _pkg_dirs:
            _abs_pkg = os.path.join(sandbox_base, _pkg_dir.replace("/", os.sep))
            _scan_cmd = (
                f'if (Test-Path "{_abs_pkg}" -PathType Container) {{ '
                f'Get-ChildItem -Path "{_abs_pkg}" -Filter "*.py" -File | '
                f'ForEach-Object {{ $_.Name }} '
                f'}} else {{ "" }}'
            )
            _scan_result = client.shell_run(_scan_cmd, timeout_seconds=10)
            if _scan_result.stdout:
                for _fname in _scan_result.stdout.strip().split("\n"):
                    _fname = _fname.strip()
                    if _fname:
                        _existing_sandbox_files.add(f"{_pkg_dir}/{_fname}")
        if _existing_sandbox_files:
            logger.info(
                "[arch_exec] v5.11 Found %d existing .py files on sandbox for import validation: %s",
                len(_existing_sandbox_files), sorted(_existing_sandbox_files),
            )
        # v5.15: Also scan PARENT directories so the LLM knows what modules
        # are available via relative `..` imports. Without this, the LLM sees
        # sibling modules but not parent-level modules like implementer.py,
        # spec_resolution.py, sandbox_client.py â€” and hallucinates absolute
        # import paths instead of using the correct `from ..X import Y`.
        _parent_module_files: set = set()
        for _pkg_dir in _pkg_dirs:
            _pkg_norm = _pkg_dir.replace("\\", "/")
            _parent_parts = _pkg_norm.rsplit("/", 1)
            if len(_parent_parts) == 2:
                _parent_dir = _parent_parts[0]
            else:
                _parent_dir = "."
            _abs_parent = os.path.join(sandbox_base, _parent_dir.replace("/", os.sep))
            _parent_scan_cmd = (
                f'if (Test-Path "{_abs_parent}" -PathType Container) {{ '
                f'Get-ChildItem -Path "{_abs_parent}" -Filter "*.py" -File | '
                f'ForEach-Object {{ $_.Name }} '
                f'}} else {{ "" }}'
            )
            _parent_result = client.shell_run(_parent_scan_cmd, timeout_seconds=10)
            if _parent_result.stdout:
                for _fname in _parent_result.stdout.strip().split("\n"):
                    _fname = _fname.strip()
                    if _fname:
                        _parent_module_files.add(f"{_parent_dir}/{_fname}")
        if _parent_module_files:
            logger.info(
                "[arch_exec] v5.15 Found %d parent-level .py modules for `..` import evidence: %s",
                len(_parent_module_files), sorted(_parent_module_files),
            )
    except Exception as _scan_err:
        logger.warning("[arch_exec] v5.11 Sandbox file scan failed: %s", _scan_err)
        _parent_module_files = set()
    
    # v5.12: Also add ALL files from this segment's task list as "planned".
    # When _executor.py imports from process_task_loop_part1.py (file [2/5]),
    # the Job Checker needs to know that file is about to be created even though
    # it doesn't exist on disk yet. This prevents false "import not found" errors
    # for intra-segment cross-file imports.
    for _task in all_tasks:
        _task_path = _task["info"]["path"].replace("\\", "/")
        _existing_sandbox_files.add(_task_path)
    logger.info(
        "[arch_exec] v5.12 Total known files for import validation: %d (sandbox + planned)",
        len(_existing_sandbox_files),
    )
    # v5.15: Build available-modules evidence string for Implementer prompts.
    # This tells the LLM exactly which sibling AND parent modules exist
    # so it never invents imports to non-existent files.
    _available_modules_evidence = ""
    if _existing_sandbox_files or _parent_module_files:
        _evidence_parts = []
        _evidence_parts.append(
            "\n\n## Available Modules (DO NOT invent imports outside this list)\n"
        )
        if _existing_sandbox_files:
            _sorted_siblings = sorted(_existing_sandbox_files)
            _sib_lines = [f"  - `{m}`" for m in _sorted_siblings]
            _evidence_parts.append(
                "### Sibling modules (use `from .module import ...`)\n"
                "These are in the same package. Import with a single dot.\n\n"
                + "\n".join(_sib_lines)
                + "\n"
            )
        if _parent_module_files:
            _sorted_parents = sorted(_parent_module_files)
            _par_lines = [f"  - `{m}`" for m in _sorted_parents]
            _evidence_parts.append(
                "\n### Parent modules (use `from ..module import ...`)\n"
                "These are in the parent package directory. Import with double dot `..`.\n"
                "Do NOT use absolute imports like `from app.x.y import Z`. "
                "Use RELATIVE imports: `from ..module_name import ClassName`.\n\n"
                + "\n".join(_par_lines)
                + "\n"
            )
        _evidence_parts.append(
            "\n**CRITICAL**: Do NOT invent imports to files not listed above. "
            "Do NOT use absolute imports (e.g. `from app.models.X`) when a relative "
            "import from the parent package exists (e.g. `from ..X import Y`). "
            "Every import MUST resolve to a file in one of these lists.\n"
        )
        _available_modules_evidence = "\n".join(_evidence_parts)

    # v2.5: Identify boundary between CREATE and MODIFY tasks for two-pass
    create_count = len(new_files)
    
    for i, task in enumerate(all_tasks, 1):
        file_info = task["info"]
        action = task["action"]
        rel_path = file_info["path"]
        # v2.2: Multi-root path resolution
        # Frontend files (orb-desktop/ prefix) resolve to D:\orb-desktop
        # Backend files resolve to sandbox_base (D:\Orb)
        abs_path = _resolve_multi_root_path(rel_path, sandbox_base)
        
        logger.info("[arch_exec] [%d/%d] %s: %s", i, total_operations, action.upper(), rel_path)
        print(f"[ARCH_EXEC] [{i}/{total_operations}] {action.upper()}: {rel_path}")
        
        add_trace("FILE_TASK_START", "processing", {
            "operation": action,
            "relative_path": rel_path,
            "absolute_path": abs_path,
            "task_number": i,
        })
        
        # =====================================================================
        # v5.13: Quarantine-aware skip for MODIFY/DELETE on quarantined files
        # When a fileâ†’package refactor is in progress, the quarantine system
        # moves the old .py file into a .quarantined/ folder BEFORE segments
        # execute. If the architecture says to DELETE or MODIFY a file that
        # has already been quarantined, there's nothing for the Implementer
        # to do â€” the quarantine already handled it. Skip the task.
        # =====================================================================
        if action == "modify":
            _rel_norm = rel_path.replace("\\", "/")
            # Check if the file description indicates a DELETE operation
            _desc_lower = file_info.get("description", "").lower()
            _is_delete = any(kw in _desc_lower for kw in [
                "delete", "remove entirely", "superseded", "replaced by",
                "no longer exists", "remove this file",
            ])
            # Also check if the architecture section says DELETE
            if not _is_delete:
                _file_section = extract_section_for_file(architecture_content, rel_path)
                if _file_section:
                    _sec_lower = _file_section.lower()
                    _is_delete = any(phrase in _sec_lower for phrase in [
                        "delete this file", "remove entirely", "removed entirely",
                        "file is removed", "this file is superseded",
                        "this file must be deleted", "must not exist",
                        "no longer exists", "must be removed",
                    ])
            # Check if the file is already in quarantine
            if _is_delete:
                # Build quarantine path: parent_dir/.quarantined/filename
                _path_parts = _rel_norm.rsplit("/", 1)
                if len(_path_parts) == 2:
                    _q_dir = f"{_path_parts[0]}/.quarantined"
                    _q_file = _path_parts[1]
                    _q_abs = _resolve_multi_root_path(f"{_q_dir}/{_q_file}", sandbox_base)
                else:
                    _q_abs = _resolve_multi_root_path(f".quarantined/{_rel_norm}", sandbox_base)
                try:
                    _q_check_cmd = (
                        f'if (Test-Path -Path "{_q_abs}" -PathType Leaf) '
                        f'{{ "QUARANTINED" }} else {{ "NONE" }}'
                    )
                    _q_check = client.shell_run(_q_check_cmd, timeout_seconds=10)
                    if _q_check.stdout and "QUARANTINED" in _q_check.stdout:
                        # File is in quarantine â€” also verify the original is gone
                        _orig_check_cmd = (
                            f'if (Test-Path -Path "{abs_path}" -PathType Leaf) '
                            f'{{ "EXISTS" }} else {{ "GONE" }}'
                        )
                        _orig_check = client.shell_run(_orig_check_cmd, timeout_seconds=10)
                        _orig_gone = _orig_check.stdout and "GONE" in _orig_check.stdout
                        if _orig_gone:
                            logger.info(
                                "[arch_exec] v5.13 QUARANTINE SKIP: %s â€” file already quarantined at %s",
                                rel_path, _q_abs,
                            )
                            print(
                                f"[ARCH_EXEC] v5.13 âœ“ SKIP (quarantined): {rel_path} â€” "
                                f"already moved to .quarantined/ by package_quarantine"
                            )
                            add_trace("QUARANTINE_SKIP", "success", {
                                "path": rel_path,
                                "quarantine_path": _q_abs,
                                "reason": "File quarantined by package_quarantine, no action needed",
                            })
                            # Count as success â€” the quarantine did the work
                            files_modified_count += 1
                            artifacts_written.append(abs_path)
                            continue  # Skip to next task
                except Exception as _q_err:
                    logger.warning(
                        "[arch_exec] v5.13 Quarantine check failed for %s: %s â€” proceeding normally",
                        rel_path, _q_err,
                    )
        
        # =====================================================================
        # Three-strike error loop
        # =====================================================================
        task_success = False
        last_error = None
        _job_checker_strike_errors: list = []  # v2.2: Accumulate checker feedback across strikes
        
        for strike in range(1, MAX_STRIKES_PER_TASK + 1):
            logger.info("[arch_exec] %s strike %d/%d", rel_path, strike, MAX_STRIKES_PER_TASK)
            
            # --- v2.6: Skip LLM for auto-generated __init__.py files ---
            if rel_path.endswith("__init__.py") and file_info.get("description", "").startswith("v2.6 auto-created"):
                file_content = "# Auto-generated by architecture executor v2.6\n"
                logger.info("[arch_exec] v2.6 Direct-writing __init__.py: %s", rel_path)
                
                try:
                    impl_result = await run_implementer_task(
                        path=abs_path,
                        content=file_content,
                        action="create",
                        ensure_parents=True,
                        client=client,
                    )
                    if impl_result.success:
                        task_success = True
                    else:
                        last_error = f"Implementer write failed for __init__.py: {impl_result.error}"
                except Exception as e:
                    last_error = f"__init__.py write exception: {e}"
                break  # No retries needed for __init__.py
            
            # --- Generate content via Implementer LLM ---
            try:
                file_context = extract_section_for_file(architecture_content, rel_path)
                
                if not file_context:
                    last_error = f"No architecture context found for {rel_path}"
                    logger.warning("[arch_exec] %s", last_error)
                    break  # No point retrying â€” architecture doesn't mention this file
                
                # v2.3/v2.5: Build cross-file context section (with resolved endpoints)
                job_context_section = _format_job_context(job_context, router_registrations)
                
                use_edit_mode = False  # v1.13: default, overridden in MODIFY branch for large files
                verbatim_content = None  # v1.13: set if verbatim extraction succeeds
                
                # =============================================================
                # v5.23: Extract contract signatures for this specific file
                # and inject them directly into the Implementer prompt.
                # This is the PRIMARY fix for seg-06 failures: the Implementer
                # now sees the exact required signatures from the skeleton
                # contract, not just the architecture's paraphrase.
                # =============================================================
                _per_file_contract_block = ""
                try:
                    from ..signature_checker import extract_contract_signatures_for_file as _extract_sigs
                    _file_contract_sigs = _extract_sigs(interface_contract, rel_path)
                    # v5.23b: Also extract bare export names (no def prefix)
                    # These are equally important — e.g. "execute_approved_segment_architecture"
                    _bare_export_names = []
                    try:
                        import re as _re
                        _fc_lines = interface_contract.split("\n")
                        _in_file = False
                        _in_exports = False
                        _file_norm = rel_path.replace("\\", "/").strip()
                        for _cl in _fc_lines:
                            _cs = _cl.strip()
                            # Use original indentation to distinguish file entries (2-space)
                            # from export symbols (6-space)
                            _indent = len(_cl) - len(_cl.lstrip())
                            # v5.24: Normalise backslashes in contract line (skeleton stores Windows paths)
                            _cs_norm = _cs.replace("\\", "/")
                            if f"`{_file_norm}`" in _cs_norm and _indent <= 4:
                                _in_file = True
                                _in_exports = False
                                continue
                            if _in_file:
                                if "MUST EXPORT" in _cs:
                                    _in_exports = True
                                    continue
                                if _cs.startswith("###") or _cs.startswith("## "):
                                    _in_file = False
                                    _in_exports = False
                                    continue
                                # Detect new file entry (low indent, contains a path)
                                if _indent <= 4 and _cs.startswith("- `"):
                                    _cm = _re.match(r'^-\s*`([^`]+)`', _cs)
                                    if _cm:
                                        _cv = _cm.group(1).strip().replace("\\", "/")
                                        _is_fp = ("/" in _cv or _cv.endswith(".py"))
                                        if _is_fp and _cv != _file_norm:
                                            _in_file = False
                                            _in_exports = False
                                            continue
                                # Collect bare export names (high indent, no def prefix)
                                if _in_exports and _indent >= 4 and _cs.startswith("- `"):
                                    _cm = _re.match(r'^-\s*`([^`]+)`', _cs)
                                    if _cm:
                                        _cv = _cm.group(1).strip()
                                        if not (_cv.startswith("def ") or _cv.startswith("async def ")):
                                            # Skip if it looks like a file path
                                            if "/" not in _cv and not _cv.endswith(".py"):
                                                _bare_export_names.append(_cv)
                    except Exception:
                        pass  # Bare name extraction is best-effort
                    if _file_contract_sigs or _bare_export_names:
                        _sig_lines = []
                        _sig_lines.append("## BINDING CONTRACT — Required Exports (NON-NEGOTIABLE)")
                        _sig_lines.append("")
                        _sig_lines.append(f"The following exports are REQUIRED for `{rel_path}`.")
                        _sig_lines.append("The downstream signature checker will reject any deviation.")
                        _sig_lines.append("")
                        if _file_contract_sigs:
                            _sig_lines.append("### Required Function Signatures (copy EXACTLY)")
                            _sig_lines.append("```python")
                            for _sig in _file_contract_sigs:
                                _sig_lines.append(f"{_sig}")
                                _sig_lines.append("    ...")
                                _sig_lines.append("")
                            _sig_lines.append("```")
                            _sig_lines.append("")
                        if _bare_export_names:
                            _sig_lines.append("### Required Export Names (MUST be defined/importable)")
                            _sig_lines.append("These symbols must be importable from this file:")
                            for _bn in _bare_export_names:
                                _sig_lines.append(f"  - `{_bn}` (define as a function, class, or top-level variable)")
                            _sig_lines.append("")
                        _sig_lines.append("**STRICT RULES — violations cause automatic rejection:**")
                        _sig_lines.append("1. DO NOT rename functions. `_find_latest_arch` must be `_find_latest_arch`, never `find_latest_architecture` or any other name.")
                        _sig_lines.append("2. DO NOT change parameter names. If the signature says `seg_dir: str`, never write `job_dir`, `segment_dir`, or `directory`.")
                        _sig_lines.append("3. DO NOT change parameter types. If the signature says `str`, never use `Path`. If it says `dict`, never use `Dict[str, Any]`.")
                        _sig_lines.append("4. DO NOT change return types. If the signature says `-> int`, never return `None`. If it says `-> Optional[str]`, never return `Optional[Path]`.")
                        _sig_lines.append("5. NEVER add `async` to a sync function or remove `async` from an async function.")
                        _sig_lines.append("6. DO NOT import these functions from sibling modules and pass them off as your own — define them directly in this file, UNLESS the architecture specification explicitly instructs you to re-export them from another module in this package.")
                        _sig_lines.append("7. NEVER omit required constants or variables listed in the contract. If `SEGMENT_LOOP_BUILD_ID` is required, it must be defined.")
                        _sig_lines.append("8. These names come from an existing codebase being refactored. They are not suggestions — they are the actual names used by callers. Renaming them will break all call sites.")
                        _sig_lines.append("")
                        _per_file_contract_block = "\n".join(_sig_lines)
                        logger.info(
                            "[arch_exec] v5.23 CONTRACT_INJECT for %s: %d signature(s) + %d bare name(s)",
                            rel_path, len(_file_contract_sigs), len(_bare_export_names),
                        )
                        print(
                            f"[ARCH_EXEC] v5.23 CONTRACT_INJECT: {rel_path} — "
                            f"{len(_file_contract_sigs)} sig(s), {len(_bare_export_names)} bare name(s)"
                        )
                        add_trace("CONTRACT_INJECT", "injected", {
                            "path": rel_path,
                            "signatures": _file_contract_sigs,
                            "bare_names": _bare_export_names,
                            "count": len(_file_contract_sigs) + len(_bare_export_names),
                        })
                    else:
                        logger.debug(
                            "[arch_exec] v5.23 No contract exports for %s",
                            rel_path,
                        )
                except ImportError:
                    logger.debug("[arch_exec] v5.23 signature_checker not available for contract injection")
                except Exception as _ci_err:
                    logger.warning("[arch_exec] v5.23 Contract injection failed (non-fatal): %s", _ci_err)

                if action == "create":
                    # v1.13: Try verbatim extraction before LLM call
                    verbatim_content = _extract_verbatim_code_from_architecture(
                        file_context, rel_path,
                    )
                    if verbatim_content:
                        print(
                            f"[ARCH_EXEC] v1.13 VERBATIM extraction: {rel_path} "
                            f"({len(verbatim_content)} chars) â€” skipping LLM"
                        )
                        logger.info(
                            "[arch_exec] v1.13 Verbatim extraction for %s: %d chars",
                            rel_path, len(verbatim_content),
                        )
                        add_trace("VERBATIM_EXTRACTION", "success", {
                            "path": rel_path, "chars": len(verbatim_content),
                        })
                    
                    # v5.23: Build prompt with contract signatures FIRST (highest priority)
                    user_prompt = f"Generate the complete content for a new file: `{rel_path}`\n\n"
                    if _per_file_contract_block:
                        user_prompt += f"{_per_file_contract_block}\n\n"
                    user_prompt += f"## Architecture Specification\n\n{file_context}\n\n"
                    if job_context_section:
                        user_prompt += f"{job_context_section}\n\n"
                    
                    # v3.0: Detect and inject source file context for extraction jobs
                    try:
                        source_files = _detect_source_files_from_architecture(
                            file_section=file_context,
                            architecture_content=architecture_content,
                            rel_path=rel_path,
                        )
                        if source_files:
                            print(f"[ARCH_EXEC] v3.0 Detected source files for {rel_path}: {source_files}")
                            logger.info("[arch_exec] v3.0 Source files for %s: %s", rel_path, source_files)
                            source_context = await _read_source_context(client, source_files, sandbox_base)
                            if source_context:
                                user_prompt += f"{source_context}\n\n"
                                print(f"[ARCH_EXEC] v3.0 Injected {len(source_context)} chars of source context")
                    except Exception as e:
                        # Non-fatal â€” proceed without source context if detection/read fails
                        logger.warning("[arch_exec] v3.0 Source context failed for %s: %s", rel_path, e)
                    
                    if _available_modules_evidence:
                        user_prompt += _available_modules_evidence
                    user_prompt += "Output ONLY the file content. No markdown fences, no explanations."
                    system_prompt = IMPLEMENTER_NEW_FILE_SYSTEM
                else:
                    # Modify: read existing file first (Overwatcher is allowed to read)
                    existing_content = await _read_existing_file(client, abs_path)
                    if existing_content is None:
                        last_error = f"Cannot read existing file for modification: {abs_path}"
                        logger.error("[arch_exec] %s", last_error)
                        break  # File doesn't exist â€” can't modify
                    
                    # v1.13: File size guardrail + edit mode decision
                    file_char_count = len(existing_content)
                    use_edit_mode = file_char_count >= MODIFY_EDIT_MODE_THRESHOLD
                    
                    if file_char_count > 150_000:
                        logger.warning("[arch_exec] v3.0 Very large MODIFY target: %s (%d chars)", rel_path, file_char_count)
                        print(f"[ARCH_EXEC] âš ï¸ Very large MODIFY: {rel_path} ({file_char_count:,} chars) â€” using edit mode")
                    elif use_edit_mode:
                        print(f"[ARCH_EXEC] v1.13 Large MODIFY: {rel_path} ({file_char_count:,} chars) â€” using edit mode")
                    
                    if use_edit_mode:
                        # v1.13: EDIT MODE â€” ask LLM for JSON edit pairs, not full file
                        logger.info("[arch_exec] v1.13 Edit mode for %s (%d chars)", rel_path, file_char_count)
                        add_trace("MODIFY_EDIT_MODE", "enabled", {
                            "path": rel_path, "chars": file_char_count,
                        })
                        
                        user_prompt = (
                            f"Apply the following modifications to `{rel_path}` ({file_char_count:,} chars).\n\n"
                            f"## Current File Content\n```\n{existing_content}\n```\n\n"
                        )
                        
                        existing_imports = _extract_existing_imports(existing_content, rel_path)
                        if existing_imports:
                            user_prompt += (
                                f"## Existing Imports\n"
                                f"Follow the same import patterns for any new imports.\n"
                                f"```\n{existing_imports}\n```\n\n"
                            )
                        
                        # v5.23: Inject contract signatures for MODIFY edit-mode path
                        if _per_file_contract_block:
                            user_prompt += f"{_per_file_contract_block}\n\n"
                        user_prompt += f"## Modification Instructions\n\n{file_context}\n\n"
                        if job_context_section:
                            user_prompt += f"{job_context_section}\n\n"
                        user_prompt += (
                            "Output ONLY a JSON array of edit objects. "
                            "Each object has \"old_text\" (exact unique snippet from the file) "
                            "and \"new_text\" (replacement). No markdown fences."
                        )
                        system_prompt = IMPLEMENTER_MODIFY_EDIT_SYSTEM
                    else:
                        # Standard full-file rewrite (small files)
                        user_prompt = (
                            f"Apply the following modifications to `{rel_path}`.\n\n"
                            f"## Current File Content\n```\n{existing_content}\n```\n\n"
                        )
                        
                        existing_imports = _extract_existing_imports(existing_content, rel_path)
                        if existing_imports:
                            user_prompt += (
                                f"## Existing Imports\n"
                                f"The file currently uses these imports. Follow the same "
                                f"import patterns and module paths for any new imports you add.\n"
                                f"```\n{existing_imports}\n```\n\n"
                            )
                        
                        # v5.23: Inject contract signatures for MODIFY path
                        if _per_file_contract_block:
                            user_prompt += f"{_per_file_contract_block}\n\n"
                        user_prompt += f"## Modification Instructions\n\n{file_context}\n\n"
                        if job_context_section:
                            user_prompt += f"{job_context_section}\n\n"
                        if _available_modules_evidence:
                            user_prompt += _available_modules_evidence
                        user_prompt += "Output the COMPLETE modified file. No markdown fences."
                        system_prompt = IMPLEMENTER_MODIFY_FILE_SYSTEM
                
                # v1.13: Skip LLM call if verbatim extraction succeeded
                if verbatim_content and strike == 1:
                    file_content = verbatim_content
                    logger.info(
                        "[arch_exec] v1.13 Using verbatim content: %d chars for %s",
                        len(file_content), rel_path,
                    )
                else:
                    # Verbatim not available or retry â€” use LLM
                    if verbatim_content and strike > 1:
                        logger.info(
                            "[arch_exec] v1.13 Verbatim failed verification, falling back to LLM for %s",
                            rel_path,
                        )
                        verbatim_content = None  # Don't retry verbatim
                    # v5.23: Add error context for retry strikes — PROMINENTLY at the top
                    # Previous version appended errors at the END of the prompt where
                    # they were buried under architecture context. Now we prepend them.
                    if strike > 1 and last_error:
                        _strike_error_block = (
                            f"## STRIKE {strike}/{MAX_STRIKES_PER_TASK} — LAST CHANCE, FIX NOW\n\n"
                            f"Your previous attempt was REJECTED. The exact error was:\n\n"
                            f"```\n{last_error}\n```\n\n"
                            f"DO NOT repeat this mistake. The checker runs automatically and will reject you again for the same issue.\n\n"
                            f"If the error says a function name is wrong: DO NOT rename functions. Use the name from the BINDING CONTRACT section above, character for character.\n"
                            f"If the error says a parameter type is wrong: DO NOT change types. If the contract says `str`, never use `Path`.\n"
                            f"If the error says a constant is missing: you MUST define it as a top-level variable in this file.\n"
                            f"If the error says a signature does not match: copy the ENTIRE `def` line from the BINDING CONTRACT section and do not alter a single character.\n\n"
                        )
                        # Prepend error to the beginning of user_prompt (after the file name line)
                        _first_newline = user_prompt.find("\n\n")
                        if _first_newline > 0:
                            user_prompt = user_prompt[:_first_newline + 2] + _strike_error_block + user_prompt[_first_newline + 2:]
                        else:
                            user_prompt = _strike_error_block + user_prompt
                        logger.info(
                            "[arch_exec] v5.23 Strike %d error prepended to prompt for %s (%d chars)",
                            strike, rel_path, len(_strike_error_block),
                        )
                    
                    # v3.0: Inject experience memory into implementer prompt
                    try:
                        from app.experience.retrieval import retrieve_for_stage, format_injection
                        from app.db import get_db_session as _get_mem_db
                        _mem_db = _get_mem_db()
                        _impl_patterns = retrieve_for_stage(
                            _mem_db, stage="implementer",
                            context=f"Implementing {rel_path} ({action}): {file_context[:150]}",
                            language=_infer_lang_from_path(rel_path),
                            error_signature=None,  # TODO: wire error_sig from strike tracker
                            max_results=5,
                        )
                        if _impl_patterns:
                            _impl_memory = format_injection(_impl_patterns, stage="implementer")
                            if _impl_memory:
                                system_prompt += f"\n\n{_impl_memory}"
                        _mem_db.close()
                    except Exception:
                        pass

                    # v3.0: Inject codebase RAG context
                    try:
                        from app.rag.vector_store import retrieve_code_context
                        from app.db import get_db_session as _get_rag_db
                        _rag_db = _get_rag_db()
                        _rag_context = retrieve_code_context(
                            _rag_db,
                            stage="implementer",
                            context=f"{rel_path}: {file_context[:200]}",
                            file_scope=[rel_path] if action == "modify" else None,
                            max_results=3,
                        )
                        if _rag_context:
                            system_prompt += f"\n\n{_rag_context}"
                        _rag_db.close()
                    except Exception:
                        pass

                    llm_result = await llm_call_fn(
                        provider_id=impl_provider,
                        model_id=impl_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=impl_max_tokens,
                        timeout_seconds=600,  # v3.2: Large file gen (up to 60k tokens) needs room
                    )
                    
                    file_content = _extract_llm_content(llm_result)
                    file_content = _strip_markdown_fences(file_content)
                    
                    if not file_content or len(file_content.strip()) < 10:
                        last_error = "LLM returned empty/minimal content"
                        logger.warning("[arch_exec] Strike %d: %s for %s", strike, last_error, rel_path)
                        add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                            "path": rel_path, "error": last_error,
                        })
                        continue
                    
                    logger.info(
                        "[arch_exec] LLM generated %d chars for %s (strike %d)",
                        len(file_content), rel_path, strike,
                    )
                
            except Exception as e:
                last_error = f"LLM call failed: {e}"
                logger.exception("[arch_exec] Strike %d: %s", strike, last_error)
                add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                    "path": rel_path, "error": last_error,
                })
                continue
            
            # --- Delegate write to Implementer ---
            try:
                if use_edit_mode:
                    # v1.13: Parse edit pairs and apply targeted edits
                    edit_pairs = _parse_edit_pairs(file_content)
                    
                    if edit_pairs is None:
                        # Parsing failed â€” fall back to full-file write
                        logger.warning(
                            "[arch_exec] v1.13 Edit pair parsing failed for %s, "
                            "falling back to full-file write",
                            rel_path,
                        )
                        print(f"[ARCH_EXEC] v1.13 Edit parse failed for {rel_path} â€” falling back to full write")
                        add_trace("EDIT_PARSE_FALLBACK", "parse_failed", {
                            "path": rel_path,
                        })
                        # Try writing as full file (may truncate, but better than nothing)
                        impl_result = await run_implementer_task(
                            path=abs_path,
                            content=file_content,
                            action=action,
                            ensure_parents=True,
                            client=client,
                        )
                    else:
                        logger.info(
                            "[arch_exec] v1.13 Applying %d targeted edits to %s",
                            len(edit_pairs), rel_path,
                        )
                        print(f"[ARCH_EXEC] v1.13 Applying {len(edit_pairs)} targeted edits to {rel_path}")
                        
                        edit_result = await run_implementer_edit_task(
                            path=abs_path,
                            edits=edit_pairs,
                            client=client,
                        )
                        
                        # Convert EditTaskResult to match expected interface
                        class _EditResultAdapter:
                            def __init__(self, er):
                                self.success = er.success
                                self.chars_written = er.chars_after
                                self.verified = er.verified
                                self.error = er.error
                        
                        impl_result = _EditResultAdapter(edit_result)
                        
                        if edit_result.edits_failed > 0:
                            logger.warning(
                                "[arch_exec] v1.13 %d/%d edits failed for %s: %s",
                                edit_result.edits_failed,
                                edit_result.edits_applied + edit_result.edits_failed,
                                rel_path,
                                edit_result.failed_edits[:3],
                            )
                            add_trace("EDIT_PARTIAL", "some_failed", {
                                "path": rel_path,
                                "applied": edit_result.edits_applied,
                                "failed": edit_result.edits_failed,
                            })
                else:
                    # Standard full-file write
                    impl_result = await run_implementer_task(
                        path=abs_path,
                        content=file_content,
                        action=action,
                        ensure_parents=True,
                        client=client,
                    )
                
                if not impl_result.success:
                    last_error = f"Implementer write failed: {impl_result.error}"
                    logger.warning("[arch_exec] Strike %d: %s", strike, last_error)
                    add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                        "path": rel_path, "error": last_error,
                    })
                    continue
                
                logger.info(
                    "[arch_exec] Implementer wrote %s: %d chars, verified=%s",
                    rel_path, impl_result.chars_written, impl_result.verified,
                )
                
            except Exception as e:
                last_error = f"Implementer exception: {e}"
                logger.exception("[arch_exec] Strike %d: %s", strike, last_error)
                add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                    "path": rel_path, "error": last_error,
                })
                continue
            
            # --- Independent verification (Overwatcher reads to verify) ---
            verify = _verify_file_via_sandbox(client, abs_path, expected_min_chars=10)
            
            if not verify["valid"]:
                last_error = f"Overwatcher verification failed: {verify['error'] or 'file too short/missing'}"
                logger.warning("[arch_exec] Strike %d: %s", strike, last_error)
                add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                    "path": rel_path, "error": last_error,
                })
                continue
            
            # --- v5.5 PHASE 4A: Job Checker â€” verify against arch spec + contract ---
            try:
                from ..job_checker import check_written_file
                _check_arch = extract_section_for_file(architecture_content, rel_path) or ""
                _check_result = await check_written_file(
                    file_path=rel_path,
                    file_content=file_content,
                    arch_section=_check_arch,
                    interface_contract=interface_contract,
                    sandbox_base=sandbox_base,
                    existing_sandbox_files=_existing_sandbox_files,
                    previous_strike_errors=_job_checker_strike_errors if _job_checker_strike_errors else None,
                )
                if _check_result.skipped:
                    logger.debug("[arch_exec] v5.5 Job check skipped for %s: %s",
                                 rel_path, _check_result.skip_reason)
                elif not _check_result.passed:
                    _blocking = _check_result.blocking_issues
                    _block_desc = "; ".join(i.description for i in _blocking[:3])
                    last_error = f"Job Checker FAILED: {len(_blocking)} blocking issue(s): {_block_desc}"
                    _job_checker_strike_errors.append(_block_desc)  # v2.2: accumulate for next strike
                    logger.warning("[arch_exec] v5.5 Strike %d: %s", strike, last_error)
                    print(f"[ARCH_EXEC] v5.5 JOB_CHECK FAIL: {rel_path} â€” {_block_desc[:120]}")
                    add_trace("JOB_CHECK_FAIL", f"strike_{strike}", {
                        "path": rel_path,
                        "blocking": len(_blocking),
                        "warnings": len(_check_result.warning_issues),
                        "issues": [i.to_dict() for i in _check_result.issues[:5]],
                    })
                    continue  # Use existing three-strike retry
                else:
                    _warns = len(_check_result.warning_issues)
                    if _warns:
                        logger.info("[arch_exec] v5.5 Job check PASSED with %d warning(s): %s",
                                    _warns, rel_path)
                    add_trace("JOB_CHECK_PASS", "verified", {
                        "path": rel_path,
                        "warnings": _warns,
                    })
            except ImportError:
                logger.debug("[arch_exec] v5.5 job_checker not available â€” skipping")
            except Exception as _jc_err:
                logger.warning("[arch_exec] v5.5 Job checker error (non-fatal): %s", _jc_err)
            
            # --- v5.22 PHASE 4B: Deterministic Signature Verification ---
            # Layer 3 safety net: after the LLM-based job checker, run a
            # deterministic AST comparison of function signatures against
            # the skeleton contract.  Zero LLM calls.  Catches the exact
            # class of bug that broke the architecture executor refactor.
            try:
                from ..signature_checker import (
                    check_file_signatures,
                    extract_contract_signatures_for_file,
                )
                _contract_sigs = extract_contract_signatures_for_file(
                    interface_contract, rel_path,
                )
                if _contract_sigs:
                    _sig_result = check_file_signatures(
                        file_content=file_content,
                        file_path=rel_path,
                        contract_signatures=_contract_sigs,
                    )
                    if not _sig_result.passed:
                        _mismatch_details = []
                        for _mm in _sig_result.mismatches:
                            _mismatch_details.append(
                                f"SIGNATURE MISMATCH: {_mm.function_name}\n"
                                f"  Contract requires: {_mm.expected_signature}\n"
                                f"  Implementation has: {_mm.actual_signature}\n"
                                f"  Differences: {'; '.join(_mm.differences)}"
                            )
                        for _mf in _sig_result.missing_functions:
                            _mismatch_details.append(
                                f"MISSING FUNCTION: {_mf} — required by contract but not found"
                            )
                        _sig_error = (
                            f"Signature checker FAILED: "
                            f"{len(_sig_result.mismatches)} mismatch(es), "
                            f"{len(_sig_result.missing_functions)} missing.\n"
                            + "\n".join(_mismatch_details)
                        )
                        last_error = _sig_error
                        _job_checker_strike_errors.append(_sig_error)
                        logger.warning(
                            "[arch_exec] v5.22 Sig check strike %d: %s",
                            strike, _sig_error[:200],
                        )
                        print(
                            f"[ARCH_EXEC] v5.22 SIG_CHECK FAIL: {rel_path} — "
                            f"{len(_sig_result.mismatches)} mismatch(es), "
                            f"{len(_sig_result.missing_functions)} missing"
                        )
                        add_trace("SIGNATURE_CHECK_FAIL", f"strike_{strike}", {
                            "path": rel_path,
                            "mismatches": len(_sig_result.mismatches),
                            "missing": len(_sig_result.missing_functions),
                            "details": [m.to_dict() for m in _sig_result.mismatches[:5]],
                        })
                        continue  # Triggers next strike with exact signature in error
                    else:
                        logger.debug(
                            "[arch_exec] v5.22 Sig check PASSED for %s (%d sigs verified)",
                            rel_path, len(_contract_sigs),
                        )
                        add_trace("SIGNATURE_CHECK_PASS", "verified", {
                            "path": rel_path,
                            "signatures_checked": len(_contract_sigs),
                        })
            except ImportError:
                logger.debug("[arch_exec] v5.22 signature_checker not available — skipping")
            except Exception as _sc_err:
                logger.warning("[arch_exec] v5.22 Signature check error (non-fatal): %s", _sc_err)
            # SUCCESS â€” all checks passed
            task_success = True
            break
        
        # --- Record task result ---
        if task_success:
            if action == "create":
                files_created += 1
                # v2.5: Store content for two-pass re-extraction
                created_file_contents[rel_path] = file_content
            else:
                files_modified_count += 1
                # v1.13: For edit mode, file_content is JSON edits, not actual content.
                # Read the actual file for cross-file context extraction.
                if use_edit_mode:
                    try:
                        _actual = await _read_existing_file(client, abs_path)
                        if _actual:
                            file_content = _actual
                    except Exception:
                        pass  # Non-fatal â€” proceed with what we have
                # v2.5: Capture router registrations from modified files (e.g. main.py)
                if rel_path.endswith('.py'):
                    try:
                        regs = _extract_router_registrations(file_content)
                        if regs:
                            router_registrations.update(regs)
                            logger.info(
                                "[arch_exec] v2.5 Captured router registrations from %s: %s",
                                rel_path, regs,
                            )
                    except Exception as e:
                        logger.warning("[arch_exec] v2.5 Router registration extraction failed: %s", e)
            artifacts_written.append(abs_path)
            _existing_sandbox_files.add(rel_path.replace("\\", "/"))  # v5.11: track for import validation
            # v2.3: Capture interfaces for cross-file context
            try:
                interface_summary = _extract_file_interfaces(rel_path, file_content)
                job_context[rel_path] = interface_summary
                logger.info(
                    "[arch_exec] v2.3 Captured interfaces for %s (%d chars)",
                    rel_path, len(interface_summary),
                )
            except Exception as e:
                # Non-fatal â€” we still succeeded, just couldn't extract interfaces
                logger.warning(
                    "[arch_exec] v2.3 Interface extraction failed for %s: %s",
                    rel_path, e,
                )
            
            logger.info("[arch_exec] âœ“ %s %s", action.upper(), rel_path)
            print(f"[ARCH_EXEC] âœ“ {action.upper()} {rel_path}")
            
            add_trace("FILE_TASK_SUCCESS", action, {
                "path": rel_path,
                "absolute_path": abs_path,
                "job_context_files": list(job_context.keys()),  # v2.3
            })
        
        else:
            # Task FAILED after exhausting all strikes
            files_failed += 1
            logger.error(
                "[arch_exec] \u2717 %s %s FAILED after %d strikes: %s",
                action.upper(), rel_path, MAX_STRIKES_PER_TASK, last_error,
            )
            print(f"[ARCH_EXEC] \u2717 {action.upper()} {rel_path} FAILED: {last_error}")

            add_trace("FILE_TASK_FAILED", "exhausted_strikes", {
                "path": rel_path,
                "strikes": MAX_STRIKES_PER_TASK,
                "last_error": last_error,
            })

        # =====================================================================
        # v2.5: Two-pass boundary - after all CREATEs, re-extract interfaces
        # This ensures MODIFY tasks get the FULL cross-file context from all
        # created files, not just the ones that happened to be created earlier.
        # =====================================================================
        if i == create_count and created_file_contents:
            logger.info(
                "[arch_exec] v2.5 Two-pass: re-extracting interfaces from %d created files",
                len(created_file_contents),
            )
            print(f"[ARCH_EXEC] v2.5 Two-pass: refreshing context from {len(created_file_contents)} created files")
            for created_path, created_content in created_file_contents.items():
                try:
                    refreshed = _extract_file_interfaces(created_path, created_content)
                    job_context[created_path] = refreshed
                except Exception as e:
                    logger.warning("[arch_exec] v2.5 Two-pass extraction failed for %s: %s", created_path, e)
            add_trace("TWO_PASS_CONTEXT_REFRESH", "success", {
                "files_refreshed": list(created_file_contents.keys()),
            })
    
    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    total_succeeded = files_created + files_modified_count
    success = total_succeeded > 0 and files_failed == 0
    
    summary = {
        "total_operations": total_operations,
        "files_created": files_created,
        "files_modified": files_modified_count,
        "files_failed": files_failed,
        "total_succeeded": total_succeeded,
        "elapsed_ms": elapsed_ms(),
    }
    
    if success:
        status_label = "âœ“ SUCCESS"
    elif total_succeeded > 0:
        status_label = f"âš  PARTIAL ({total_succeeded}/{total_operations})"
    else:
        status_label = "âœ— FAILED"
    
    logger.info(
        "[arch_exec] %s: %d created, %d modified, %d failed (%dms)",
        status_label, files_created, files_modified_count, files_failed, elapsed_ms(),
    )
    print(
        f"[ARCH_EXEC] {status_label}: "
        f"{files_created} created, {files_modified_count} modified, "
        f"{files_failed} failed ({elapsed_ms()}ms)"
    )
    
    add_trace(
        "ARCHITECTURE_EXECUTION_COMPLETE",
        "success" if success else "partial" if total_succeeded > 0 else "failed",
        summary,
    )

    # =========================================================================
    # Step 6: Backend boot check with retry loop (v2.9)
    # After all file operations, verify the backend can still start.
    # If boot fails, identify the broken file from the traceback, feed the
    # error back to the Implementer for a targeted fix, and retry.
    # Three-strike limit per unique error. New errors reset the counter.
    # Critical contract: fixes must not destroy working functionality.
    # =========================================================================
    BOOT_MAX_STRIKES = 3

    def _run_boot_check(cl: SandboxClient, sb: str) -> tuple:
        """Run boot check, return (passed: bool, error: str, full_output: str).
        
        v3.1: Fixed error reporting - when boot fails, report the actual
        failure from stdout (import errors, syntax errors) not just stderr
        warnings. stderr often contains non-fatal warnings like
        'MemoryService not available' that are red herrings.
        """
        venv_python = sb + "\\.venv\\Scripts\\python.exe"
        boot_cmd = (
            f'cd "{sb}" ; '
            f'& "{venv_python}" -c '
            f'"import sys; sys.path.insert(0, r\'{sb}\'); '
            f'from main import app; print(\'BOOT_CHECK_PASS\')"'
        )
        result = cl.shell_run(boot_cmd, timeout_seconds=30)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        passed = "BOOT_CHECK_PASS" in stdout
        
        if passed:
            return passed, "", stderr
        
        # v3.1: Build a useful error message - prefer traceback/import errors
        error_keywords = (
            'Error', 'Traceback', 'ImportError', 'ModuleNotFoundError',
            'SyntaxError', 'AttributeError', 'NameError', 'TypeError',
            'File "', 'cannot import', 'No module named',
        )
        error_parts = []
        for line in (stdout + "\n" + stderr).split('\n'):
            line_s = line.strip()
            if any(kw in line_s for kw in error_keywords):
                error_parts.append(line_s)
        
        if error_parts:
            error_msg = '\n'.join(error_parts[:10])
        else:
            error_msg = f"stdout(tail): {stdout[-500:]}\nstderr(tail): {stderr[-500:]}"
        
        full_output = stdout + "\n---STDERR---\n" + stderr
        return passed, error_msg[:1000], full_output

    def _parse_broken_file_from_traceback(tb: str, written: list) -> Optional[str]:
        """Extract the broken file path from a Python traceback.
        Only returns paths that were written by this job (artifacts_written)."""
        import re
        # Match 'File "<path>"' lines in traceback
        file_matches = re.findall(r'File "([^"]+)"', tb)
        # Walk backwards â€” the deepest frame is most likely the broken file
        written_set = {p.replace("/", "\\") for p in written}
        for fpath in reversed(file_matches):
            normalised = fpath.replace("/", "\\")
            if normalised in written_set:
                return normalised
        return None

    if skip_boot_check:
        logger.info("[arch_exec] v3.2 Boot check SKIPPED (skip_boot_check=True, intermediate segment)")
        print("[ARCH_EXEC] â­ï¸ Boot check skipped (intermediate segment)")
        add_trace("BOOT_CHECK_COMPLETE", "skipped_intermediate")
    elif success or total_succeeded > 0:
        logger.info("[arch_exec] v2.9 Running backend boot check...")
        print("[ARCH_EXEC] ðŸ” Running backend boot check...")
        add_trace("BOOT_CHECK_START", "running")

        boot_passed = False
        last_boot_error = None
        same_error_count = 0

        try:
            for boot_strike in range(1, BOOT_MAX_STRIKES + 1):
                passed, boot_error, full_stderr = _run_boot_check(client, sandbox_base)

                if passed:
                    logger.info("[arch_exec] v2.9 âœ“ Backend boot check PASSED (strike %d)", boot_strike)
                    print(f"[ARCH_EXEC] âœ… Backend boot check PASSED (attempt {boot_strike})")
                    add_trace("BOOT_CHECK_COMPLETE", "pass", {"attempt": boot_strike})
                    boot_passed = True
                    break

                # Boot failed â€” check if same error or new error
                logger.error("[arch_exec] v2.9 âœ— Boot check FAILED (strike %d): %s", boot_strike, boot_error[:200])
                print(f"[ARCH_EXEC] âŒ Boot check FAILED (attempt {boot_strike}/{BOOT_MAX_STRIKES}): {boot_error[:200]}")
                add_trace("BOOT_CHECK_FAIL", f"strike_{boot_strike}", {
                    "error": boot_error[:500],
                })

                # Track same-error vs new-error
                if boot_error == last_boot_error:
                    same_error_count += 1
                else:
                    same_error_count = 1
                    last_boot_error = boot_error

                if same_error_count >= BOOT_MAX_STRIKES:
                    logger.error("[arch_exec] v2.9 Same boot error %d times â€” giving up", same_error_count)
                    print(f"[ARCH_EXEC] âŒ Same boot error {same_error_count} times â€” giving up")
                    break

                # Last strike â€” don't retry, just fail
                if boot_strike >= BOOT_MAX_STRIKES:
                    break

                # --- Attempt to fix the broken file ---
                broken_file = _parse_broken_file_from_traceback(full_stderr, artifacts_written)
                if not broken_file:
                    logger.warning("[arch_exec] v2.9 Cannot identify broken file from traceback â€” cannot auto-fix")
                    print("[ARCH_EXEC] âš ï¸ Cannot identify broken file from traceback")
                    break

                logger.info("[arch_exec] v2.9 Broken file identified: %s â€” attempting fix", broken_file)
                print(f"[ARCH_EXEC] ðŸ”§ Attempting fix on: {broken_file}")
                add_trace("BOOT_FIX_ATTEMPT", f"strike_{boot_strike}", {
                    "broken_file": broken_file,
                    "error": boot_error[:300],
                })

                # Read the current (broken) content from sandbox
                broken_content = await _read_existing_file(client, broken_file)
                if not broken_content:
                    logger.warning("[arch_exec] v2.9 Cannot read broken file: %s", broken_file)
                    break

                # Get the architecture section for this file
                broken_rel = broken_file
                for prefix in [sandbox_base + "\\", "D:\\orb-desktop\\"]:
                    if broken_file.startswith(prefix):
                        broken_rel = broken_file[len(prefix):]
                        break
                arch_section = extract_section_for_file(architecture_content, broken_rel)

                # Build a targeted fix prompt
                fix_prompt = (
                    f"## BOOT CHECK FIX â€” Strike {boot_strike}\n\n"
                    f"The backend failed to start after your changes. "
                    f"You MUST fix this file while preserving ALL existing functionality.\n\n"
                    f"### Boot Error\n```\n{boot_error}\n```\n\n"
                    f"### Full Traceback\n```\n{full_stderr[:2000]}\n```\n\n"
                    f"### Current File Content (broken)\n```\n{broken_content}\n```\n\n"
                    f"### Architecture Specification For This File\n{arch_section}\n\n"
                    f"### CRITICAL RULES\n"
                    f"1. Output ONLY the complete fixed file â€” no markdown fences, no explanations.\n"
                    f"2. Fix the boot error shown above.\n"
                    f"3. DO NOT remove or break any existing imports, functions, or functionality.\n"
                    f"4. The fix must integrate the new feature while keeping everything that already worked.\n"
                    f"5. If an import path doesn't exist, remove it or fix it â€” don't guess.\n"
                    f"6. Preserve the file's existing code style and patterns.\n"
                )

                # Call the Implementer to fix
                from ..implementer import run_implementer_task, run_implementer_edit_task
                try:
                    fix_result = await llm_call_fn(
                        provider_id=impl_provider,
                        model_id=impl_model,
                        messages=[
                            {"role": "system", "content": IMPLEMENTER_MODIFY_FILE_SYSTEM},
                            {"role": "user", "content": fix_prompt},
                        ],
                        max_tokens=IMPLEMENTER_MAX_TOKENS,
                        timeout_seconds=600,  # v3.2: Fix attempts can be large
                    )

                    fix_content = _extract_llm_content(fix_result)
                    fix_content = _strip_markdown_fences(fix_content)

                    if not fix_content or len(fix_content) < 50:
                        logger.warning("[arch_exec] v2.9 Fix produced empty/minimal content")
                        continue

                    # Write the fix to sandbox
                    write_result = await run_implementer_task(
                        path=broken_file,
                        content=fix_content,
                        action="modify",
                        client=client,
                    )

                    if write_result.success:
                        logger.info("[arch_exec] v2.9 Fix written: %s (%d chars)", broken_file, len(fix_content))
                        print(f"[ARCH_EXEC] âœ“ Fix applied to {broken_file} ({len(fix_content)} chars)")
                    else:
                        logger.error("[arch_exec] v2.9 Fix write failed: %s", write_result.error)
                        break

                except Exception as e:
                    logger.error("[arch_exec] v2.9 Fix LLM call failed: %s", e)
                    break

            # Final status
            if not boot_passed:
                boot_error_final = last_boot_error or "Boot check failed"
                add_trace("BOOT_CHECK_COMPLETE", "fail", {
                    "error": boot_error_final[:500],
                    "strikes": boot_strike,
                })
                success = False
                files_failed += total_succeeded
                summary["boot_check"] = "FAILED"
                summary["boot_error"] = boot_error_final[:500]

        except Exception as e:
            logger.warning("[arch_exec] v2.9 Boot check could not run: %s", e)
            print(f"[ARCH_EXEC] âš ï¸ Boot check skipped: {e}")
            add_trace("BOOT_CHECK_COMPLETE", "skipped", {"error": str(e)})

    # =========================================================================
    # Step 7: Final result
    # =========================================================================
    error_msg = None
    if not success:
        if total_succeeded == 0:
            error_msg = f"Architecture execution failed: 0/{total_operations} file operations succeeded"
        else:
            error_msg = (
                f"Architecture execution partial: {total_succeeded}/{total_operations} "
                f"succeeded, {files_failed} failed"
            )
    
    return {
        "success": success,
        "decision": "PASS" if success else "FAIL",
        "error": error_msg,
        "trace": trace,
        "artifacts_written": artifacts_written,
        "summary": summary,
    }


__all__ = [
    "run_architecture_execution",
    "parse_file_inventory",
    "extract_section_for_file",
    "_extract_file_interfaces",
    "_extract_existing_imports",
    "_extract_router_registrations",
    "_build_resolved_endpoints",
    "_format_job_context",
    "_ensure_python_init_files",
    "ARCHITECTURE_EXECUTOR_BUILD_ID",
]

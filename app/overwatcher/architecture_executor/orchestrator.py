"""
Main orchestrator for architecture execution.

Thin coordinator that delegates to step modules:
1. Parse & validate  (inline — small)
2. Shadow preflight  (step_shadow_check)
3. Sandbox scanning  (step_sandbox_scan)
4. Process tasks     (step_process_task)
5. Boot check        (step_boot_check)

All utility functions are imported from sibling modules.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

from ..spec_resolution import ResolvedSpec
from ..sandbox_client import get_sandbox_client

from .constants import ARCHITECTURE_EXECUTOR_BUILD_ID, IMPLEMENTER_MAX_TOKENS
from .execution_state import ExecutionContext
from .parsing import parse_file_inventory, extract_section_for_file
from .sandbox_ops import _resolve_sandbox_base
from .step_shadow_check import run_init_file_creation, run_shadow_preflight
from .step_sandbox_scan import scan_sandbox_files
from .step_process_task import process_all_tasks
from .step_boot_check import run_boot_check_loop

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
    manifest_all_files: Optional[Set[str]] = None,
    scaffold_result: Any = None,
) -> Dict[str, Any]:
    """Supervise architecture-level spec execution.

    The Overwatcher (this function) is the supervisor.  It:
    1. Parses the architecture document to find file operations
    2. For each file, calls the Implementer LLM to generate content
    3. Delegates each write to the Implementer via run_implementer_task()
    4. Reads back from sandbox to independently verify
    5. Implements three-strike error handling per task

    The Implementer LLM generates the code.
    The Implementer module (implementer.py) writes it to the sandbox.
    The Overwatcher only reads for verification.
    """
    # -----------------------------------------------------------------
    # Initialise execution context
    # -----------------------------------------------------------------
    ctx = ExecutionContext(
        spec_id=spec.spec_id,
        job_id=job_id,
        architecture_content=architecture_content,
        architecture_path=architecture_path,
        artifact_root=artifact_root,
        interface_contract=interface_contract,
        skip_boot_check=skip_boot_check,
        manifest_all_files=manifest_all_files,
        llm_call_fn=llm_call_fn,
        scaffold_result=scaffold_result,
    )

    ctx.add_trace("ARCHITECTURE_EXECUTION_START", "started", {
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

    # -----------------------------------------------------------------
    # Step 1: Parse file inventory
    # -----------------------------------------------------------------
    ctx.new_files, ctx.modified_files = parse_file_inventory(architecture_content)
    ctx.total_operations = len(ctx.new_files) + len(ctx.modified_files)

    logger.info("[arch_exec] Files: %d new, %d modified", len(ctx.new_files), len(ctx.modified_files))
    print(f"[ARCH_EXEC] Files: {len(ctx.new_files)} new, {len(ctx.modified_files)} modified")

    ctx.add_trace("ARCHITECTURE_PARSE", "success", {
        "new_files": [f["path"] for f in ctx.new_files],
        "modified_files": [f["path"] for f in ctx.modified_files],
        "total_operations": ctx.total_operations,
    })

    if ctx.total_operations == 0:
        error_msg = "No file operations found in architecture document."
        logger.error("[arch_exec] v3.1 HARD FAIL: %s (arch_length=%d chars)", error_msg, len(architecture_content or ""))
        ctx.add_trace("ARCHITECTURE_PARSE", "failed", {"error": error_msg})
        return _fail(ctx, error_msg)

    # -----------------------------------------------------------------
    # Step 2: Validate prerequisites
    # -----------------------------------------------------------------
    if llm_call_fn is None:
        return _fail(ctx, "LLM function required for architecture execution")

    client = get_sandbox_client()
    if not client.is_connected():
        return _fail(ctx, "SAFETY: Sandbox not available")
    ctx.add_trace("SANDBOX_CONNECTED", "success")

    # Implementer LLM config
    try:
        from app.llm.stage_models import get_implementer_config
        cfg = get_implementer_config()
        ctx.impl_provider = cfg.provider
        ctx.impl_model = cfg.model
        ctx.impl_max_tokens = cfg.max_output_tokens or IMPLEMENTER_MAX_TOKENS
    except Exception as e:
        logger.warning("[arch_exec] Could not load implementer config: %s — using defaults", e)

    # -----------------------------------------------------------------
    # Step 3: Resolve sandbox base + init files + shadow check
    # -----------------------------------------------------------------
    ctx.sandbox_base = _resolve_sandbox_base(client)
    logger.info("[arch_exec] Sandbox base: %s", ctx.sandbox_base)
    ctx.add_trace("SANDBOX_BASE_RESOLVED", "success", {"base_path": ctx.sandbox_base})

    run_init_file_creation(ctx, client)
    run_shadow_preflight(ctx, client)

    # -----------------------------------------------------------------
    # Step 4: Scan sandbox + process all file tasks
    # -----------------------------------------------------------------
    all_tasks = (
        [{"info": f, "action": "create"} for f in ctx.new_files]
        + [{"info": f, "action": "modify"} for f in ctx.modified_files]
    )
    scan_sandbox_files(ctx, client, all_tasks)

    await process_all_tasks(ctx, client)

    # -----------------------------------------------------------------
    # Step 5: Summary
    # -----------------------------------------------------------------
    summary = {
        "total_operations": ctx.total_operations,
        "files_created": ctx.files_created,
        "files_modified": ctx.files_modified,
        "files_failed": ctx.files_failed,
        "total_succeeded": ctx.total_succeeded,
        "elapsed_ms": ctx.elapsed_ms(),
    }

    if ctx.success:
        status_label = "✓ SUCCESS"
    elif ctx.total_succeeded > 0:
        status_label = f"⚠ PARTIAL ({ctx.total_succeeded}/{ctx.total_operations})"
    else:
        status_label = "✗ FAILED"

    logger.info(
        "[arch_exec] %s: %d created, %d modified, %d failed (%dms)",
        status_label, ctx.files_created, ctx.files_modified, ctx.files_failed, ctx.elapsed_ms(),
    )
    print(
        f"[ARCH_EXEC] {status_label}: "
        f"{ctx.files_created} created, {ctx.files_modified} modified, "
        f"{ctx.files_failed} failed ({ctx.elapsed_ms()}ms)"
    )
    ctx.add_trace(
        "ARCHITECTURE_EXECUTION_COMPLETE",
        "success" if ctx.success else "partial" if ctx.total_succeeded > 0 else "failed",
        summary,
    )

    # -----------------------------------------------------------------
    # Step 6: Boot check
    # -----------------------------------------------------------------
    await run_boot_check_loop(ctx, client)

    # -----------------------------------------------------------------
    # Step 7: Final result
    # -----------------------------------------------------------------
    success = ctx.success
    error_msg = None
    if not success:
        if ctx.total_succeeded == 0:
            error_msg = f"Architecture execution failed: 0/{ctx.total_operations} file operations succeeded"
        else:
            error_msg = (
                f"Architecture execution partial: {ctx.total_succeeded}/{ctx.total_operations} "
                f"succeeded, {ctx.files_failed} failed"
            )

    return {
        "success": success,
        "decision": "PASS" if success else "FAIL",
        "error": error_msg,
        "trace": ctx.trace,
        "artifacts_written": ctx.artifacts_written,
        "summary": summary,
        "impl_provider": ctx.impl_provider,
        "impl_model": ctx.impl_model,
    }


def _fail(ctx: ExecutionContext, error_msg: str) -> Dict[str, Any]:
    """Return a standard failure result."""
    ctx.add_trace("ARCHITECTURE_EXECUTION", "failed", {"error": error_msg})
    return {
        "success": False,
        "decision": "FAIL",
        "error": error_msg,
        "trace": ctx.trace,
        "artifacts_written": [],
    }


__all__ = [
    "run_architecture_execution",
    "parse_file_inventory",
    "extract_section_for_file",
    "ARCHITECTURE_EXECUTOR_BUILD_ID",
]

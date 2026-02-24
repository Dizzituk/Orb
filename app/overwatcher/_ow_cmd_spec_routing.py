# FILE: app/overwatcher/_ow_cmd_spec_routing.py
"""
Overwatcher command — spec routing helpers.

Extracted from overwatcher_command.py to reduce file size.
Handles POT spec execution + build validation, and architecture spec routing.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


async def execute_pot_spec(
    *,
    spec: Any,
    job_id: str,
    llm_call_fn: Optional[Callable],
    artifact_root: str,
    result: Any,
) -> bool:
    """Execute a POT spec and run build validation.

    Mutates *result* in place (sets success, decision, error, trace, artifacts).
    Returns True if POT execution was handled (caller should return result).
    """
    from .pot_spec_executor import run_pot_spec_execution

    pot_tasks = spec.pot_tasks
    logger.info("[ow_cmd] POT SPEC DETECTED: %d tasks", len(pot_tasks.tasks))
    result.add_trace(
        "SPEC_RESOLVE", "success_pot",
        {
            "spec_id": spec.spec_id, "is_pot_spec": True,
            "task_count": len(pot_tasks.tasks),
            "search_term": pot_tasks.search_term,
            "replace_term": pot_tasks.replace_term,
        },
    )

    pot_result = await run_pot_spec_execution(
        spec=spec, pot_tasks=pot_tasks, job_id=job_id,
        llm_call_fn=llm_call_fn, artifact_root=artifact_root,
    )

    result.success = pot_result.get("success", False)
    result.overwatcher_decision = pot_result.get("decision", "PASS" if pot_result.get("success") else "FAIL")
    result.error = pot_result.get("error")
    result.stage_trace.extend(pot_result.get("trace", []))
    result.artifacts_written = pot_result.get("artifacts_written", [])

    logger.info(
        "[ow_cmd] POT execution complete: success=%s, completed=%d/%d, already_applied=%d",
        result.success, pot_result.get("tasks_completed", 0),
        pot_result.get("total_tasks", 0), pot_result.get("tasks_already_applied", 0),
    )

    if not pot_result.get("success", False):
        logger.warning("[ow_cmd] POT writes failed — skipping build validation")
        return True

    # Build validation
    modified_files = pot_result.get("affected_files", []) or pot_result.get("artifacts_written", [])
    if not modified_files:
        result.add_trace("BUILD_VALIDATION", "skipped", {"reason": "no_artifacts_written"})
        return True

    try:
        from .sandbox_build_validator import run_build_validation_loop, MAX_BUILD_FIX_ATTEMPTS
        from .sandbox_client import get_sandbox_client

        build_client = get_sandbox_client()
        if not build_client.is_connected():
            result.add_trace("BUILD_VALIDATION", "skipped_no_sandbox", {"reason": "sandbox_unavailable"})
            return True
    except Exception as e:
        result.add_trace("BUILD_VALIDATION", "skipped_error", {"reason": str(e)})
        return True

    build_passed, build_results, fix_history = await run_build_validation_loop(
        client=build_client, modified_files=modified_files,
        spec_content=spec.spec_content or "", pot_result=pot_result,
        llm_call_fn=llm_call_fn, add_trace=result.add_trace,
    )

    if build_passed:
        result.success = True
        result.overwatcher_decision = "PASS"
        result.add_trace("BUILD_VALIDATION_COMPLETE", "passed", {
            "build_results": [r.to_dict() for r in build_results] if build_results else [],
            "fix_attempts": len(fix_history),
        })
    else:
        result.success = False
        result.overwatcher_decision = "FAIL"
        failed_summaries = [
            f"{r.project_type}: {r.error_summary or 'unknown error'}"
            for r in (build_results or []) if not r.passed
        ]
        from .sandbox_build_validator import MAX_BUILD_FIX_ATTEMPTS
        result.error = (
            f"Build validation failed after {len(fix_history)} fix attempts "
            f"(max {MAX_BUILD_FIX_ATTEMPTS}). Errors: {'; '.join(failed_summaries)}"
        )
        result.add_trace("BUILD_VALIDATION_COMPLETE", "failed", {
            "error": result.error,
            "build_results": [r.to_dict() for r in build_results] if build_results else [],
            "fix_history": fix_history,
        })

    return True


async def execute_architecture_spec(
    *,
    spec: Any,
    job_id: str,
    llm_call_fn: Optional[Callable],
    artifact_root: str,
    result: Any,
) -> bool:
    """Execute an architecture spec via Critical Pipeline architecture document.

    Mutates *result* in place. Returns True if handled, False to fall through.
    """
    from app.overwatcher._overwatcher_command_utils_2 import _find_architecture_for_spec

    logger.info("[ow_cmd] v5.2 ARCHITECTURE SPEC: %s — searching for arch doc", spec.spec_id)

    arch_artifacts = _find_architecture_for_spec(
        spec_id=spec.spec_id, artifact_root=artifact_root,
    )

    if not arch_artifacts or not arch_artifacts.get("architecture"):
        result.error = (
            f"Spec {spec.spec_id} is an architecture spec but no Critical Pipeline "
            f"architecture document was found. Run the Critical Pipeline first."
        )
        result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error, "is_architecture_spec": True})
        logger.error("[ow_cmd] %s", result.error)
        return True

    arch_path = arch_artifacts["architecture"]
    try:
        arch_content = Path(arch_path).read_text(encoding="utf-8")
    except Exception as e:
        result.error = f"Found architecture at {arch_path} but failed to read: {e}"
        result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
        return True

    result.spec = spec
    result.add_trace("SPEC_RESOLVE", "success_architecture", {
        "spec_id": spec.spec_id, "is_architecture_spec": True,
        "architecture_path": arch_path, "architecture_chars": len(arch_content),
    })

    from .architecture_executor import run_architecture_execution

    arch_result = await run_architecture_execution(
        spec=spec, architecture_content=arch_content,
        architecture_path=arch_path, job_id=job_id,
        llm_call_fn=llm_call_fn, artifact_root=artifact_root,
    )

    result.success = arch_result.get("success", False)
    result.overwatcher_decision = arch_result.get("decision", "PASS" if arch_result.get("success") else "FAIL")
    result.error = arch_result.get("error")
    result.stage_trace.extend(arch_result.get("trace", []))
    result.artifacts_written = arch_result.get("artifacts_written", [])
    return True


__all__ = ["execute_pot_spec", "execute_architecture_spec"]

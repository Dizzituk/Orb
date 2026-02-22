# FILE: app/overwatcher/overwatcher_command.py
"""Overwatcher Command Handler: Entry point for 'run overwatcher' command.

v5.1 (2026-02-04): Idempotent re-run + path inference support.
    - Build validation uses affected_files (written + already_applied)
      so it runs even on idempotent POT re-runs
    - Works with pot_spec_executor v2.0 idempotent task handling
v5.0 (2026-02-03): Build validation + self-correction loop.
    - After POT execution, runs build validation in sandbox
    - If build fails: LLM diagnoses error, generates fix, re-validates
    - Bounded retry loop (max 3 attempts, configurable)
    - Escalates to user with full evidence if stuck
v4.4 (2026-02-03): Fixed POT spec routing order bug.
    - POT spec check now happens BEFORE get_target_file() call
    - Fixed import: pot_spec_executor (was missing from overwatcher.py)
    - Added pot_spec_executor.py for POT atomic task execution
    - Better error messages for failed POT parsing
v4.3 (2026-01): Refactored into modules:
    - spec_parsing.py: Parsing logic
    - spec_resolution.py: DB resolution
    - implementer.py: Execution and verification

This file handles orchestration only.

SAFETY INVARIANT:
    - ASTRA may ONLY write to Windows Sandbox
    - NO host filesystem writes permitted
    - If sandbox unavailable → FAIL (no local fallback)
    - Build validation bounded to MAX_BUILD_FIX_ATTEMPTS retries
    - All diagnostic fixes go through sandbox_client only
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from app.overwatcher.evidence import EvidenceBundle, FileChange
from app.overwatcher.overwatcher import (
    run_overwatcher,
    OverwatcherOutput,
    Decision,
)

# Import from refactored modules
from .spec_parsing import ParsedDeliverable, parse_spec_content, DEFAULT_TARGET
from .spec_resolution import (
    ResolvedSpec,
    SpecMissingDeliverableError,
    resolve_latest_spec,
    create_smoke_test_spec,
)
from .implementer import (
    ImplementerResult,
    VerificationResult,
    run_implementer,
    run_verification,
)
from app.overwatcher._overwatcher_command_utils_2 import ALLOWED_HOST_WRITE_PATH, FileExistenceError, OverwatcherCommandResult, SpecParseError, _find_architecture_for_spec, build_overwatcher_evidence, load_critical_pipeline_artifacts
from app.overwatcher._overwatcher_command_utils_3 import DEFAULT_ARTIFACT_ROOT

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


# =============================================================================
# Exceptions (re-exported for backwards compatibility)
# =============================================================================


# =============================================================================
# Artifact Loading
# =============================================================================


# =============================================================================
# v5.2: Architecture Document Discovery
# =============================================================================


# =============================================================================
# Evidence Building
# =============================================================================


# =============================================================================
# Main Command Handler
# =============================================================================


async def run_overwatcher_command(
    *,
    project_id: int = 0,
    job_id: Optional[str] = None,
    message: str = "",
    db_session=None,
    llm_call_fn: Optional[Callable] = None,
    use_smoke_test: bool = False,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    # NEW (from overwatcher_stream)
    evidence_bundle: Optional[Any] = None,
    artifact_bindings: Optional[Dict[str, Any]] = None,
) -> OverwatcherCommandResult:
    """Execute the 'run overwatcher' command.

    FAILS HARD if:
    - No spec found (unless use_smoke_test=True)
    - Spec has no parseable deliverable (for non-smoke specs)
    - File doesn't exist when action=modify with must_exist=True

    New params:
    - evidence_bundle: Optional pre-built EvidenceBundle (or dict) from caller.
      If provided and compatible, it is used; otherwise we fall back to
      building evidence from the spec (no hard failure).
    - artifact_bindings: Accepted for compatibility with streaming callers.
      Currently unused here to avoid behavioural changes.
    """
    job_id = job_id or str(uuid4())
    result = OverwatcherCommandResult(success=False, job_id=job_id)

    # Set journal context so all pipeline stages can emit entries
    try:
        from app.experience.context import set_job_context, clear_job_context
        _job_dir = os.path.join(artifact_root, "jobs", job_id)
        set_job_context(job_id=job_id, job_dir=_job_dir)
    except Exception:
        pass

    logger.info("[overwatcher_command] ========================================")
    logger.info(
        "[overwatcher_command] Starting job=%s, project=%s", job_id, project_id
    )
    logger.info("[overwatcher_command] use_smoke_test=%s", use_smoke_test)
    logger.info("[overwatcher_command] ========================================")

    result.add_trace(
        "OVERWATCHER_COMMAND_START",
        "started",
        {
            "project_id": project_id,
            "use_smoke_test": use_smoke_test,
        },
    )

    # ==========================================================================
    # Step 1: Resolve spec
    # ==========================================================================
    spec = resolve_latest_spec(project_id, db_session)

    if spec is None:
        if use_smoke_test:
            logger.info(
                "[overwatcher_command] No spec found for project %s, using SMOKE TEST",
                project_id,
            )
            spec = create_smoke_test_spec()
            result.add_trace(
                "SPEC_RESOLVE",
                "smoke_test",
                {"reason": "no_spec_found"},
            )
        else:
            result.error = (
                "No validated spec found. Run Spec Gate first, or use "
                "use_smoke_test=True."
            )
            result.add_trace(
                "SPEC_RESOLVE",
                "failed",
                {"error": result.error},
            )
            logger.error("[overwatcher_command] %s", result.error)
            return result
    else:
        # ======================================================================
        # v4.4 FIX: POT spec routing MUST happen BEFORE get_target_file()
        # because POT specs have no deliverable and get_target_file() crashes.
        # ======================================================================

        # First: check for POT spec and route early
        logger.info(
            f"[overwatcher_command] POT check: is_pot_spec={spec.is_pot_spec}, "
            f"pot_tasks={spec.pot_tasks is not None}, "
            f"is_valid={spec.pot_tasks.is_valid if spec.pot_tasks else 'N/A'}"
        )

        if spec.is_pot_spec and spec.pot_tasks and spec.pot_tasks.is_valid:
            # POT spec with valid tasks → route to POT executor immediately
            result.spec = spec
            logger.info(
                "[overwatcher_command] POT SPEC DETECTED: %d tasks to execute",
                len(spec.pot_tasks.tasks)
            )
            result.add_trace(
                "SPEC_RESOLVE",
                "success_pot",
                {
                    "spec_id": spec.spec_id,
                    "is_pot_spec": True,
                    "task_count": len(spec.pot_tasks.tasks),
                    "search_term": spec.pot_tasks.search_term,
                    "replace_term": spec.pot_tasks.replace_term,
                },
            )
            result.add_trace(
                "POT_SPEC_DETECTED",
                "routing_to_pot_execution",
                {
                    "task_count": len(spec.pot_tasks.tasks),
                    "search_term": spec.pot_tasks.search_term,
                    "replace_term": spec.pot_tasks.replace_term,
                },
            )

            # Import POT execution handler from pot_spec_executor
            from .pot_spec_executor import run_pot_spec_execution

            # Execute POT spec with sequential task processing
            pot_result = await run_pot_spec_execution(
                spec=spec,
                pot_tasks=spec.pot_tasks,
                job_id=job_id,
                llm_call_fn=llm_call_fn,
                artifact_root=artifact_root,
            )

            # Map POT result to OverwatcherCommandResult
            result.success = pot_result.get("success", False)
            result.overwatcher_decision = pot_result.get(
                "decision", "PASS" if pot_result.get("success") else "FAIL"
            )
            result.error = pot_result.get("error")
            result.stage_trace.extend(pot_result.get("trace", []))
            result.artifacts_written = pot_result.get("artifacts_written", [])

            logger.info(
                "[overwatcher_command] POT execution complete: success=%s, "
                "tasks_completed=%d/%d, already_applied=%d",
                result.success,
                pot_result.get("tasks_completed", 0),
                pot_result.get("total_tasks", 0),
                pot_result.get("tasks_already_applied", 0),
            )

            # ==================================================================
            # v5.0: Build Validation + Self-Correction Loop
            # ==================================================================
            # If POT writes failed, return immediately (nothing to validate)
            if not pot_result.get("success", False):
                logger.warning(
                    "[overwatcher_command] POT writes failed — skipping build validation"
                )
                return result

            # POT writes succeeded → validate the build in sandbox
            from .sandbox_build_validator import run_build_validation_loop
            from .sandbox_client import get_sandbox_client, SandboxError

            # Use affected_files (includes already-applied) so build validation
            # runs even on idempotent re-runs; fall back to artifacts_written
            modified_files = pot_result.get("affected_files", []) or pot_result.get("artifacts_written", [])

            if not modified_files:
                logger.info(
                    "[overwatcher_command] No artifacts written — skipping build validation"
                )
                result.add_trace(
                    "BUILD_VALIDATION", "skipped",
                    {"reason": "no_artifacts_written"},
                )
                return result

            # Get sandbox client for build validation
            try:
                build_client = get_sandbox_client()
                if not build_client.is_connected():
                    logger.warning(
                        "[overwatcher_command] Sandbox not available for build validation "
                        "— returning POT result as-is (warning)"
                    )
                    result.add_trace(
                        "BUILD_VALIDATION", "skipped_no_sandbox",
                        {"reason": "sandbox_unavailable"},
                    )
                    return result
            except Exception as e:
                logger.warning(
                    "[overwatcher_command] Could not connect to sandbox for build "
                    "validation: %s — returning POT result as-is", e
                )
                result.add_trace(
                    "BUILD_VALIDATION", "skipped_error",
                    {"reason": str(e)},
                )
                return result

            # Run the full build validation + diagnostic/retry loop
            build_passed, build_results, fix_history = await run_build_validation_loop(
                client=build_client,
                modified_files=modified_files,
                spec_content=spec.spec_content or "",
                pot_result=pot_result,
                llm_call_fn=llm_call_fn,
                add_trace=result.add_trace,
            )

            if build_passed:
                result.success = True
                result.overwatcher_decision = "PASS"
                logger.info(
                    "[overwatcher_command] ✓ POT execution + build validation PASSED"
                )
                result.add_trace(
                    "BUILD_VALIDATION_COMPLETE", "passed",
                    {
                        "build_results": [
                            r.to_dict() for r in build_results
                        ] if build_results else [],
                        "fix_attempts": len(fix_history),
                    },
                )
            else:
                result.success = False
                result.overwatcher_decision = "FAIL"

                # Build comprehensive error message for user
                from .sandbox_build_validator import MAX_BUILD_FIX_ATTEMPTS
                failed_summaries = [
                    f"{r.project_type}: {r.error_summary or 'unknown error'}"
                    for r in (build_results or []) if not r.passed
                ]
                result.error = (
                    f"Build validation failed after {len(fix_history)} fix attempts "
                    f"(max {MAX_BUILD_FIX_ATTEMPTS}). "
                    f"Errors: {'; '.join(failed_summaries)}"
                )

                logger.error(
                    "[overwatcher_command] ✗ Build validation FAILED: %s",
                    result.error,
                )
                result.add_trace(
                    "BUILD_VALIDATION_COMPLETE", "failed",
                    {
                        "error": result.error,
                        "build_results": [
                            r.to_dict() for r in build_results
                        ] if build_results else [],
                        "fix_history": fix_history,
                    },
                )

            return result  # Exit — POT execution + build validation complete

        # Non-POT path: validate deliverable exists
        if not spec.is_smoke_test and spec.deliverable is None:
            # =============================================================
            # v5.2: Architecture spec routing (Bug 1 fix)
            # When SpecGate grounded-create produces rich markdown but
            # empty JSON, the spec gets flagged as is_architecture_spec.
            # Route to the Implementer via the Critical Pipeline
            # architecture document instead of hard-failing.
            # =============================================================
            if spec.is_architecture_spec:
                logger.info(
                    "[overwatcher_command] v5.2 ARCHITECTURE SPEC ROUTING: "
                    "spec %s has no deliverable but is an architecture spec. "
                    "Looking for Critical Pipeline architecture document...",
                    spec.spec_id,
                )
                print(
                    f">>> [ARCH_ROUTE] Architecture spec detected for {spec.spec_id} — "
                    f"searching for Critical Pipeline architecture document"
                )
                
                # Try to find the architecture document
                # The job_id used by the Critical Pipeline is cp-{hash}
                # We need to scan for it since the Overwatcher doesn't receive
                # the CP job_id directly.
                arch_artifacts = _find_architecture_for_spec(
                    spec_id=spec.spec_id,
                    artifact_root=artifact_root,
                )
                
                if arch_artifacts and arch_artifacts.get("architecture"):
                    arch_path = arch_artifacts["architecture"]
                    logger.info(
                        "[overwatcher_command] v5.2 Found architecture document: %s",
                        arch_path,
                    )
                    print(f">>> [ARCH_ROUTE] \u2713 Found architecture: {arch_path}")
                    
                    # Read the architecture document
                    try:
                        arch_content = Path(arch_path).read_text(encoding="utf-8")
                        logger.info(
                            "[overwatcher_command] v5.2 Architecture document loaded: %d chars",
                            len(arch_content),
                        )
                    except Exception as e:
                        result.error = (
                            f"Found architecture document at {arch_path} but failed to read it: {e}"
                        )
                        result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
                        logger.error("[overwatcher_command] %s", result.error)
                        return result
                    
                    result.spec = spec
                    result.add_trace(
                        "SPEC_RESOLVE",
                        "success_architecture",
                        {
                            "spec_id": spec.spec_id,
                            "is_architecture_spec": True,
                            "architecture_path": arch_path,
                            "architecture_chars": len(arch_content),
                            "spec_markdown_chars": len(spec.architecture_markdown or ""),
                        },
                    )
                    result.add_trace(
                        "ARCHITECTURE_SPEC_DETECTED",
                        "routing_to_architecture_execution",
                        {
                            "architecture_path": arch_path,
                            "architecture_chars": len(arch_content),
                        },
                    )
                    
                    # Route to architecture-based implementation
                    # For now, pass to the LLM Overwatcher with the full
                    # architecture as context, then let it proceed to
                    # implementer for multi-file execution.
                    # This is the bridge — the architecture document IS
                    # the execution plan.
                    from .architecture_executor import run_architecture_execution
                    
                    arch_result = await run_architecture_execution(
                        spec=spec,
                        architecture_content=arch_content,
                        architecture_path=arch_path,
                        job_id=job_id,
                        llm_call_fn=llm_call_fn,
                        artifact_root=artifact_root,
                    )
                    
                    result.success = arch_result.get("success", False)
                    result.overwatcher_decision = arch_result.get(
                        "decision", "PASS" if arch_result.get("success") else "FAIL"
                    )
                    result.error = arch_result.get("error")
                    result.stage_trace.extend(arch_result.get("trace", []))
                    result.artifacts_written = arch_result.get("artifacts_written", [])
                    
                    logger.info(
                        "[overwatcher_command] v5.2 Architecture execution complete: success=%s",
                        result.success,
                    )
                    return result
                else:
                    logger.warning(
                        "[overwatcher_command] v5.2 Architecture spec detected but no "
                        "architecture document found. Has the Critical Pipeline been run?"
                    )
                    result.error = (
                        f"Spec {spec.spec_id} is an architecture spec but no Critical Pipeline "
                        f"architecture document was found. Run the Critical Pipeline first "
                        f"('Astra, command: run critical pipeline') to generate the architecture."
                    )
                    result.add_trace(
                        "SPEC_RESOLVE",
                        "failed",
                        {"error": result.error, "is_architecture_spec": True},
                    )
                    logger.error("[overwatcher_command] %s", result.error)
                    return result
            
            # Check if this was a POT spec that failed to parse
            elif spec.is_pot_spec:
                pot_errors = spec.pot_tasks.errors if spec.pot_tasks else ["No POT tasks parsed"]
                result.error = (
                    f"Spec {spec.spec_id} is a POT spec but parsing failed: "
                    f"{pot_errors}. Check POT markdown format in content_markdown."
                )
            else:
                result.error = (
                    f"Spec {spec.spec_id} has no parseable deliverable. "
                    "Cannot determine target file. Check spec content format."
                )
            result.add_trace(
                "SPEC_RESOLVE",
                "failed",
                {"error": result.error},
            )
            logger.error("[overwatcher_command] %s", result.error)
            return result

        try:
            filename, content, action = spec.get_target_file()
            result.add_trace(
                "SPEC_RESOLVE",
                "success",
                {
                    "spec_id": spec.spec_id,
                    "is_smoke_test": spec.is_smoke_test,
                    "target_file": filename,
                    "target_content": (content[:50] if content else ""),
                    "action": action,
                    "must_exist": spec.get_must_exist(),
                },
            )
            logger.info(
                "[overwatcher_command] Spec target: %s '%s' with '%s'",
                action,
                filename,
                (content[:50] if content else ""),
            )
        except SpecMissingDeliverableError as e:
            result.error = str(e)
            result.add_trace(
                "SPEC_RESOLVE",
                "failed",
                {"error": result.error},
            )
            logger.error("[overwatcher_command] %s", result.error)
            return result

    result.spec = spec

    # ==========================================================================
    # Step 2: Load artifacts (NON-POT path)
    # ==========================================================================
    artifacts = load_critical_pipeline_artifacts(job_id, artifact_root)
    result.add_trace(
        "ARTIFACTS_LOAD",
        "success" if artifacts["exists"] else "none",
        artifacts,
    )

    # ==========================================================================
    # Step 3: Evidence (use provided bundle if present, else build)
    # ==========================================================================

    def _build_evidence_from_spec() -> EvidenceBundle:
        ev = build_overwatcher_evidence(
            job_id=job_id,
            spec=spec,
            artifacts=artifacts,
        )
        result.add_trace(
            "EVIDENCE_BUILD",
            "success",
            {
                "chunk_id": ev.chunk_id,
                "source": "spec",
            },
        )
        return ev

    if evidence_bundle is not None:
        # Caller can pass a dict or an EvidenceBundle; normalise to EvidenceBundle.
        if isinstance(evidence_bundle, EvidenceBundle):
            evidence = evidence_bundle
            result.add_trace(
                "EVIDENCE_BUILD",
                "provided",
                {
                    "chunk_id": getattr(evidence, "chunk_id", None),
                    "source": "caller",
                    "type": "EvidenceBundle",
                },
            )
        elif isinstance(evidence_bundle, dict):
            # Try to coerce; if it fails, fall back to building from spec
            try:
                # Some callers include keys EvidenceBundle doesn't know about,
                # like 'artifacts' – strip unknown keys first.
                allowed_keys = EvidenceBundle.__dataclass_fields__.keys() \
                    if hasattr(EvidenceBundle, "__dataclass_fields__") else None
                if allowed_keys is not None:
                    filtered = {k: v for k, v in evidence_bundle.items() if k in allowed_keys}
                else:
                    filtered = evidence_bundle
                evidence = EvidenceBundle(**filtered)
                result.add_trace(
                    "EVIDENCE_BUILD",
                    "provided",
                    {
                        "chunk_id": getattr(evidence, "chunk_id", None),
                        "source": "caller",
                        "type": "dict_coerced",
                    },
                )
            except TypeError as e:
                logger.warning(
                    "[overwatcher_command] Evidence bundle dict not compatible with "
                    "EvidenceBundle (%s); rebuilding from spec instead.",
                    e,
                )
                evidence = _build_evidence_from_spec()
        else:
            logger.warning(
                "[overwatcher_command] Unsupported evidence_bundle type: %s; "
                "rebuilding from spec.",
                type(evidence_bundle),
            )
            evidence = _build_evidence_from_spec()
    else:
        # No evidence provided → use spec-driven evidence
        evidence = _build_evidence_from_spec()

    # NOTE: artifact_bindings is currently accepted but unused here to avoid
    # changing existing behaviour. It remains available for future wiring.

    # ==========================================================================
    # Step 4: Run Overwatcher (LLM analysis / policy gate)
    # ==========================================================================
    if llm_call_fn:
        try:
            logger.info("[overwatcher_command] Running Overwatcher LLM analysis...")
            result.add_trace("OVERWATCHER_ENTER", "running")

            ow_output: OverwatcherOutput = await run_overwatcher(
                evidence=evidence,
                llm_call_fn=llm_call_fn,
                job_artifact_root=artifact_root,
            )

            result.overwatcher_decision = ow_output.decision.value
            result.decision = ow_output.decision.value  # keep alias in sync
            result.overwatcher_diagnosis = ow_output.diagnosis

            # Handle decisions explicitly
            if ow_output.decision == Decision.PASS:
                result.add_trace(
                    "OVERWATCHER_EXIT",
                    "complete",
                    {
                        "decision": ow_output.decision.value,
                        "confidence": ow_output.confidence,
                    },
                )
                logger.info(
                    "[overwatcher_command] Overwatcher decision: %s",
                    ow_output.decision.value,
                )

            elif getattr(Decision, "NEEDS_INFO", None) is not None and ow_output.decision == Decision.NEEDS_INFO:  # type: ignore[attr-defined]
                # Soft-pass: log + trace, but DO NOT block implementer.
                logger.warning(
                    "[overwatcher_command] Overwatcher returned NEEDS_INFO: %s "
                    "- treating as SOFT PASS for this run.",
                    ow_output.diagnosis,
                )
                result.add_trace(
                    "OVERWATCHER_NEEDS_INFO",
                    "soft_pass",
                    {
                        "decision": ow_output.decision.value,
                        "diagnosis": ow_output.diagnosis,
                        "confidence": ow_output.confidence,
                    },
                )

            else:
                # Hard reject for explicit FAIL / BLOCK / whatever other enums exist
                result.add_trace(
                    "OVERWATCHER_EXIT",
                    "complete",
                    {
                        "decision": ow_output.decision.value,
                        "confidence": ow_output.confidence,
                    },
                )
                logger.info(
                    "[overwatcher_command] Overwatcher decision: %s",
                    ow_output.decision.value,
                )

                result.error = f"Overwatcher rejected: {ow_output.diagnosis}"
                result.add_trace(
                    "OVERWATCHER_REJECT",
                    "failed",
                    {"diagnosis": ow_output.diagnosis},
                )
                return result

        except Exception as e:  # noqa: BLE001
            logger.exception("[overwatcher_command] Overwatcher failed: %s", e)
            result.error = f"Overwatcher failed: {e}"
            result.add_trace(
                "OVERWATCHER_ERROR",
                "failed",
                {"error": str(e)},
            )
            return result
    else:
        # No LLM - only auto-approve for explicit smoke tests
        if spec.is_smoke_test:
            result.overwatcher_decision = Decision.PASS.value
            result.decision = Decision.PASS.value
            result.overwatcher_diagnosis = "Auto-approved (smoke test, no LLM)"
            result.add_trace(
                "OVERWATCHER_SKIP",
                "auto_approved",
                {"reason": "smoke_test"},
            )
        else:
            result.error = "LLM function required for non-smoke-test jobs"
            result.add_trace(
                "OVERWATCHER_ERROR",
                "failed",
                {"error": result.error},
            )
            return result

    # ==========================================================================
    # Step 5: Run Implementer
    # ==========================================================================
    logger.info("[overwatcher_command] Running Implementer...")
    result.add_trace("IMPLEMENTER_ENTER", "running")

    impl_output = OverwatcherOutput(
        decision=Decision.PASS,
        diagnosis=result.overwatcher_diagnosis or "Approved",
    )

    impl_result = await run_implementer(spec=spec, output=impl_output)
    result.implementer_result = impl_result

    if impl_result.success:
        # Record artifact(s) written so streaming/UI can show them
        result.artifacts_written = []
        if getattr(impl_result, "output_path", None):
            result.artifacts_written.append(impl_result.output_path)

        result.add_trace(
            "IMPLEMENTER_EXIT",
            "success",
            {
                "output_path": impl_result.output_path,
                "filename": impl_result.filename,
                "action": impl_result.action_taken,
            },
        )
        logger.info(
            "[overwatcher_command] Implementer success: %s",
            impl_result.output_path,
        )
    else:
        result.error = f"Implementer failed: {impl_result.error}"
        result.add_trace(
            "IMPLEMENTER_EXIT",
            "failed",
            {"error": impl_result.error},
        )
        logger.error("[overwatcher_command] %s", result.error)
        return result

    # ==========================================================================
    # Step 6: Verification
    # ==========================================================================
    logger.info("[overwatcher_command] Running verification...")
    result.add_trace("VERIFICATION_ENTER", "running")

    verify_result = await run_verification(impl_result=impl_result, spec=spec)
    result.verification_result = verify_result

    if verify_result.passed:
        result.success = True
        result.add_trace(
            "VERIFICATION_EXIT",
            "passed",
            verify_result.to_dict(),
        )
        result.add_trace(
            "JOB_COMPLETE",
            "success",
            {"job_id": job_id},
        )
        logger.info("[overwatcher_command] ✓ Job %s COMPLETE - PASSED", job_id)
    else:
        result.error = f"Verification failed: {verify_result.error}"
        result.add_trace(
            "VERIFICATION_EXIT",
            "failed",
            {"error": verify_result.to_dict()},
        )
        logger.error(
            "[overwatcher_command] ✗ Job %s FAILED verification: %s",
            job_id,
            result.error,
        )

    return result


# =============================================================================
# Exports (backwards compatible)
# =============================================================================

__all__ = [
    # From spec_parsing
    "ParsedDeliverable",
    "parse_spec_content",
    "DEFAULT_TARGET",
    # From spec_resolution
    "ResolvedSpec",
    "SpecMissingDeliverableError",
    "resolve_latest_spec",
    "create_smoke_test_spec",
    # From implementer
    "ImplementerResult",
    "VerificationResult",
    "run_implementer",
    "run_verification",
    # Local
    "OverwatcherCommandResult",
    "SpecParseError",
    "FileExistenceError",
    "load_critical_pipeline_artifacts",
    "build_overwatcher_evidence",
    "run_overwatcher_command",
    "DEFAULT_ARTIFACT_ROOT",
    "ALLOWED_HOST_WRITE_PATH",
]

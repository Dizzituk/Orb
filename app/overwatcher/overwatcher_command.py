# FILE: app/overwatcher/overwatcher_command.py
"""Overwatcher Command Handler: Entry point for 'run overwatcher' command.

v5.3 (2026-02-24): Refactored — POT/arch routing extracted to _ow_cmd_spec_routing.py
v5.2: Architecture document discovery
v5.1: Idempotent re-run + path inference
v5.0: Build validation + self-correction loop
v4.4: Fixed POT spec routing order bug
v4.3: Refactored into spec_parsing, spec_resolution, implementer modules

SAFETY INVARIANT:
    - ASTRA may ONLY write to Windows Sandbox
    - NO host filesystem writes permitted
    - Build validation bounded to MAX_BUILD_FIX_ATTEMPTS retries
"""

from __future__ import annotations

import logging
import os
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
from app.overwatcher._overwatcher_command_utils_2 import (
    ALLOWED_HOST_WRITE_PATH,
    FileExistenceError,
    OverwatcherCommandResult,
    SpecParseError,
    _find_architecture_for_spec,
    build_overwatcher_evidence,
    load_critical_pipeline_artifacts,
)
from app.overwatcher._overwatcher_command_utils_3 import DEFAULT_ARTIFACT_ROOT
from app.overwatcher._ow_cmd_spec_routing import execute_pot_spec, execute_architecture_spec

logger = logging.getLogger(__name__)


async def run_overwatcher_command(
    *,
    project_id: int = 0,
    job_id: Optional[str] = None,
    message: str = "",
    db_session=None,
    llm_call_fn: Optional[Callable] = None,
    use_smoke_test: bool = False,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    evidence_bundle: Optional[Any] = None,
    artifact_bindings: Optional[Dict[str, Any]] = None,
) -> OverwatcherCommandResult:
    """Execute the 'run overwatcher' command."""
    job_id = job_id or str(uuid4())
    result = OverwatcherCommandResult(success=False, job_id=job_id)

    # Set journal context
    try:
        from app.experience.context import set_job_context
        set_job_context(job_id=job_id, job_dir=os.path.join(artifact_root, "jobs", job_id))
    except Exception:
        pass

    logger.info("[ow_cmd] Starting job=%s, project=%s, smoke_test=%s", job_id, project_id, use_smoke_test)
    result.add_trace("OVERWATCHER_COMMAND_START", "started", {"project_id": project_id, "use_smoke_test": use_smoke_test})

    # Step 1: Resolve spec
    spec = resolve_latest_spec(project_id, db_session)

    if spec is None:
        if use_smoke_test:
            spec = create_smoke_test_spec()
            result.add_trace("SPEC_RESOLVE", "smoke_test", {"reason": "no_spec_found"})
        else:
            result.error = "No validated spec found. Run Spec Gate first, or use use_smoke_test=True."
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            return result
    else:
        # v4.4: POT spec routing BEFORE get_target_file()
        if spec.is_pot_spec and spec.pot_tasks and spec.pot_tasks.is_valid:
            result.spec = spec
            handled = await execute_pot_spec(
                spec=spec, job_id=job_id, llm_call_fn=llm_call_fn,
                artifact_root=artifact_root, result=result,
            )
            if handled:
                return result

        # Non-POT: validate deliverable exists
        if not spec.is_smoke_test and spec.deliverable is None:
            # v5.2: Architecture spec routing
            if spec.is_architecture_spec:
                handled = await execute_architecture_spec(
                    spec=spec, job_id=job_id, llm_call_fn=llm_call_fn,
                    artifact_root=artifact_root, result=result,
                )
                if handled:
                    return result
            elif spec.is_pot_spec:
                pot_errors = spec.pot_tasks.errors if spec.pot_tasks else ["No POT tasks parsed"]
                result.error = f"Spec {spec.spec_id} is a POT spec but parsing failed: {pot_errors}"
            else:
                result.error = f"Spec {spec.spec_id} has no parseable deliverable."
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            return result

        try:
            filename, content, action = spec.get_target_file()
            result.add_trace("SPEC_RESOLVE", "success", {
                "spec_id": spec.spec_id, "is_smoke_test": spec.is_smoke_test,
                "target_file": filename, "action": action,
            })
        except SpecMissingDeliverableError as e:
            result.error = str(e)
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            return result

    result.spec = spec

    # Step 2: Load artifacts
    artifacts = load_critical_pipeline_artifacts(job_id, artifact_root)
    result.add_trace("ARTIFACTS_LOAD", "success" if artifacts["exists"] else "none", artifacts)

    # Step 3: Evidence
    evidence = _resolve_evidence(evidence_bundle, job_id, spec, artifacts, result)

    # Step 4: Run Overwatcher (LLM analysis / policy gate)
    ow_ok = await _run_overwatcher_gate(spec, evidence, llm_call_fn, artifact_root, result)
    if not ow_ok:
        return result

    # Step 5: Run Implementer
    logger.info("[ow_cmd] Running Implementer...")
    result.add_trace("IMPLEMENTER_ENTER", "running")

    impl_output = OverwatcherOutput(
        decision=Decision.PASS,
        diagnosis=result.overwatcher_diagnosis or "Approved",
    )
    impl_result = await run_implementer(spec=spec, output=impl_output)
    result.implementer_result = impl_result

    if impl_result.success:
        result.artifacts_written = []
        if getattr(impl_result, "output_path", None):
            result.artifacts_written.append(impl_result.output_path)
        result.add_trace("IMPLEMENTER_EXIT", "success", {
            "output_path": impl_result.output_path,
            "filename": impl_result.filename,
            "action": impl_result.action_taken,
        })
    else:
        result.error = f"Implementer failed: {impl_result.error}"
        result.add_trace("IMPLEMENTER_EXIT", "failed", {"error": impl_result.error})
        return result

    # Step 6: Verification
    logger.info("[ow_cmd] Running verification...")
    result.add_trace("VERIFICATION_ENTER", "running")

    verify_result = await run_verification(impl_result=impl_result, spec=spec)
    result.verification_result = verify_result

    if verify_result.passed:
        result.success = True
        result.add_trace("VERIFICATION_EXIT", "passed", verify_result.to_dict())
        result.add_trace("JOB_COMPLETE", "success", {"job_id": job_id})
        logger.info("[ow_cmd] ✓ Job %s COMPLETE", job_id)
    else:
        result.error = f"Verification failed: {verify_result.error}"
        result.add_trace("VERIFICATION_EXIT", "failed", {"error": verify_result.to_dict()})

    return result


def _resolve_evidence(evidence_bundle, job_id, spec, artifacts, result) -> EvidenceBundle:
    """Resolve evidence from provided bundle or build from spec."""
    def _build_from_spec():
        ev = build_overwatcher_evidence(job_id=job_id, spec=spec, artifacts=artifacts)
        result.add_trace("EVIDENCE_BUILD", "success", {"chunk_id": ev.chunk_id, "source": "spec"})
        return ev

    if evidence_bundle is None:
        return _build_from_spec()

    if isinstance(evidence_bundle, EvidenceBundle):
        result.add_trace("EVIDENCE_BUILD", "provided", {"source": "caller", "type": "EvidenceBundle"})
        return evidence_bundle

    if isinstance(evidence_bundle, dict):
        try:
            allowed_keys = EvidenceBundle.__dataclass_fields__.keys() if hasattr(EvidenceBundle, "__dataclass_fields__") else None
            filtered = {k: v for k, v in evidence_bundle.items() if k in allowed_keys} if allowed_keys else evidence_bundle
            ev = EvidenceBundle(**filtered)
            result.add_trace("EVIDENCE_BUILD", "provided", {"source": "caller", "type": "dict_coerced"})
            return ev
        except TypeError:
            return _build_from_spec()

    return _build_from_spec()


async def _run_overwatcher_gate(spec, evidence, llm_call_fn, artifact_root, result) -> bool:
    """Run LLM Overwatcher gate. Returns True if approved, False if rejected."""
    if llm_call_fn:
        try:
            result.add_trace("OVERWATCHER_ENTER", "running")
            ow_output: OverwatcherOutput = await run_overwatcher(
                evidence=evidence, llm_call_fn=llm_call_fn, job_artifact_root=artifact_root,
            )
            result.overwatcher_decision = ow_output.decision.value
            result.decision = ow_output.decision.value
            result.overwatcher_diagnosis = ow_output.diagnosis

            if ow_output.decision == Decision.PASS:
                result.add_trace("OVERWATCHER_EXIT", "complete", {"decision": "PASS", "confidence": ow_output.confidence})
                return True
            elif getattr(Decision, "NEEDS_INFO", None) is not None and ow_output.decision == Decision.NEEDS_INFO:
                result.add_trace("OVERWATCHER_NEEDS_INFO", "soft_pass", {"diagnosis": ow_output.diagnosis})
                return True
            else:
                result.add_trace("OVERWATCHER_EXIT", "complete", {"decision": ow_output.decision.value})
                result.error = f"Overwatcher rejected: {ow_output.diagnosis}"
                result.add_trace("OVERWATCHER_REJECT", "failed", {"diagnosis": ow_output.diagnosis})
                return False
        except Exception as e:
            logger.exception("[ow_cmd] Overwatcher failed: %s", e)
            result.error = f"Overwatcher failed: {e}"
            result.add_trace("OVERWATCHER_ERROR", "failed", {"error": str(e)})
            return False
    else:
        if spec.is_smoke_test:
            result.overwatcher_decision = Decision.PASS.value
            result.decision = Decision.PASS.value
            result.overwatcher_diagnosis = "Auto-approved (smoke test, no LLM)"
            result.add_trace("OVERWATCHER_SKIP", "auto_approved", {"reason": "smoke_test"})
            return True
        else:
            result.error = "LLM function required for non-smoke-test jobs"
            result.add_trace("OVERWATCHER_ERROR", "failed", {"error": result.error})
            return False


# Exports (backwards compatible)
__all__ = [
    "ParsedDeliverable", "parse_spec_content", "DEFAULT_TARGET",
    "ResolvedSpec", "SpecMissingDeliverableError", "resolve_latest_spec", "create_smoke_test_spec",
    "ImplementerResult", "VerificationResult", "run_implementer", "run_verification",
    "OverwatcherCommandResult", "SpecParseError", "FileExistenceError",
    "load_critical_pipeline_artifacts", "build_overwatcher_evidence",
    "run_overwatcher_command", "DEFAULT_ARTIFACT_ROOT", "ALLOWED_HOST_WRITE_PATH",
]

# FILE: app/overwatcher/overwatcher_command.py
"""Overwatcher Command Handler: Entry point for 'run overwatcher' command.

v4.2 (2026-01): Refactored into modules:
    - spec_parsing.py: Parsing logic
    - spec_resolution.py: DB resolution
    - implementer.py: Execution and verification

This file handles orchestration only.

SAFETY INVARIANT:
    - ASTRA may ONLY write to Windows Sandbox
    - NO host filesystem writes permitted
    - If sandbox unavailable → FAIL (no local fallback)
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

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ARTIFACT_ROOT = os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs")
ALLOWED_HOST_WRITE_PATH = Path("D:/Tools/zobie_mapper/out")


# =============================================================================
# Exceptions (re-exported for backwards compatibility)
# =============================================================================

class SpecParseError(Exception):
    """Raised when spec content cannot be parsed to extract deliverable."""
    pass


class FileExistenceError(Exception):
    """Raised when a file that should exist doesn't, or vice versa."""
    pass


# =============================================================================
# Artifact Loading
# =============================================================================

def load_critical_pipeline_artifacts(
    job_id: str,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Load artifacts from Critical Pipeline if they exist."""
    artifacts = {
        "architecture": None,
        "critique": None,
        "plan": None,
        "exists": False,
    }
    
    job_dir = Path(artifact_root) / job_id
    if not job_dir.exists():
        return artifacts
    
    for name, paths in [
        ("architecture", ["architecture/latest.md", "arch_v1.md"]),
        ("critique", ["critique/latest.json", "critique_v1.json"]),
        ("plan", ["plan/chunk_plan.json"]),
    ]:
        for rel_path in paths:
            if (job_dir / rel_path).exists():
                artifacts[name] = str(job_dir / rel_path)
                break
    
    artifacts["exists"] = any([artifacts["architecture"], artifacts["critique"], artifacts["plan"]])
    return artifacts


# =============================================================================
# Evidence Building
# =============================================================================

def build_overwatcher_evidence(
    *,
    job_id: str,
    spec: ResolvedSpec,
    artifacts: Dict[str, Any],
    strike_number: int = 1,
    chunk_id: Optional[str] = None,
) -> EvidenceBundle:
    """Build EvidenceBundle from spec content.
    
    Raises SpecMissingDeliverableError if spec has no deliverable.
    """
    stage_run_id = str(uuid4())
    chunk_id = chunk_id or f"chunk-{uuid4().hex[:8]}"
    
    filename, content, action = spec.get_target_file()
    description = spec.get_task_description()
    
    logger.info(f"[build_evidence] File: {filename}, Action: {action}, Content: {content[:50] if content else ''}...")
    
    file_changes = [FileChange(
        path=filename,
        action=action,
        intent=description,
    )]
    
    return EvidenceBundle(
        job_id=job_id,
        chunk_id=chunk_id,
        stage_run_id=stage_run_id,
        spec_id=spec.spec_id,
        spec_hash=spec.spec_hash,
        strike_number=strike_number,
        file_changes=file_changes,
        chunk_title=spec.title or "Overwatcher Job",
        chunk_objective=description,
        verification_commands=[],
    )


# =============================================================================
# Main Command Handler
# =============================================================================

@dataclass
class OverwatcherCommandResult:
    """Complete result from 'run overwatcher' command."""
    success: bool
    job_id: str
    spec: Optional[ResolvedSpec] = None
    overwatcher_decision: Optional[str] = None
    overwatcher_diagnosis: Optional[str] = None
    implementer_result: Optional[ImplementerResult] = None
    verification_result: Optional[VerificationResult] = None
    error: Optional[str] = None
    stage_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "spec": self.spec.to_dict() if self.spec else None,
            "overwatcher_decision": self.overwatcher_decision,
            "overwatcher_diagnosis": self.overwatcher_diagnosis,
            "implementer_result": self.implementer_result.to_dict() if self.implementer_result else None,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "error": self.error,
            "stage_trace": self.stage_trace,
        }
    
    def add_trace(self, stage: str, status: str, details: Optional[Dict] = None):
        self.stage_trace.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })


async def run_overwatcher_command(
    *,
    project_id: int = 0,
    job_id: Optional[str] = None,
    message: str = "",
    db_session=None,
    llm_call_fn: Optional[Callable] = None,
    use_smoke_test: bool = False,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> OverwatcherCommandResult:
    """Execute the 'run overwatcher' command.
    
    FAILS HARD if:
    - No spec found (unless use_smoke_test=True)
    - Spec has no parseable deliverable (for non-smoke specs)
    - File doesn't exist when action=modify with must_exist=True
    """
    job_id = job_id or str(uuid4())
    result = OverwatcherCommandResult(success=False, job_id=job_id)
    
    logger.info(f"[overwatcher_command] ========================================")
    logger.info(f"[overwatcher_command] Starting job={job_id}, project={project_id}")
    logger.info(f"[overwatcher_command] use_smoke_test={use_smoke_test}")
    logger.info(f"[overwatcher_command] ========================================")
    
    result.add_trace("OVERWATCHER_COMMAND_START", "started", {
        "project_id": project_id,
        "use_smoke_test": use_smoke_test,
    })
    
    # ==========================================================================
    # Step 1: Resolve spec
    # ==========================================================================
    spec = resolve_latest_spec(project_id, db_session)
    
    if spec is None:
        if use_smoke_test:
            logger.info("[overwatcher_command] No spec found, using SMOKE TEST")
            spec = create_smoke_test_spec()
            result.add_trace("SPEC_RESOLVE", "smoke_test", {"reason": "no_spec_found"})
        else:
            result.error = "No validated spec found. Run Spec Gate first, or use use_smoke_test=True."
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            logger.error(f"[overwatcher_command] {result.error}")
            return result
    else:
        # Validate deliverable exists for NON-smoke specs
        if not spec.is_smoke_test and spec.deliverable is None:
            result.error = (
                f"Spec {spec.spec_id} has no parseable deliverable. "
                f"Cannot determine target file. Check spec content format."
            )
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            logger.error(f"[overwatcher_command] {result.error}")
            return result
        
        # Try to get target file
        try:
            filename, content, action = spec.get_target_file()
            result.add_trace("SPEC_RESOLVE", "success", {
                "spec_id": spec.spec_id,
                "is_smoke_test": spec.is_smoke_test,
                "target_file": filename,
                "target_content": content[:50] if content else "",
                "action": action,
                "must_exist": spec.get_must_exist(),
            })
            logger.info(f"[overwatcher_command] Spec target: {action} '{filename}' with '{content}'")
        except SpecMissingDeliverableError as e:
            result.error = str(e)
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            logger.error(f"[overwatcher_command] {result.error}")
            return result
    
    result.spec = spec
    
    # ==========================================================================
    # Step 2: Load artifacts
    # ==========================================================================
    artifacts = load_critical_pipeline_artifacts(job_id, artifact_root)
    result.add_trace("ARTIFACTS_LOAD", "success" if artifacts["exists"] else "none", artifacts)
    
    # ==========================================================================
    # Step 3: Build evidence
    # ==========================================================================
    try:
        evidence = build_overwatcher_evidence(
            job_id=job_id,
            spec=spec,
            artifacts=artifacts,
        )
        result.add_trace("EVIDENCE_BUILD", "success", {"chunk_id": evidence.chunk_id})
    except SpecMissingDeliverableError as e:
        result.error = str(e)
        result.add_trace("EVIDENCE_BUILD", "failed", {"error": result.error})
        return result
    
    # ==========================================================================
    # Step 4: Run Overwatcher
    # ==========================================================================
    if llm_call_fn:
        try:
            logger.info("[overwatcher_command] Running Overwatcher LLM analysis...")
            result.add_trace("OVERWATCHER_ENTER", "running")
            
            ow_output = await run_overwatcher(
                evidence=evidence,
                llm_call_fn=llm_call_fn,
                job_artifact_root=artifact_root,
            )
            
            result.overwatcher_decision = ow_output.decision.value
            result.overwatcher_diagnosis = ow_output.diagnosis
            
            result.add_trace("OVERWATCHER_EXIT", "complete", {
                "decision": ow_output.decision.value,
                "confidence": ow_output.confidence,
            })
            
            logger.info(f"[overwatcher_command] Overwatcher decision: {ow_output.decision.value}")
            
            if ow_output.decision != Decision.PASS:
                result.error = f"Overwatcher rejected: {ow_output.diagnosis}"
                result.add_trace("OVERWATCHER_REJECT", "failed", {"diagnosis": ow_output.diagnosis})
                return result
                
        except Exception as e:
            logger.exception(f"[overwatcher_command] Overwatcher failed: {e}")
            result.error = f"Overwatcher failed: {e}"
            result.add_trace("OVERWATCHER_ERROR", "failed", {"error": str(e)})
            return result
    else:
        # No LLM - only auto-approve for explicit smoke tests
        if spec.is_smoke_test:
            result.overwatcher_decision = Decision.PASS.value
            result.overwatcher_diagnosis = "Auto-approved (smoke test, no LLM)"
            result.add_trace("OVERWATCHER_SKIP", "auto_approved", {"reason": "smoke_test"})
        else:
            result.error = "LLM function required for non-smoke-test jobs"
            result.add_trace("OVERWATCHER_ERROR", "failed", {"error": result.error})
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
        result.add_trace("IMPLEMENTER_EXIT", "success", {
            "output_path": impl_result.output_path,
            "filename": impl_result.filename,
            "action": impl_result.action_taken,
        })
        logger.info(f"[overwatcher_command] Implementer success: {impl_result.output_path}")
    else:
        result.error = f"Implementer failed: {impl_result.error}"
        result.add_trace("IMPLEMENTER_EXIT", "failed", {"error": impl_result.error})
        logger.error(f"[overwatcher_command] {result.error}")
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
        result.add_trace("VERIFICATION_EXIT", "passed", verify_result.to_dict())
        result.add_trace("JOB_COMPLETE", "success", {"job_id": job_id})
        logger.info(f"[overwatcher_command] ✓ Job {job_id} COMPLETE - PASSED")
    else:
        result.error = f"Verification failed: {verify_result.error}"
        result.add_trace("VERIFICATION_EXIT", "failed", verify_result.to_dict())
        logger.error(f"[overwatcher_command] ✗ {verify_result.error}")
    
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

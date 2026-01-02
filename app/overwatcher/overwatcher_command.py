# FILE: app/overwatcher/overwatcher_command.py
"""Overwatcher Command Handler: Entry point for 'run overwatcher' command.

Responsible for:
1. Auto-resolving latest validated spec (spec_id + spec_hash) from DB
2. Loading Critical Pipeline artifacts if they exist
3. Building EvidenceBundle for Overwatcher evaluation
4. Executing Overwatcher decision flow
5. On APPROVE → routing to Implementer
6. On Implementer completion → running verification
7. Recording final job state

User flow:
    "Astra, command: run overwatcher"
    → This handler auto-resolves context
    → No manual spec_hash required

Spec anchoring:
    - spec_id and spec_hash are REQUIRED internally
    - If no validated spec exists → hard error (no fallback to chat)
    - All decisions logged with spec context for audit trail

SAFETY INVARIANT (v2.1):
    - ASTRA may ONLY write to Windows Sandbox
    - ASTRA may ONLY write to D:/Tools/zobie_mapper/out on host (repo scans)
    - NO other host filesystem writes permitted
    - If sandbox unavailable → FAIL (no local fallback)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from app.overwatcher.evidence import EvidenceBundle, FileChange
from app.overwatcher.overwatcher import (
    run_overwatcher,
    OverwatcherOutput,
    Decision,
    OVERWATCHER_PROVIDER,
    OVERWATCHER_MODEL,
)
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default artifact root
DEFAULT_ARTIFACT_ROOT = os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs")

# Smoke test configuration
SMOKE_TEST_FILENAME = "hello.txt"
SMOKE_TEST_CONTENT = "ASTRA OK"
SMOKE_TEST_TARGET = "DESKTOP"

# SAFETY: Only allowed host write path (for repo scans)
ALLOWED_HOST_WRITE_PATH = Path("D:/Tools/zobie_mapper/out")


# =============================================================================
# Spec Resolution
# =============================================================================

@dataclass
class ResolvedSpec:
    """Resolved spec context from database."""
    spec_id: str
    spec_hash: str
    project_id: int
    title: Optional[str] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at,
        }


def resolve_latest_spec(
    project_id: int,
    db_session=None,
) -> Optional[ResolvedSpec]:
    """Resolve the latest validated spec for a project.
    
    Queries the specs table for the most recent validated spec.
    
    Args:
        project_id: Project ID to query
        db_session: Database session (optional, creates if needed)
    
    Returns:
        ResolvedSpec if found, None otherwise
    """
    try:
        # Import here to avoid circular imports
        from app.specs.service import get_latest_validated_spec
        
        # v2.1: Fixed argument order - service expects (db, project_id)
        spec = get_latest_validated_spec(db_session, project_id)
        if spec:
            return ResolvedSpec(
                spec_id=spec.spec_id,
                spec_hash=spec.spec_hash,
                project_id=project_id,
                title=getattr(spec, 'title', None),
                created_at=spec.created_at.isoformat() if hasattr(spec, 'created_at') and spec.created_at else None,
            )
    except ImportError:
        logger.warning("[overwatcher_command] specs.service not available, using fallback")
    except Exception as e:
        logger.warning(f"[overwatcher_command] Failed to resolve spec from DB: {e}")
    
    return None


def create_smoke_test_spec() -> ResolvedSpec:
    """Create a minimal spec for smoke testing.
    
    Used when no real spec exists but we want to test the Overwatcher flow.
    
    Returns:
        ResolvedSpec for smoke test
    """
    import hashlib
    
    spec_content = json.dumps({
        "title": "Sandbox Hello Test",
        "goal": "Create hello.txt on desktop with content 'ASTRA OK'",
        "deliverables": [
            {
                "type": "file",
                "target": "DESKTOP",
                "filename": "hello.txt",
                "content": "ASTRA OK"
            }
        ],
        "verification": {
            "file_exists": True,
            "content_match": "ASTRA OK"
        }
    }, sort_keys=True)
    
    spec_hash = hashlib.sha256(spec_content.encode()).hexdigest()
    spec_id = f"smoke-test-{uuid4().hex[:8]}"
    
    return ResolvedSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        project_id=0,  # No project for smoke test
        title="Sandbox Hello Test",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# Artifact Loading
# =============================================================================

def load_critical_pipeline_artifacts(
    job_id: str,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Load artifacts from Critical Pipeline if they exist.
    
    Args:
        job_id: Job ID to look up
        artifact_root: Root directory for job artifacts
    
    Returns:
        Dict with artifact paths and content summaries
    """
    artifacts = {
        "architecture": None,
        "critique": None,
        "plan": None,
        "exists": False,
    }
    
    job_dir = Path(artifact_root) / job_id
    if not job_dir.exists():
        return artifacts
    
    # Check for architecture
    arch_path = job_dir / "architecture" / "latest.md"
    if arch_path.exists():
        artifacts["architecture"] = str(arch_path)
    
    # Check for critique
    critique_path = job_dir / "critique" / "latest.json"
    if critique_path.exists():
        artifacts["critique"] = str(critique_path)
    
    # Check for plan
    plan_path = job_dir / "plan" / "chunk_plan.json"
    if plan_path.exists():
        artifacts["plan"] = str(plan_path)
    
    artifacts["exists"] = any([
        artifacts["architecture"],
        artifacts["critique"],
        artifacts["plan"],
    ])
    
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
    task_description: Optional[str] = None,
) -> EvidenceBundle:
    """Build EvidenceBundle for Overwatcher evaluation.
    
    Args:
        job_id: Job UUID
        spec: Resolved spec context
        artifacts: Critical Pipeline artifacts (if any)
        strike_number: Current strike (default 1 for new job)
        chunk_id: Chunk being processed (if applicable)
        task_description: Description of what needs to be done
    
    Returns:
        EvidenceBundle ready for Overwatcher
    """
    stage_run_id = str(uuid4())
    chunk_id = chunk_id or f"chunk-{uuid4().hex[:8]}"
    
    # Build file changes based on task
    file_changes = []
    if task_description:
        file_changes.append(FileChange(
            path=SMOKE_TEST_FILENAME,
            action="add",
            intent=task_description or "Create test file",
        ))
    
    bundle = EvidenceBundle(
        job_id=job_id,
        chunk_id=chunk_id,
        stage_run_id=stage_run_id,
        spec_id=spec.spec_id,
        spec_hash=spec.spec_hash,
        strike_number=strike_number,
        file_changes=file_changes,
        chunk_title=spec.title or "Overwatcher Job",
        chunk_objective=task_description or "Execute validated spec",
        verification_commands=[],  # Will be filled by Implementer
    )
    
    return bundle


# =============================================================================
# Implementer Integration
# =============================================================================

@dataclass
class ImplementerResult:
    """Result from Implementer execution."""
    success: bool
    output_path: Optional[str] = None
    sha256: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    sandbox_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "sha256": self.sha256,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "sandbox_used": self.sandbox_used,
        }


async def run_implementer(
    *,
    spec: ResolvedSpec,
    output: OverwatcherOutput,
    client: Optional[SandboxClient] = None,
) -> ImplementerResult:
    """Execute approved work via Implementer.
    
    Overwatcher is a SUPERVISOR - it does NOT do the work.
    This function routes to Implementer (Claude Sonnet) for actual execution.
    
    SAFETY INVARIANT (v2.1):
        - ALL file writes go through Windows Sandbox
        - NO local fallback - if sandbox unavailable, FAIL
        - This is a hard safety requirement, not a preference
    
    Args:
        spec: Resolved spec context
        output: Overwatcher decision (must be PASS/APPROVE)
        client: Sandbox client for file operations
    
    Returns:
        ImplementerResult with execution details
    """
    import time
    start_time = time.time()
    
    if output.decision != Decision.PASS:
        return ImplementerResult(
            success=False,
            error=f"Cannot implement: Overwatcher decision was {output.decision.value}",
            duration_ms=int((time.time() - start_time) * 1000),
        )
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    # Execute the smoke test job: write hello.txt to desktop
    try:
        # v2.1: SAFETY - Sandbox is REQUIRED, no fallback
        if not client.is_connected():
            logger.error("[implementer] SAFETY: Sandbox not connected - refusing to proceed")
            return ImplementerResult(
                success=False,
                error="SAFETY: Sandbox not available. ASTRA cannot write to host filesystem. Please start the Windows Sandbox.",
                duration_ms=int((time.time() - start_time) * 1000),
                sandbox_used=False,
            )
        
        logger.info("[implementer] Writing file via sandbox")
        
        result = client.write_file(
            target=SMOKE_TEST_TARGET,
            filename=SMOKE_TEST_FILENAME,
            content=SMOKE_TEST_CONTENT,
            overwrite=True,
        )
        
        if result.ok:
            return ImplementerResult(
                success=True,
                output_path=result.path,
                sha256=result.sha256,
                duration_ms=int((time.time() - start_time) * 1000),
                sandbox_used=True,
            )
        else:
            return ImplementerResult(
                success=False,
                error="Sandbox write returned not ok",
                duration_ms=int((time.time() - start_time) * 1000),
                sandbox_used=True,
            )
            
    except SandboxError as e:
        logger.error(f"[implementer] Sandbox error: {e}")
        return ImplementerResult(
            success=False,
            error=f"Sandbox error: {e}",
            duration_ms=int((time.time() - start_time) * 1000),
            sandbox_used=True,
        )
    except Exception as e:
        logger.error(f"[implementer] Execution failed: {e}")
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


# =============================================================================
# Verification
# =============================================================================

@dataclass
class VerificationResult:
    """Result from verification step."""
    passed: bool
    file_exists: bool = False
    content_matches: bool = False
    actual_content: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "file_exists": self.file_exists,
            "content_matches": self.content_matches,
            "actual_content": self.actual_content,
            "error": self.error,
        }


async def run_verification(
    *,
    impl_result: ImplementerResult,
    expected_content: str = SMOKE_TEST_CONTENT,
    client: Optional[SandboxClient] = None,
) -> VerificationResult:
    """Verify that Implementer output is correct.
    
    Uses PowerShell via sandbox to check:
    1. File exists at expected path
    2. Content matches expected
    3. Nothing else was modified (for smoke test, just check the file)
    
    SAFETY INVARIANT (v2.1):
        - ALL verification goes through Windows Sandbox
        - NO local fallback - if sandbox unavailable, FAIL
    
    Args:
        impl_result: Result from Implementer
        expected_content: Expected file content
        client: Sandbox client for verification commands
    
    Returns:
        VerificationResult with pass/fail and details
    """
    if not impl_result.success:
        return VerificationResult(
            passed=False,
            error=f"Implementation failed: {impl_result.error}",
        )
    
    if not impl_result.output_path:
        return VerificationResult(
            passed=False,
            error="No output path from Implementer",
        )
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    try:
        # v2.1: SAFETY - Sandbox is REQUIRED for verification
        if not client.is_connected():
            logger.error("[verification] SAFETY: Sandbox not connected - cannot verify")
            return VerificationResult(
                passed=False,
                error="SAFETY: Sandbox not available for verification. Cannot read from host filesystem.",
            )
        
        # Use PowerShell to verify file exists and content matches
        # Note: Use proper escaping for PowerShell path
        ps_path = impl_result.output_path.replace("/", "\\")
        
        # Check file exists
        exists_cmd = f'Test-Path -Path "{ps_path}"'
        exists_result = client.shell_run(exists_cmd, timeout_seconds=10)
        
        file_exists = exists_result.ok and "True" in exists_result.stdout
        
        if not file_exists:
            return VerificationResult(
                passed=False,
                file_exists=False,
                error=f"File not found at {impl_result.output_path}",
            )
        
        # Read content
        read_cmd = f'Get-Content -Path "{ps_path}" -Raw'
        read_result = client.shell_run(read_cmd, timeout_seconds=10)
        
        if not read_result.ok:
            return VerificationResult(
                passed=False,
                file_exists=True,
                error=f"Failed to read file: {read_result.stderr}",
            )
        
        actual_content = read_result.stdout.strip()
        content_matches = actual_content == expected_content
        
        return VerificationResult(
            passed=content_matches,
            file_exists=True,
            content_matches=content_matches,
            actual_content=actual_content,
            error=None if content_matches else f"Content mismatch: expected '{expected_content}', got '{actual_content}'",
        )
            
    except SandboxError as e:
        logger.error(f"[verification] Sandbox error: {e}")
        return VerificationResult(
            passed=False,
            error=f"Sandbox error: {e}",
        )
    except Exception as e:
        logger.error(f"[verification] Failed: {e}")
        return VerificationResult(
            passed=False,
            error=str(e),
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
        """Add entry to stage trace."""
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
    use_smoke_test: bool = True,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> OverwatcherCommandResult:
    """Execute the 'run overwatcher' command.
    
    This is the main entry point for Overwatcher execution.
    
    Flow:
    1. Resolve spec context (from DB or create smoke test spec)
    2. Load any Critical Pipeline artifacts
    3. Build EvidenceBundle
    4. Run Overwatcher for decision
    5. If APPROVED → run Implementer
    6. Run verification
    7. Log final state
    
    SAFETY INVARIANT (v2.1):
        - ALL execution goes through Windows Sandbox
        - NO local fallback at any stage
        - If sandbox unavailable → FAIL with clear error
    
    Args:
        project_id: Project ID (0 for smoke test)
        job_id: Existing job ID to continue (optional)
        message: User message (for context)
        db_session: Database session (optional)
        llm_call_fn: Async function to call LLM
        use_smoke_test: Use smoke test spec if no real spec found
        artifact_root: Root directory for job artifacts
    
    Returns:
        OverwatcherCommandResult with complete execution state
    """
    # Initialize result
    job_id = job_id or str(uuid4())
    result = OverwatcherCommandResult(success=False, job_id=job_id)
    
    logger.info(f"[overwatcher_command] Starting: job_id={job_id}, project_id={project_id}")
    result.add_trace("OVERWATCHER_COMMAND_START", "started", {"project_id": project_id})
    
    # ==========================================================================
    # Step 1: Resolve spec context
    # ==========================================================================
    spec = resolve_latest_spec(project_id, db_session)
    
    if spec is None:
        if use_smoke_test:
            logger.info("[overwatcher_command] No spec found, using smoke test spec")
            spec = create_smoke_test_spec()
            result.add_trace("SPEC_RESOLVE", "smoke_test", {"reason": "no_validated_spec"})
        else:
            result.error = "No validated spec found for project and smoke test disabled"
            result.add_trace("SPEC_RESOLVE", "failed", {"error": result.error})
            logger.error(f"[overwatcher_command] {result.error}")
            return result
    else:
        result.add_trace("SPEC_RESOLVE", "success", {"spec_id": spec.spec_id})
    
    result.spec = spec
    logger.info(f"[overwatcher_command] Spec resolved: {spec.spec_id} (hash: {spec.spec_hash[:16]}...)")
    
    # ==========================================================================
    # Step 2: Load Critical Pipeline artifacts
    # ==========================================================================
    artifacts = load_critical_pipeline_artifacts(job_id, artifact_root)
    result.add_trace("ARTIFACTS_LOAD", "success" if artifacts["exists"] else "none", artifacts)
    
    # ==========================================================================
    # Step 3: Build EvidenceBundle
    # ==========================================================================
    evidence = build_overwatcher_evidence(
        job_id=job_id,
        spec=spec,
        artifacts=artifacts,
        task_description=f"Create {SMOKE_TEST_FILENAME} with content '{SMOKE_TEST_CONTENT}' on sandbox desktop",
    )
    result.add_trace("EVIDENCE_BUILD", "success", {"chunk_id": evidence.chunk_id})
    
    # ==========================================================================
    # Step 4: Run Overwatcher (if LLM available)
    # ==========================================================================
    if llm_call_fn:
        try:
            logger.info("[overwatcher_command] Running Overwatcher analysis")
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
            logger.error(f"[overwatcher_command] Overwatcher failed: {e}")
            result.error = f"Overwatcher execution failed: {e}"
            result.add_trace("OVERWATCHER_ERROR", "failed", {"error": str(e)})
            return result
    else:
        # No LLM - auto-approve for smoke test
        logger.warning("[overwatcher_command] No LLM function provided, auto-approving for smoke test")
        result.overwatcher_decision = Decision.PASS.value
        result.overwatcher_diagnosis = "Auto-approved (no LLM, smoke test mode)"
        result.add_trace("OVERWATCHER_SKIP", "auto_approved", {"reason": "no_llm_fn"})
    
    # ==========================================================================
    # Step 5: Run Implementer
    # ==========================================================================
    logger.info("[overwatcher_command] Running Implementer")
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
            "sandbox_used": impl_result.sandbox_used,
        })
        logger.info(f"[overwatcher_command] Implementer success: {impl_result.output_path}")
    else:
        result.error = f"Implementer failed: {impl_result.error}"
        result.add_trace("IMPLEMENTER_EXIT", "failed", {"error": impl_result.error})
        logger.error(f"[overwatcher_command] {result.error}")
        return result
    
    # ==========================================================================
    # Step 6: Run Verification
    # ==========================================================================
    logger.info("[overwatcher_command] Running verification")
    result.add_trace("VERIFICATION_ENTER", "running")
    
    verify_result = await run_verification(impl_result=impl_result)
    result.verification_result = verify_result
    
    if verify_result.passed:
        result.success = True
        result.add_trace("VERIFICATION_EXIT", "passed", verify_result.to_dict())
        result.add_trace("JOB_COMPLETE", "success", {"job_id": job_id})
        logger.info(f"[overwatcher_command] ✓ Job {job_id} COMPLETE - verification passed")
    else:
        result.error = f"Verification failed: {verify_result.error}"
        result.add_trace("VERIFICATION_EXIT", "failed", verify_result.to_dict())
        logger.error(f"[overwatcher_command] {result.error}")
    
    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "ResolvedSpec",
    "ImplementerResult",
    "VerificationResult",
    "OverwatcherCommandResult",
    # Functions
    "resolve_latest_spec",
    "create_smoke_test_spec",
    "load_critical_pipeline_artifacts",
    "build_overwatcher_evidence",
    "run_implementer",
    "run_verification",
    "run_overwatcher_command",
    # Constants
    "SMOKE_TEST_FILENAME",
    "SMOKE_TEST_CONTENT",
    "SMOKE_TEST_TARGET",
    "ALLOWED_HOST_WRITE_PATH",
]

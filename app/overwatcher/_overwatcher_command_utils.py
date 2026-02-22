import logging
import os
from .implementer import ImplementerResult, VerificationResult
from .spec_resolution import ResolvedSpec
from app.overwatcher.evidence import EvidenceBundle, FileChange
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
DEFAULT_ARTIFACT_ROOT = os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs")


ALLOWED_HOST_WRITE_PATH = Path("D:/Tools/zobie_mapper/out")

class SpecParseError(Exception):
    """Raised when spec content cannot be parsed to extract deliverable."""
    pass

class FileExistenceError(Exception):
    """Raised when a file that should exist doesn't, or vice versa."""
    pass

def load_critical_pipeline_artifacts(
    job_id: str,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Load artifacts from Critical Pipeline if they exist.
    
    v5.2: Searches multiple directory layouts:
    - {root}/{job_id}/architecture/latest.md (original layout)
    - {root}/{job_id}/arch/arch_v1.md (actual CP output layout)
    - {root}/jobs/{job_id}/arch/arch_v1.md (nested jobs/ layout)
    """
    artifacts: Dict[str, Any] = {
        "architecture": None,
        "critique": None,
        "plan": None,
        "exists": False,
    }

    # v5.2: Try multiple directory layouts for job artifacts
    candidate_dirs = [
        Path(artifact_root) / job_id,
        Path(artifact_root) / "jobs" / job_id,
    ]
    
    job_dir = None
    for candidate in candidate_dirs:
        if candidate.exists():
            job_dir = candidate
            break
    
    if job_dir is None:
        logger.debug("[artifact_load] No job directory found for %s in %s", job_id, artifact_root)
        return artifacts

    for name, paths in [
        ("architecture", [
            "architecture/latest.md",
            "arch/arch_v3.md",
            "arch/arch_v2.md",
            "arch/arch_v1.md",
            "arch_v1.md",
        ]),
        ("critique", [
            "critique/latest.json",
            "critique/critique_v1.json",
            "critique_v1.json",
        ]),
        ("plan", ["plan/chunk_plan.json"]),
    ]:
        for rel_path in paths:
            candidate = job_dir / rel_path
            if candidate.exists():
                artifacts[name] = str(candidate)
                logger.debug("[artifact_load] Found %s at %s", name, candidate)
                break

    artifacts["exists"] = any(
        [artifacts["architecture"], artifacts["critique"], artifacts["plan"]]
    )
    return artifacts

def _find_architecture_for_spec(
    spec_id: str,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> Optional[Dict[str, Any]]:
    """Find the Critical Pipeline architecture document for a given spec.
    
    v5.2: The Critical Pipeline stores architecture docs in:
        {artifact_root}/jobs/cp-{hash}/arch/arch_v{N}.md
    
    Since the Overwatcher doesn't receive the CP job_id directly, we scan
    recent CP job directories and check if their architecture document
    references the spec_id.
    
    Strategy:
    1. Scan for cp-* directories in the jobs folder
    2. Sort by modification time (most recent first)
    3. Check if the architecture doc references the spec_id
    4. Return the first match
    """
    jobs_dir = Path(artifact_root) / "jobs"
    if not jobs_dir.exists():
        # Also try flat layout
        jobs_dir = Path(artifact_root)
    
    # Find all cp-* directories
    cp_dirs = []
    try:
        for entry in jobs_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("cp-"):
                cp_dirs.append(entry)
    except OSError as e:
        logger.warning("[arch_find] Failed to scan %s: %s", jobs_dir, e)
        return None
    
    if not cp_dirs:
        logger.info("[arch_find] No cp-* directories found in %s", jobs_dir)
        return None
    
    # Sort by modification time, most recent first
    cp_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    logger.info("[arch_find] Scanning %d cp-* dirs for spec %s", len(cp_dirs), spec_id)
    
    # Check each for an architecture document referencing our spec
    for cp_dir in cp_dirs[:20]:  # Limit scan to 20 most recent
        # Find the latest arch file
        arch_dir = cp_dir / "arch"
        if not arch_dir.exists():
            continue
        
        # Get the highest-version arch file
        arch_files = sorted(arch_dir.glob("arch_v*.md"), reverse=True)
        if not arch_files:
            continue
        
        arch_path = arch_files[0]  # Latest version
        
        # Quick check: does the architecture reference our spec_id?
        try:
            # Read just the first 2000 chars for spec_id check
            with open(arch_path, 'r', encoding='utf-8') as f:
                header = f.read(2000)
            
            if spec_id in header:
                logger.info(
                    "[arch_find] \u2713 Match: %s references spec %s",
                    arch_path, spec_id,
                )
                return load_critical_pipeline_artifacts(
                    job_id=cp_dir.name,
                    artifact_root=str(jobs_dir.parent) if jobs_dir.name == "jobs" else artifact_root,
                )
        except Exception as e:
            logger.debug("[arch_find] Failed to read %s: %s", arch_path, e)
            continue
    
    # Fallback: return the most recent CP's architecture if we have one
    # (in case the spec_id wasn't in the header)
    for cp_dir in cp_dirs[:5]:
        result = load_critical_pipeline_artifacts(
            job_id=cp_dir.name,
            artifact_root=str(jobs_dir.parent) if jobs_dir.name == "jobs" else artifact_root,
        )
        if result.get("architecture"):
            logger.info(
                "[arch_find] Fallback: using most recent architecture from %s",
                cp_dir.name,
            )
            return result
    
    logger.warning("[arch_find] No architecture document found for spec %s", spec_id)
    return None

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

    logger.info(
        "[build_evidence] File: %s, Action: %s, Content: %s...",
        filename,
        action,
        (content[:50] if content else ""),
    )

    file_changes = [
        FileChange(
            path=filename,
            action=action,
            intent=description,
        )
    ]

    # NOTE: this is deliberately minimal – Overwatcher may ask for more info
    # via NEEDS_INFO. For now, we handle NEEDS_INFO upstream in the command.
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
    # For UI / streaming layer: list of artifact paths created/updated
    artifacts_written: List[str] = field(default_factory=list)

    # --- Compatibility aliases for older callers ---

    # Some callers expect `result.decision`
    @property
    def decision(self) -> Optional[str]:
        return self.overwatcher_decision

    @decision.setter
    def decision(self, value: Optional[str]) -> None:
        self.overwatcher_decision = value

    # Some callers expect `result.reason` (typically for error text)
    @property
    def reason(self) -> Optional[str]:
        return self.error

    @reason.setter
    def reason(self, value: Optional[str]) -> None:
        self.error = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "spec": self.spec.to_dict() if self.spec else None,
            "overwatcher_decision": self.overwatcher_decision,
            "overwatcher_diagnosis": self.overwatcher_diagnosis,
            "implementer_result": self.implementer_result.to_dict()
            if self.implementer_result
            else None,
            "verification_result": self.verification_result.to_dict()
            if self.verification_result
            else None,
            "error": self.error,
            "reason": self.reason,
            "decision": self.decision,
            "stage_trace": self.stage_trace,
            "artifacts_written": self.artifacts_written,
        }

    def add_trace(self, stage: str, status: str, details: Optional[Dict] = None):
        self.stage_trace.append(
            {
                "stage": stage,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": details or {},
            }
        )
        # Emit to Build Journal (fire-and-forget, never crashes pipeline)
        try:
            from app.experience.journal_writer import emit_from_trace
            _job_dir = os.path.join(DEFAULT_ARTIFACT_ROOT, "jobs", self.job_id)
            emit_from_trace(
                job_id=self.job_id,
                job_dir=_job_dir,
                trace_stage=stage,
                trace_status=status,
                trace_details=details,
            )
        except Exception:
            pass  # Journal must never crash the pipeline

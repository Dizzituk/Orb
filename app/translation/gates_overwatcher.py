# FILE: app/translation/gates_overwatcher.py
# Purpose: Overwatcher gate (validated spec + completed pipeline checks) — split from gates.py.
# Called-by: app.translation.gates
# Depends-on: app.translation.schemas
# Last-renovated: 2026-06-21
"""Overwatcher-specific gating: validated spec exists + Critical Pipeline completed (v1.2)."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from .schemas import ContextGateResult

logger = logging.getLogger(__name__)


# =============================================================================
# OVERWATCHER GATE (v1.2)
# =============================================================================
# Overwatcher-specific gating that checks:
# - REQUIRED: validated spec exists (spec_id + spec_hash resolvable)
# - REQUIRED: Critical Pipeline completed for that spec
# - NOT REQUIRED: change_set_id (Overwatcher derives internally)
# - NOT REQUIRED: zero blocking issues (Overwatcher evaluates these)
# =============================================================================

def check_overwatcher_gate(
    project_id: int,
    db_session: Any = None,
) -> ContextGateResult:
    """
    Check Overwatcher-specific requirements.
    
    Unlike the generic context gate, this performs actual DB lookups to verify:
    1. A validated spec exists for this project
    2. Critical Pipeline has completed for that spec
    
    Args:
        project_id: The project ID to check
        db_session: SQLAlchemy session (optional - will create if needed)
        
    Returns:
        ContextGateResult indicating if Overwatcher can run
    """
    missing = []
    provided = {"project_id": project_id}
    
    # Check project_id
    if not project_id:
        return ContextGateResult(
            passed=False,
            gate_name="overwatcher",
            reason="Missing required context: project_id",
            missing_context=["project_id"],
            provided_context={},
        )
    
    # Try to resolve validated spec from DB
    spec_info = _resolve_validated_spec(project_id, db_session)
    
    if spec_info is None:
        return ContextGateResult(
            passed=False,
            gate_name="overwatcher",
            reason="No validated spec found for this project. Run Spec Gate first.",
            missing_context=["validated_spec"],
            provided_context=provided,
        )
    
    provided["spec_id"] = spec_info.get("spec_id")
    provided["spec_hash"] = spec_info.get("spec_hash")
    
    # Check if Critical Pipeline completed for this spec
    pipeline_completed = _check_critical_pipeline_completed(
        project_id, 
        spec_info.get("spec_id"),
        db_session
    )
    
    if not pipeline_completed:
        return ContextGateResult(
            passed=False,
            gate_name="overwatcher",
            reason=f"Critical Pipeline has not completed for spec {spec_info.get('spec_id')}. Run 'Astra, command: run critical pipeline' first.",
            missing_context=["critical_pipeline_completion"],
            provided_context=provided,
        )
    
    provided["pipeline_completed"] = True
    
    # All checks passed
    logger.info(f"[overwatcher_gate] PASSED: project={project_id}, spec={spec_info.get('spec_id')}")
    
    return ContextGateResult(
        passed=True,
        gate_name="overwatcher",
        reason="Overwatcher gate passed: validated spec and completed pipeline found",
        missing_context=[],
        provided_context=provided,
    )


def _resolve_validated_spec(
    project_id: int,
    db_session: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve the latest validated spec for a project.
    
    Returns:
        Dict with spec_id, spec_hash, spec_version if found
        None if no validated spec exists
    """
    try:
        # Try to import spec service
        from app.spec_gate.spec_persistence import get_latest_validated_spec
        
        if db_session is None:
            # Create session if not provided
            from app.database import SessionLocal
            db_session = SessionLocal()
            close_session = True
        else:
            close_session = False
        
        try:
            spec = get_latest_validated_spec(db_session, project_id)
            if spec:
                return {
                    "spec_id": spec.spec_id,
                    "spec_hash": spec.spec_hash,
                    "spec_version": spec.version,
                    "status": spec.status,
                }
            return None
        finally:
            if close_session:
                db_session.close()
                
    except ImportError as e:
        logger.warning(f"[overwatcher_gate] Could not import spec_persistence: {e}")
        # Fallback: try to find spec in jobs directory
        return _resolve_spec_from_jobs(project_id)
    except Exception as e:
        logger.error(f"[overwatcher_gate] Error resolving spec: {e}")
        return None


def _resolve_spec_from_jobs(project_id: int) -> Optional[Dict[str, Any]]:
    """
    Fallback: resolve spec from jobs directory if DB unavailable.
    Looks for the most recent spec_v*.json in the project's job folders.
    """
    import os
    import json
    from pathlib import Path
    
    jobs_root = Path(os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs/jobs"))
    
    if not jobs_root.exists():
        return None
    
    # Find spec files
    latest_spec = None
    latest_time = 0
    
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        
        spec_dir = job_dir / "spec"
        if not spec_dir.exists():
            continue
        
        for spec_file in spec_dir.glob("spec_v*.json"):
            try:
                mtime = spec_file.stat().st_mtime
                if mtime > latest_time:
                    with open(spec_file, "r", encoding="utf-8") as f:
                        spec_data = json.load(f)
                    
                    # Check if validated
                    if spec_data.get("status") == "validated":
                        latest_spec = {
                            "spec_id": spec_data.get("spec_id"),
                            "spec_hash": spec_data.get("spec_hash"),
                            "spec_version": spec_data.get("version", 1),
                            "status": "validated",
                        }
                        latest_time = mtime
            except Exception as e:
                logger.debug(f"[overwatcher_gate] Could not read {spec_file}: {e}")
                continue
    
    return latest_spec


def _check_critical_pipeline_completed(
    project_id: int,
    spec_id: str,
    db_session: Any = None,
) -> bool:
    """
    Check if Critical Pipeline has completed for the given spec.
    
    Returns:
        True if pipeline completed, False otherwise
    """
    try:
        # Try DB lookup first
        from app.jobs.service import get_completed_pipeline_for_spec
        
        if db_session is None:
            from app.database import SessionLocal
            db_session = SessionLocal()
            close_session = True
        else:
            close_session = False
        
        try:
            pipeline = get_completed_pipeline_for_spec(db_session, spec_id)
            return pipeline is not None
        finally:
            if close_session:
                db_session.close()
                
    except ImportError:
        logger.debug("[overwatcher_gate] get_completed_pipeline_for_spec not available, using fallback")
        # Fallback: check jobs directory for architecture artifacts
        return _check_pipeline_from_jobs(spec_id)
    except Exception as e:
        logger.error(f"[overwatcher_gate] Error checking pipeline: {e}")
        # Fallback to jobs directory check
        return _check_pipeline_from_jobs(spec_id)


def _check_pipeline_from_jobs(spec_id: str) -> bool:
    """
    Fallback: check if pipeline completed by looking for architecture artifacts.
    """
    import os
    import json
    from pathlib import Path
    
    jobs_root = Path(os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs/jobs"))
    
    if not jobs_root.exists():
        return False
    
    # Look for cp-* (critical pipeline) job directories
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        if not job_dir.name.startswith("cp-"):
            continue
        
        # Check for architecture artifact
        arch_dir = job_dir / "arch"
        if not arch_dir.exists():
            continue
        
        # Check if any arch file references our spec_id
        for arch_file in arch_dir.glob("arch_v*.md"):
            try:
                content = arch_file.read_text(encoding="utf-8", errors="ignore")
                if spec_id in content:
                    logger.debug(f"[overwatcher_gate] Found pipeline completion in {arch_file}")
                    return True
            except Exception:
                continue
        
        # Also check arch_v*.json
        for arch_file in arch_dir.glob("arch_v*.json"):
            try:
                with open(arch_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("spec_id") == spec_id:
                    logger.debug(f"[overwatcher_gate] Found pipeline completion in {arch_file}")
                    return True
            except Exception:
                continue
    
    return False

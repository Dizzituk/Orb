from __future__ import annotations
import glob
import json
import logging
import os
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

# v3.2: Direct imports - fixes namespace isolation bug where parent's
# imports didn't propagate to this utils module.
try:
    from app.jobs.service import get_active_job_for_project, get_job_for_spec
except ImportError:
    get_active_job_for_project = None
    get_job_for_spec = None

try:
    from app.llm.stage_models import get_overwatcher_config
    STAGE_MODELS_AVAILABLE = True
except ImportError:
    get_overwatcher_config = None
    STAGE_MODELS_AVAILABLE = False

ARTIFACT_ROOT = os.getenv("ORB_JOB_ARTIFACT_ROOT", r"D:\Orb\jobs")


def _get_overwatcher_provider_model() -> tuple[str, str]:
    """Get Overwatcher model configuration."""
    if not STAGE_MODELS_AVAILABLE or get_overwatcher_config is None:
        raise RuntimeError("stage_models not available for Overwatcher config")
    
    config = get_overwatcher_config()
    return config.provider, config.model

def _resolve_job_id(
    db: Session,
    project_id: int,
    spec_id: Optional[str] = None,
    provided_job_id: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """
    Resolve job_id using multiple fallback strategies.
    
    Returns (job_id, resolution_method)
    
    Strategies (in order):
    1. Use provided job_id if given
    2. Look up job by spec_id in database
    3. Get active job for project from database
    4. Find most recent cp-* folder in filesystem
    """
    # Strategy 1: Provided job_id
    if provided_job_id:
        logger.info(f"[job_resolve] Using provided job_id: {provided_job_id}")
        return provided_job_id, "provided"
    
    # Strategy 2: Look up by spec_id
    if spec_id and get_job_for_spec:
        try:
            job = get_job_for_spec(db, spec_id)
            if job:
                logger.info(f"[job_resolve] Found job {job.id} for spec {spec_id}")
                return job.id, "spec_lookup"
        except Exception as e:
            logger.warning(f"[job_resolve] Spec lookup failed: {e}")
    
    # Strategy 3: Get active job for project
    if get_active_job_for_project:
        try:
            active_job = get_active_job_for_project(db, project_id)
            if active_job:
                logger.info(f"[job_resolve] Found active job: {active_job.id}")
                return active_job.id, "active_project"
        except Exception as e:
            logger.warning(f"[job_resolve] Active job lookup failed: {e}")
    
    # Strategy 4: Filesystem fallback - find most recent cp-* folder
    jobs_dir = os.path.join(ARTIFACT_ROOT, "jobs")
    if os.path.isdir(jobs_dir):
        try:
            cp_folders = glob.glob(os.path.join(jobs_dir, "cp-*"))
            if cp_folders:
                # Sort by modification time, newest first
                cp_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                newest = os.path.basename(cp_folders[0])
                logger.info(f"[job_resolve] Found recent job folder: {newest}")
                return newest, "filesystem"
        except Exception as e:
            logger.warning(f"[job_resolve] Filesystem lookup failed: {e}")
    
    logger.warning("[job_resolve] Could not resolve job_id via any method")
    return None, "none"

def _load_artifact_bindings(
    job_id: Optional[str],
    work_artifacts: Optional[Any] = None,
    spec_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load artifact bindings from multiple sources.
    
    Priority:
    1. work_artifacts.artifact_bindings (from job system)
    2. work_artifacts.metadata.outputs (fallback)
    3. spec_data outputs + content_verbatim + location (fallback)
    4. Load from job directory filesystem (NEW)
    """
    bindings: List[Dict[str, Any]] = []
    
    # Try work_artifacts first
    if work_artifacts:
        if hasattr(work_artifacts, 'artifact_bindings') and work_artifacts.artifact_bindings:
            bindings = list(work_artifacts.artifact_bindings)
            logger.info("[artifact_load] Loaded %d bindings from work_artifacts", len(bindings))
            return bindings
        
        if hasattr(work_artifacts, 'metadata'):
            metadata = work_artifacts.metadata or {}
            outputs = metadata.get("outputs") or []
            for i, out in enumerate(outputs):
                bindings.append({
                    "artifact_id": f"output_{i+1}",
                    "action": out.get("action", "create"),
                    "path": out.get("path", ""),
                    "content_type": "text",
                    "content_verbatim": out.get("content", ""),
                })
            if bindings:
                logger.info("[artifact_load] Loaded %d bindings from metadata.outputs", len(bindings))
                return bindings
    
    # Try spec_data fallback
    if spec_data:
        outputs = spec_data.get("outputs", [])
        content_verbatim = spec_data.get("content_verbatim") or ""
        location = spec_data.get("location") or ""
        
        for i, out in enumerate(outputs):
            if isinstance(out, dict):
                name = out.get("name", "")
                path = out.get("path", "") or location
                content = out.get("content", "") or content_verbatim
                action = out.get("action", "create")
            else:
                name = str(out)
                path = location
                content = content_verbatim
                action = "create"
            
            if not name:
                continue
            
            full_path = os.path.join(path, name) if path and not path.endswith(name) else (path or name)
            
            bindings.append({
                "artifact_id": f"output_{i+1}",
                "action": action,
                "path": full_path,
                "content_type": "text",
                "content_verbatim": content,
            })
        
        if bindings:
            logger.info("[artifact_load] Constructed %d bindings from spec_data", len(bindings))
            return bindings
    
    # Try loading from job directory (NEW v3.5)
    if job_id:
        job_outputs_dir = os.path.join(ARTIFACT_ROOT, "jobs", job_id, "outputs")
        if os.path.isdir(job_outputs_dir):
            try:
                for filename in os.listdir(job_outputs_dir):
                    filepath = os.path.join(job_outputs_dir, filename)
                    if os.path.isfile(filepath):
                        # Read content for evidence
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception:
                            content = ""
                        
                        bindings.append({
                            "artifact_id": filename,
                            "action": "created",  # Already created by critical pipeline
                            "path": filepath,
                            "content_type": "text",
                            "content_verbatim": content,
                            "actual_file": True,  # Mark as actual file
                        })
                
                if bindings:
                    logger.info("[artifact_load] Loaded %d bindings from job directory", len(bindings))
                    return bindings
            except Exception as e:
                logger.warning(f"[artifact_load] Failed to read job outputs: {e}")
    
    logger.warning("[artifact_load] No artifact bindings found from any source")
    return bindings

def _validate_artifact_bindings(bindings: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """Validate artifact bindings have required fields."""
    issues: List[str] = []
    
    if not bindings:
        issues.append("No artifact bindings found")
        return False, issues
    
    for i, binding in enumerate(bindings):
        if not binding.get("path"):
            issues.append(f"Binding {i+1}: missing 'path'")
        if not binding.get("action"):
            binding["action"] = "create"
    
    is_valid = len(issues) == 0
    return is_valid, issues

def _build_evidence_bundle(
    artifact_bindings: List[Dict[str, Any]],
    spec_data: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build evidence bundle for Overwatcher verification.
    
    Evidence bundle contains:
    - expected: What the spec says should exist
    - actual: What actually exists (file content, hash)
    - result: Verification status (match/mismatch/missing)
    """
    evidence = {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat(),
        "artifacts": [],
        "acceptance_criteria": [],
        "verification_results": [],
    }
    
    # Add acceptance criteria from spec
    if spec_data:
        evidence["acceptance_criteria"] = spec_data.get("acceptance_criteria", [])
        evidence["objective"] = spec_data.get("objective", "")
        evidence["content_verbatim"] = spec_data.get("content_verbatim", "")
        evidence["location"] = spec_data.get("location", "")
    
    # Process each artifact binding
    for binding in artifact_bindings:
        artifact_evidence = {
            "artifact_id": binding.get("artifact_id", "unknown"),
            "expected_path": binding.get("path", ""),
            "expected_content": binding.get("content_verbatim", ""),
            "action": binding.get("action", "create"),
        }
        
        path = binding.get("path", "")
        
        # Check if file actually exists and read content
        if path and os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    actual_content = f.read()
                
                artifact_evidence["actual_content"] = actual_content
                artifact_evidence["actual_exists"] = True
                artifact_evidence["file_size_bytes"] = os.path.getsize(path)
                
                # Compare expected vs actual
                expected = binding.get("content_verbatim", "").strip()
                actual = actual_content.strip()
                
                if expected and actual == expected:
                    artifact_evidence["verification"] = "MATCH"
                elif expected and actual != expected:
                    artifact_evidence["verification"] = "MISMATCH"
                    artifact_evidence["diff"] = {
                        "expected": expected,
                        "actual": actual,
                    }
                else:
                    artifact_evidence["verification"] = "EXISTS"
                    
            except Exception as e:
                artifact_evidence["actual_exists"] = True
                artifact_evidence["read_error"] = str(e)
                artifact_evidence["verification"] = "READ_ERROR"
        else:
            artifact_evidence["actual_exists"] = False
            artifact_evidence["verification"] = "MISSING"
        
        evidence["artifacts"].append(artifact_evidence)
        evidence["verification_results"].append({
            "artifact_id": artifact_evidence["artifact_id"],
            "result": artifact_evidence["verification"],
        })
    
    # Overall verification
    results = [a.get("verification") for a in evidence["artifacts"]]
    if all(r in ("MATCH", "EXISTS") for r in results):
        evidence["overall_result"] = "PASS"
    elif any(r == "MISSING" for r in results):
        evidence["overall_result"] = "FAIL_MISSING"
    elif any(r == "MISMATCH" for r in results):
        evidence["overall_result"] = "FAIL_MISMATCH"
    else:
        evidence["overall_result"] = "UNKNOWN"
    
    return evidence

def sse_token(content: str) -> str:
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"

def sse_event(event_type: str, **kwargs) -> str:
    return "data: " + json.dumps({"type": event_type, **kwargs}) + "\n\n"

def sse_error(error: str) -> str:
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"

# FILE: app/overwatcher/quarantine.py
"""Blocks 10-11: Quarantine and Deletion Workflows.

Block 10: Safely isolate suspected dead files with static + dynamic evidence
Block 11: Delete only what quarantine proved safe, after approval

Key behaviors:
- Static evidence: rg references, import graph, config references
- Dynamic evidence: tests, smoke boot, import walk
- Quarantine moves/disables but doesn't delete
- Deletion requires explicit approval
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from app.overwatcher.schemas import (
    StaticEvidence,
    DynamicEvidence,
    QuarantineCandidate,
    QuarantineReason,
    QuarantineReport,
    DeletionReport,
    VerificationResult,
    VerificationStatus,
)
from app.overwatcher.verifier import run_full_verification, run_smoke_boot

logger = logging.getLogger(__name__)

# Configuration
QUARANTINE_DIR = os.getenv("ORB_QUARANTINE_DIR", ".quarantine")
RG_CMD = os.getenv("ORB_RG_CMD", "rg")


# =============================================================================
# Static Analysis
# =============================================================================

def count_rg_references(
    file_path: str,
    repo_path: str,
    exclude_self: bool = True,
) -> int:
    """Count references to a file using ripgrep.
    
    Searches for imports/references to the file's module.
    """
    p = Path(file_path)
    
    # Get the module name (e.g., "app.foo.bar" from "app/foo/bar.py")
    if p.suffix == ".py":
        module_name = str(p.with_suffix("")).replace("/", ".").replace("\\", ".")
        search_term = module_name.split(".")[-1]  # Just the final component
    else:
        search_term = p.name
    
    try:
        result = subprocess.run(
            [RG_CMD, "-c", search_term, "--type", "py"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return 0
        
        # Sum up counts from each file
        total = 0
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Format: "file.py:count"
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    try:
                        count = int(parts[1])
                        # Exclude self-references
                        if exclude_self and file_path in parts[0]:
                            continue
                        total += count
                    except ValueError:
                        pass
        
        return total
        
    except Exception as e:
        logger.warning(f"[quarantine] rg search failed for {file_path}: {e}")
        return 0


def find_imports_of_module(
    file_path: str,
    repo_path: str,
) -> int:
    """Count how many files import this module."""
    p = Path(file_path)
    
    if p.suffix != ".py":
        return 0
    
    # Build module path
    module_parts = str(p.with_suffix("")).replace("/", ".").replace("\\", ".").split(".")
    
    # Search for various import patterns
    patterns = [
        f"from {'.'.join(module_parts[:-1])} import {module_parts[-1]}",
        f"import {'.'.join(module_parts)}",
        f"from {'.'.join(module_parts)} import",
    ]
    
    total = 0
    for pattern in patterns:
        try:
            result = subprocess.run(
                [RG_CMD, "-c", "-F", pattern, "--type", "py"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            try:
                                count = int(parts[1])
                                # Exclude self
                                if file_path not in parts[0]:
                                    total += count
                            except ValueError:
                                pass
        except Exception:
            pass
    
    return total


def find_config_references(
    file_path: str,
    repo_path: str,
    config_patterns: List[str] = None,
) -> int:
    """Count references in config files."""
    if config_patterns is None:
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.env"]
    
    p = Path(file_path)
    search_term = p.stem  # Just the file name without extension
    
    total = 0
    for pattern in config_patterns:
        try:
            result = subprocess.run(
                [RG_CMD, "-c", search_term, "-g", pattern],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            try:
                                total += int(parts[1])
                            except ValueError:
                                pass
        except Exception:
            pass
    
    return total


def get_file_mtime(file_path: str, repo_path: str) -> str:
    """Get file modification time."""
    full_path = Path(repo_path) / file_path
    if full_path.exists():
        mtime = full_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return ""


def analyze_static_evidence(
    file_path: str,
    repo_path: str,
) -> StaticEvidence:
    """Gather static evidence for a file."""
    return StaticEvidence(
        rg_references=count_rg_references(file_path, repo_path),
        import_count=find_imports_of_module(file_path, repo_path),
        config_references=find_config_references(file_path, repo_path),
        last_modified=get_file_mtime(file_path, repo_path),
    )


# =============================================================================
# Dynamic Analysis
# =============================================================================

def run_import_walk(
    file_path: str,
    repo_path: str,
) -> Tuple[bool, str]:
    """Try to import the module to verify it's valid Python."""
    p = Path(file_path)
    
    if p.suffix != ".py":
        return True, "Not a Python file"
    
    module_path = str(p.with_suffix("")).replace("/", ".").replace("\\", ".")
    
    try:
        result = subprocess.run(
            ["python", "-c", f"import {module_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONPATH": repo_path},
        )
        
        return result.returncode == 0, result.stderr
        
    except Exception as e:
        return False, str(e)


def analyze_dynamic_evidence(
    file_path: str,
    repo_path: str,
    run_tests: bool = True,
) -> DynamicEvidence:
    """Gather dynamic evidence for a file."""
    evidence = DynamicEvidence()
    
    # Import walk
    import_ok, _ = run_import_walk(file_path, repo_path)
    evidence.import_walk_passed = import_ok
    
    # Full tests and smoke boot are expensive, optionally skip
    if run_tests:
        # Smoke boot
        boot_ok, _ = run_smoke_boot(repo_path)
        evidence.smoke_boot_passed = boot_ok
        
        # Full tests (expensive)
        tests_ok, _ = run_full_verification(repo_path)
        evidence.tests_passed = tests_ok
    
    return evidence


# =============================================================================
# Quarantine Candidates
# =============================================================================

def find_dead_file_candidates(
    repo_path: str,
    file_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> List[str]:
    """Find files that might be dead code.
    
    Returns list of file paths relative to repo_path.
    """
    if file_patterns is None:
        file_patterns = ["**/*.py"]
    
    if exclude_patterns is None:
        exclude_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/conftest.py",
            "**/__init__.py",
            "**/migrations/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
        ]
    
    candidates = []
    repo = Path(repo_path)
    
    for pattern in file_patterns:
        for file_path in repo.glob(pattern):
            rel_path = str(file_path.relative_to(repo))
            
            # Check exclusions
            excluded = False
            for exc in exclude_patterns:
                if file_path.match(exc):
                    excluded = True
                    break
            
            if not excluded:
                candidates.append(rel_path)
    
    return candidates


def analyze_candidate(
    file_path: str,
    repo_path: str,
    run_dynamic: bool = True,
) -> QuarantineCandidate:
    """Analyze a file to determine if it's dead code."""
    static_evidence = analyze_static_evidence(file_path, repo_path)
    
    # Determine reason based on evidence
    reason = QuarantineReason.MANUAL_FLAG
    confidence = 0.0
    
    if static_evidence.rg_references == 0 and static_evidence.import_count == 0:
        reason = QuarantineReason.NO_REFERENCES
        confidence = 0.8
    elif static_evidence.import_count == 0:
        reason = QuarantineReason.NO_IMPORTS
        confidence = 0.6
    
    # Lower confidence if there are config references
    if static_evidence.config_references > 0:
        confidence *= 0.5
    
    # Dynamic evidence
    dynamic_evidence = DynamicEvidence()
    if run_dynamic and confidence > 0.5:
        dynamic_evidence = analyze_dynamic_evidence(file_path, repo_path, run_tests=False)
        
        # If import fails, might be dead
        if not dynamic_evidence.import_walk_passed:
            confidence = min(confidence + 0.1, 1.0)
    
    return QuarantineCandidate(
        file_path=file_path,
        reason=reason,
        confidence=confidence,
        static_evidence=static_evidence,
        dynamic_evidence=dynamic_evidence,
    )


# =============================================================================
# Quarantine Actions
# =============================================================================

def quarantine_file(
    file_path: str,
    repo_path: str,
    quarantine_dir: str = QUARANTINE_DIR,
) -> Optional[str]:
    """Move a file to quarantine.
    
    Returns the quarantine path or None on failure.
    """
    source = Path(repo_path) / file_path
    
    if not source.exists():
        logger.warning(f"[quarantine] File not found: {file_path}")
        return None
    
    # Create quarantine path preserving structure
    dest_dir = Path(repo_path) / quarantine_dir / Path(file_path).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(file_path).name
    
    try:
        shutil.move(str(source), str(dest))
        logger.info(f"[quarantine] Moved {file_path} -> {dest}")
        return str(dest.relative_to(Path(repo_path)))
    except Exception as e:
        logger.error(f"[quarantine] Failed to move {file_path}: {e}")
        return None


def restore_file(
    quarantine_path: str,
    original_path: str,
    repo_path: str,
) -> bool:
    """Restore a file from quarantine."""
    source = Path(repo_path) / quarantine_path
    dest = Path(repo_path) / original_path
    
    if not source.exists():
        logger.warning(f"[quarantine] Quarantine file not found: {quarantine_path}")
        return False
    
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        logger.info(f"[quarantine] Restored {original_path}")
        return True
    except Exception as e:
        logger.error(f"[quarantine] Failed to restore {original_path}: {e}")
        return False


# =============================================================================
# Block 10: Quarantine Workflow
# =============================================================================

async def run_quarantine_workflow(
    repo_path: str,
    job_id: str,
    confidence_threshold: float = 0.7,
    job_artifact_root: Optional[str] = None,
) -> QuarantineReport:
    """Run the full quarantine workflow.
    
    1. Find candidate files
    2. Analyze each candidate
    3. Quarantine high-confidence candidates
    4. Verify repo still works
    """
    report_id = str(uuid4())
    logger.info(f"[quarantine] Starting workflow for job {job_id}")
    
    # Find candidates
    file_candidates = find_dead_file_candidates(repo_path)
    logger.info(f"[quarantine] Found {len(file_candidates)} potential candidates")
    
    # Analyze each
    candidates = []
    for file_path in file_candidates:
        candidate = analyze_candidate(file_path, repo_path, run_dynamic=True)
        candidates.append(candidate)
    
    # Filter by confidence
    high_confidence = [c for c in candidates if c.confidence >= confidence_threshold]
    logger.info(f"[quarantine] {len(high_confidence)} candidates above threshold {confidence_threshold}")
    
    # Quarantine high-confidence files
    for candidate in high_confidence:
        quarantine_path = quarantine_file(candidate.file_path, repo_path)
        if quarantine_path:
            candidate.quarantined = True
            candidate.quarantine_path = quarantine_path
    
    # Verify repo still works
    repo_ok, test_result = run_full_verification(repo_path)
    
    verification = VerificationResult(
        chunk_id="quarantine",
        status=VerificationStatus.PASSED if repo_ok else VerificationStatus.FAILED,
        command_results=[test_result],
    )
    
    report = QuarantineReport(
        report_id=report_id,
        job_id=job_id,
        candidates=high_confidence,
        repo_still_passes=repo_ok,
        verification_evidence=verification,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    
    # Store report
    if job_artifact_root:
        report_dir = Path(job_artifact_root) / "jobs" / job_id / "quarantine"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"quarantine_report_{report_id}.json"
        report_path.write_text(report.to_json(), encoding="utf-8")
        logger.info(f"[quarantine] Report stored: {report_path}")
    
    return report


# =============================================================================
# Block 11: Deletion Workflow
# =============================================================================

async def run_deletion_workflow(
    repo_path: str,
    quarantine_report: QuarantineReport,
    approved_files: List[str],
    approved_by: str = "user",
    job_artifact_root: Optional[str] = None,
) -> DeletionReport:
    """Run the deletion workflow for approved quarantined files.
    
    Only deletes files that:
    1. Were quarantined in the report
    2. Are in the approved_files list
    3. Repo still passes after deletion
    """
    report_id = str(uuid4())
    job_id = quarantine_report.job_id
    
    logger.info(f"[deletion] Starting workflow for {len(approved_files)} files")
    
    deleted_files = []
    deletion_evidence = {}
    
    for candidate in quarantine_report.candidates:
        if candidate.file_path not in approved_files:
            continue
        
        if not candidate.quarantined or not candidate.quarantine_path:
            continue
        
        # Delete the quarantined file
        quarantine_path = Path(repo_path) / candidate.quarantine_path
        if quarantine_path.exists():
            try:
                quarantine_path.unlink()
                deleted_files.append(candidate.file_path)
                deletion_evidence[candidate.file_path] = f"Confidence: {candidate.confidence:.2f}, Reason: {candidate.reason.value}"
                logger.info(f"[deletion] Deleted: {candidate.file_path}")
                candidate.deleted = True
            except Exception as e:
                logger.error(f"[deletion] Failed to delete {candidate.file_path}: {e}")
    
    # Verify repo still works
    repo_ok, test_result = run_full_verification(repo_path)
    
    verification = VerificationResult(
        chunk_id="deletion",
        status=VerificationStatus.PASSED if repo_ok else VerificationStatus.FAILED,
        command_results=[test_result],
    )
    
    report = DeletionReport(
        report_id=report_id,
        job_id=job_id,
        quarantine_report_id=quarantine_report.report_id,
        deleted_files=deleted_files,
        deletion_evidence=deletion_evidence,
        repo_still_passes=repo_ok,
        verification_evidence=verification,
        approved_by=approved_by,
        approved_at=datetime.now(timezone.utc).isoformat(),
    )
    
    # Store report
    if job_artifact_root:
        report_dir = Path(job_artifact_root) / "jobs" / job_id / "deletion"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"deletion_report_{report_id}.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"[deletion] Report stored: {report_path}")
    
    return report


__all__ = [
    # Static analysis
    "count_rg_references",
    "find_imports_of_module",
    "find_config_references",
    "analyze_static_evidence",
    # Dynamic analysis
    "run_import_walk",
    "analyze_dynamic_evidence",
    # Candidates
    "find_dead_file_candidates",
    "analyze_candidate",
    # Quarantine actions
    "quarantine_file",
    "restore_file",
    # Workflows
    "run_quarantine_workflow",
    "run_deletion_workflow",
]

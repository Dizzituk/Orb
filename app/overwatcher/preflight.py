"""
Overwatcher Pre-Flight Check — Per-Segment Coherence Verification.

Called by segment_loop BEFORE the Overwatcher executes a segment.
Verifies the segment's architecture against the skeleton contract
using deterministic checks (Layer 1 from cohesion_check.py).

If pre-flight fails, produces a StructuredRejection with full context
for routing back to the Critical Pipeline.

v1.0 (2026-02-12): Initial implementation — deterministic pre-flight.

BUILD_ID: 2026-02-12-v1.0-overwatcher-preflight
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PREFLIGHT_BUILD_ID = "2026-02-12-v1.0-overwatcher-preflight"
print(f"[OVERWATCHER_PREFLIGHT_LOADED] BUILD_ID={PREFLIGHT_BUILD_ID}")


# =============================================================================
# STRUCTURED REJECTION
# =============================================================================

@dataclass
class StructuredRejection:
    """Full-context rejection when a segment fails coherence or execution.
    
    Captures everything needed to:
    1. Route back to the correct pipeline stage
    2. Give the re-generating LLM all failure context
    3. Save as a data point for the Experience Database
    """
    # Identity
    job_id: str = ""
    segment_id: str = ""
    rejection_id: str = ""
    
    # What failed
    stage: str = ""           # "preflight", "implementation", "phase_checkout"
    category: str = ""        # "coherence_fail", "scope_violation", "phantom_segment",
                              # "boot_fail", "interface_mismatch", "implementation_error"
    
    # Failure details
    issues: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    
    # Context for re-generation
    architecture_content: str = ""     # The arch that failed
    architecture_path: str = ""
    skeleton_contract: str = ""        # The skeleton markdown for this segment
    spec_markdown: str = ""            # The parent SPoT spec
    
    # Routing
    route_to: str = ""                 # "critical_pipeline", "specgate", "implementer"
    route_segment_only: bool = True    # Only re-run this segment, not all
    
    # Metadata
    timestamp: str = ""
    attempt_number: int = 1
    max_attempts: int = 3
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.rejection_id:
            from uuid import uuid4
            self.rejection_id = f"rej-{uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "segment_id": self.segment_id,
            "rejection_id": self.rejection_id,
            "stage": self.stage,
            "category": self.category,
            "issues": self.issues,
            "summary": self.summary,
            "architecture_path": self.architecture_path,
            "skeleton_contract_length": len(self.skeleton_contract),
            "spec_markdown_length": len(self.spec_markdown),
            "route_to": self.route_to,
            "route_segment_only": self.route_segment_only,
            "timestamp": self.timestamp,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredRejection":
        return cls(**{k: v for k, v in data.items() 
                      if k in cls.__dataclass_fields__})


# =============================================================================
# PER-SEGMENT PRE-FLIGHT CHECK
# =============================================================================

def run_segment_preflight(
    segment_id: str,
    architecture_content: str,
    skeleton_json: Optional[str] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
    job_id: str = "",
    architecture_path: str = "",
    skeleton_contract_markdown: str = "",
    spec_markdown: str = "",
    attempt_number: int = 1,
) -> Optional[StructuredRejection]:
    """
    Run deterministic pre-flight check for a single segment before implementation.
    
    This is the Overwatcher's coherence gate. Called BEFORE run_architecture_execution().
    
    Args:
        segment_id: The segment being checked
        architecture_content: The architecture markdown to verify
        skeleton_json: Full skeleton contract JSON (from SkeletonContractSet.to_json())
        manifest_dict: Raw manifest dict
        job_id: Job identifier
        architecture_path: Path to the architecture file on disk
        skeleton_contract_markdown: The formatted skeleton for this segment
        spec_markdown: The parent SPoT spec (for rejection context)
        attempt_number: Which attempt this is (for retry tracking)
    
    Returns:
        None if pre-flight passes (proceed to implementation)
        StructuredRejection if pre-flight fails (route back to Critical Pipeline)
    """
    from app.orchestrator.cohesion_check import run_skeleton_compliance
    
    # Run deterministic checks for this single segment
    issues = run_skeleton_compliance(
        architectures={segment_id: architecture_content},
        skeleton_json=skeleton_json,
        manifest_dict=manifest_dict,
    )
    
    # Filter to blocking issues only (warnings don't stop execution)
    blocking = [i for i in issues if i.severity == "blocking"]
    
    if not blocking:
        logger.info("[preflight] %s: PASSED (%d warnings)", 
                     segment_id, len(issues))
        return None
    
    # Build structured rejection
    logger.warning("[preflight] %s: FAILED — %d blocking issue(s)", 
                    segment_id, len(blocking))
    
    issue_dicts = []
    summary_parts = []
    for issue in blocking:
        issue_dicts.append(issue.to_dict())
        summary_parts.append(f"[{issue.category}] {issue.description}")
    
    rejection = StructuredRejection(
        job_id=job_id,
        segment_id=segment_id,
        stage="preflight",
        category=blocking[0].category if len(blocking) == 1 else "multiple_violations",
        issues=issue_dicts,
        summary=f"{len(blocking)} blocking issue(s): " + "; ".join(summary_parts),
        architecture_content=architecture_content,
        architecture_path=architecture_path,
        skeleton_contract=skeleton_contract_markdown,
        spec_markdown=spec_markdown,
        route_to="critical_pipeline",
        route_segment_only=True,
        attempt_number=attempt_number,
    )
    
    return rejection


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_rejection(rejection: StructuredRejection, job_dir: str) -> str:
    """Save rejection to disk for the Experience Database.
    
    Stored alongside the segment's arch files so the full context
    is available for re-generation and later analysis.
    """
    seg_dir = os.path.join(job_dir, "segments", rejection.segment_id)
    os.makedirs(seg_dir, exist_ok=True)
    
    path = os.path.join(seg_dir, f"rejection_{rejection.rejection_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        # Save the full dict including architecture content for context
        full_data = rejection.to_dict()
        # Also include the full content for re-generation context
        full_data["architecture_content"] = rejection.architecture_content
        full_data["skeleton_contract"] = rejection.skeleton_contract
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    
    logger.info("[preflight] Rejection saved: %s", path)
    return path


def load_rejections(job_dir: str, segment_id: str) -> List[StructuredRejection]:
    """Load all rejections for a segment."""
    seg_dir = os.path.join(job_dir, "segments", segment_id)
    rejections = []
    
    if not os.path.isdir(seg_dir):
        return rejections
    
    for fname in sorted(os.listdir(seg_dir)):
        if fname.startswith("rejection_") and fname.endswith(".json"):
            path = os.path.join(seg_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rejections.append(StructuredRejection.from_dict(data))
            except Exception as e:
                logger.warning("[preflight] Failed to load %s: %s", path, e)
    
    return rejections


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "StructuredRejection",
    "run_segment_preflight",
    "save_rejection",
    "load_rejections",
    "PREFLIGHT_BUILD_ID",
]

# FILE: app/orchestrator/construction_skeleton.py
"""
Construction Skeleton — Phase-to-Phase Interface Contracts (Tier 1).

Bridges the ConstructionPlan (Stage 3) and the per-segment skeleton
contracts. For multi-phase jobs, generates inter-phase contracts that:

1. Define what each phase promises to deliver (exports)
2. Define what each phase can import from completed phases
3. Verify completed phase deliverables exist on disk
4. Produce markdown blocks for architecture prompts showing
   "these files already exist from Phase N, import from them"

For single-phase jobs, this is a no-op — the existing segment
skeleton handles everything.

v1.0 (2026-02-14): Initial implementation — Tier 1.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .construction_planner_models import ConstructionPlan, PhaseDefinition

logger = logging.getLogger(__name__)

CONSTRUCTION_SKELETON_BUILD_ID = "2026-02-14-v1.0-initial"
print(f"[CONSTRUCTION_SKELETON_LOADED] BUILD_ID={CONSTRUCTION_SKELETON_BUILD_ID}")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PhaseDeliverable:
    """A file delivered by a completed phase, verified on disk."""
    file_path: str
    from_phase: str
    exists_on_disk: bool = False
    line_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "from_phase": self.from_phase,
            "exists_on_disk": self.exists_on_disk,
            "line_count": self.line_count,
        }


@dataclass
class PhaseInterfaceContract:
    """What a phase can import from completed upstream phases."""
    phase_id: str
    phase_number: int
    upstream_deliverables: List[PhaseDeliverable] = field(default_factory=list)
    missing_deliverables: List[str] = field(default_factory=list)
    all_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "phase_number": self.phase_number,
            "upstream_deliverables": [d.to_dict() for d in self.upstream_deliverables],
            "missing_deliverables": self.missing_deliverables,
            "all_verified": self.all_verified,
        }


# =============================================================================
# CONTRACT GENERATION
# =============================================================================

def generate_phase_interface(
    plan: ConstructionPlan,
    current_phase: PhaseDefinition,
    sandbox_base: str = r"D:\Orb",
) -> PhaseInterfaceContract:
    """
    Generate an interface contract for a phase about to execute.

    Scans completed upstream phases, verifies their exported files exist
    on disk, and produces a contract listing available imports.

    Args:
        plan: The full construction plan
        current_phase: The phase about to start
        sandbox_base: Root path for file verification

    Returns:
        PhaseInterfaceContract with verified deliverables
    """
    deliverables: List[PhaseDeliverable] = []
    missing: List[str] = []

    # Collect all completed upstream phases
    completed_phases = [
        p for p in plan.phases
        if p.status == "complete" and p.phase_number < current_phase.phase_number
    ]

    for upstream in completed_phases:
        if not upstream.contract:
            continue

        for file_path in upstream.contract.exports:
            abs_path = _resolve_path(file_path, sandbox_base)
            exists = abs_path is not None and os.path.isfile(abs_path)
            line_count = 0

            if exists:
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        line_count = f.read().count("\n") + 1
                except Exception:
                    pass

            d = PhaseDeliverable(
                file_path=file_path,
                from_phase=upstream.phase_id,
                exists_on_disk=exists,
                line_count=line_count,
            )
            deliverables.append(d)

            if not exists:
                missing.append(file_path)

    all_verified = len(missing) == 0 and len(deliverables) > 0

    contract = PhaseInterfaceContract(
        phase_id=current_phase.phase_id,
        phase_number=current_phase.phase_number,
        upstream_deliverables=deliverables,
        missing_deliverables=missing,
        all_verified=all_verified,
    )

    logger.info(
        "[construction_skeleton] Phase %d interface: %d deliverables, %d missing",
        current_phase.phase_number, len(deliverables), len(missing),
    )

    return contract


# =============================================================================
# MARKDOWN RENDERING
# =============================================================================

def format_phase_contract_markdown(contract: PhaseInterfaceContract) -> str:
    """
    Render the phase interface contract as markdown for the architecture
    prompt. Tells the LLM what files already exist from earlier phases.
    """
    if not contract.upstream_deliverables:
        return ""

    parts = []
    parts.append("## Upstream Phase Deliverables (AVAILABLE FOR IMPORT)\n")
    parts.append(
        f"This is Phase {contract.phase_number}. Earlier phases have already "
        f"built the following files. They exist on disk and are available for "
        f"import. **Use these directly — do NOT recreate or duplicate them.**\n"
    )

    # Group by source phase
    by_phase: Dict[str, List[PhaseDeliverable]] = {}
    for d in contract.upstream_deliverables:
        by_phase.setdefault(d.from_phase, []).append(d)

    for phase_id, phase_deliverables in by_phase.items():
        parts.append(f"### From `{phase_id}`\n")
        for d in phase_deliverables:
            status = "✅" if d.exists_on_disk else "❌ MISSING"
            parts.append(f"  - `{d.file_path}` ({d.line_count} lines) {status}")
        parts.append("")

    if contract.missing_deliverables:
        parts.append("### ⚠️ Missing Deliverables\n")
        parts.append(
            "The following files were promised by upstream phases but are "
            "NOT found on disk. You may need workarounds or stubs.\n"
        )
        for fp in contract.missing_deliverables:
            parts.append(f"  - `{fp}`")
        parts.append("")

    parts.append("### Import Rules\n")
    parts.append("1. Import from upstream files using their exact paths.")
    parts.append("2. Do NOT modify upstream files — they belong to earlier phases.")
    parts.append("3. If an upstream file is missing, add a TODO comment and stub the import.")
    parts.append("")

    return "\n".join(parts)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_phase_deliverables(
    plan: ConstructionPlan,
    phase: PhaseDefinition,
    sandbox_base: str = r"D:\Orb",
) -> Dict[str, Any]:
    """
    Quick check: did this phase deliver everything it promised?

    Returns a dict with pass/fail and any missing files.
    Called by Phase Checkout to validate contract compliance.
    """
    if not phase.contract:
        return {"status": "pass", "reason": "no contract defined"}

    missing = []
    for file_path in phase.contract.exports:
        abs_path = _resolve_path(file_path, sandbox_base)
        if not abs_path or not os.path.isfile(abs_path):
            missing.append(file_path)

    return {
        "status": "fail" if missing else "pass",
        "phase_id": phase.phase_id,
        "total_exports": len(phase.contract.exports),
        "delivered": len(phase.contract.exports) - len(missing),
        "missing": missing,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _resolve_path(rel_path: str, sandbox_base: str) -> Optional[str]:
    """Resolve relative path against sandbox base."""
    normalised = rel_path.replace("/", os.sep).replace("\\", os.sep)
    abs_path = os.path.join(sandbox_base, normalised)
    return abs_path

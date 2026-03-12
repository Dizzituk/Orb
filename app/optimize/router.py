# FILE: app/optimize/router.py
"""
Optimize Tab API Router.

FastAPI endpoints for the Electron/React frontend.

Endpoints:
  POST /optimize/run          — Run full A-C pass (decompose→profile→propose)
  GET  /optimize/proposals     — Get current proposals
  POST /optimize/approve       — Approve proposals for execution
  POST /optimize/execute       — Execute approved proposals
  GET  /optimize/report        — Get latest report
  GET  /optimize/patterns      — Get learned patterns
  GET  /optimize/stats         — Get optimize system stats

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize", tags=["optimize"])

# In-memory state for the current session
_current_report = None
_current_proposals = []


class RunRequest(BaseModel):
    target_id: str = "astra-backend"
    auto_approve_low_risk: bool = False


class ApproveRequest(BaseModel):
    proposal_ids: List[str]


class ExecuteRequest(BaseModel):
    target_id: str = "astra-backend"


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/run")
async def run_optimize(request: RunRequest):
    """Run phases A-C: Decompose → Profile → Propose."""
    global _current_report, _current_proposals

    from app.optimize.orchestrator import run_optimize_pass

    def emit(msg):
        logger.info("[optimize] %s", msg)

    report = await run_optimize_pass(
        target_id=request.target_id,
        auto_approve_low_risk=request.auto_approve_low_risk,
        emit=emit,
    )

    _current_report = report
    _current_proposals = report.proposals

    return {
        "status": "complete",
        "target": report.target,
        "chunks": report.manifest.total_files if report.manifest else 0,
        "bottlenecks": len(report.profile.bottlenecks) if report.profile else 0,
        "proposals": len(report.proposals),
        "executed": report.executed_count,
        "passed": report.success_count,
        "duration_seconds": report.total_duration_seconds,
    }


@router.get("/proposals")
async def get_proposals():
    """Get current proposals from the last run."""
    if not _current_proposals:
        return {"proposals": []}

    return {
        "proposals": [
            {
                "proposal_id": p.proposal_id,
                "category": p.category.value,
                "title": p.title,
                "description": p.description,
                "target_chunks": p.target_chunks,
                "predicted_improvement": p.predicted_improvement,
                "risk": p.risk.value,
                "status": p.status.value,
                "impact_score": p.impact_score,
                "confidence": p.confidence,
                "estimated_token_cost": p.estimated_token_cost,
            }
            for p in _current_proposals
        ]
    }


@router.post("/approve")
async def approve_proposals(request: ApproveRequest):
    """Approve specific proposals for execution."""
    from app.optimize.models import ProposalStatus

    approved = 0
    for p in _current_proposals:
        if p.proposal_id in request.proposal_ids:
            p.status = ProposalStatus.APPROVED
            approved += 1

    return {"approved": approved, "total": len(request.proposal_ids)}


@router.post("/execute")
async def execute_approved(request: ExecuteRequest):
    """Execute all approved proposals."""
    from app.optimize.orchestrator import execute_approved as exec_fn
    from app.optimize.models import ProposalStatus

    approved = [p for p in _current_proposals if p.status == ProposalStatus.APPROVED]
    if not approved:
        raise HTTPException(400, "No approved proposals to execute")

    def emit(msg):
        logger.info("[optimize] %s", msg)

    results = await exec_fn(
        proposals=_current_proposals,
        target_id=request.target_id,
        emit=emit,
    )

    return {
        "executed": len(results),
        "passed": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "ready_for_promotion": sum(1 for r in results if r.ready_for_promotion),
    }


@router.get("/report")
async def get_report():
    """Get the latest optimisation report."""
    if not _current_report:
        return {"status": "no_report"}

    r = _current_report
    return {
        "target": r.target,
        "manifest_summary": r.manifest.summary() if r.manifest else "",
        "bottlenecks": [
            {"path": b.path, "metric": b.metric, "value": b.value,
             "impact": b.impact_score, "description": b.description}
            for b in (r.profile.bottlenecks if r.profile else [])
        ],
        "proposals_count": len(r.proposals),
        "executed_count": r.executed_count,
        "success_count": r.success_count,
        "duration_seconds": r.total_duration_seconds,
        "token_cost": r.total_token_cost,
    }


@router.get("/patterns")
async def get_patterns():
    """Get learned optimisation patterns."""
    from app.optimize.pattern_learner import get_pattern_learner
    learner = get_pattern_learner()
    patterns = learner.get_patterns()
    return {
        "patterns": [p.to_dict() for p in patterns],
        "stats": learner.get_stats(),
    }


@router.get("/stats")
async def get_stats():
    """Get optimize system statistics."""
    stats = {
        "has_report": _current_report is not None,
        "proposals_count": len(_current_proposals),
    }

    try:
        from app.optimize.pattern_learner import get_pattern_learner
        stats["patterns"] = get_pattern_learner().get_stats()
    except Exception:
        stats["patterns"] = {}

    return stats

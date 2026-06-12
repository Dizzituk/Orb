# Purpose: router
# Called-by: main
# Depends-on: app.optimize.code_learner, app.optimize.loop_router, app.optimize.models, app.optimize.orchestrator (+2 more)
# Last-renovated: 2026-06-11
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.optimize.loop_router import router as loop_router
from app.optimize.target_registry import get_target_definition, list_target_definitions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize", tags=["optimize"])
router.include_router(loop_router)

_current_report = None
_current_proposals = []


class RunRequest(BaseModel):
    target_id: str = "astra-backend:optimize"
    auto_approve_low_risk: bool = False


class ApproveRequest(BaseModel):
    proposal_ids: List[str]


class ExecuteRequest(BaseModel):
    target_id: str = "astra-backend:optimize"


@router.get("/targets")
async def get_targets():
    return {"targets": [target.to_api_dict() for target in list_target_definitions()]}


@router.post("/run")
async def run_optimize(request: RunRequest):
    global _current_report, _current_proposals
    from app.optimize.orchestrator import run_optimize_pass

    target = get_target_definition(request.target_id)

    def emit(msg):
        logger.info("[optimize] %s", msg)

    report = await run_optimize_pass(
        target_id=target.target_id,
        auto_approve_low_risk=request.auto_approve_low_risk,
        emit=emit,
    )
    _current_report = report
    _current_proposals = report.proposals
    return {
        "status": "complete",
        "target": report.target,
        "target_label": target.display_label,
        "user_outcome": target.user_outcome,
        "verification": target.verification,
        "chunks": report.manifest.total_files if report.manifest else 0,
        "bottlenecks": len(report.profile.bottlenecks) if report.profile else 0,
        "proposals": len(report.proposals),
        "executed": report.executed_count,
        "passed": report.success_count,
        "duration_seconds": report.total_duration_seconds,
    }


@router.get("/proposals")
async def get_proposals():
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
    from app.optimize.models import ProposalStatus
    approved = 0
    for proposal in _current_proposals:
        if proposal.proposal_id in request.proposal_ids:
            proposal.status = ProposalStatus.APPROVED
            approved += 1
    return {"approved": approved, "total": len(request.proposal_ids)}


@router.post("/execute")
async def execute_approved(request: ExecuteRequest):
    from app.optimize.models import ProposalStatus
    from app.optimize.orchestrator import execute_approved as exec_fn

    approved = [proposal for proposal in _current_proposals if proposal.status == ProposalStatus.APPROVED]
    if not approved:
        raise HTTPException(400, "No approved proposals to execute")

    def emit(msg):
        logger.info("[optimize] %s", msg)

    results = await exec_fn(proposals=_current_proposals, target_id=request.target_id, emit=emit)
    return {
        "executed": len(results),
        "passed": sum(1 for result in results if result.success),
        "failed": sum(1 for result in results if not result.success),
        "ready_for_promotion": sum(1 for result in results if result.ready_for_promotion),
    }


@router.get("/report")
async def get_report():
    if not _current_report:
        return {"status": "no_report"}
    report = _current_report
    return {
        "target": report.target,
        "manifest_summary": report.manifest.summary() if report.manifest else "",
        "bottlenecks": [
            {
                "path": bottleneck.path,
                "metric": bottleneck.metric,
                "value": bottleneck.value,
                "impact": bottleneck.impact_score,
                "description": bottleneck.description,
            }
            for bottleneck in (report.profile.bottlenecks if report.profile else [])
        ],
        "proposals_count": len(report.proposals),
        "executed_count": report.executed_count,
        "success_count": report.success_count,
        "duration_seconds": report.total_duration_seconds,
        "token_cost": report.total_token_cost,
    }


@router.get("/patterns")
async def get_patterns():
    from app.optimize.pattern_learner import get_pattern_learner
    learner = get_pattern_learner()
    patterns = learner.get_patterns()
    return {"patterns": [pattern.to_dict() for pattern in patterns], "stats": learner.get_stats()}


@router.get("/stats")
async def get_stats():
    stats = {"has_report": _current_report is not None, "proposals_count": len(_current_proposals)}
    try:
        from app.optimize.pattern_learner import get_pattern_learner
        stats["patterns"] = get_pattern_learner().get_stats()
    except Exception:
        stats["patterns"] = {}
    return stats


@router.get("/lessons")
async def get_lessons(category: str = ""):
    """Get code lessons learned from successful optimisations."""
    from app.optimize.code_learner import get_lesson_store
    store = get_lesson_store()
    if category:
        lessons = store.get_lessons_for_category(category)
    else:
        lessons = store.get_all_lessons()
    return {
        "lessons": [lesson.to_dict() for lesson in lessons],
        "stats": store.get_stats(),
    }


@router.get("/lessons/rules")
async def get_reusable_rules():
    """Get distilled reusable rules from all lessons."""
    from app.optimize.code_learner import get_lesson_store
    store = get_lesson_store()
    rules = store.get_reusable_rules()
    return {"rules": rules, "count": len(rules)}

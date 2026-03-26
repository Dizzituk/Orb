from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.optimize.loop_engine import get_loop_history, get_loop_status, run_recursive_loop, stop_loop
from app.optimize.scope_manager import approve_scope_expansion, reject_scope_expansion
from app.optimize.self_test import run_self_test
from app.optimize.trust_gate import TrustLevel, get_trust_level, set_trust_level

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize/loop", tags=["optimize-loop"])


class LoopStartRequest(BaseModel):
    target_id: str = "routing"
    max_passes: int = 3
    trust_level: TrustLevel = TrustLevel.SEMI_AUTO


class ScopeApprovalRequest(BaseModel):
    flag_id: str
    approved: bool = True


class TrustLevelRequest(BaseModel):
    trust_level: TrustLevel


@router.post("/start")
async def start_recursive_optimization(request: LoopStartRequest):
    set_trust_level(request.trust_level)
    return run_recursive_loop(request.target_id, request.max_passes, request.trust_level)


@router.get("/status")
async def loop_status():
    return get_loop_status()


@router.post("/approve-scope")
async def approve_scope(request: ScopeApprovalRequest):
    flag = approve_scope_expansion(request.flag_id) if request.approved else reject_scope_expansion(request.flag_id)
    if not flag:
        raise HTTPException(status_code=404, detail="Scope flag not found")
    return {"flag": flag.to_dict()}


@router.post("/stop")
async def stop_recursive_optimization():
    return stop_loop()


@router.get("/history")
async def loop_history():
    return {"passes": get_loop_history()}


@router.get("/self-test")
async def self_test(target: str = "routing"):
    return run_self_test(target).to_dict()


@router.get("/trust")
async def trust_status():
    return {"trust_level": get_trust_level().value}


@router.post("/trust")
async def update_trust(request: TrustLevelRequest):
    return {"trust_level": set_trust_level(request.trust_level).value}

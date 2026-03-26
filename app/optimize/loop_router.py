# FILE: app/optimize/loop_router.py
"""
Router for recursive optimisation loop.

Wires the orchestrator's run_recursive_optimize to HTTP endpoints.
The loop runs as a background task so the HTTP response returns
immediately with a status handle.

v2.0 (2026-03-26): Rewired to use real orchestrator recursive loop.
v1.0: Original simulated loop via loop_engine.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.optimize.trust_gate import TrustLevel, get_trust_level, set_trust_level
from app.optimize.self_test import run_self_test

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/loop", tags=["optimize-loop"])


# ══════════════════════════════════════════════════════════
# STATE — tracks the running loop from outside
# ══════════════════════════════════════════════════════════

_loop_state = {
    "running": False,
    "target_id": None,
    "max_passes": 0,
    "status": "idle",
    "stop_requested": False,
    "result": None,
    "log": [],
}


def _emit(msg: str) -> None:
    """Capture loop output into the state log."""
    logger.info("[loop] %s", msg)
    _loop_state["log"].append(msg)


def _reset_state(target_id: str, max_passes: int) -> None:
    _loop_state.update({
        "running": True,
        "target_id": target_id,
        "max_passes": max_passes,
        "status": "running",
        "stop_requested": False,
        "result": None,
        "log": [],
    })


# ══════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════

class LoopStartRequest(BaseModel):
    target_id: str = "astra-backend:memory"
    max_passes: int = 5
    trust_level: TrustLevel = TrustLevel.SEMI_AUTO


class TrustLevelRequest(BaseModel):
    trust_level: TrustLevel


# ══════════════════════════════════════════════════════════
# BACKGROUND TASK
# ══════════════════════════════════════════════════════════

async def _run_loop_background(target_id: str, max_passes: int) -> None:
    """Run the recursive loop in the background."""
    try:
        from app.optimize.orchestrator import run_recursive_optimize
        result = await run_recursive_optimize(
            target_id=target_id,
            max_passes=max_passes,
            emit=_emit,
            should_stop=lambda: _loop_state["stop_requested"],
        )
        _loop_state["result"] = result.to_dict()
        _loop_state["status"] = "complete"
    except Exception as e:
        logger.error("[loop] Recursive loop failed: %s", e, exc_info=True)
        _loop_state["status"] = "error"
        _loop_state["result"] = {"error": str(e)}
    finally:
        _loop_state["running"] = False


# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

@router.post("/start")
async def start_recursive_optimization(request: LoopStartRequest):
    """Kick off a recursive optimisation loop (runs in background)."""
    if _loop_state["running"]:
        raise HTTPException(409, "A loop is already running")

    set_trust_level(request.trust_level)
    _reset_state(request.target_id, request.max_passes)

    asyncio.create_task(_run_loop_background(
        request.target_id, request.max_passes,
    ))

    return {
        "status": "started",
        "target_id": request.target_id,
        "max_passes": request.max_passes,
        "trust_level": request.trust_level.value,
    }


@router.get("/status")
async def loop_status():
    """Get current loop state and result if complete."""
    return {
        "running": _loop_state["running"],
        "target_id": _loop_state["target_id"],
        "max_passes": _loop_state["max_passes"],
        "status": _loop_state["status"],
        "result": _loop_state["result"],
    }


@router.get("/log")
async def loop_log():
    """Get the real-time log output from the running loop."""
    return {
        "running": _loop_state["running"],
        "status": _loop_state["status"],
        "log": _loop_state["log"],
        "log_count": len(_loop_state["log"]),
    }


@router.post("/stop")
async def stop_recursive_optimization():
    """Request the loop to stop after the current pass."""
    if not _loop_state["running"]:
        raise HTTPException(400, "No loop is running")
    _loop_state["stop_requested"] = True
    return {"status": "stop_requested"}


@router.get("/history")
async def loop_history():
    """Get the result from the last completed loop."""
    if not _loop_state["result"]:
        return {"passes": [], "status": "no_history"}
    return _loop_state["result"]


@router.get("/self-test")
async def self_test(target: str = "routing"):
    return run_self_test(target).to_dict()


@router.get("/trust")
async def trust_status():
    return {"trust_level": get_trust_level().value}


@router.post("/trust")
async def update_trust(request: TrustLevelRequest):
    return {"trust_level": set_trust_level(request.trust_level).value}


# FILE: app/gpu/router.py
# Purpose: GPU orchestrator endpoints — state/status, Electron desired-state poll, GAMING toggle.
# Called-by: app.router_registry (registered local-trusted); orb-desktop main.js polls /gpu/desired
# Depends-on: app.gpu.orchestrator
# Last-renovated: 2026-07-02 (LANE E)
"""
GPU orchestrator HTTP surface (LANE E, Task 4d).

  GET  /gpu/state    full status: state, per-component desired vs running,
                     nvidia-smi VRAM map, visual ingest queue depth
  GET  /gpu/desired  minimal desired-components poll for the Electron
                     reconciler (it starts/stops the Chatterbox child)
  POST /gpu/gaming   manual GAMING toggle {"enabled": true|false}

Registered with require_local_or_secret (loopback-bound backend; same trust
tier as transcribe/audio). GAMING frees the whole 4080; the 3090 fleet is
never touched by design.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.gpu.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gpu", tags=["gpu"])


class GamingToggle(BaseModel):
    enabled: bool


@router.get("/state")
async def gpu_state():
    """Full orchestrator status (desktop status indicator polls this).
    Probes + nvidia-smi block for a few seconds — keep off the event loop."""
    return await asyncio.to_thread(get_orchestrator().status, True)


@router.get("/desired")
async def gpu_desired():
    """Cheap desired-state poll for the Electron Chatterbox reconciler —
    no probes, no nvidia-smi, safe at a few-second cadence."""
    orch = get_orchestrator()
    return {
        "state": orch.current_state(),
        "components": orch.desired_components(),
    }


@router.post("/activity")
async def gpu_activity():
    """CARRYOVER §2: activity = ANY user interaction. The desktop calls this
    for interactions that don't create a message (tab focus, clicks) so the
    10-minute park timer resets and a parked Chatterbox wakes."""
    from app.idle.governor import record_user_activity
    record_user_activity()
    return {"ok": True}


@router.post("/gaming")
async def gpu_gaming(body: GamingToggle):
    """Manual GAMING toggle: everything on the 4080 unloads (Chatterbox via
    the Electron poll, vLLM instances + worker directly). Flip back to
    restore INTERACTIVE — first TTS reply pays the reload, by design.
    Converge shells out to WSL (up to ~30s) — run on a worker thread so the
    event loop keeps serving; the response returns the CONVERGED state."""
    status = await asyncio.to_thread(get_orchestrator().set_gaming, body.enabled)
    return {"ok": True, "state": status["state"], "reason": status["reason"]}

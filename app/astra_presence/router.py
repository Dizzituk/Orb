# FILE: app/astra_presence/router.py
# Purpose: FastAPI surface for ASTRA presence — WS /astra/ws (push state changes on
#          connect + every change) + GET/POST /astra/state (read + manual/testing set).
# Called-by: main (include_router), Room AstraStateClient.cs
# Depends-on: app.astra_presence.state
# Last-renovated: 2026-06-13
"""ASTRA presence API (prefix /astra).

Local-trusted like the Room's /scene/* endpoints (the renderer holds no token).
WS /astra/ws is push-only: on connect it sends the current state, then one
{"type":"astra_state","state":...,"version":n} frame per change.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.astra_presence.state import VALID_STATES, normalise_state, presence_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/astra", tags=["ASTRA Presence"])


class StateBody(BaseModel):
    state: str


@router.get("/state")
def get_state():
    """Current presence state + version."""
    return presence_state.current_payload()


@router.post("/state")
def set_state(body: StateBody):
    """Set ASTRA's presence state (manual/testing + other backend code via HTTP)."""
    if normalise_state(body.state) is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown state '{body.state}'; valid: {list(VALID_STATES)}",
        )
    version = presence_state.set_state(body.state)
    return {"ok": True, "state": presence_state.get_state(), "version": version}


@router.websocket("/ws")
async def astra_ws(ws: WebSocket):
    """Push the current state on connect, then every change as it happens."""
    await ws.accept()
    queue = presence_state.subscribe()
    logger.info("[presence] client connected (%d subscriber(s))", presence_state.subscriber_count)
    try:
        await ws.send_json(presence_state.current_payload())
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        logger.info("[presence] client disconnected")
    except Exception:
        logger.exception("[presence] websocket error")
    finally:
        presence_state.unsubscribe(queue)
        try:
            await ws.close()
        except Exception:
            pass

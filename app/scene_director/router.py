# FILE: app/scene_director/router.py
# Purpose: FastAPI surface for ASTRA Room — compose/current/catalog/narrate/events
#          REST + the /scene/ws websocket that pushes SceneDocs to the renderer.
# Called-by: main (include_router), orb-desktop RoomView, Unity AstraClient
# Depends-on: app.auth, app.scene_director.director/state/asset_catalog/schemas, httpx
# Last-renovated: 2026-06-12
"""Scene Director API (prefix /scene).

AUTH MODEL (v1): POST /scene/compose costs LLM money and is desktop-initiated,
so it requires the normal Bearer auth. The renderer-facing endpoints
(current/catalog/narrate/ws/events) are local-trusted like the Electron
web_automation polling — the Unity app holds no token. Future Quest builds
reach these over the tailnet, which is the same trust boundary as /bridge.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.auth import require_auth
from app.scene_director import asset_catalog, director
from app.scene_director.schemas import ComposeRequest, ComposeResponse
from app.scene_director.state import scene_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scene", tags=["Scene Director"])

# Same service + endpoint chat_and_speak.py uses — ONE TTS path in this house.
_TTS_BASE = "http://127.0.0.1:8002"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\n)")


class NarrateBody(BaseModel):
    text: str = Field(min_length=1, max_length=1500)


# ─────────────────────────────────────────────────────────────────────────────
# REST
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/compose", response_model=ComposeResponse,
             dependencies=[Depends(require_auth)])
async def compose(body: ComposeRequest) -> ComposeResponse:
    """Compose a SceneDoc from intent, run the scene critic to enrich/fix it (v3),
    store it as current, push to renderers."""
    # v3: optional research rail (gate -> facts) before composing.
    fact_pack = None
    try:
        from app.scene_director.research import maybe_research
        fact_pack = await maybe_research(body.intent, era=body.era)
    except Exception as _research_err:  # research is best-effort; never blocks compose
        logger.debug("[scene] research skipped: %s", _research_err)

    result = await director.compose_scene(body.intent, era=body.era, fact_pack=fact_pack)

    # v3: the scene critic interrogates the draft and revises it (bounded passes).
    try:
        from app.scene_director import critic
        result.scene = await critic.critique_and_revise(
            result.scene, body.intent, result.warnings, era=body.era)
    except Exception as _critic_err:
        logger.warning("[scene] critic skipped: %s", _critic_err)

    scene_state.set_scene(result.scene)
    return result


@router.get("/current")
def current_scene():
    """The current SceneDoc, or 404 when nothing has been composed yet."""
    doc = scene_state.get_scene()
    if doc is None:
        raise HTTPException(status_code=404, detail="no scene composed yet")
    return {"type": "scene", "doc": doc.model_dump(mode="json")}


@router.get("/catalog")
def get_catalog():
    """The asset catalogue — Unity PrefabCatalog reconciliation + Room tab read this."""
    return asset_catalog.load_catalog()


@router.get("/events")
def get_events(limit: int = 50):
    """Recent renderer status breadcrumbs (scene_loaded, actor_done, error …)."""
    return {"events": scene_state.status_events(limit=limit),
            "subscribers": scene_state.subscriber_count}


@router.post("/narrate")
async def narrate(body: NarrateBody):
    """Synthesise one narration line via Chatterbox (:8002 /tts/sentence — the
    chat_and_speak path) and return a single MP3 body. Sentence-batched: each
    sentence is synthesised separately and the MP3 chunks concatenated, which
    players (and Unity's MPEG AudioClip loader) handle fine."""
    sentences = [s.strip() for s in _SENTENCE_END.split(body.text) if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="no speakable text")
    chunks: list[bytes] = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for sentence in sentences:
                resp = await client.post(f"{_TTS_BASE}/tts/sentence", json={"text": sentence})
                resp.raise_for_status()
                chunks.append(resp.content)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502,
                            detail=f"Chatterbox TTS error {exc.response.status_code}")
    except httpx.HTTPError:
        raise HTTPException(status_code=503,
                            detail="Chatterbox TTS is not reachable on :8002 — narration unavailable")
    return Response(content=b"".join(chunks), media_type="audio/mpeg")


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — the renderer's feed
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws")
async def scene_ws(ws: WebSocket):
    """On connect: send the current scene immediately (if any), then push every
    newly composed scene. Client→server messages {"type":"status", ...} are
    recorded as debug breadcrumbs (GET /scene/events)."""
    await ws.accept()
    queue = scene_state.subscribe()
    logger.info("[scene] renderer connected (%d subscriber(s))", scene_state.subscriber_count)
    recv_task: Optional[asyncio.Task] = None
    push_task: Optional[asyncio.Task] = None
    try:
        doc = scene_state.get_scene()
        if doc is not None:
            await ws.send_json({"type": "scene", "doc": doc.model_dump(mode="json")})
        recv_task = asyncio.create_task(ws.receive_json())
        push_task = asyncio.create_task(queue.get())
        while True:
            done, _ = await asyncio.wait({recv_task, push_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if recv_task in done:
                try:
                    msg = recv_task.result()
                except WebSocketDisconnect:
                    raise
                except Exception as exc:  # malformed frame — log, keep listening
                    logger.warning("[scene] bad websocket frame from renderer: %s", exc)
                    msg = None
                if isinstance(msg, dict) and msg.get("type") == "status":
                    scene_state.add_status_event(msg)
                elif msg is not None:
                    logger.info("[scene] unrecognised renderer message: %s", str(msg)[:200])
                recv_task = asyncio.create_task(ws.receive_json())
            if push_task in done:
                new_doc = push_task.result()
                await ws.send_json({"type": "scene", "doc": new_doc.model_dump(mode="json")})
                push_task = asyncio.create_task(queue.get())
    except WebSocketDisconnect:
        logger.info("[scene] renderer disconnected")
    except Exception:
        logger.exception("[scene] websocket error")
    finally:
        scene_state.unsubscribe(queue)
        for task in (recv_task, push_task):
            if task is not None and not task.done():
                task.cancel()
        try:
            await ws.close()
        except Exception:
            pass

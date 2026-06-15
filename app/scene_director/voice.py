# FILE: app/scene_director/voice.py
# Purpose: Room conversation endpoint — local-trusted POST /scene/ask: text in, ASTRA's
#          spoken reply out (one audio/mpeg body) reusing the bridge chat brain + TTS.
# Called-by: main (include_router), Room VoiceLoop.cs
# Depends-on: app.bridge.capability_layer.run_astra_chat, app.bridge.chat_and_speak (TTS helpers),
#             app.memory (Room project + messages), app.astra_presence.state
# Last-renovated: 2026-06-13
"""Room voice conversation (v2).

WHY A SEPARATE ENDPOINT (verified 2026-06-13): the obvious reuse target,
/bridge/chat-and-speak, is Bearer-auth'd (require_bridge_auth, token from
/bridge/login). The Unity Room is a tokenless local renderer (same trust model
as /scene/* and /scene/narrate), so it cannot carry a bridge session token.
This endpoint REUSES the exact bridge brain (run_astra_chat) and the exact TTS
sentence helpers (chat_and_speak._split_into_sentences/_synthesise_sentence) but
is local-trusted and returns ONE complete audio/mpeg body (like /scene/narrate)
for reliable Unity DownloadHandlerAudioClip(AudioType.MPEG) playback — it does
NOT refactor the bridge. Conversation continuity rides a single dedicated
"ASTRA Room" project (configurable via ASTRA_ROOM_PROJECT_NAME), resolved by
name (DB ids are not stable across resets).

Streaming the reply (first-audio-sooner) is a documented future enhancement;
v2 returns a full body because that is the playback path already proven in
Unity by /scene/narrate.
"""
from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import quote as url_quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scene", tags=["Scene Director"])

ROOM_PROJECT_NAME = os.getenv("ASTRA_ROOM_PROJECT_NAME", "ASTRA Room")


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    project_id: int | None = None


def _resolve_room_project(db: Session):
    """Fixed, persistent Room project so all Room turns share one memory/session
    scope. Resolved by NAME (ids are DB auto-increment, not stable). Created once."""
    from app.memory.schemas import ProjectCreate
    from app.memory._service_utils_2 import create_project, get_project_by_name

    existing = get_project_by_name(db, ROOM_PROJECT_NAME)
    if existing:
        return existing
    return create_project(db, ProjectCreate(
        name=ROOM_PROJECT_NAME,
        description="Immersive ASTRA Room conversation (voice)",
        type="room",
    ))


def _presence(name: str) -> None:
    """Best-effort presence broadcast — never fails the conversation."""
    try:
        from app.astra_presence.state import presence_state
        presence_state.set_state(name)
    except Exception as exc:  # pragma: no cover — presence is optional enrichment
        logger.debug("[scene.ask] presence set %s skipped: %s", name, exc)


async def _idle_after(seconds: float) -> None:
    """Return presence to idle after an estimated speak duration, unless a newer
    state has superseded 'speaking' in the meantime."""
    try:
        await asyncio.sleep(seconds)
        from app.astra_presence.state import presence_state
        if presence_state.get_state() == "speaking":
            presence_state.set_state("idle")
    except Exception:  # pragma: no cover
        pass


@router.post("/ask")
async def ask(req: AskRequest, db: Session = Depends(get_db)):
    """Text question -> ASTRA's spoken reply (audio/mpeg body) + X-Full-Text header.

    Presence: thinking (while the brain runs) -> speaking (audio ready) ->
    idle (after an estimated duration). The Room also drives its orb LOCALLY
    from its own voice-loop lifecycle, which is authoritative for the Room; this
    presence feed is the enrichment that lets other surfaces watch too.
    """
    from app.memory.service import create_message, get_project
    from app.memory.schemas import MessageCreate
    from app.memory._service_utils_2 import list_messages
    from app.bridge.capability_layer import run_astra_chat
    from app.bridge.chat_and_speak import _split_into_sentences, _synthesise_sentence

    if req.project_id:
        project = get_project(db, req.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"project {req.project_id} not found")
    else:
        project = _resolve_room_project(db)

    _presence("thinking")
    try:
        create_message(db, MessageCreate(
            project_id=project.id, role="user",
            content=req.message, provider="room", model="voice-input",
        ))
        history = [
            {"role": m.role, "content": m.content}
            for m in list_messages(db, project.id, limit=20)
            if m.role in ("user", "assistant")
        ]
        result = await run_astra_chat(
            message=req.message, project_id=project.id,
            history=history, db=db, source="room",
        )
        reply = ((result or {}).get("reply") or "").strip()
        if not reply:
            raise RuntimeError("empty reply from chat brain")
        create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=reply, provider=(result or {}).get("provider", "room"),
            model=(result or {}).get("model", "room"),
        ))
    except Exception as exc:
        logger.warning("[scene.ask] chat brain failed: %s", exc)
        _presence("error")
        asyncio.get_running_loop().create_task(_idle_after(2.0))
        raise HTTPException(status_code=503, detail=f"ASTRA chat unavailable: {exc}")

    # Strip [[astra:...]] directives so they are never spoken or shown.
    display_text = reply
    try:
        from app.bridge.directives import extract_directives, strip_directives
        if extract_directives(display_text):
            display_text = strip_directives(display_text)
    except Exception:  # pragma: no cover — directives are optional
        pass

    base_headers = {
        "X-Full-Text": url_quote(display_text[:8000]),
        "X-Project-Id": str(project.id),
        "X-Project-Name": url_quote(project.name or ""),
    }

    sentences = _split_into_sentences(display_text)
    chunks: list[bytes] = []
    try:
        for sentence in sentences:
            chunks.append(await _synthesise_sentence(sentence))
    except Exception as exc:
        # Chatterbox down: still return the TEXT so the Room can show the reply.
        logger.warning("[scene.ask] TTS failed (%s) — returning text-only", exc)
        _presence("idle")
        return Response(content=b"", media_type="audio/mpeg",
                        headers={**base_headers, "X-Tts-Error": "chatterbox-unreachable"})

    audio = b"".join(chunks)
    _presence("speaking")
    words = max(1, len(display_text.split()))
    asyncio.get_running_loop().create_task(_idle_after(min(60.0, max(2.0, words * 0.42))))
    return Response(content=audio, media_type="audio/mpeg", headers=base_headers)

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.memory._service_utils_2 import list_messages

from . import tts_cache
from .schemas import BridgeArtifactRef, require_bridge_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])


class MissedReplyOut(BaseModel):
    id: int
    content: str
    created_at: str
    has_audio: bool
    # Set when complete cached audio exists on the PC — the phone fetches it
    # via this auth-gated path instead of re-synthesising the reply.
    audio_url: str | None = None
    # 2026-06-13: artifact chips for recovered replies. The direct response
    # paths strip [ASTRA_ARTIFACT:...] markers and ship refs via X-Artifacts,
    # but this path returned the RAW marker text with no attachment — an
    # image reply recovered after a dead zone showed the bracketed filename
    # and the image itself never reached the phone (road bug 2026-06-12).
    attachments: List[BridgeArtifactRef] = []


@router.get("/missed-replies", response_model=List[MissedReplyOut])
async def get_missed_replies(
    since_id: int,
    project_id: int,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    # Same marker→attachment processing as the direct chat responses
    # (lazy import keeps module import light; router never imports us back).
    from .router import _process_artifacts

    messages = list_messages(db, project_id, limit=500)
    replies = []
    for message in messages:
        if message.id <= since_id or message.role != "assistant":
            continue
        stripped, refs = _process_artifacts(message.content)
        replies.append(MissedReplyOut(
            id=message.id,
            content=stripped,
            created_at=message.created_at.isoformat() if message.created_at else "",
            has_audio=bool(stripped.strip()),
            audio_url=(
                f"/bridge/tts/audio/{message.id}"
                if tts_cache.has_audio(message.id) else None
            ),
            attachments=refs,
        ))
    logger.info("[bridge] missed replies since %s for project %s -> %s", since_id, project_id, len(replies))
    return replies

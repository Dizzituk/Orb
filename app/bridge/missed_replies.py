from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.memory._service_utils_2 import list_messages

from .schemas import require_bridge_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])


class MissedReplyOut(BaseModel):
    id: int
    content: str
    created_at: str
    has_audio: bool


@router.get("/missed-replies", response_model=List[MissedReplyOut])
async def get_missed_replies(
    since_id: int,
    project_id: int,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    messages = list_messages(db, project_id, limit=500)
    replies = [
        MissedReplyOut(
            id=message.id,
            content=message.content,
            created_at=message.created_at.isoformat() if message.created_at else "",
            has_audio=bool(getattr(message, "content", "").strip()),
        )
        for message in messages
        if message.id > since_id and message.role == "assistant"
    ]
    logger.info("[bridge] missed replies since %s for project %s -> %s", since_id, project_id, len(replies))
    return replies

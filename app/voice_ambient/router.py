# FILE: app/voice_ambient/router.py
# Purpose: FastAPI router for the ambient voice WebSocket.
# Called-by: app.voice_ambient, main
# Depends-on: app.voice_ambient.session
# Last-renovated: 2026-06-11
"""
FastAPI router for the ambient voice WebSocket.

Route:
  WS  /api/voice-ambient/stream
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.voice_ambient.session import AmbientVoiceSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice-ambient", tags=["Voice Ambient"])


@router.websocket("/stream")
async def ambient_voice_stream(ws: WebSocket):
    """WebSocket endpoint for ambient wake-word + utterance capture."""
    await ws.accept()
    logger.info("[voice_ambient] WebSocket connection accepted")
    session = AmbientVoiceSession(ws)
    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("[voice_ambient] client disconnected")
    except Exception:
        logger.exception("[voice_ambient] WebSocket error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
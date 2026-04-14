# FILE: app/bridge/tts_proxy.py
"""
TTS proxy endpoints for the Bridge API.

Proxies to the Chatterbox TTS microservice on port 8002 so AstraBridge
only needs one backend URL.  All TTS traffic goes through bridge auth.

Extracted from router.py during modularisation (2026-04-06).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response, StreamingResponse

from .schemas import BridgeTTSRequest, require_bridge_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bridge/tts", tags=["Bridge TTS"])

TTS_BASE = "http://127.0.0.1:8002"


@router.get("/voices")
async def bridge_tts_voices(
    _auth: bool = Depends(require_bridge_auth),
):
    """List available TTS voices (proxied from TTS microservice)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{TTS_BASE}/tts/voices")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("[bridge] TTS voices proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/speak")
async def bridge_tts_speak(
    req: BridgeTTSRequest,
    _auth: bool = Depends(require_bridge_auth),
):
    """Synthesise speech (proxied from TTS microservice).  Returns MP3 audio bytes."""
    import httpx
    payload = {"text": req.text}
    if req.voice_name:
        payload["voice"] = req.voice_name
    if req.speed is not None:
        payload["speed"] = req.speed

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{TTS_BASE}/tts/speak", json=payload)
            resp.raise_for_status()
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "audio/mpeg"),
            )
    except Exception as e:
        logger.error("[bridge] TTS speak proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/stream")
async def bridge_tts_stream(
    req: BridgeTTSRequest,
    request: Request,
    _auth: bool = Depends(require_bridge_auth),
):
    """Progressive TTS proxy — streams a growing MP3 from Chatterbox.

    v2.0 (2026-04-13): Client-disconnect handling. When the Bridge app
    drops connection mid-stream (signal loss, app backgrounded with radio
    killed, manual cancel), we now:
      1. Poll request.is_disconnected() between chunks and break cleanly
      2. Treat CancelledError / RemoteProtocolError as INFO not ERROR
      3. Cancel the upstream Chatterbox request so it stops generating
         audio for a listener that will never hear it (token waste).
    """
    import asyncio
    import httpx

    payload = {"text": req.text}
    if req.voice_name:
        payload["voice"] = req.voice_name
    if req.speed is not None:
        payload["speed"] = req.speed

    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(
            connect=10.0, read=120.0, write=10.0, pool=10.0,
        ))

        async def stream_from_chatterbox():
            bytes_sent = 0
            try:
                async with client.stream("POST", f"{TTS_BASE}/tts/progressive", json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        if await request.is_disconnected():
                            logger.info(
                                "[bridge] TTS client disconnected after %d bytes — aborting upstream",
                                bytes_sent,
                            )
                            return
                        yield chunk
                        bytes_sent += len(chunk)
            except asyncio.CancelledError:
                logger.info("[bridge] TTS stream cancelled after %d bytes", bytes_sent)
                raise
            except httpx.RemoteProtocolError as e:
                logger.info("[bridge] TTS upstream closed after %d bytes: %s", bytes_sent, e)
            except Exception as e:
                logger.error("[bridge] TTS progressive proxy error after %d bytes: %s", bytes_sent, e)
            finally:
                await client.aclose()

        return StreamingResponse(stream_from_chatterbox(), media_type="audio/mpeg")
    except Exception as e:
        logger.error("[bridge] TTS progressive proxy failed: %s", e)
        raise HTTPException(502, f"TTS stream unavailable: {e}")


@router.post("/voices/select")
async def bridge_tts_select_voice(
    voice_name: str,
    _auth: bool = Depends(require_bridge_auth),
):
    """Set the active TTS voice (proxied from TTS microservice)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{TTS_BASE}/tts/voices/select",
                json={"voice": voice_name},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("[bridge] TTS voice select proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/preview")
async def bridge_tts_preview(
    voice_name: str,
    _auth: bool = Depends(require_bridge_auth),
):
    """Preview a TTS voice with sample text (proxied from TTS microservice)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{TTS_BASE}/tts/preview",
                json={"voice": voice_name},
            )
            resp.raise_for_status()
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "audio/mpeg"),
            )
    except Exception as e:
        logger.error("[bridge] TTS preview proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")
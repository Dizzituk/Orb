# FILE: app/content/video_pipeline/heygen_client.py
# Purpose: HeyGen API client for AI avatar video generation.
# Called-by: app.content.video_pipeline.asset_resolver, app.content.video_pipeline.router
# Depends-on: app.db, app.settings.service
# Last-renovated: 2026-06-11
"""
HeyGen API client for AI avatar video generation.

Generates talking-head avatar clips from text using HeyGen's API.
Used for ASTRA persona narration segments.

API key from encrypted settings: HEYGEN_API_KEY
Avatar ID from env: HEYGEN_AVATAR_ID
"""
import json
import os
import time
import logging
import httpx
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

HEYGEN_BASE_URL = "https://api.heygen.com"
DOWNLOAD_DIR = Path("data/content/video_pipeline/downloads/heygen")


def _get_api_key() -> str:
    """Get HeyGen API key from environment."""
    key = os.getenv("HEYGEN_API_KEY", "")
    if not key:
        raise ValueError(
            "HEYGEN_API_KEY not set. Add it in Settings > API Keys."
        )
    return key


def _headers() -> Dict[str, str]:
    return {
        "X-Api-Key": _get_api_key(),
        "Content-Type": "application/json",
    }


async def list_avatars() -> list:
    """List available avatars in the account."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HEYGEN_BASE_URL}/v2/avatars",
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("data", {}).get("avatars", [])


async def list_voices() -> list:
    """List available voices in the account."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HEYGEN_BASE_URL}/v2/voices",
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("data", {}).get("voices", [])


async def generate_avatar_video(
    text: str,
    avatar_id: Optional[str] = None,
    voice_id: Optional[str] = None,
    background_color: str = "#0a0a14",  # Dark blue-black fallback
    aspect_ratio: str = "16:9",
) -> Dict[str, Any]:
    """
    Generate an avatar video from text.

    Returns dict with: video_id, status, video_url (when complete)
    """
    avatar_id = avatar_id or os.getenv("HEYGEN_AVATAR_ID", "")
    voice_id = voice_id or os.getenv("HEYGEN_VOICE_ID", "")

    # Fallback: if voice_id still empty, read directly from DB
    # This handles the case where env sync missed the key on startup
    if not voice_id:
        try:
            from app.db import SessionLocal
            from app.settings.service import get_key_value
            with SessionLocal() as _db:
                voice_id = get_key_value(_db, "heygen_voice_id") or ""
                if voice_id:
                    # Also sync to env so future calls don't hit DB
                    os.environ["HEYGEN_VOICE_ID"] = voice_id
                    logger.info(
                        f"[heygen] voice_id loaded from DB fallback: "
                        f"{voice_id[:8]}..."
                    )
        except Exception as e:
            logger.warning(f"[heygen] DB fallback for voice_id failed: {e}")

    if not avatar_id:
        raise ValueError(
            "HEYGEN_AVATAR_ID not set. Configure your avatar in "
            "Settings > API Keys > HeyGen Avatar ID."
        )

    # Determine avatar type.
    # Stock avatars have names like "Daisy-insuit-20220818".
    # Photo avatars (talking photos) have hex IDs like "571dcfbf...".
    # If the ID looks like a hex string (32 chars, no hyphens with letters),
    # treat it as a Photo Avatar / talking_photo.
    is_photo_avatar = (
        len(avatar_id) == 32
        and all(c in '0123456789abcdef' for c in avatar_id.lower())
    )

    # Build character config based on avatar type
    if is_photo_avatar:
        # Photo Avatar: use talking_photo type with talking_photo_id
        # Enable Avatar IV for higher quality (better motion, lip sync)
        character = {
            "type": "talking_photo",
            "talking_photo_id": avatar_id,
        }
        logger.info(f"[heygen] Using Photo Avatar (talking_photo) with Avatar IV: {avatar_id}")
    else:
        # Stock/Instant avatar: use standard avatar type
        character = {
            "type": "avatar",
            "avatar_id": avatar_id,
            "avatar_style": "normal",
        }
        logger.info(f"[heygen] Using standard avatar: {avatar_id}")

    # Build voice config — voice_id is REQUIRED for Photo Avatars
    if not voice_id:
        raise ValueError(
            "HEYGEN_VOICE_ID not set. Photo Avatars require a voice_id. "
            "Set it in Settings > API Keys > HeyGen Voice ID. "
            "Use /content/video-pipeline/heygen-voices to find voice IDs."
        )

    voice_config = {
        "type": "text",
        "input_text": text,
        "voice_id": voice_id,
    }

    logger.info(
        f"[heygen] Voice config: voice_id={voice_id}, "
        f"text_length={len(text)} chars"
    )

    # Build the video generation request
    payload = {
        "video_inputs": [
            {
                "character": character,
                "voice": voice_config,
                "background": {
                    "type": "color",
                    "value": background_color,
                },
            }
        ],
        "dimension": {
            "width": 1920 if aspect_ratio == "16:9" else 1080,
            "height": 1080 if aspect_ratio == "16:9" else 1920,
        },
    }

    # Enable Avatar IV for Photo Avatars (higher quality rendering)
    if is_photo_avatar:
        payload["video_inputs"][0]["character"]["use_avatar_iv_model"] = True

    # Use a tech-themed background image instead of plain colour
    # if a background_url is provided
    avatar_bg_url = os.getenv("HEYGEN_AVATAR_BG_URL", "")
    if avatar_bg_url:
        payload["video_inputs"][0]["background"] = {
            "type": "image",
            "url": avatar_bg_url,
        }
        logger.info(f"[heygen] Using image background: {avatar_bg_url[:80]}...")

    logger.info(
        f"[heygen] Generating avatar video: "
        f"avatar={avatar_id}, voice={voice_id}, "
        f"avatar_iv={is_photo_avatar}, text='{text[:50]}...'"
    )
    logger.debug(f"[heygen] Full payload: {json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{HEYGEN_BASE_URL}/v2/video/generate",
            headers=_headers(),
            json=payload,
        )
        if resp.status_code != 200:
            error_body = resp.text
            logger.error(
                f"[heygen] Video generation failed ({resp.status_code}): "
                f"{error_body[:500]}"
            )
            raise RuntimeError(
                f"HeyGen {resp.status_code}: {error_body[:200]}"
            )
        data = resp.json()

    video_id = data.get("data", {}).get("video_id", "")
    if not video_id:
        raise RuntimeError(f"[heygen] No video_id in response: {data}")

    logger.info(f"[heygen] Video submitted: {video_id}")
    return {"video_id": video_id, "status": "processing"}


async def poll_video_status(
    video_id: str,
    max_wait: int = 1800,
    poll_interval: int = 10,
) -> Dict[str, Any]:
    """
    Poll HeyGen for video completion.
    Returns dict with video_url when ready.
    """
    elapsed = 0

    async with httpx.AsyncClient(timeout=15) as client:
        while elapsed < max_wait:
            resp = await client.get(
                f"{HEYGEN_BASE_URL}/v1/video_status.get",
                headers=_headers(),
                params={"video_id": video_id},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            status = data.get("status", "")

            if status == "completed":
                video_url = data.get("video_url", "")
                duration = data.get("duration", 0)
                logger.info(
                    f"[heygen] Video ready: {video_id} "
                    f"({duration}s, {video_url[:60]}...)"
                )
                return {
                    "video_id": video_id,
                    "status": "completed",
                    "video_url": video_url,
                    "duration_s": duration,
                }
            elif status == "failed":
                error = data.get("error", {}).get("message", "Unknown")
                raise RuntimeError(
                    f"[heygen] Video generation failed: {error}"
                )

            time.sleep(poll_interval)
            elapsed += poll_interval

    raise TimeoutError(
        f"[heygen] Video {video_id} timed out after {max_wait}s"
    )


async def download_avatar_video(
    video_url: str,
    segment_id: str,
    text_hash: str = "",
) -> Optional[str]:
    """Download a completed avatar video to local cache."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Include text hash in filename so cache matches on content, not just segment ID
    hash_part = f"_{text_hash}" if text_hash else ""
    filename = f"heygen_{segment_id}{hash_part}_{int(time.time())}.mp4"
    filepath = DOWNLOAD_DIR / filename

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)

    logger.info(
        f"[heygen] Downloaded: {filepath} "
        f"({len(resp.content) / 1024 / 1024:.1f} MB)"
    )
    return str(filepath)


async def generate_and_download(
    text: str,
    segment_id: str,
    avatar_id: Optional[str] = None,
    voice_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full flow: generate avatar video, poll until done, download.
    Returns dict with file_path, duration_s, cost estimate.
    """
    result = await generate_avatar_video(
        text=text,
        avatar_id=avatar_id,
        voice_id=voice_id,
    )
    video_id = result["video_id"]

    status = await poll_video_status(video_id)
    video_url = status["video_url"]
    duration = status.get("duration_s", 0)

    # Include text hash in filename for content-based caching
    import hashlib
    text_hash = hashlib.md5(text.strip().encode()).hexdigest()[:10]
    file_path = await download_avatar_video(video_url, segment_id, text_hash)

    # HeyGen costs ~1 credit per minute (standard)
    cost_estimate = max(duration / 60.0, 0.02)

    return {
        "file_path": file_path,
        "video_url": video_url,
        "duration_s": duration,
        "cost_usd": cost_estimate,
        "video_id": video_id,
    }

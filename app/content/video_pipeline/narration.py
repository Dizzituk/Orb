# FILE: app/content/video_pipeline/narration.py
"""
Narration Generator — TTS audio for b-roll segments.

Uses HeyGen's Starfish TTS API with the same voice_id as the
avatar (Alba) so the voice is consistent across the whole video.
Avatar segments are SKIPPED — HeyGen embeds audio in those.

Features:
- TTS cache: same script text → instant TTS, zero credits
- 5 retries with exponential backoff
- Aborts pipeline if TTS fails (wrong voice = broken video)
"""
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from app.content.video_pipeline.models import ScenePlan

logger = logging.getLogger(__name__)


async def generate_narration(
    scene_plan: ScenePlan,
    output_dir: str,
    emit: Callable,
) -> dict:
    """
    Generate TTS narration for B-ROLL segments only.

    Returns:
        narration_map: {segment_id: audio_file_path}
    """
    import httpx

    heygen_key = os.getenv("HEYGEN_API_KEY", "")
    voice_id = os.getenv("HEYGEN_VOICE_ID", "")

    if not voice_id:
        try:
            from app.db import SessionLocal
            from app.settings.service import get_key_value
            with SessionLocal() as _db:
                voice_id = get_key_value(_db, "heygen_voice_id") or ""
        except Exception:
            pass

    use_heygen_tts = bool(heygen_key and voice_id)
    if use_heygen_tts:
        logger.info(f"[narration] Using HeyGen Starfish TTS (voice_id={voice_id[:8]}...)")
    else:
        logger.warning("[narration] HeyGen TTS unavailable")

    narration_map = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts_cache_dir = Path("data/content/video_pipeline/tts_cache")
    tts_cache_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped_avatar = 0
    cached_hits = 0

    for i, segment in enumerate(scene_plan.segments):
        if not segment.script_text.strip():
            continue

        if segment.requires_avatar:
            skipped_avatar += 1
            continue

        output_path = out_dir / f"{segment.segment_id}.mp3"

        # TTS cache check
        _cache_key = hashlib.sha256(
            f"{segment.script_text}|{voice_id}".encode()
        ).hexdigest()[:16]
        _cache_path = tts_cache_dir / f"tts_{_cache_key}.mp3"

        if _cache_path.exists() and _cache_path.stat().st_size > 100:
            shutil.copy2(str(_cache_path), str(output_path))
            narration_map[segment.segment_id] = str(output_path)
            generated += 1
            cached_hits += 1
            logger.info(f"[narration] TTS CACHED for {segment.segment_id}")
            continue

        success = False
        max_retries = 5

        if use_heygen_tts:
            import asyncio as _aio
            for _attempt in range(1, max_retries + 1):
                if success:
                    break
                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        resp = await client.post(
                            "https://api.heygen.com/v1/audio/text_to_speech",
                            headers={
                                "X-Api-Key": heygen_key,
                                "Content-Type": "application/json",
                            },
                            json={
                                "text": segment.script_text,
                                "voice_id": voice_id,
                            },
                        )
                        if resp.status_code == 200:
                            tts_data = resp.json()
                            audio_url = tts_data.get("data", {}).get("audio_url", "")
                            if audio_url:
                                audio_resp = await client.get(audio_url)
                                if audio_resp.status_code == 200:
                                    output_path.write_bytes(audio_resp.content)
                                    narration_map[segment.segment_id] = str(output_path)
                                    generated += 1
                                    success = True
                                    # Save to cache
                                    try:
                                        shutil.copy2(str(output_path), str(_cache_path))
                                    except Exception:
                                        pass
                        else:
                            logger.warning(
                                f"[narration] HeyGen TTS {segment.segment_id}: "
                                f"{resp.status_code} (attempt {_attempt}/{max_retries})"
                            )
                except Exception as e:
                    logger.warning(
                        f"[narration] HeyGen TTS failed: {e} "
                        f"(attempt {_attempt}/{max_retries})"
                    )
                if not success and _attempt < max_retries:
                    await _aio.sleep(2 * _attempt)

        if not success:
            error_msg = (
                f"HeyGen TTS FAILED after {max_retries} attempts "
                f"for {segment.segment_id}. Pipeline aborted."
            )
            logger.error(f"[narration] {error_msg}")
            raise RuntimeError(error_msg)

        if generated % 5 == 0 and generated > 0:
            await emit(
                "narration", "progress",
                f"Generated {generated} narrations "
                f"({skipped_avatar} avatar segments skipped)",
            )

    logger.info(
        f"[narration] Complete: {generated} TTS generated "
        f"({cached_hits} from cache), "
        f"{skipped_avatar} avatar segments skipped"
    )
    return narration_map


async def generate_narration_single(
    segment,
    output_dir: str,
) -> Optional[str]:
    """Generate TTS for a single segment (used for split segments)."""
    import httpx

    if not segment.script_text.strip():
        return None
    if segment.requires_avatar:
        return None

    heygen_key = os.getenv("HEYGEN_API_KEY", "")
    voice_id = os.getenv("HEYGEN_VOICE_ID", "")
    if not voice_id:
        try:
            from app.db import SessionLocal
            from app.settings.service import get_key_value
            with SessionLocal() as _db:
                voice_id = get_key_value(_db, "heygen_voice_id") or ""
        except Exception:
            pass

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{segment.segment_id}.mp3"

    if heygen_key and voice_id:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.heygen.com/v1/audio/text_to_speech",
                    headers={
                        "X-Api-Key": heygen_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": segment.script_text,
                        "voice_id": voice_id,
                    },
                )
                if resp.status_code == 200:
                    tts_data = resp.json()
                    audio_url = tts_data.get("data", {}).get("audio_url", "")
                    if audio_url:
                        audio_resp = await client.get(audio_url)
                        if audio_resp.status_code == 200:
                            output_path.write_bytes(audio_resp.content)
                            return str(output_path)
        except Exception as e:
            logger.warning(f"[narration] TTS failed for {segment.segment_id}: {e}")

    return None

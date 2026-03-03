# FILE: app/briefing/briefing_audio.py
"""
Briefing Audio — Dual-voice TTS generation for podcast-style audio.

Takes the audio script from a CompiledBriefing and generates an MP3
file using two alternating Google Cloud TTS voices:
  - Voice A (headlines): Reads headlines, section headers, transitions
  - Voice B (analysis):  Reads summaries, context, analysis

Inserts configurable pauses between segments to create a natural
podcast rhythm.

Uses the existing TTS server (app.voice.tts_server) via HTTP,
or falls back to direct Google API calls.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.briefing.briefing_compiler import AudioSegment, CompiledBriefing
from app.briefing.briefing_config import get_voice_config

logger = logging.getLogger(__name__)


# =========================================================================
# Audio output directory
# =========================================================================

AUDIO_OUTPUT_DIR = os.getenv(
    "BRIEFING_AUDIO_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "briefings"),
)


def _ensure_output_dir() -> Path:
    """Create the briefing audio output directory if needed."""
    path = Path(AUDIO_OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================================================================
# Silence generation
# =========================================================================

def _generate_silence_mp3(duration_ms: int) -> bytes:
    """Generate a silent MP3 frame of approximately the given duration.

    This is a minimal approach — generates a very short silent MP3 frame.
    For exact durations, a proper audio library would be needed, but this
    keeps dependencies minimal.
    """
    if duration_ms <= 0:
        return b""
    # Minimal valid MP3 frame (128kbps, 44100Hz, mono, ~26ms per frame)
    # MPEG1 Layer3, 128kbps, 44100Hz, mono
    silent_frame = (
        b"\xff\xfb\x90\x00"  # MP3 sync + header
        + b"\x00" * 413      # Padding (one frame at 128kbps/44100Hz)
    )
    # Repeat frame to approximate duration
    frames_needed = max(1, duration_ms // 26)
    return silent_frame * frames_needed


# =========================================================================
# TTS synthesis
# =========================================================================

async def _synthesize_segment(
    text: str,
    voice_name: str,
    speed: float = 1.0,
) -> bytes:
    """
    Synthesise a text segment via the TTS server or direct Google API.

    Tries the local TTS server first (localhost:8001), falls back to
    direct Google Cloud TTS API call.
    """
    if not text or not text.strip():
        return b""

    # Try local TTS server first
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://localhost:8001/tts/speak",
                json={"text": text, "voice": voice_name, "speed": speed},
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning("[briefing_audio] TTS server returned %d", resp.status_code)
    except Exception as e:
        logger.debug("[briefing_audio] TTS server not available: %s", e)

    # Fallback: direct Google API call
    try:
        import httpx
        import base64
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            logger.error("[briefing_audio] No GOOGLE_API_KEY for direct TTS")
            return b""

        parts = voice_name.split("-")
        lang_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-GB"

        payload = {
            "input": {"text": text},
            "voice": {"languageCode": lang_code, "name": voice_name},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": speed},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}",
                json=payload,
            )
        if resp.status_code != 200:
            logger.error("[briefing_audio] Google TTS API error: %d", resp.status_code)
            return b""

        audio_b64 = resp.json().get("audioContent", "")
        return base64.b64decode(audio_b64) if audio_b64 else b""

    except Exception as e:
        logger.error("[briefing_audio] Direct TTS failed: %s", e)
        return b""


# =========================================================================
# Full briefing audio generation
# =========================================================================

async def generate_briefing_audio(
    briefing: CompiledBriefing,
) -> Optional[str]:
    """
    Generate a complete podcast-style MP3 from a compiled briefing.

    Processes each audio segment, alternating between the two configured
    voices, with pauses between segments.

    Returns:
        Path to the generated MP3 file, or None on failure.
    """
    voice_config = get_voice_config()
    segments = briefing.audio_script
    if not segments:
        logger.warning("[briefing_audio] No audio segments to process")
        return None

    output_dir = _ensure_output_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"briefing_{timestamp}.mp3"

    logger.info(
        "[briefing_audio] Generating audio: %d segments, voices=%s/%s",
        len(segments), voice_config.voice_headlines, voice_config.voice_analysis,
    )

    audio_chunks: list[bytes] = []
    errors = 0

    for i, segment in enumerate(segments):
        # Select voice based on role
        if segment.voice_role == "headlines":
            voice = voice_config.voice_headlines
        else:
            voice = voice_config.voice_analysis

        # Synthesise
        try:
            chunk = await _synthesize_segment(
                text=segment.text,
                voice_name=voice,
                speed=voice_config.speed,
            )
            if chunk:
                audio_chunks.append(chunk)
            else:
                errors += 1
        except Exception as e:
            logger.warning("[briefing_audio] Segment %d failed: %s", i, e)
            errors += 1

        # Add pause after segment
        if segment.pause_after_ms > 0:
            silence = _generate_silence_mp3(segment.pause_after_ms)
            if silence:
                audio_chunks.append(silence)

    if not audio_chunks:
        logger.error("[briefing_audio] No audio generated (all segments failed)")
        return None

    # Concatenate all chunks into single MP3
    combined = b"".join(audio_chunks)
    output_path.write_bytes(combined)

    logger.info(
        "[briefing_audio] Audio generated: %s (%.1f KB, %d segments, %d errors)",
        output_path.name, len(combined) / 1024, len(segments), errors,
    )

    return str(output_path)


__all__ = [
    "generate_briefing_audio",
    "AUDIO_OUTPUT_DIR",
]

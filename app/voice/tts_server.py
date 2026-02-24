# FILE: app/voice/tts_server.py
"""
Google Cloud TTS microservice — FastAPI on port 8001.

Uses the Google Cloud TTS REST API with API key auth (no service account needed).
Serves audio to the Electron frontend via ttsApi.ts.

Endpoints:
  GET  /ping              — Health check
  POST /tts/speak         — Synthesise text → MP3 audio
  POST /tts/preview       — Preview a voice with sample text
  GET  /tts/voices        — List available voices + current selection
  POST /tts/voices/select — Set the active voice

v1.0 (2026-02-24): Initial implementation.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass  # dotenv not required if env vars set externally

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

TTS_BUILD_ID = "2026-02-24-v1.0-google-cloud-tts"

# ── Config ──────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICES_API_URL = "https://texttospeech.googleapis.com/v1/voices"

# Default voice — British English, male, WaveNet quality
DEFAULT_VOICE = "en-GB-WaveNet-B"
DEFAULT_LANGUAGE = "en-GB"
DEFAULT_SPEED = 1.0

# Supported language prefixes (en-GB first, then en-US)
SUPPORTED_LANGUAGES = ["en-GB", "en-US"]

# Tier display order (best first)
TIER_ORDER = ["Chirp3-HD", "Chirp-HD", "Studio", "Wavenet", "Neural2", "News", "Standard"]

# Cached voice list from Google API
_voice_cache: list = []
_voice_cache_time: float = 0
VOICE_CACHE_TTL = 3600  # Refresh every hour

# ── State ───────────────────────────────────────────────────────────────

_selected_voice = DEFAULT_VOICE

# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="ASTRA TTS Service", version=TTS_BUILD_ID)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──────────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None


class PreviewRequest(BaseModel):
    voice: str


class SelectVoiceRequest(BaseModel):
    voice: str


# ── Voice catalogue (live from Google API) ──────────────────────────────

def _extract_tier(voice_name: str) -> str:
    """Extract tier from voice ID, e.g. 'en-GB-Chirp3-HD-Achird' → 'Chirp3-HD'."""
    # Remove language prefix (e.g. 'en-GB-')
    parts = voice_name.split("-", 2)
    if len(parts) < 3:
        return "Other"
    remainder = parts[2]  # e.g. 'Chirp3-HD-Achird' or 'Wavenet-B'
    for tier in TIER_ORDER:
        if remainder.startswith(tier):
            return tier
    return "Other"


def _make_display_name(voice_name: str, gender: str) -> str:
    """Create a friendly display name from the voice ID."""
    tier = _extract_tier(voice_name)
    # Extract the letter/name suffix
    parts = voice_name.split("-")
    suffix = parts[-1] if parts else ""
    lang = "-".join(parts[:2]) if len(parts) >= 2 else ""
    region = "British" if lang == "en-GB" else "US" if lang == "en-US" else lang
    gender_label = "Male" if gender == "MALE" else "Female"
    return f"{region} {gender_label} — {tier} {suffix}"


async def _fetch_voices_from_google() -> list:
    """Fetch all supported English voices from Google TTS API."""
    global _voice_cache, _voice_cache_time
    import time

    now = time.time()
    if _voice_cache and (now - _voice_cache_time) < VOICE_CACHE_TTL:
        return _voice_cache

    if not GOOGLE_API_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{VOICES_API_URL}?key={GOOGLE_API_KEY}"
            )
        if resp.status_code != 200:
            logger.error("[tts] Failed to fetch voices: %d", resp.status_code)
            return _voice_cache  # Return stale cache on error

        all_voices = resp.json().get("voices", [])

        catalogue = []
        for v in all_voices:
            langs = v.get("languageCodes", [])
            # Only include supported languages
            matching_lang = None
            for lang in SUPPORTED_LANGUAGES:
                if lang in langs:
                    matching_lang = lang
                    break
            if not matching_lang:
                continue

            voice_id = v["name"]
            gender = v.get("ssmlGender", "NEUTRAL")
            tier = _extract_tier(voice_id)
            display = _make_display_name(voice_id, gender)

            catalogue.append({
                "name": display,
                "voice_id": voice_id,
                "gender": gender,
                "tier": tier,
                "language": matching_lang,
            })

        # Sort: by language (en-GB first), then tier order, then gender, then name
        def sort_key(v):
            lang_idx = SUPPORTED_LANGUAGES.index(v["language"]) if v["language"] in SUPPORTED_LANGUAGES else 99
            tier_idx = TIER_ORDER.index(v["tier"]) if v["tier"] in TIER_ORDER else 99
            return (lang_idx, tier_idx, v["gender"], v["voice_id"])

        catalogue.sort(key=sort_key)
        _voice_cache = catalogue
        _voice_cache_time = now
        logger.info("[tts] Fetched %d voices from Google API", len(catalogue))
        return catalogue

    except Exception as e:
        logger.error("[tts] Voice fetch error: %s", e)
        return _voice_cache


# ── Google TTS call ─────────────────────────────────────────────────────

async def _synthesize(text: str, voice_name: str, speed: float = 1.0) -> bytes:
    """Call Google Cloud TTS REST API and return MP3 bytes."""
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY not set in environment")

    # Determine language code from voice name (e.g. "en-GB-WaveNet-B" → "en-GB")
    parts = voice_name.split("-")
    lang_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else DEFAULT_LANGUAGE

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": lang_code,
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": max(0.25, min(4.0, speed)),
        },
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{TTS_API_URL}?key={GOOGLE_API_KEY}",
            json=payload,
        )

    if resp.status_code != 200:
        detail = resp.text[:300]
        logger.error("[tts] Google API error %d: %s", resp.status_code, detail)
        raise HTTPException(resp.status_code, f"Google TTS error: {detail}")

    audio_b64 = resp.json().get("audioContent", "")
    if not audio_b64:
        raise HTTPException(500, "Google TTS returned empty audio")

    return base64.b64decode(audio_b64)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "tts", "build": TTS_BUILD_ID}


@app.post("/tts/speak")
async def speak(req: SpeakRequest):
    """Synthesise text to speech. Returns MP3 audio."""
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    voice = req.voice or _selected_voice
    speed = req.speed if req.speed is not None else DEFAULT_SPEED

    logger.info("[tts] Speak: %d chars, voice=%s, speed=%.1f", len(req.text), voice, speed)
    audio = await _synthesize(req.text, voice, speed)

    return Response(content=audio, media_type="audio/mpeg")


@app.post("/tts/preview")
async def preview(req: PreviewRequest):
    """Preview a voice with sample text."""
    sample = "Hello, I'm ASTRA. This is how I sound with this voice."
    audio = await _synthesize(sample, req.voice, DEFAULT_SPEED)
    return Response(content=audio, media_type="audio/mpeg")


@app.get("/tts/voices")
async def list_voices():
    """List available voices and current selection."""
    catalogue = await _fetch_voices_from_google()
    voices = []
    for v in catalogue:
        voices.append({
            "name": v["name"],
            "voice_id": v["voice_id"],
            "gender": v["gender"],
            "tier": v["tier"],
            "selected": v["voice_id"] == _selected_voice,
        })
    return {"voices": voices, "selected": _selected_voice}


@app.post("/tts/voices/select")
async def select_voice(req: SelectVoiceRequest):
    """Set the active voice."""
    global _selected_voice

    # Accept any voice ID — Google will validate it on speak
    _selected_voice = req.voice
    logger.info("[tts] Voice selected: %s", _selected_voice)
    return {"selected": _selected_voice}


# ── Standalone entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
